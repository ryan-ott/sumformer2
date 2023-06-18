import torch
from torch import nn
from torch.nn import functional as F


class MHSelfAttention(nn.Module):
    """Multi-head self-attention"""

    # k is the dimensionality of the embedding space (len of the input vector)
    def __init__(self, k, heads, mask):
        super().__init__()

        assert k % heads == 0  # embedding dimension must be divisible by number of heads
        self.k, self.heads, self.mask = k, heads, mask  # mask is a flag to use a look-ahead mask

        # computing queries, keys and values in parallel for all heads
        # bias=False so that we can use this as a simple projection
        self.toQueries = nn.Linear(k, k, bias=False)
        self.toKeys = nn.Linear(k, k, bias=False)
        self.toValues = nn.Linear(k, k, bias=False)

        self.unifyHeads = nn.Linear(k, k)  # W0 matrix

    def forward(self, x, padding_mask=None):
        b, t, e = x.size()
        h = self.heads

        assert e == self.k,  f'Input embedding dimension ({e}) should match layer embedding dimension ({self.k})'

        queries = self.toQueries(x)
        keys = self.toKeys(x)
        values = self.toValues(x)

        s = e // h  # s is the dimensionality of the embedding space per head

        # split the embedding space into multiple heads
        queries = queries.view(b, t, h, s)
        keys = keys.view(b, t, h, s)
        values = values.view(b, t, h, s)

        # fold heads into batch dimension so that we can bmm all heads at once
        # first swapping the time and head dimensions, then folding the heads into the batch dimension
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # scaling the queries and keys in place to save memory
        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))  # instead of scaling the dot product

        dot = torch.bmm(queries, keys.transpose(1, 2))  # (b * h, t, t)

        if self.mask:  # look-ahead mask
            # Create a mask to remove the upper half of the dot matrix, excluding the diagonal
            mask = torch.triu(torch.ones(t, t, device=dot.device), diagonal=1)
            # Set the masked positions to float('-inf') to minimize their impact on the softmax operation
            mask = mask.masked_fill(mask == 1, float('-inf'))            
            # Add the mask to the dot product matrix
            dot = dot + mask.unsqueeze(0)

        # print("Dot pre pad mask:", dot.shape, "contains NaNs:", torch.isnan(dot).any(), "\n", dot)

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).expand(-1, h, -1)  # Should be (b, h, t)
            padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, -1, t)  # Should be (b, h, t, t)
            padding_mask = padding_mask.reshape(b * h, t, t)
            # print("Padding mask:", padding_mask.shape, "\n", padding_mask)
            dot = dot.masked_fill(padding_mask == 0, float('-inf'))
        
        # print("Dot post pad mask:", dot.shape, torch.isnan(dot).any(), torch.isinf(dot).any(), "\n", dot)

        # row-wise softmax
        dot = F.softmax(dot, dim=2)  # (b * h, t, t)

        # print("Dot post softmax:", dot.shape, torch.isnan(dot).any(), torch.isinf(dot).any(), "\n", dot)

        # apply the attention weights to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap head and time dimensions back again so that we can concatenate the heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * s)

        # concatenate the heads and return
        return self.unifyHeads(out)


class MHCrossAttention(nn.Module):
    """Multi-head cross-attention"""

    def __init__(self, k, heads):
        super().__init__()

        assert k % heads == 0  # embedding dimension must be divisible by number of heads
        self.k, self.heads = k, heads

        self.toQueries = nn.Linear(k, k, bias=False)
        self.toKeys = nn.Linear(k, k, bias=False)
        self.toValues = nn.Linear(k, k, bias=False)

        self.unifyHeads = nn.Linear(k, k)

    def forward(self, x, context, padding_mask=None):
        b, t, e = x.size()
        _, t_context, _ = context.size()
        h = self.heads

        assert e == self.k, f'Input embedding dim ({e}) should match layer embedding dim ({self.k})'

        queries = self.toQueries(x)
        keys = self.toKeys(context)  # keys and values come from the encoder's latent space representation
        values = self.toValues(context)

        s = e // h  # dimensionality of the embedding space per head

        queries = queries.view(b, t, h, s).transpose(1, 2).contiguous().view(b * h, t, s)
        keys = keys.view(b, t_context, h, s).transpose(1, 2).contiguous().view(b * h, t_context, s)
        values = values.view(b, t_context, h, s).transpose(1, 2).contiguous().view(b * h, t_context, s)

        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1, 2))  # (b * h, t, t_context)

        if padding_mask is not None:  # ! THIS CREATES INF VALUES IN THE DOT PRODUCT - AVOID FOR NOW
            padding_mask = padding_mask.unsqueeze(-1)
            padding_mask = padding_mask.expand(-1, -1, t_context)  # Should be (b, t, t_context)
            dot = dot.masked_fill(padding_mask == 0, float('-inf'))
        
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, s)

        out = out.transpose(1, 2).contiguous().view(b, t, h * s)

        return self.unifyHeads(out)


class EncoderBlock(nn.Module):
    def __init__(self, emb, heads, hidden=4, dropout=0.1, GLU=False):
        super().__init__()

        self.attention = MHSelfAttention(emb, heads, mask=False)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        if GLU:  # * test GLU vs FF
            self.ff = GLUlayer(emb, hidden)
        else:
            self.ff = FFLayer(emb, hidden)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        attended = self.attention(x, padding_mask)
        attended = self.dropout(attended)
        x = self.norm1(x + attended)
        
        fedforward = self.ff(x)
        fedforward = self.dropout(fedforward)
        x = self.norm2(x + fedforward)
        
        return x


class DecoderBlock(nn.Module):
    def __init__(self, emb, heads, hidden=4, dropout=0.1, GLU=False):
        super().__init__()

        self.maskedAttention = MHSelfAttention(emb, heads, mask=True)
        self.crossAttention = MHCrossAttention(emb, heads)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.norm3 = nn.LayerNorm(emb)

        self.dropout = nn.Dropout(dropout)

        if GLU:
            self.ff = GLUlayer(emb, hidden)
        else:
            self.ff = FFLayer(emb, hidden)
        
    
    def forward(self, x, context, padding_mask=None):
        masked_attended = self.maskedAttention(x, padding_mask)
        masked_attended = self.dropout(masked_attended)
        x = self.norm1(x + masked_attended)

        cross_attended = self.crossAttention(x, context, padding_mask)
        cross_attended = self.dropout(cross_attended)
        x = self.norm2(x + cross_attended)

        fedforward = self.ff(x)
        fedforward = self.dropout(fedforward)
        x = self.norm3(x + fedforward)
        
        return x
    

class GLUlayer(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, emb_dim, hidden):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim * hidden * 2)
        self.linear2 = nn.Linear(emb_dim * hidden, emb_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.glu(x, dim=-1)
        return self.linear2(x)
    

class FFLayer(nn.Module):
    """Feed-forward layer"""
    def __init__(self, emb_dim, hidden):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim * hidden)
        self.linear2 = nn.Linear(emb_dim * hidden, emb_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        return self.linear2(x)