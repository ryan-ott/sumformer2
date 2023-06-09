import torch
import torch.nn as nn

from modules import *

class Sumformer(nn.Module):
    """Text summarization transformer."""
    def __init__(self, device, emb_dim, vocab_size, max_len, enc_heads, enc_hidden, enc_depth, enc_dropout, dec_heads, dec_hidden, dec_depth, dec_dropout):
        super().__init__()

        self.device = device

        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim).to(device)
        self.pos_embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=emb_dim).to(device)

        self.encoder = [EncoderBlock(emb_dim, enc_heads, enc_hidden, enc_dropout) for _ in range(enc_depth)]
        for module in self.encoder: module.to(device)
        self.decoder = [DecoderBlock(emb_dim, dec_heads, dec_hidden, dec_dropout) for _ in range(dec_depth)]
        for module in self.decoder: module.to(device)

        self.toProbs = nn.Linear(emb_dim, vocab_size).to(device)  # convert to probabilities over vocab


    def forward(self, source, target, source_mask=None, target_mask=None):
        tokens_source = self.token_embedding(source.to(self.device))
        tokens_target = self.token_embedding(target.to(self.device))

        b, t_s, k = tokens_source.size()
        _, t_t, _ = tokens_target.size()

        positions_source = self.pos_embedding(torch.arange(t_s, device=self.device))[None, :, :].expand(b, t_s, k)
        positions_target = self.pos_embedding(torch.arange(t_t, device=self.device))[None, :, :].expand(b, t_t, k)

        x = tokens_source + positions_source
        for layer in self.encoder:
            x = layer(x, source_mask)
        context = x

        y = tokens_target + positions_target
        for layer in self.decoder:
            y = layer(y, context, target_mask)

        return self.toProbs(y)
    
    def generate(self, source, source_mask, max_len=512):
        pass  # TODO