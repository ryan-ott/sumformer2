import numpy as np
import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from .modules import *

class Sumformer(nn.Module):
    """Text summarization transformer."""
    def __init__(self, device, emb_dim, vocab_size, max_len, pos_encoding, GLU, enc_heads, enc_hidden, enc_depth, enc_dropout, dec_heads, dec_hidden, dec_depth, dec_dropout):
        super().__init__()

        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        # self.pos_encoding = pos_encoding

        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim).to(device)
        if pos_encoding:
            self.pos_embedding = self.compute_encodings(max_len, emb_dim).to(device)
        else:
            self.pos_embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=emb_dim).to(device)

        self.encoder = [EncoderBlock(emb_dim, enc_heads, enc_hidden, enc_dropout, GLU) for _ in range(enc_depth)]
        for module in self.encoder: module.to(device)
        self.decoder = [DecoderBlock(emb_dim, dec_heads, dec_hidden, dec_dropout, GLU) for _ in range(dec_depth)]
        for module in self.decoder: module.to(device)

        self.toProbs = nn.Linear(emb_dim, vocab_size).to(device)  # convert to probabilities over vocab

    def compute_encodings(self, max_len, emb_dim):
        """Initialise positional encodings."""
        pos_encodings = torch.zeros(max_len, emb_dim).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(np.log(10000.0) / emb_dim))

        pos_encodings[:, 0::2] = torch.sin(position * div_term)
        pos_encodings[:, 1::2] = torch.cos(position * div_term)

        return nn.Embedding.from_pretrained(pos_encodings, freeze=True)

    def encode(self, source, source_mask=None):
        tokens_source = self.token_embedding(source.to(self.device))
        b, t_s, k = tokens_source.size()
        positions_source = self.pos_embedding(torch.arange(t_s, device=self.device))[None, :, :].expand(b, t_s, k)
        x = tokens_source + positions_source
        for enc_layer in self.encoder:
            x = checkpoint(enc_layer, x, source_mask)
        return x
    
    def decode(self, target, context, target_mask=None):
        tokens_target = self.token_embedding(target.to(self.device))
        b, t_t, k = tokens_target.size()
        positions_target = self.pos_embedding(torch.arange(t_t, device=self.device))[None, :, :].expand(b, t_t, k)
        y = tokens_target + positions_target
        for dec_layer in self.decoder:
            y = checkpoint(dec_layer, y, context, target_mask)
        return self.toProbs(y)

    def forward(self, source, target, source_mask=None, target_mask=None):
        context = self.encode(source, source_mask)
        return self.decode(target, context, target_mask)
    
    def greedy(self, source, start_token=0, end_token=1, max_len=256, source_mask=None, logits=False):
        self.eval()
        logit_list = []

        context = self.encode(source, source_mask)

        # Initialize the generated sequence with the start token
        generated = torch.full((source.size(0), 1), start_token, dtype=torch.long, device=self.device)
        
        for _ in range(max_len):
            output = self.decode(generated, context)

            next_token_logits = output[:, -1, :]  # (b, t, vocab_size) -> (b, 1, vocab_size) (take the last token of the sequence)

            # Select the token with the highest probability
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Concatenate the token to the sequence
            generated = torch.cat((generated, next_token), dim=-1)

            logit_list.append(next_token_logits)

            # Stop if the end token is generated
            if torch.eq(next_token, end_token).all():
                break
        
        if logits:
            return generated, torch.stack(logit_list, dim=1)
        else:
            return generated


    def beam(self, source, start_token=0, end_token=1, max_len=256, source_mask=None, num_beams=3, length_penalty=0.6):
        self.eval()
        with torch.no_grad():
            generated = torch.full((num_beams, 1), start_token, dtype=torch.long, device=self.device)
            scores = torch.zeros((num_beams,), dtype=torch.float, device=self.device)

            for _ in range(max_len):
                output = self.forward(source.repeat(num_beams, 1), generated, source_mask=source_mask.repeat(num_beams, 1, 1))
                next_token_logits = output[:, -1, :]  # (b, t, vocab_size) -> (b, vocab_size)

                # Apply a softmax to convert the logits into probabilities
                probs = F.softmax(next_token_logits, dim=-1)  # (b, vocab_size)
                
                # Multiply the probabilities by the scores and find the top num_beams sequences
                # Apply length penalty: the longer the sentence, the smaller the score.
                scores = scores.view(-1, 1) * probs / (generated.size(-1)**length_penalty)  # (b, vocab_size)
                scores, indices = scores.view(-1).topk(num_beams)

                # Convert flat indices to actual token indices
                next_tokens = indices % probs.size(-1)  # tokens
                beam_indices = indices // probs.size(-1)  # beam indices
                
                # Add the most probable tokens to the sequence
                generated = torch.cat([generated[beam_indices], next_tokens.unsqueeze(-1)], dim=-1)

                # Stop when end_token is generated in all num_beams sequences
                if all(next_token == end_token for next_token in next_tokens):
                    break

        return generated[0]  # return the first beam
