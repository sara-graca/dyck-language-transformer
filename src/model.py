import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=80):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class DyckTransformer(nn.Module):
    def __init__(self, vocab_size, pad_idx, num_edit_labels, d_model=128, n_heads=4, n_layers=2, max_len=80, dropout=0.1):
        super().__init__()

        self.pad_idx = pad_idx

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(d_model, 2)               
        self.token_head = nn.Linear(d_model, num_edit_labels)    

    def forward(self, x):
        pad_mask = (x == self.pad_idx)

        x = self.embed(x)
        x = self.pos(x)

        h = self.encoder(x, src_key_padding_mask=pad_mask)

        cls_repr = self.dropout(h[:, 0, :])
        cls_logits = self.cls_head(cls_repr)                     
        token_logits = self.token_head(self.dropout(h))             

        return cls_logits, token_logits                       