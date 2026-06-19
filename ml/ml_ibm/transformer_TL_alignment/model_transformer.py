"""
model_transformer.py
====================
Transformer Encoder + Attention Pooling 架構
輸入 : (batch, seq_len=30, input_size=10)
輸出 : 未來 7 天消費加總（scaled）
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size=10, d_model=64, nhead=4, num_layers=2, output_size=1, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention = nn.Linear(d_model, 1)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_size)
        self.relu = nn.ReLU()

    def encode(self, x) -> torch.Tensor:
        x = self.input_proj(x)           # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)              # (B, T, d_model)
        attn_w = torch.softmax(self.attention(x), dim=1)
        context = (x * attn_w).sum(dim=1)  # (B, d_model)
        return self.layer_norm(context)

    def forward(self, x):
        context = self.encode(x)
        out = self.dropout(context)
        out = self.relu(self.fc1(out))
        return self.fc2(out)
