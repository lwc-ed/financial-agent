"""
model_bigru.py
==============
BiGRU + Attention 架構
輸入 : 10 個 aligned features
輸出 : 未來 7 天消費加總（scaled）
"""

import torch
import torch.nn as nn


class BiGRUWithAttention(nn.Module):
    def __init__(self, input_size=10, hidden_size=48, num_layers=2, output_size=1, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        bi_hidden = hidden_size * 2
        self.attention = nn.Linear(bi_hidden, 1)
        self.layer_norm = nn.LayerNorm(bi_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(bi_hidden, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def encode(self, x) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        attn_w = torch.softmax(self.attention(gru_out), dim=1)
        context = (gru_out * attn_w).sum(dim=1)
        return self.layer_norm(context)

    def forward(self, x):
        context = self.encode(x)
        out = self.dropout(context)
        out = self.relu(self.fc1(out))
        return self.fc2(out)
