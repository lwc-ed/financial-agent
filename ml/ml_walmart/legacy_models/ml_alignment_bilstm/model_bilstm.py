"""
model_bilstm.py
===============
Bi-LSTM + Attention 架構
輸入 : 7 個 aligned rolling z-score features
輸出 : 未來 7 天消費加總（scaled）
"""

import torch
import torch.nn as nn


class BiLSTMWithAttention(nn.Module):
    """
    雙向 LSTM + Temporal Attention + LayerNorm
    相比 GRU 的改進：
      - bidirectional=True → 30天窗口內正反向都看，資訊更完整
      - output dim = hidden_size * 2（正向 + 反向拼接）
    """
    def __init__(self, input_size=7, hidden_size=64,
                 num_layers=2, output_size=1, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        bi_hidden = hidden_size * 2            # 128
        self.attention  = nn.Linear(bi_hidden, 1)
        self.layer_norm = nn.LayerNorm(bi_hidden)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(bi_hidden, hidden_size)   # 128 → 64
        self.fc2        = nn.Linear(hidden_size, output_size) # 64  → 1
        self.relu       = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                            # (B, T, 128)
        attn_w      = torch.softmax(self.attention(lstm_out), dim=1)
        context     = (lstm_out * attn_w).sum(dim=1)          # (B, 128)
        context     = self.layer_norm(context)
        out         = self.dropout(context)
        out         = self.relu(self.fc1(out))
        return self.fc2(out)
