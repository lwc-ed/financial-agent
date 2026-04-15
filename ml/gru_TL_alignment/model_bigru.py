"""
model_bigru.py
==============
BiGRU + Attention 架構
輸入 : 10 個 aligned rolling z-score features
輸出 : 未來 7 天消費加總（scaled）

vs 單向 GRU：
  - bidirectional=True → 30 天窗口正反向都看，資訊更完整
  - output dim = hidden_size * 2（正向 + 反向拼接）

vs BiLSTM：
  - GRU 每個 cell 3 個 gate，LSTM 有 4 個
  - 參數量少約 25%，較不容易 overfit（個人資料量少時有優勢）
"""

import torch
import torch.nn as nn


class BiGRUWithAttention(nn.Module):
    """
    雙向 GRU + Temporal Attention + LayerNorm

    預設 hidden_size=48 而非 64，因為 bidirectional 後輸出是 hidden_size*2=96，
    與原本單向 GRU hidden_size=64 的表達力相近，但參數量更小。
    """
    def __init__(self, input_size=10, hidden_size=48,
                 num_layers=2, output_size=1, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        bi_hidden = hidden_size * 2          # 96
        self.attention  = nn.Linear(bi_hidden, 1)
        self.layer_norm = nn.LayerNorm(bi_hidden)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(bi_hidden, hidden_size)    # 96 → 48
        self.fc2        = nn.Linear(hidden_size, output_size)  # 48 → 1
        self.relu       = nn.ReLU()

    def encode(self, x) -> torch.Tensor:
        """回傳 attended hidden representation（用於 MMD）"""
        gru_out, _ = self.gru(x)                              # (B, T, 96)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)            # (B, 96)
        return self.layer_norm(context)

    def forward(self, x):
        context = self.encode(x)
        out     = self.dropout(context)
        out     = self.relu(self.fc1(out))
        return self.fc2(out)
