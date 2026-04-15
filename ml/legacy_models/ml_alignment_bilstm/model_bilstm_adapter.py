"""
model_bilstm_adapter.py
=======================
Bi-LSTM + Input Adapter 架構

Pretrain 階段：input_size=7（z-score aligned features）
Finetune 階段：raw features (5) → Adapter(5→7) → 同一個 BiLSTM body

這樣做的好處：
  - BiLSTM body 完整繼承 Walmart pretrain 的時序先驗知識
  - Finetune 時改用資訊量更豐富的 raw features（含絕對金額）
  - Adapter 是唯一從頭學的部分（只有 5×7=35 個參數）
"""

import torch
import torch.nn as nn


class BiLSTMWithAdapter(nn.Module):
    def __init__(self, raw_input_size=5, pretrain_input_size=7,
                 hidden_size=64, num_layers=2, output_size=1, dropout=0.4):
        super().__init__()

        # ── Input Adapter：把 raw features 投影到 pretrain 的 feature 空間 ──
        self.adapter      = nn.Linear(raw_input_size, pretrain_input_size)
        self.adapter_norm = nn.LayerNorm(pretrain_input_size)
        self.adapter_act  = nn.ReLU()

        # ── Bi-LSTM body（pretrain weights 載入這裡）───────────────────────
        self.lstm = nn.LSTM(
            pretrain_input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        bi_hidden = hidden_size * 2                      # 128
        self.attention  = nn.Linear(bi_hidden, 1)
        self.layer_norm = nn.LayerNorm(bi_hidden)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(bi_hidden, hidden_size)
        self.fc2        = nn.Linear(hidden_size, output_size)
        self.relu       = nn.ReLU()

    def forward(self, x):
        # x: (B, T, 5)  ← raw features
        x = self.adapter_act(self.adapter_norm(self.adapter(x)))  # (B, T, 7)
        lstm_out, _ = self.lstm(x)                                 # (B, T, 128)
        attn_w  = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_w).sum(dim=1)                   # (B, 128)
        context = self.layer_norm(context)
        out     = self.dropout(context)
        out     = self.relu(self.fc1(out))
        return self.fc2(out)

    def load_pretrained_body(self, pretrain_path: str, device):
        """
        從 pretrain checkpoint 載入 BiLSTM body weights
        Adapter 層保持隨機初始化（讓它從 raw features 學習投影方式）
        """
        ckpt           = torch.load(pretrain_path, map_location=device)
        pretrain_state = ckpt["model_state"]
        current_state  = self.state_dict()

        loaded, skipped = [], []
        for key in current_state:
            if key.startswith("adapter"):          # adapter 跳過，保持隨機初始化
                skipped.append(key)
            elif key in pretrain_state:
                current_state[key] = pretrain_state[key]
                loaded.append(key)
            else:
                skipped.append(key)

        self.load_state_dict(current_state)
        print(f"  ✅ 繼承 pretrain weights：{len(loaded)} 個參數組")
        print(f"  🔀 隨機初始化（adapter）：{len(skipped)} 個參數組")
