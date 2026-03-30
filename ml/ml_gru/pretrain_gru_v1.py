"""
版本 V1：GRU Pretrain（增強版）
================================
改進點：
  1. 模型容量：hidden_size 64 → 128
  2. 加入 Temporal Attention Mechanism（替代直接取最後 hidden state）
  3. 更深 FC Head：128 → 64 → 1 with ReLU + LayerNorm
  4. 損失函數：MSELoss → Huber Loss（對離群值更穩健）
  5. AdamW + Weight Decay 1e-4
  6. 更長訓練（100 epochs, patience=15）
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os

ARTIFACTS_DIR = "artificats"
INPUT_SIZE    = 7
HIDDEN_SIZE   = 128      # V1: 64 → 128
NUM_LAYERS    = 2
DROPOUT       = 0.3      # V1: 0.2 → 0.3
OUTPUT_SIZE   = 1
BATCH_SIZE    = 64
EPOCHS        = 100      # V1: 50 → 100
LEARNING_RATE = 0.001
PATIENCE      = 15       # V1: 10 → 15
WEIGHT_DECAY  = 1e-4     # V1: 新增
HUBER_DELTA   = 1.0      # V1: 新增 Huber Loss delta

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple M1 MPS 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("⚠️  使用 CPU")

print("\n📂 載入資料...")
X_train = np.load(f"{ARTIFACTS_DIR}/walmart_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/walmart_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/walmart_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/walmart_y_val.npy")

print(f"  X_train : {X_train.shape}  → (樣本數, 30天, 7特徵)")
print(f"  y_train : {y_train.shape}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=BATCH_SIZE, shuffle=False)


class GRUWithAttention(nn.Module):
    """
    V1 架構：GRU + Temporal Attention + 深層 FC Head

    改進說明：
    - Temporal Attention: 讓模型學習對 30 天序列中不同時間點給予不同權重
      而非只看最後一個 hidden state，可以更好地捕捉消費模式的關鍵節點
    - LayerNorm: 穩定訓練，取代部分 Dropout 的正則化角色
    - 深層 FC Head: hidden → hidden//2 → 1，增加非線性表達能力
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        # Temporal Attention
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        # Deeper FC Head
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)                          # (batch, seq, hidden)
        attn_w     = torch.softmax(
            self.attention(gru_out), dim=1
        )                                                  # (batch, seq, 1)
        context    = (gru_out * attn_w).sum(dim=1)        # (batch, hidden)
        context    = self.layer_norm(context)
        out        = self.dropout(context)
        out        = self.relu(self.fc1(out))
        return self.fc2(out)


model = GRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n🏗️  V1 模型參數量 : {total_params:,} 個")

# V1: Huber Loss（對離群消費更穩健）
criterion = nn.HuberLoss(delta=HUBER_DELTA)
# V1: AdamW + Weight Decay
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

print("\n🚀 開始 V1 Pretrain（hidden=128, Attention, Huber Loss）...")
print("=" * 65)

best_val_loss    = float("inf")
patience_counter = 0
best_model_path  = f"{ARTIFACTS_DIR}/pretrain_gru_v1.pth"
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            val_loss += criterion(model(X_b), y_b).item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    lr = optimizer.param_groups[0]["lr"]
    print(f"  Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            "epoch"      : epoch,
            "model_state": model.state_dict(),
            "val_loss"   : best_val_loss,
            "version"    : "v1",
            "hyperparams": {
                "input_size"  : INPUT_SIZE,
                "hidden_size" : HIDDEN_SIZE,
                "num_layers"  : NUM_LAYERS,
                "output_size" : OUTPUT_SIZE,
                "dropout"     : DROPOUT,
            }
        }, best_model_path)
        print(f"             ✅ 儲存最佳模型（val_loss={best_val_loss:.6f}）")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping！")
            break

print("=" * 65)
print(f"\n🎉 V1 Pretrain 完成！最佳 Val Loss：{best_val_loss:.6f}")

with open(f"{ARTIFACTS_DIR}/pretrain_history_v1.pkl", "wb") as f:
    pickle.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

print("\n下一步：執行 finetune_gru_v1.py")
