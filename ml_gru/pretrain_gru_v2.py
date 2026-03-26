"""
版本 V2：GRU Pretrain（修正過擬合版）
========================================
相對 V1 的改動：
  1. hidden_size 128 → 64（縮小容量，對齊個人資料集規模）
     理由：V1 模型有 160K 參數，但只有 4067 筆訓練資料（參數/樣本比 1:25）
           縮回 64 讓比例回到 1:60，更適合 16 人的小型資料集
  2. Dropout 0.3 → 0.4（加強正則化）
     理由：V1 最佳 epoch 在第 5 輪就發生，代表正則化嚴重不足
  3. Weight decay 1e-4 → 5e-4（加強 L2 正則）
  4. FC Head 縮小：128→64→1 改為 64→32→1
     理由：容量一致縮小，避免深層 FC 在小資料集中記憶 noise

  保留 V1 的改進：
  - Temporal Attention（已證實改善 SMAPE）
  - LayerNorm
  - HuberLoss
  - AdamW

  注意：V2 pretrain 也供 V3、V4 使用（相同架構）
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle

ARTIFACTS_DIR = "artificats"
INPUT_SIZE    = 7
HIDDEN_SIZE   = 64       # V2: 128 → 64
NUM_LAYERS    = 2
DROPOUT       = 0.4      # V2: 0.3 → 0.4
OUTPUT_SIZE   = 1
BATCH_SIZE    = 64
EPOCHS        = 100
LEARNING_RATE = 0.001
PATIENCE      = 15
WEIGHT_DECAY  = 5e-4     # V2: 1e-4 → 5e-4
HUBER_DELTA   = 1.0

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
print(f"  X_train : {X_train.shape}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=BATCH_SIZE, shuffle=False)


class GRUWithAttention(nn.Module):
    """V2 架構：hidden=64, Attention, FC(64→32→1), Dropout=0.4"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0,
                          batch_first=True)
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)   # 64→32
        self.fc2 = nn.Linear(hidden_size // 2, output_size)   # 32→1
        self.relu = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        context    = self.layer_norm(context)
        out        = self.dropout(context)
        out        = self.relu(self.fc1(out))
        return self.fc2(out)


model        = GRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n🏗️  V2 模型參數量 : {total_params:,} 個（vs V1: 160,386）")

criterion = nn.HuberLoss(delta=HUBER_DELTA)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

print("\n🚀 開始 V2 Pretrain（hidden=64, Attention, Dropout=0.4）...")
print("=" * 65)

best_val_loss    = float("inf")
patience_counter = 0
best_model_path  = f"{ARTIFACTS_DIR}/pretrain_gru_v2.pth"
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
            "version"    : "v2",
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
print(f"\n🎉 V2 Pretrain 完成！最佳 Val Loss：{best_val_loss:.6f}")

with open(f"{ARTIFACTS_DIR}/pretrain_history_v2.pkl", "wb") as f:
    pickle.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

print("\n下一步：執行 finetune_gru_v2.py")
