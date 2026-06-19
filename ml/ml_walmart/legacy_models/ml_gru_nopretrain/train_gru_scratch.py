"""
ml_gru_nopretrain/train_gru_scratch.py
=======================================
GRU 從頭訓練（完全無 Walmart pretrain）。

目的：作為對照組，驗證「Walmart pretrain 到底是幫助還是拖累」。
  - 如果 GRU scratch 接近或優於原版 pretrain GRU (Test MAE ~5,773)：
    → Walmart pretrain 沒有幫助（甚至有害），應放棄 transfer learning 策略
  - 如果 GRU scratch 明顯更差：
    → Walmart pretrain 確實帶來了一些通用序列特徵

架構：與 ml_gru V2 相同（hidden=64, layers=2, Attention, Dropout=0.4）
差異：直接用 random init 訓練，不載入任何 pretrain 權重
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import json
from datetime import datetime

ARTIFACTS_DIR = "artifacts"
INPUT_SIZE    = 7
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
OUTPUT_SIZE   = 1
DROPOUT       = 0.4
BATCH_SIZE    = 16
EPOCHS        = 200
PATIENCE      = 30
LR            = 1e-3
WEIGHT_DECAY  = 5e-4

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple MPS 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("⚠️  使用 CPU")


class GRUWithAttention(nn.Module):
    """與 ml_gru V2 相同架構，但從頭訓練"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, num_layers,
                                 dropout=dropout if num_layers > 1 else 0,
                                 batch_first=True)
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2        = nn.Linear(hidden_size // 2, output_size)
        self.relu       = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        context    = self.layer_norm(context)
        out        = self.relu(self.fc1(self.dropout(context)))
        return self.fc2(out)


# ─────────────────────────────────────────
# 載入資料
# ─────────────────────────────────────────
print("\n📂 載入序列資料...")
X_train = np.load(f"{ARTIFACTS_DIR}/gru_X_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/gru_X_val.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/y_train_sc.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/y_val_sc.npy")
print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
    batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────
# 模型（無任何預載入權重）
# ─────────────────────────────────────────
model = GRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n📐 GRU from scratch")
print(f"   架構: hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, attention=True")
print(f"   參數量: {total_params:,}  |  無 Walmart pretrain")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
huber     = nn.HuberLoss(delta=1.0)
mse_fn    = nn.MSELoss()
criterion = lambda p, t: 0.7 * huber(p, t) + 0.3 * mse_fn(p, t)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=40, T_mult=2, eta_min=1e-7)

# ─────────────────────────────────────────
# 訓練
# ─────────────────────────────────────────
best_val_loss    = float("inf")
patience_counter = 0
train_losses, val_losses = [], []

print(f"\n🚀 開始訓練（從頭，無 pretrain）...")
print("=" * 70)

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

    scheduler.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"  Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.2e}")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_loss": best_val_loss,
            "pretrained": False,
            "hyperparams": {
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "output_size": OUTPUT_SIZE,
                "dropout": DROPOUT,
            }
        }, f"{ARTIFACTS_DIR}/gru_scratch.pth")
        print(f"             ✅ 儲存最佳模型（val_loss={best_val_loss:.6f}）")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch}！")
            break

print("=" * 70)
print(f"\n🎉 GRU scratch 訓練完成！最佳 Val Loss：{best_val_loss:.6f}")

with open(f"{ARTIFACTS_DIR}/gru_scratch_history.pkl", "wb") as f:
    pickle.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

print("\n下一步：python predict.py")
