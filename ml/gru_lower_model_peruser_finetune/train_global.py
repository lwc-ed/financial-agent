"""
gru_lower_model_peruser_finetune/train_global.py
==================================================
Step 1：訓練小型 GRU 全域模型（所有使用者合併）。

架構刻意縮小：hidden=32, layers=1
理由：
  - 原版 V2 hidden=64, layers=2 有 ~70K 參數，資料量才 2,850 筆 → 易過擬合
  - hidden=32, layers=1 約 15K 參數，參數/樣本比從 1:40 改善到 1:190
  - 每個 user fine-tune 時資料更少（~178 筆），更小的模型才不會 overfit

此全域模型學習所有使用者的通用消費模式，
再由 finetune_peruser.py 針對每個使用者個別微調。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
from datetime import datetime

ARTIFACTS_DIR = "artifacts"
INPUT_SIZE    = 7
HIDDEN_SIZE   = 32    # 刻意縮小（原版 V2 是 64）
NUM_LAYERS    = 1     # 減少層數（原版 V2 是 2）
OUTPUT_SIZE   = 1
DROPOUT       = 0.3   # layers=1 時 GRU dropout 無效，這裡用在 FC 前
BATCH_SIZE    = 32
EPOCHS        = 150
PATIENCE      = 25
LR            = 5e-4
WEIGHT_DECAY  = 5e-4

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple MPS 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("⚠️  使用 CPU")


class SmallGRU(nn.Module):
    """小型 GRU + Attention：hidden=32, layers=1"""
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        # layers=1 時 GRU dropout 不適用，dropout 放在 FC 前
        self.gru        = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2        = nn.Linear(hidden_size // 2, output_size)
        self.relu       = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)                               # (B, T, H)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)             # (B, H)
        context    = self.layer_norm(context)
        out        = self.relu(self.fc1(self.dropout(context)))
        return self.fc2(out)

    def get_context(self, x):
        """回傳 attention 加權後的 context vector，供 per-user fine-tune 時檢查"""
        with torch.no_grad():
            gru_out, _ = self.gru(x)
            attn_w     = torch.softmax(self.attention(gru_out), dim=1)
            return (gru_out * attn_w).sum(dim=1)


# ─────────────────────────────────────────
# 載入資料
# ─────────────────────────────────────────
print("\n📂 載入全域訓練資料...")
X_train = np.load(f"{ARTIFACTS_DIR}/X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/y_val.npy")
print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
    batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────
# 模型
# ─────────────────────────────────────────
model = SmallGRU(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n📐 SmallGRU | hidden={HIDDEN_SIZE}, layers=1 | 參數量: {total_params:,}")
print(f"   資料筆數: {len(X_train)} → 參數/樣本比 = 1:{len(X_train) // total_params}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
huber     = nn.HuberLoss(delta=1.0)
mse_fn    = nn.MSELoss()
criterion = lambda p, t: 0.7 * huber(p, t) + 0.3 * mse_fn(p, t)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

# ─────────────────────────────────────────
# 訓練
# ─────────────────────────────────────────
best_val_loss    = float("inf")
patience_counter = 0
train_losses, val_losses = [], []

print(f"\n🚀 開始全域訓練（所有使用者合併）...")
print("=" * 65)

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
            "hyperparams": {
                "input_size":  INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "output_size": OUTPUT_SIZE,
                "dropout":     DROPOUT,
            }
        }, f"{ARTIFACTS_DIR}/global_model.pth")
        print(f"             ✅ 儲存最佳模型（val_loss={best_val_loss:.6f}）")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch}！")
            break

print("=" * 65)
print(f"\n🎉 全域訓練完成！最佳 Val Loss：{best_val_loss:.6f}")

with open(f"{ARTIFACTS_DIR}/global_train_history.pkl", "wb") as f:
    pickle.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

print("\n下一步：python finetune_peruser.py")
