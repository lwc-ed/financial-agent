"""
Step 1：Bi-LSTM Aligned Pretrain（Walmart）
==========================================
使用與 ml_alignment_lwc 完全相同的 aligned features（rolling z-score）
只換架構：GRU → Bi-LSTM
直接複用 ml_alignment_lwc/artifacts_aligned/ 的前處理結果
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from model_bilstm import BiLSTMWithAttention

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
# 前處理資料直接複用 ml_alignment_lwc（features 完全相同，不重跑）
SRC_ARTIFACTS  = "../ml_alignment_lwc/artifacts_aligned"
SAVE_DIR       = "artifacts_bilstm"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 超參數 ────────────────────────────────────────────────────────────────────
INPUT_SIZE    = 7
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.4
OUTPUT_SIZE   = 1
BATCH_SIZE    = 64
EPOCHS        = 150
LEARNING_RATE = 0.0001   # 與修正後的 GRU pretrain 一致
PATIENCE      = 20
WEIGHT_DECAY  = 5e-4

# ── 裝置 ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps"); print("✅ Apple MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda"); print("✅ CUDA")
else:
    device = torch.device("cpu"); print("⚠️  CPU")

# ── 載入 Walmart aligned 資料 ─────────────────────────────────────────────────
print("\n📂 載入 Walmart aligned 資料（來自 ml_alignment_lwc）...")
X_train = np.load(f"{SRC_ARTIFACTS}/walmart_aligned_X_train.npy")
y_train = np.load(f"{SRC_ARTIFACTS}/walmart_aligned_y_train.npy")
X_val   = np.load(f"{SRC_ARTIFACTS}/walmart_aligned_X_val.npy")
y_val   = np.load(f"{SRC_ARTIFACTS}/walmart_aligned_y_val.npy")
print(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=BATCH_SIZE, shuffle=True
)
val_t = torch.tensor(X_val).to(device)
val_y = torch.tensor(y_val).to(device)

# ── 模型 ─────────────────────────────────────────────────────────────────────
model     = BiLSTMWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
n_params  = sum(p.numel() for p in model.parameters())
print(f"\n🏗️  Bi-LSTM 參數量 : {n_params:,}（GRU 版約 42K）")

criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

# ── 訓練 ─────────────────────────────────────────────────────────────────────
print(f"\n🚀 開始 Bi-LSTM Pretrain（LR={LEARNING_RATE}，patience={PATIENCE}）...")
print("=" * 60)

best_val_loss = float("inf")
patience_cnt  = 0
best_path     = f"{SAVE_DIR}/pretrain_bilstm.pth"

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(val_t), val_y).item()
    scheduler.step(val_loss)

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"  Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_cnt  = 0
        torch.save({
            "epoch"      : epoch,
            "model_state": model.state_dict(),
            "val_loss"   : best_val_loss,
            "hyperparams": {
                "input_size" : INPUT_SIZE, "hidden_size": HIDDEN_SIZE,
                "num_layers" : NUM_LAYERS, "dropout"    : DROPOUT,
            }
        }, best_path)
        print(f"             ✅ 儲存最佳模型（val_loss={best_val_loss:.6f}）")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break

print("=" * 60)
print(f"\n🎉 Pretrain 完成！最佳 Val Loss：{best_val_loss:.6f}")
print(f"   儲存至：{best_path}")
print(f"\n下一步：執行 2_finetune_bilstm.py")
