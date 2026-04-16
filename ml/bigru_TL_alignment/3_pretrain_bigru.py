"""
Step 3：BiGRU Aligned Pretrain
==============================
使用 aligned Walmart 特徵預訓練 BiGRU
Loss = HuberLoss + λ * MMD_loss
"""

import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS
from model_bigru import BiGRUWithAttention

ARTIFACTS_DIR = "artifacts_bigru_tl"

INPUT_SIZE = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.4
OUTPUT_SIZE = 1
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.0001
PATIENCE = 20
WEIGHT_DECAY = 5e-4
HUBER_DELTA = 1.0
MMD_LAMBDA = 0.1

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Apple M1 MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ CUDA")
else:
    device = torch.device("cpu")
    print("⚠️  CPU")

print("\n📂 載入 aligned Walmart 資料...")
X_train = np.load(f"{ARTIFACTS_DIR}/walmart_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/walmart_y_train.npy")
X_val = np.load(f"{ARTIFACTS_DIR}/walmart_X_val.npy")
y_val = np.load(f"{ARTIFACTS_DIR}/walmart_y_val.npy")
print(f"  X_train: {X_train.shape}")

print("\n📂 載入個人資料（用於 MMD alignment）...")
X_personal = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
print(f"  X_personal: {X_personal.shape}")

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)
personal_loader = DataLoader(TensorDataset(torch.tensor(X_personal, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)


def compute_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    m = y.size(0)
    rx = (x**2).sum(dim=1, keepdim=True)
    ry = (y**2).sum(dim=1, keepdim=True)
    dist_xx = rx + rx.t() - 2 * torch.mm(x, x.t())
    dist_yy = ry + ry.t() - 2 * torch.mm(y, y.t())
    dist_xy = rx + ry.t() - 2 * torch.mm(x, y.t())
    all_dist = torch.cat([dist_xx.reshape(-1), dist_yy.reshape(-1), dist_xy.reshape(-1)])
    bandwidth = all_dist.median().clamp(min=1e-6)
    k = torch.exp(-0.5 * dist_xx / bandwidth)
    l = torch.exp(-0.5 * dist_yy / bandwidth)
    p = torch.exp(-0.5 * dist_xy / bandwidth)
    mmd = (k.sum() - k.trace()) / (n * (n - 1) + 1e-8)
    mmd += (l.sum() - l.trace()) / (m * (m - 1) + 1e-8)
    mmd -= 2 * p.mean()
    return mmd.clamp(min=0)


model = BiGRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n🏗️  BiGRU 參數量: {total_params:,}")

criterion = nn.HuberLoss(delta=HUBER_DELTA)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

print(f"\n🚀 開始 BiGRU Aligned Pretrain（MMD_LAMBDA={MMD_LAMBDA}）...")
print("=" * 75)
print(f"  {'Epoch':>5}  {'HuberLoss':>10}  {'MMD_loss':>10}  {'Total':>10}  {'Val':>10}  {'LR':>8}")
print("=" * 75)

best_val_loss = float("inf")
patience_counter = 0
best_model_path = f"{ARTIFACTS_DIR}/pretrain_bigru.pth"
train_losses, val_losses = [], []
personal_iter = iter(personal_loader)

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_huber = 0.0
    epoch_mmd = 0.0

    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        try:
            (X_p,) = next(personal_iter)
        except StopIteration:
            personal_iter = iter(personal_loader)
            (X_p,) = next(personal_iter)
        X_p = X_p.to(device)

        optimizer.zero_grad()
        huber_loss = criterion(model(X_b), y_b)
        rep_walmart = model.encode(X_b)
        rep_personal = model.encode(X_p)
        mmd_loss = compute_mmd(rep_walmart, rep_personal)
        mmd_scale = huber_loss.detach() / (mmd_loss.detach() + 1e-8)
        total_loss = huber_loss + MMD_LAMBDA * mmd_scale * mmd_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_huber += huber_loss.item()
        epoch_mmd += mmd_loss.item()

    epoch_huber /= len(train_loader)
    epoch_mmd /= len(train_loader)
    epoch_total = epoch_huber + MMD_LAMBDA * epoch_mmd

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            val_loss += criterion(model(X_b.to(device)), y_b.to(device)).item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    train_losses.append(epoch_total)
    val_losses.append(val_loss)

    lr = optimizer.param_groups[0]["lr"]
    print(f"  {epoch:5d}  {epoch_huber:10.6f}  {epoch_mmd:10.6f}  {epoch_total:10.6f}  {val_loss:10.6f}  {lr:8.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": best_val_loss,
                "version": "bigru_mmd",
                "mmd_lambda": MMD_LAMBDA,
                "hyperparams": {
                    "input_size": INPUT_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "output_size": OUTPUT_SIZE,
                    "dropout": DROPOUT,
                    "feature_cols": ALIGNED_FEATURE_COLS,
                    "model_type": "BiGRUWithAttention",
                },
            },
            best_model_path,
        )
        print(f"             ✅ 儲存最佳模型（val_loss={best_val_loss:.6f}）")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping（patience={PATIENCE}）")
            break

print("=" * 75)
print(f"\n🎉 Pretrain 完成！最佳 Val Loss：{best_val_loss:.6f}")

with open(f"{ARTIFACTS_DIR}/pretrain_history.pkl", "wb") as f:
    pickle.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

print("  ✅ pretrain_bigru.pth")
print("  ✅ pretrain_history.pkl")
print("\n下一步：執行 4_finetune_bigru.py")
