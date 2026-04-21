"""
Step 4：BiGRU Aligned Finetune (路徑鎖定版)
==============================
載入 aligned pretrained BiGRU，在個人 aligned 資料上 finetune
Loss = HuberLoss + λ * MMD_loss
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ── 1. 路徑鎖定 ──────────────────────────────────────────────────────────
MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts_bigru_tl"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 確保能 import 同資料夾下的模組
sys.path.insert(0, str(MY_DIR))
from alignment_utils import ALIGNED_FEATURE_COLS
from model_bigru import BiGRUWithAttention

# ── 2. 超參數設定 ────────────────────────────────────────────────────────
INPUT_SIZE = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.4
OUTPUT_SIZE = 1
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 3e-4
PATIENCE = 20
WEIGHT_DECAY = 1e-4
HUBER_DELTA = 1.0
MMD_LAMBDA = 0.1

# 縮減 Seed 數量以便快速測試 (你可以依需求加回來)
SEEDS = [42, 123, 777, 456, 789, 999, 2024]

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Apple MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ CUDA")
else:
    device = torch.device("cpu")
    print("⚠️  CPU")

# ── 3. 載入資料 ──────────────────────────────────────────────────────────
print(f"📂 正在從 {ARTIFACTS_DIR} 載入個人資料...")
X_train = np.load(ARTIFACTS_DIR / "personal_X_train.npy")
y_train = np.load(ARTIFACTS_DIR / "personal_y_train.npy")
X_val   = np.load(ARTIFACTS_DIR / "personal_X_val.npy")
y_val   = np.load(ARTIFACTS_DIR / "personal_y_val.npy")
print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")

# ⚠️ 注意：Walmart 資料與預訓練權重通常是由 Step 3 產生的
# 如果你手邊沒有這兩個檔案，程式會在這裡報錯
WALMART_DATA_PATH = ARTIFACTS_DIR / "walmart_X_train.npy"
PRETRAIN_WEIGHT_PATH = ARTIFACTS_DIR / "pretrain_bigru.pth"

if not WALMART_DATA_PATH.exists():
    print(f"❌ 找不到 Walmart 資料檔: {WALMART_DATA_PATH}")
    print("💡 請確保你已經跑過 Step 3 (Walmart Pretrain) 或已將檔案放入該路徑。")
    sys.exit()

print("📂 載入 Walmart 資料（用於 MMD alignment）...")
X_walmart = np.load(WALMART_DATA_PATH)
walmart_loader = DataLoader(TensorDataset(torch.tensor(X_walmart, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)

def compute_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n, m = x.size(0), y.size(0)
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

def load_pretrained():
    if not PRETRAIN_WEIGHT_PATH.exists():
        raise FileNotFoundError(f"❌ 找不到預訓練模型: {PRETRAIN_WEIGHT_PATH}")
    ckpt = torch.load(PRETRAIN_WEIGHT_PATH, map_location=device, weights_only=True)
    model = BiGRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT)
    model.load_state_dict(ckpt["model_state"])
    return model

# ── 4. 訓練迴圈 ──────────────────────────────────────────────────────────
print(f"\n🚀 開始微調訓練 (MMD_LAMBDA={MMD_LAMBDA})")

for seed in SEEDS:
    save_path = ARTIFACTS_DIR / f"finetune_bigru_seed{seed}.pth"
    if save_path.exists():
        print(f"⏩ Seed {seed} 已有權重檔，跳過")
        continue

    print(f"\n🔥 Seed {seed} 訓練中...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        model = load_pretrained().to(device)
    except Exception as e:
        print(e)
        break

    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)

    walmart_iter = iter(walmart_loader)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_huber, epoch_mmd = 0.0, 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            try:
                (X_w,) = next(walmart_iter)
            except StopIteration:
                walmart_iter = iter(walmart_loader)
                (X_w,) = next(walmart_iter)
            X_w = X_w.to(device)

            optimizer.zero_grad()
            huber_loss = criterion(model(X_b), y_b)
            rep_personal = model.encode(X_b)
            rep_walmart = model.encode(X_w)
            mmd_loss = compute_mmd(rep_personal, rep_walmart)
            mmd_scale = huber_loss.detach() / (mmd_loss.detach() + 1e-8)
            total_loss = huber_loss + MMD_LAMBDA * mmd_scale * mmd_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_huber += huber_loss.item()
            epoch_mmd += mmd_loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b.to(device)), y_b.to(device)).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:02d} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "val_loss": best_val_loss,
                "seed": seed,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE: break

    print(f"✅ Seed {seed} 完成！最佳 Val Loss: {best_val_loss:.6f}")

print("\n🎉 微調流程結束！")