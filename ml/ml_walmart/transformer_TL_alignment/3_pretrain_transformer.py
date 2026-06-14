"""
Step 3：Transformer Pretrain（Walmart 資料 + MMD 對齊）
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts_transformer_tl"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MY_DIR))
from alignment_utils import ALIGNED_FEATURE_COLS
from model_transformer import TransformerModel

INPUT_SIZE    = len(ALIGNED_FEATURE_COLS)
D_MODEL       = 64
NHEAD         = 4
NUM_LAYERS    = 2
DROPOUT       = 0.4
OUTPUT_SIZE   = 1
BATCH_SIZE    = 128
EPOCHS        = 30
LEARNING_RATE = 0.0005
HUBER_DELTA   = 1.0
MMD_LAMBDA    = 0.1

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"⚙️  目前使用設備: {device}")

print("📂 載入資料中...")
X_train    = np.load(ARTIFACTS_DIR / "walmart_X_train.npy")
y_train    = np.load(ARTIFACTS_DIR / "walmart_y_train.npy")
X_val      = np.load(ARTIFACTS_DIR / "walmart_X_val.npy")
y_val      = np.load(ARTIFACTS_DIR / "walmart_y_val.npy")
X_personal = np.load(ARTIFACTS_DIR / "personal_X_train.npy")

train_loader    = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader      = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE)
personal_loader = DataLoader(TensorDataset(torch.tensor(X_personal, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)


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


model     = TransformerModel(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
criterion = nn.HuberLoss(delta=HUBER_DELTA)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"🚀 開始預訓練（預計跑 {EPOCHS} 輪）...")
best_val_loss = float("inf")
personal_iter = iter(personal_loader)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_huber = 0

    for i, (X_b, y_b) in enumerate(train_loader):
        X_b, y_b = X_b.to(device), y_b.to(device)
        try:
            (X_p,) = next(personal_iter)
        except StopIteration:
            personal_iter = iter(personal_loader)
            (X_p,) = next(personal_iter)
        X_p = X_p.to(device)

        optimizer.zero_grad()
        huber_loss = criterion(model(X_b), y_b)
        rep_walmart  = model.encode(X_b)
        rep_personal = model.encode(X_p)
        mmd_loss  = compute_mmd(rep_walmart, rep_personal)
        mmd_scale = huber_loss.detach() / (mmd_loss.detach() + 1e-8)
        loss = huber_loss + MMD_LAMBDA * mmd_scale * mmd_loss
        loss.backward()
        optimizer.step()
        total_huber += huber_loss.item()

        if i % 50 == 0:
            print(f"  > Epoch {epoch} [{i}/{len(train_loader)}] Huber: {huber_loss.item():.4f}", end='\r')

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_v, y_v in val_loader:
            val_loss += criterion(model(X_v.to(device)), y_v.to(device)).item()
    val_loss /= len(val_loader)

    print(f"✅ Epoch {epoch:2d} | Avg Huber: {total_huber/len(train_loader):.6f} | Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({"model_state": model.state_dict()}, ARTIFACTS_DIR / "pretrain_transformer.pth")
        print(f"  🌟 已存下目前最好的模型大腦")

print(f"\n🎉 預訓練完成！最佳 Val Loss: {best_val_loss:.6f}")
print("接下來請執行 python 2_preprocess_personal.py → python 4_finetune_transformer.py")
