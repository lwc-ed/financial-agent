"""
[NO-ALIGN pipeline] Step 3：在 IBM(raw 特徵) 上預訓練 BiGRU
==========================================================
與原 3_pretrain_bigru.py 相同超參數，僅讀寫 ml_temp/artifacts_noalign，
並重用原本的 model_bigru.BiGRUWithAttention（只讀 import，不改原碼）。

輸出 → ml_temp/artifacts_noalign/pretrain_bigru.pth
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ORIG = HERE.parent / "ml_ibm" / "bigru_TL_alignment"
for p in (HERE, ORIG):
    sys.path.insert(0, str(p))

from model_bigru import BiGRUWithAttention  # noqa: E402  （只讀重用）
from no_alignment_utils import RAW_FEATURE_COLS  # noqa: E402

ART = HERE / "artifacts_noalign"
ART.mkdir(parents=True, exist_ok=True)

INPUT_SIZE = len(RAW_FEATURE_COLS)  # 10
HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.4
OUTPUT_SIZE = 1
BATCH_SIZE = 512
EPOCHS = 150
LEARNING_RATE = 0.00003
HUBER_DELTA = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚙️  設備: {device}")

X_train = np.load(ART / "ibm_X_train.npy")
y_train = np.load(ART / "ibm_y_train.npy")
X_val = np.load(ART / "ibm_X_val.npy")
y_val = np.load(ART / "ibm_y_val.npy")

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE)

model = BiGRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
criterion = nn.HuberLoss(delta=HUBER_DELTA)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

checkpoint_path = ART / "pretrain_checkpoint.pth"
start_epoch, best_val_loss = 1, float("inf")
if checkpoint_path.exists():
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    print(f"🚀 從 Epoch {start_epoch} 續訓")

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_v, y_v in val_loader:
            val_loss += criterion(model(X_v.to(device)), y_v.to(device)).item()
    val_loss /= len(val_loader)
    print(f"✅ Epoch {epoch:2d} | Train: {total_loss/len(train_loader):.4f} | Val: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({"model_state": model.state_dict()}, ART / "pretrain_bigru.pth")
    torch.save(
        {"epoch": epoch, "model_state": model.state_dict(),
         "optimizer_state": optimizer.state_dict(), "best_val_loss": best_val_loss},
        checkpoint_path,
    )

print(f"\n🎉 預訓練完成！最佳 Val Loss: {best_val_loss:.6f}")
if checkpoint_path.exists():
    os.remove(checkpoint_path)
