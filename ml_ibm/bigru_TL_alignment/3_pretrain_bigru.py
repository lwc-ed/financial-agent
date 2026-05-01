"""
Step 3：BiGRU Aligned Pretrain (IBM 全量跑 - 移除 MMD 加速版)
======================================================
"""

import os
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── 1. 路徑鎖定 ──────────────────────────────────────────────────────────
MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts_bigru_tl"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MY_DIR))
from alignment_utils import ALIGNED_FEATURE_COLS
from model_bigru import BiGRUWithAttention

# ── 2. 超參數設定 (正式訓練設定) ─────────────────────────────────────────
INPUT_SIZE = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.4
OUTPUT_SIZE = 1
BATCH_SIZE = 512      # 全量跑時，加大 Batch Size 可以跑更快
EPOCHS = 30           # 【關鍵】設定為正式要求的 30 輪
LEARNING_RATE = 0.0005
PATIENCE = 10
WEIGHT_DECAY = 5e-4
HUBER_DELTA = 1.0

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"⚙️  目前使用設備: {device} | 模式：全量訓練 (無 MMD)")

# ── 3. 載入資料 ──────────────────────────────────────────────────────────
print(f"📂 載入 IBM 資料中...")
X_train = np.load(ARTIFACTS_DIR / "ibm_X_train.npy")
y_train = np.load(ARTIFACTS_DIR / "ibm_y_train.npy")
X_val   = np.load(ARTIFACTS_DIR / "ibm_X_val.npy")
y_val   = np.load(ARTIFACTS_DIR / "ibm_y_val.npy")

# 注意：不再需要載入 X_personal，因為不跑 MMD 了，這能大幅節省記憶體！

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE)

model = BiGRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
criterion = nn.HuberLoss(delta=HUBER_DELTA)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ── 4. 訓練迴圈 ──────────────────────────────────────────────────────────
print(f"🚀 開始預訓練 (預計跑 {EPOCHS} 輪)...")
best_val_loss = float("inf")
best_model_path = ARTIFACTS_DIR / "pretrain_bigru.pth"

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    
    for i, (X_b, y_b) in enumerate(train_loader):
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        
        # 只計算 Huber Loss，移除耗時的 MMD 計算
        outputs = model(X_b)
        loss = criterion(outputs, y_b)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"  > Epoch {epoch} [{i}/{len(train_loader)}] Loss: {loss.item():.4f}", end='\r')

    # 驗證
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_v, y_v in val_loader:
            val_loss += criterion(model(X_v.to(device)), y_v.to(device)).item()
    val_loss /= len(val_loader)
    
    print(f"✅ Epoch {epoch:2d} | Train Loss: {total_loss/len(train_loader):.6f} | Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({"model_state": model.state_dict()}, best_model_path)
        print(f"  🌟 已存下目前最好的 IBM 模型大腦")

print(f"\n🎉 預訓練完成！最佳 Val Loss: {best_val_loss:.6f}")