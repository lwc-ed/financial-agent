"""
Step 4：BiGRU Aligned Finetune (IBM 版修正)
==============================
修正內容：
1. 將 WALMART_DATA_PATH 修正為 IBM_DATA_PATH (讀取 ibm_X_train.npy)
2. 將 MMD_LAMBDA 設為 0.0 (依據朋友建議提速)
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
EPOCHS = 50           # 微調不需要跑太久，50 輪搭配 Early Stopping 足夠
LEARNING_RATE = 3e-4
PATIENCE = 10         # 提早觸發 Early Stopping 節省時間
WEIGHT_DECAY = 1e-4
HUBER_DELTA = 1.0
MMD_LAMBDA = 0.0      # 【核心修正】設為 0.0 以提速

SEEDS = [
    42, 123, 777, 456, 789, 999, 2024,
    0, 7, 13, 21, 100, 314, 1234, 9999,
    11, 22, 33, 44, 55, 66, 77, 88, 99,
    111, 222, 333, 444, 555, 666
]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"⚙️  使用設備: {device} | MMD_LAMBDA: {MMD_LAMBDA}")

# ── 3. 載入資料 ──────────────────────────────────────────────────────────
print(f"📂 載入個人資料...")
X_train = np.load(ARTIFACTS_DIR / "personal_X_train.npy")
y_train = np.load(ARTIFACTS_DIR / "personal_y_train.npy")
X_val   = np.load(ARTIFACTS_DIR / "personal_X_val.npy")
y_val   = np.load(ARTIFACTS_DIR / "personal_y_val.npy")

# 【路徑修正】這裡必須指向 IBM 的資料
IBM_DATA_PATH = ARTIFACTS_DIR / "ibm_X_train.npy"
PRETRAIN_WEIGHT_PATH = ARTIFACTS_DIR / "pretrain_bigru.pth"

if not IBM_DATA_PATH.exists():
    print(f"❌ 找不到 IBM 資料檔: {IBM_DATA_PATH}")
    sys.exit()

# ── 4. 載入預訓練模型 ──────────────────────────────────────────────────────────
def load_pretrained():
    if not PRETRAIN_WEIGHT_PATH.exists():
        raise FileNotFoundError(f"❌ 找不到預訓練大腦: {PRETRAIN_WEIGHT_PATH}")
    ckpt = torch.load(PRETRAIN_WEIGHT_PATH, map_location=device, weights_only=True)
    model = BiGRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT)
    model.load_state_dict(ckpt["model_state"])
    return model

# ── 5. 訓練迴圈 ──────────────────────────────────────────────────────────
print(f"\n🚀 開始微調 30 個 Seeds...")

for seed in SEEDS:
    save_path = ARTIFACTS_DIR / f"finetune_bigru_seed{seed}.pth"
    if save_path.exists():
        print(f"⏩ Seed {seed} 已存在，跳過")
        continue

    print(f"🔥 Seed {seed} 訓練中...", end=" ")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_pretrained().to(device)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            # 純 Huber Loss，不跑 MMD 數學運算
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for X_v, y_v in val_loader:
                v_loss += criterion(model(X_v.to(device)), y_v.to(device)).item()
        v_loss /= len(val_loader)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            torch.save({"model_state": model.state_dict()}, save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE: break
    
    print(f"完成！最佳 Val Loss: {best_val_loss:.6f}")

print("\n🎉 微調流程結束！你現在可以跑 python 5_predict_bigru.py 了！")
