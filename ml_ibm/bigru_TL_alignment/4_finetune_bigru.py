"""
Step 4：BiGRU Aligned Finetune (方案 1 - 完全凍結 Encoder 版)
==============================
策略：
1. 凍結 bigru、attention、layer_norm 等所有特徵萃取層 (Encoder)
2. 只解凍並訓練 fc1, fc2 等全連接層 (Head)
3. 使用 L1Loss 直球最佳化 MAE
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

# 方案 1 訓練參數 (單階段，全凍結)
EPOCHS = 50               # 讓 Head 有充足時間適應
LEARNING_RATE = 1e-3      # 只有 FC 層要學，學習率可以給稍微大一點點
PATIENCE = 15             # 容忍度給高一點，讓 L1 Loss 慢慢收斂
WEIGHT_DECAY = 1e-5       # 調低正則化，避免壓抑預測數值

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
print(f"⚙️  使用設備: {device}")

# ── 3. 載入資料 ──────────────────────────────────────────────────────────
print(f"📂 載入個人資料...")
X_train = np.load(ARTIFACTS_DIR / "personal_X_train.npy")
y_train = np.load(ARTIFACTS_DIR / "personal_y_train.npy")
X_val   = np.load(ARTIFACTS_DIR / "personal_X_val.npy")
y_val   = np.load(ARTIFACTS_DIR / "personal_y_val.npy")

PRETRAIN_WEIGHT_PATH = ARTIFACTS_DIR / "pretrain_bigru.pth"

# ── 4. 載入預訓練模型 ──────────────────────────────────────────────────────────
def load_pretrained():
    if not PRETRAIN_WEIGHT_PATH.exists():
        raise FileNotFoundError(f"❌ 找不到預訓練大腦: {PRETRAIN_WEIGHT_PATH}")
    ckpt = torch.load(PRETRAIN_WEIGHT_PATH, map_location=device, weights_only=True)
    model = BiGRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT)
    model.load_state_dict(ckpt["model_state"])
    return model

# ── 5. 訓練迴圈 ──────────────────────────────────────────────────────────
print(f"\n🚀 開始微調 (方案1：完全凍結 Encoder，只訓練 Head)...")

for seed in SEEDS:
    save_path = ARTIFACTS_DIR / f"finetune_bigru_seed{seed}.pth"
    if save_path.exists():
        print(f"⏩ Seed {seed} 已存在，跳過")
        continue

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_pretrained().to(device)
    
    # 【直球對決】使用 L1 Loss 來最佳化 MAE
    criterion = nn.L1Loss()
    
    # ==========================================
    # 🛑 方案 1: 完全凍結 Encoder
    # ==========================================
    for name, param in model.named_parameters():
        if "fc" not in name:  # 名稱裡沒有 fc 的一律凍結
            param.requires_grad = False
            
    # Optimizer 裡面的過濾器：只給它 requires_grad=True 的參數 (即 fc1, fc2)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"🔥 Seed {seed} 訓練中...", end=" ")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
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
            if patience_counter >= PATIENCE: 
                break
    
    print(f"完成！最佳 Val L1-Loss: {best_val_loss:.6f}")

print("\n🎉 微調流程結束！你現在可以跑 python 5_predict_bigru.py 了！")