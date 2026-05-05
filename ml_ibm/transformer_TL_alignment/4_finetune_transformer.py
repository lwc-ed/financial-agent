"""
Step 4：Transformer Finetune（個人資料）
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

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
BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 3e-4
PATIENCE      = 10
WEIGHT_DECAY  = 1e-4
HUBER_DELTA   = 1.0

SEEDS = [42, 123, 777, 456, 789, 999, 2024]

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚙️  使用設備: {device}")

print(f"📂 載入個人資料...")
X_train = np.load(ARTIFACTS_DIR / "personal_X_train.npy")
y_train = np.load(ARTIFACTS_DIR / "personal_y_train.npy")
X_val   = np.load(ARTIFACTS_DIR / "personal_X_val.npy")
y_val   = np.load(ARTIFACTS_DIR / "personal_y_val.npy")

PRETRAIN_WEIGHT_PATH = ARTIFACTS_DIR / "pretrain_transformer.pth"


def load_pretrained():
    if not PRETRAIN_WEIGHT_PATH.exists():
        raise FileNotFoundError(f"❌ 找不到預訓練權重: {PRETRAIN_WEIGHT_PATH}")
    ckpt = torch.load(PRETRAIN_WEIGHT_PATH, map_location=device, weights_only=True)
    model = TransformerModel(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, OUTPUT_SIZE, DROPOUT)
    model.load_state_dict(ckpt["model_state"])
    return model


print(f"\n🚀 開始微調 {len(SEEDS)} 個 Seeds...")

for seed in SEEDS:
    save_path = ARTIFACTS_DIR / f"finetune_transformer_seed{seed}.pth"
    if save_path.exists():
        print(f"⏩ Seed {seed} 已存在，跳過")
        continue

    print(f"🔥 Seed {seed} 訓練中...", end=" ")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_pretrained().to(device)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE)

    best_val_loss = float("inf")
    patience_counter = 0

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

    print(f"完成！最佳 Val Loss: {best_val_loss:.6f}")

print("\n🎉 微調流程結束！你現在可以跑 python 5_predict_transformer.py 了！")
