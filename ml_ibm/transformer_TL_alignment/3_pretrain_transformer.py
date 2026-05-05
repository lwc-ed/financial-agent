"""
Step 3：Transformer Pretrain（IBM 資料）
"""

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
BATCH_SIZE    = 512
EPOCHS        = 30
LEARNING_RATE = 0.0005
HUBER_DELTA   = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚙️  目前使用設備: {device}")

X_train = np.load(ARTIFACTS_DIR / "ibm_X_train.npy")
y_train = np.load(ARTIFACTS_DIR / "ibm_y_train.npy")
X_val   = np.load(ARTIFACTS_DIR / "ibm_X_val.npy")
y_val   = np.load(ARTIFACTS_DIR / "ibm_y_val.npy")

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE)

model     = TransformerModel(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
criterion = nn.HuberLoss(delta=HUBER_DELTA)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

checkpoint_path = ARTIFACTS_DIR / "pretrain_checkpoint.pth"
start_epoch = 1
best_val_loss = float("inf")

if checkpoint_path.exists():
    print(f"📦 偵測到上次沒跑完的存檔！正在載入...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"🚀 已從 Epoch {start_epoch} 恢復訓練！")

print(f"🚀 預計練功到 Epoch {EPOCHS}...")

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
    avg_train = total_loss / len(train_loader)
    print(f"Epoch {epoch:03d}/{EPOCHS} | Train: {avg_train:.6f} | Val: {val_loss:.6f}")

    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, checkpoint_path)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({"model_state": model.state_dict()}, ARTIFACTS_DIR / "pretrain_transformer.pth")
        print(f"  ✅ 新最佳模型已儲存（Val Loss: {best_val_loss:.6f}）")

print(f"\n🎉 Pretrain 完成！最佳 Val Loss: {best_val_loss:.6f}")
print("接下來請執行 python 2_preprocess_personal.py → python 4_finetune_transformer.py")
