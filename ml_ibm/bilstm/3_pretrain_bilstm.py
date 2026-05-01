import os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ── 1. 路徑設定 ──────────────────────────────────────────────────────────
MY_DIR = Path(__file__).resolve().parent
# 引用 Step 1 產出的 IBM 數據
DATA_DIR = MY_DIR.parent / "bigru_TL_alignment" / "artifacts_bigru_tl"
SAVE_DIR = MY_DIR / "artifacts_bilstm"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MY_DIR))
from model_bilstm import MyBiLSTM

# ── 2. 超參數 (全量跑優化) ──────────────────────────────────────────
DEBUG_MODE = False    # 【關鍵】關閉測試模式，跑全量數據
EPOCHS = 30           # 【關鍵】依照正式規格跑 30 輪
BATCH_SIZE = 512      # 全量跑時加大 Batch Size 可以顯著提速
LEARNING_RATE = 0.001

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️  設備: {device} | 模式: FULL RUN (全量訓練)")

# ── 3. 載入 IBM 資料 ──────────────────────────────────────────────────────────
print(f"📂 載入 IBM 資料中 (這可能需要一點時間)...")
X_train = np.load(DATA_DIR / "ibm_X_train.npy")
y_train = np.load(DATA_DIR / "ibm_y_train.npy")
X_val   = np.load(DATA_DIR / "ibm_X_val.npy")
y_val   = np.load(DATA_DIR / "ibm_y_val.npy")

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE)

# ── 4. 開始訓練 ──────────────────────────────────────────────────────────
model = MyBiLSTM(X_train.shape[2], 64, 2, 1).to(device)
criterion = nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"🚀 開始 IBM 預訓練 (共 {EPOCHS} 輪)...")
best_loss = float('inf')
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for i, (X_b, y_b) in enumerate(train_loader):
        optimizer.zero_grad()
        loss = criterion(model(X_b.to(device)), y_b.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        if i % 100 == 0:
            print(f"  > Epoch {epoch} [{i}/{len(train_loader)}] Loss: {loss.item():.4f}", end='\r')
    
    model.eval()
    v_loss = 0
    with torch.no_grad():
        for X_v, y_v in val_loader:
            v_loss += criterion(model(X_v.to(device)), y_v.to(device)).item()
    v_loss /= len(val_loader)
    
    print(f"✅ Epoch {epoch:2d} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {v_loss:.6f}")
    if v_loss < best_loss:
        best_loss = v_loss
        torch.save({"model_state": model.state_dict()}, SAVE_DIR / "pretrain_bilstm.pth")
        print(f"  🌟 已存下最佳 Bi-LSTM 模型")

print(f"🎉 預訓練完成！最佳 Val Loss: {best_loss:.6f}")