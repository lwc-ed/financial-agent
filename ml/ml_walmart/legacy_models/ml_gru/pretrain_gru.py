"""
步驟二（更新版）：GRU Pretrain
================================
日級別，7個特徵，INPUT_DAYS=30
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os

ARTIFACTS_DIR = "artificats"
INPUT_SIZE    = 7     # 更新：7個特徵
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.2
OUTPUT_SIZE   = 1
BATCH_SIZE    = 64    # 資料更多，batch 可以大一點
EPOCHS        = 50
LEARNING_RATE = 0.001
PATIENCE      = 10

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple M1 MPS 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("⚠️  使用 CPU")

print("\n📂 載入資料...")
X_train = np.load(f"{ARTIFACTS_DIR}/walmart_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/walmart_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/walmart_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/walmart_y_val.npy")

print(f"  X_train : {X_train.shape}  → (樣本數, 30天, 7特徵)")
print(f"  y_train : {y_train.shape}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=BATCH_SIZE, shuffle=False)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru     = nn.GRU(input_size, hidden_size, num_layers,
                              dropout=dropout if num_layers > 1 else 0,
                              batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)


model = GRUModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n🏗️  模型參數量 : {total_params:,} 個")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

print("\n🚀 開始 Pretrain（日級別，7特徵）...")
print("=" * 65)

best_val_loss    = float("inf")
patience_counter = 0
best_model_path  = f"{ARTIFACTS_DIR}/pretrain_gru.pth"
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            val_loss += criterion(model(X_b), y_b).item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    lr = optimizer.param_groups[0]["lr"]
    print(f"  Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            "epoch"      : epoch,
            "model_state": model.state_dict(),
            "val_loss"   : best_val_loss,
            "hyperparams": {
                "input_size" : INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers" : NUM_LAYERS,
                "output_size": OUTPUT_SIZE,
                "dropout"    : DROPOUT,
            }
        }, best_model_path)
        print(f"             ✅ 儲存最佳模型（val_loss={best_val_loss:.6f}）")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping！")
            break

print("=" * 65)
print(f"\n🎉 Pretrain 完成！最佳 Val Loss：{best_val_loss:.6f}")

with open(f"{ARTIFACTS_DIR}/pretrain_history.pkl", "wb") as f:
    pickle.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

print("\n下一步：執行 preprocess_personal.py → finetune_gru.py")