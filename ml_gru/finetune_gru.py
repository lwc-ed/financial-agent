"""
步驟四（更新版）：GRU Finetune
================================
日級別，7個特徵，pretrain 和 finetune 完全對齊
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os

ARTIFACTS_DIR       = "artificats"
PERSONAL_INPUT_SIZE = 7     # 更新：7個特徵，跟 pretrain 完全一樣！
BATCH_SIZE          = 16
EPOCHS              = 150
LEARNING_RATE       = 0.0005
PATIENCE            = 20

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple M1 MPS 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("⚠️  使用 CPU")

print("\n📂 載入個人記帳資料...")
X_train = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/personal_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")

print(f"  X_train : {X_train.shape}")
print(f"  X_val   : {X_val.shape}")

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


# ─────────────────────────────────────────
# 載入 pretrain 權重
# 這次特徵數完全一樣（7→7），所以所有權重都可以直接繼承！
# ─────────────────────────────────────────
print("\n📦 載入 pretrain 權重...")
checkpoint = torch.load(f"{ARTIFACTS_DIR}/pretrain_gru.pth", map_location=device)
hp         = checkpoint["hyperparams"]

print(f"  Pretrain：input={hp['input_size']}, hidden={hp['hidden_size']}, layers={hp['num_layers']}")
print(f"  Finetune：input={PERSONAL_INPUT_SIZE}  ← 完全一致，全部權重可繼承 ✅")

model = GRUModel(
    PERSONAL_INPUT_SIZE, hp["hidden_size"],
    hp["num_layers"], hp["output_size"], hp["dropout"]
).to(device)

# 全部權重直接載入（不需要跳過任何層）
model.load_state_dict(checkpoint["model_state"])
print("  ✅ 全部 10 個權重層成功繼承")


# ─────────────────────────────────────────
# 凍結策略：只凍結第二層 GRU 的 hidden-to-hidden 權重
# 讓第一層和 fc 層 finetune 個人消費模式
# ─────────────────────────────────────────
print("\n🔒 凍結策略...")
for name, param in model.named_parameters():
    if "weight_hh_l1" in name or "bias_hh_l1" in name:
        param.requires_grad = False
        print(f"  🔒 凍結: {name}")
    else:
        param.requires_grad = True
        print(f"  🔓 訓練: {name}")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\n  可訓練參數: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

print("\n🚀 開始 Finetune...")
print("=" * 65)

best_val_loss    = float("inf")
patience_counter = 0
best_model_path  = f"{ARTIFACTS_DIR}/finetune_gru.pth"
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
                "input_size" : PERSONAL_INPUT_SIZE,
                "hidden_size": hp["hidden_size"],
                "num_layers" : hp["num_layers"],
                "output_size": hp["output_size"],
                "dropout"    : hp["dropout"],
            }
        }, best_model_path)
        print(f"             ✅ 儲存最佳模型（val_loss={best_val_loss:.6f}）")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping！")
            break

print("=" * 65)
print(f"\n🎉 Finetune 完成！最佳 Val Loss：{best_val_loss:.6f}")

with open(f"{ARTIFACTS_DIR}/finetune_history.pkl", "wb") as f:
    pickle.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

print("\n下一步：執行 predict.py")