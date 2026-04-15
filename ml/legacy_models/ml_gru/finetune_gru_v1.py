"""
版本 V1：GRU Finetune（增強版）
================================
改進點：
  1. 完整繼承 V1 pretrain 架構（hidden=128, Attention, 深層FC）
  2. 移除層凍結策略 → 全量微調（小資料集下更優）
  3. 損失函數：MSELoss → Huber Loss
  4. AdamW + Weight Decay 1e-4
  5. LR 排程：CosineAnnealingLR（更平滑的學習率衰減）
  6. 更多訓練輪數（200 epochs, patience=30）
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os

ARTIFACTS_DIR       = "artificats"
PERSONAL_INPUT_SIZE = 7
BATCH_SIZE          = 16
EPOCHS              = 200        # V1: 150 → 200
LEARNING_RATE       = 1e-4       # V1: 5e-4 → 1e-4（全量微調用更小 LR）
PATIENCE            = 30         # V1: 20 → 30
WEIGHT_DECAY        = 1e-4       # V1: 新增
HUBER_DELTA         = 1.0        # V1: 新增

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


class GRUWithAttention(nn.Module):
    """
    V1 架構：GRU + Temporal Attention + 深層 FC Head
    （與 pretrain_gru_v1.py 完全相同，確保權重可完整繼承）
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)                          # (batch, seq, hidden)
        attn_w     = torch.softmax(
            self.attention(gru_out), dim=1
        )                                                  # (batch, seq, 1)
        context    = (gru_out * attn_w).sum(dim=1)        # (batch, hidden)
        context    = self.layer_norm(context)
        out        = self.dropout(context)
        out        = self.relu(self.fc1(out))
        return self.fc2(out)


# ─────────────────────────────────────────
# 載入 V1 pretrain 權重
# ─────────────────────────────────────────
print("\n📦 載入 V1 pretrain 權重...")
checkpoint = torch.load(f"{ARTIFACTS_DIR}/pretrain_gru_v1.pth", map_location=device)
hp         = checkpoint["hyperparams"]

print(f"  Pretrain V1：input={hp['input_size']}, hidden={hp['hidden_size']}, layers={hp['num_layers']}")
print(f"  版本確認：{checkpoint.get('version', 'unknown')}")

model = GRUWithAttention(
    PERSONAL_INPUT_SIZE, hp["hidden_size"],
    hp["num_layers"], hp["output_size"], hp["dropout"]
).to(device)

# 全部權重直接載入（V1 pretrain/finetune 架構完全一致）
model.load_state_dict(checkpoint["model_state"])
print("  ✅ 全部權重成功繼承（含 Attention 層）")


# ─────────────────────────────────────────
# V1 改進：移除凍結策略，改為全量微調
# 理由：個人資料集只有 16 個使用者（約 4067 訓練樣本），
#       凍結部分層會限制模型適應個人消費模式的能力。
#       配合更小的 LR (1e-4) 和 Weight Decay 來防止遺忘 pretrain 知識。
# ─────────────────────────────────────────
print("\n🔓 V1 策略：全量微調（不凍結任何層）...")
for name, param in model.named_parameters():
    param.requires_grad = True
    print(f"  🔓 訓練: {name}")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\n  可訓練參數: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

# V1: Huber Loss
criterion = nn.HuberLoss(delta=HUBER_DELTA)
# V1: AdamW + Weight Decay
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
# V1: CosineAnnealingLR（比 ReduceLROnPlateau 更平滑）
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

print("\n🚀 開始 V1 Finetune（全量微調, Huber, CosineAnnealing）...")
print("=" * 65)

best_val_loss    = float("inf")
patience_counter = 0
best_model_path  = f"{ARTIFACTS_DIR}/finetune_gru_v1.pth"
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

    scheduler.step()
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
            "version"    : "v1",
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
print(f"\n🎉 V1 Finetune 完成！最佳 Val Loss：{best_val_loss:.6f}")

with open(f"{ARTIFACTS_DIR}/finetune_history_v1.pkl", "wb") as f:
    pickle.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

print("\n下一步：執行 predict_v1.py")
