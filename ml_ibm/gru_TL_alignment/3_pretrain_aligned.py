"""
Step 4：Aligned Pretrain
=========================
使用 Aligned Walmart 資料預訓練 GRU
Loss = HuberLoss + λ * MMD_loss
  - HuberLoss：監督式預測誤差
  - MMD_loss ：讓 Walmart 與個人資料的 hidden representation 分佈對齊
  - 兩個 loss 分開印，方便觀察 scale 再調整 λ
輸出：
  - pretrain_aligned_gru.pth
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS

ARTIFACTS_DIR = "artifacts_aligned"

# ── 超參數 ────────────────────────────────────────────────────────────────────
INPUT_SIZE    = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.4
OUTPUT_SIZE   = 1
BATCH_SIZE    = 64
EPOCHS        = 150
LEARNING_RATE = 0.0001
PATIENCE      = 20
WEIGHT_DECAY  = 5e-4
HUBER_DELTA   = 1.0

# ── 設備 ──────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple M1 MPS 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ 使用 CUDA")
else:
    device = torch.device("cpu")
    print("⚠️  使用 CPU")

# ── 載入資料 ──────────────────────────────────────────────────────────────────
print("\n📂 載入 IBM 資料...")
X_train = np.load(f"{ARTIFACTS_DIR}/ibm_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/ibm_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/ibm_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/ibm_y_val.npy")
print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
    batch_size=BATCH_SIZE, shuffle=False
)


# ── 模型架構 ──────────────────────────────────────────────────────────────────
class GRUWithAttention(nn.Module):
    """GRU + Temporal Attention + LayerNorm"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, num_layers,
                                 dropout=dropout if num_layers > 1 else 0,
                                 batch_first=True)
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2        = nn.Linear(hidden_size // 2, output_size)
        self.relu       = nn.ReLU()

    def encode(self, x) -> torch.Tensor:
        """回傳 attended hidden representation（用於 MMD）"""
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)   # (B, hidden_size)
        return self.layer_norm(context)

    def forward(self, x):
        context = self.encode(x)
        out     = self.dropout(context)
        out     = self.relu(self.fc1(out))
        return self.fc2(out)


model = GRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n🏗️  模型參數量 : {total_params:,}")

criterion = nn.HuberLoss(delta=HUBER_DELTA)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

# ── 訓練 ──────────────────────────────────────────────────────────────────────
print(f"\n🚀 開始 GRU IBM Pretrain...")
print("=" * 55)
print(f"  {'Epoch':>5}  {'HuberLoss':>10}  {'Val':>10}  {'LR':>8}")
print("=" * 55)

best_val_loss    = float("inf")
patience_counter = 0
best_model_path  = f"{ARTIFACTS_DIR}/pretrain_aligned_gru.pth"
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_huber = 0.0

    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_huber += loss.item()

    epoch_huber /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            val_loss += criterion(model(X_b.to(device)), y_b.to(device)).item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)
    train_losses.append(epoch_huber)
    val_losses.append(val_loss)

    lr = optimizer.param_groups[0]["lr"]
    print(f"  {epoch:5d}  {epoch_huber:10.6f}  {val_loss:10.6f}  {lr:8.6f}")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            "epoch"       : epoch,
            "model_state" : model.state_dict(),
            "val_loss"    : best_val_loss,
            "version"     : "ibm_pretrain",
            "hyperparams" : {
                "input_size"  : INPUT_SIZE,
                "hidden_size" : HIDDEN_SIZE,
                "num_layers"  : NUM_LAYERS,
                "output_size" : OUTPUT_SIZE,
                "dropout"     : DROPOUT,
                "feature_cols": ALIGNED_FEATURE_COLS,
            }
        }, best_model_path)
        print(f"             ✅ 儲存最佳模型（val_loss={best_val_loss:.6f}）")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping（patience={PATIENCE}）")
            break

print("=" * 75)
print(f"\n🎉 Aligned Pretrain 完成！最佳 Val Loss：{best_val_loss:.6f}")
print(f"   💡 Loss = HuberLoss only（MMD 已移除）")

with open(f"{ARTIFACTS_DIR}/pretrain_aligned_history.pkl", "wb") as f:
    pickle.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

print("  ✅ pretrain_aligned_gru.pth")
print("  ✅ pretrain_aligned_history.pkl")
print(f"\n下一步：執行 4_finetune_aligned.py")
