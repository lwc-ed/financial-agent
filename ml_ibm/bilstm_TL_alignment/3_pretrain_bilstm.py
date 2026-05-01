"""
Step 3：BiLSTM Aligned Pretrain
================================
使用 10 個 Aligned Walmart 特徵預訓練 BiLSTM
Loss = HuberLoss + λ * MMD_loss
  - HuberLoss：監督式預測誤差
  - MMD_loss ：讓 Walmart 與個人資料的 hidden representation 分佈對齊
  - 兩個 loss 分開印，方便觀察 scale 再調整 λ
輸出：
  - pretrain_bilstm.pth
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS

ARTIFACTS_DIR = "artifacts_bilstm_v2"

# ── 超參數 ────────────────────────────────────────────────────────────────────
INPUT_SIZE    = len(ALIGNED_FEATURE_COLS)   # 10
HIDDEN_SIZE   = 64                           # bidirectional → 實際 128
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
    device = torch.device("mps");  print("✅ Apple M1 MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda"); print("✅ CUDA")
else:
    device = torch.device("cpu");  print("⚠️  CPU")

# ── 載入資料 ──────────────────────────────────────────────────────────────────
print("\n📂 載入 IBM 資料...")
X_train = np.load(f"{ARTIFACTS_DIR}/ibm_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/ibm_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/ibm_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/ibm_y_val.npy")
print(f"  X_train: {X_train.shape}")

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(X_val),   torch.tensor(y_val)),
                          batch_size=BATCH_SIZE, shuffle=False)


# ── 模型架構：BiLSTM + Attention ──────────────────────────────────────────────
class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        bi_hidden = hidden_size * 2  # 128
        self.attention  = nn.Linear(bi_hidden, 1)
        self.layer_norm = nn.LayerNorm(bi_hidden)
        self.dropout    = nn.Dropout(dropout)
        self.fc1 = nn.Linear(bi_hidden, hidden_size)   # 128 → 64
        self.fc2 = nn.Linear(hidden_size, output_size) # 64 → 1
        self.relu = nn.ReLU()

    def encode(self, x) -> torch.Tensor:
        """回傳 attended hidden representation（用於 MMD）"""
        out, _ = self.lstm(x)
        attn_w  = torch.softmax(self.attention(out), dim=1)
        context = (out * attn_w).sum(dim=1)   # (B, 128)
        return self.layer_norm(context)

    def forward(self, x):
        context = self.encode(x)
        out     = self.dropout(context)
        out     = self.relu(self.fc1(out))
        return self.fc2(out)


model = BiLSTMWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n🏗️  BiLSTM 參數量: {total_params:,}  (input={INPUT_SIZE}, hidden={HIDDEN_SIZE}×2={HIDDEN_SIZE*2})")

criterion = nn.HuberLoss(delta=HUBER_DELTA)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

# ── 訓練 ──────────────────────────────────────────────────────────────────────
print(f"\n🚀 開始 BiLSTM IBM Pretrain...")
print("=" * 55)
print(f"  {'Epoch':>5}  {'HuberLoss':>10}  {'Val':>10}  {'LR':>8}")
print("=" * 55)

best_val_loss    = float("inf")
patience_counter = 0
best_model_path  = f"{ARTIFACTS_DIR}/pretrain_bilstm.pth"
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
            "epoch"      : epoch,
            "model_state": model.state_dict(),
            "val_loss"   : best_val_loss,
            "version"    : "ibm_pretrain",
            "hyperparams": {
                "input_size"  : INPUT_SIZE,
                "hidden_size" : HIDDEN_SIZE,
                "num_layers"  : NUM_LAYERS,
                "output_size" : OUTPUT_SIZE,
                "dropout"     : DROPOUT,
                "feature_cols": ALIGNED_FEATURE_COLS,
                "model_type"  : "BiLSTMWithAttention",
            }
        }, best_model_path)
        print(f"             ✅ 儲存最佳模型（val_loss={best_val_loss:.6f}）")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping（patience={PATIENCE}）")
            break

print("=" * 55)
print(f"\n🎉 Pretrain 完成！最佳 Val Loss：{best_val_loss:.6f}")

with open(f"{ARTIFACTS_DIR}/pretrain_history.pkl", "wb") as f:
    pickle.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

print("  ✅ pretrain_bilstm.pth")
print("  ✅ pretrain_history.pkl")
print(f"\n下一步：執行 4_finetune_bilstm.py")
