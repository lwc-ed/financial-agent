"""
Step 5：Aligned Finetune
=========================
載入 Aligned Pretrained GRU，在個人 Aligned 資料上做 finetune
使用 ensemble（7 seeds），與 aligned pretrain 配合
輸出：
  - finetune_aligned_gru_seed{seed}.pth（7個）
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS

ARTIFACTS_DIR = "artifacts_aligned"

# ── 超參數（finetune 用較小 LR，避免 catastrophic forgetting）─────────────────
INPUT_SIZE    = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.4
OUTPUT_SIZE   = 1
BATCH_SIZE    = 32        # 個人資料量少，小 batch
EPOCHS        = 80
LEARNING_RATE = 3e-4      # 比 pretrain 小（保留 pretrained 知識）
PATIENCE      = 20        # 個人資料收斂慢，patience 大一點
WEIGHT_DECAY  = 1e-4
HUBER_DELTA   = 1.0
SEEDS         = [42, 123, 777, 456, 789, 999, 2024]

# ── 設備 ──────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple M1 MPS 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ── 載入個人 Aligned 資料 ──────────────────────────────────────────────────────
print("📂 載入個人 Aligned 資料...")
X_train = np.load(f"{ARTIFACTS_DIR}/personal_aligned_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/personal_aligned_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/personal_aligned_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/personal_aligned_y_val.npy")
print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")


# ── 模型架構（與 pretrain 完全相同）──────────────────────────────────────────
class GRUWithAttention(nn.Module):
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

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        context    = self.layer_norm(context)
        out        = self.dropout(context)
        out        = self.relu(self.fc1(out))
        return self.fc2(out)


def load_pretrained_model():
    """載入 aligned pretrained 權重"""
    ckpt  = torch.load(f"{ARTIFACTS_DIR}/pretrain_aligned_gru.pth", map_location=device)
    model = GRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT)
    model.load_state_dict(ckpt["model_state"])
    return model


# ── Ensemble Finetune ─────────────────────────────────────────────────────────
print(f"\n🚀 Ensemble Finetune（seeds={SEEDS}）...")

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"  Seed {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_pretrained_model().to(device)

    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    best_val_loss    = float("inf")
    patience_counter = 0
    save_path        = f"{ARTIFACTS_DIR}/finetune_aligned_gru_seed{seed}.pth"

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
                val_loss += criterion(model(X_b.to(device)), y_b.to(device)).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_loss"    : best_val_loss,
                "seed"        : seed,
                "version"     : "aligned_finetune",
            }, save_path)
            print(f"             ✅ 儲存（val_loss={best_val_loss:.6f}）")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  ⏹️  Early stopping")
                break

    print(f"  Seed {seed} 最佳 Val Loss: {best_val_loss:.6f}")

print(f"\n🎉 Ensemble Finetune 完成！")
print(f"   → 下一步：執行 6_predict_aligned.py")
