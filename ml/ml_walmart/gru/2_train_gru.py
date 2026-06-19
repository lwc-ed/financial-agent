"""
Step 2：GRU baseline 訓練（30 seeds）
======================================
GRUWithAttention（單向，與 gru_TL_alignment 相同架構）
每個 seed 各訓練一次，存 per-seed checkpoint
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.4
OUTPUT_SIZE   = 1
BATCH_SIZE    = 32
EPOCHS        = 80
LEARNING_RATE = 3e-4
PATIENCE      = 20
WEIGHT_DECAY  = 1e-4

SEEDS = [
    42, 123, 777, 456, 789, 999, 2024,
    0, 7, 13, 21, 100, 314, 1234, 9999,
    11, 22, 33, 44, 55, 66, 77, 88, 99,
    111, 222, 333, 444, 555, 666,
]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"⚙️  使用裝置: {device}")


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


def train_one_seed(seed: int, X_train, y_train, X_val, y_val, input_size: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = GRUWithAttention(input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                val_loss += criterion(model(Xb.to(device)), yb.to(device)).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    ckpt_path = ARTIFACTS_DIR / f"gru_seed{seed}.pth"
    torch.save({"model_state": best_state, "val_loss": best_val_loss}, ckpt_path)
    return best_val_loss


def main():
    print("📂 載入資料...")
    X_train = np.load(ARTIFACTS_DIR / "my_X_train.npy")
    y_train = np.load(ARTIFACTS_DIR / "my_y_train_scaled.npy")
    X_val   = np.load(ARTIFACTS_DIR / "my_X_val.npy")
    y_val   = np.load(ARTIFACTS_DIR / "my_y_val_scaled.npy")
    input_size = X_train.shape[2]
    print(f"   X_train {X_train.shape}  X_val {X_val.shape}  input_size={input_size}")

    for i, seed in enumerate(SEEDS, 1):
        val_loss = train_one_seed(seed, X_train, y_train, X_val, y_val, input_size)
        print(f"  [{i:02d}/{len(SEEDS)}] seed={seed:5d}  best_val_loss={val_loss:.4f}")

    print(f"\n🎉 [Step 2] 完成！{len(SEEDS)} 個 checkpoint 存於 {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
