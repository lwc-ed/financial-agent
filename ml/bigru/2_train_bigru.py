"""
Step 2：Bi-GRU baseline 訓練（30 seeds）
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.3
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


class MyBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out    = out[:, -1, :]
        return self.fc(self.dropout(out))


def train_one_seed(seed: int, X_train, y_train, X_val, y_val, input_size: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = MyBiGRU(input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
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

    for _ in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            nn.HuberLoss(delta=1.0)(model(Xb.to(device)), yb.to(device)).backward()
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

    torch.save({"model_state": best_state, "val_loss": best_val_loss},
               ARTIFACTS_DIR / f"bigru_seed{seed}.pth")
    return best_val_loss


def main():
    print("📂 載入資料...")
    X_train    = np.load(ARTIFACTS_DIR / "my_X_train.npy")
    y_train    = np.load(ARTIFACTS_DIR / "my_y_train_scaled.npy")
    X_val      = np.load(ARTIFACTS_DIR / "my_X_val.npy")
    y_val      = np.load(ARTIFACTS_DIR / "my_y_val_scaled.npy")
    input_size = X_train.shape[2]

    for i, seed in enumerate(SEEDS, 1):
        val_loss = train_one_seed(seed, X_train, y_train, X_val, y_val, input_size)
        print(f"  [{i:02d}/{len(SEEDS)}] seed={seed:5d}  best_val_loss={val_loss:.4f}")

    print(f"\n🎉 [Step 2] 完成！{len(SEEDS)} 個 checkpoint 存於 {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
