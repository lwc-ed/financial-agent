from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── 路徑 ─────────────────────────────────────────────
MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts"
MODELS_DIR = MY_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── 超參數 ───────────────────────────────────────────
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3

# ✅ 改成 5（與 own data 對齊）
INPUT_SIZE = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


# ── 模型定義 ─────────────────────────────────────────
class BiGRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用裝置：{device}")

    # ── 載入資料 ─────────────────────────────────────
    X_train = np.load(ARTIFACTS_DIR / "ibm_X_train.npy").astype(np.float32)
    X_val = np.load(ARTIFACTS_DIR / "ibm_X_val.npy").astype(np.float32)
    y_train = np.load(ARTIFACTS_DIR / "ibm_y_train_scaled.npy").astype(np.float32)
    y_val = np.load(ARTIFACTS_DIR / "ibm_y_val_scaled.npy").astype(np.float32)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape  : {X_val.shape}")

    # ── DataLoader ───────────────────────────────────
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── 建立模型 ─────────────────────────────────────
    model = BiGRURegressor(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_path = MODELS_DIR / "bigru_ibm_pretrained.pt"

    print("🚀 [IBM Pretrain Step 2] 開始訓練 BiGRU...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # ── validation ────────────────────────────────
        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # ── 儲存最佳模型 ─────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": INPUT_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "feature_cols": [
                        "daily_expense",
                        "roll_7d_mean",
                        "roll_30d_mean",
                        "dow_sin",
                        "dow_cos",
                    ],
                },
                best_path,
            )
            print(f"✅ 儲存最佳模型：{best_path}")

    print("🎉 [IBM Pretrain Step 2] 完成！")
    print(f"最佳 validation loss：{best_val_loss:.6f}")


if __name__ == "__main__":
    main()