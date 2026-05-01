from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts"
MODELS_DIR = MY_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

PRETRAIN_PATH = MODELS_DIR / "bigru_ibm_pretrained.pt"
SAVE_PATH = MODELS_DIR / "bigru_ibm_finetuned_own.pt"

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4


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
        return self.fc(out[:, -1, :])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用裝置：{device}")

    if not PRETRAIN_PATH.exists():
        raise FileNotFoundError(f"找不到 IBM pretrained model：{PRETRAIN_PATH}")

    X_train = np.load(ARTIFACTS_DIR / "my_X_train.npy").astype(np.float32)
    X_val = np.load(ARTIFACTS_DIR / "my_X_val.npy").astype(np.float32)
    y_train = np.load(ARTIFACTS_DIR / "my_y_train_scaled.npy").astype(np.float32)
    y_val = np.load(ARTIFACTS_DIR / "my_y_val_scaled.npy").astype(np.float32)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape  : {X_val.shape}")

    ckpt = torch.load(PRETRAIN_PATH, map_location=device)

    input_size = ckpt["input_size"]
    hidden_size = ckpt["hidden_size"]
    num_layers = ckpt["num_layers"]
    dropout = ckpt["dropout"]

    if X_train.shape[2] != input_size:
        raise ValueError(
            f"Feature 數量不一致：pretrain input_size={input_size}, "
            f"但 own data X_train.shape[2]={X_train.shape[2]}。"
            "請先把 own data preprocessing 改成跟 IBM 一樣的 6 個特徵。"
        )

    model = BiGRURegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print("✅ 已載入 IBM pretrained 權重")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    print("🚀 開始用 own data finetune...")

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

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "source_pretrain": "IBM Credit Card Transactions",
                },
                SAVE_PATH,
            )
            print(f"✅ 儲存最佳 finetune 模型：{SAVE_PATH}")

    print("🎉 Finetune 完成！")
    print(f"最佳 validation loss：{best_val_loss:.6f}")


if __name__ == "__main__":
    main()