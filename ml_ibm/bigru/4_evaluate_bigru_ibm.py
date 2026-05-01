import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── 路徑設定 ─────────────────────────────────────────────
MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts"
MODELS_DIR = MY_DIR / "models"
OUTPUT_DIR = MY_DIR.parent / "model_outputs" / "bigru"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "bigru_ibm_finetuned_own.pt"


# ── 模型參數：必須跟 pretrain / finetune 一致 ─────────────
INPUT_SIZE = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


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


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def main():
    print("🚀 [IBM BigRU Evaluate] 開始評估 IBM pretrain + own finetune 模型...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 使用裝置：{device}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到 finetune 模型：{MODEL_PATH}")

    # ── 載入 own test data ─────────────────────────────
    X_test = np.load(ARTIFACTS_DIR / "my_X_test.npy").astype(np.float32)
    y_test_raw = np.load(ARTIFACTS_DIR / "my_y_test_raw.npy").astype(np.float32)

    with open(ARTIFACTS_DIR / "my_target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)

    metadata = pd.read_csv(ARTIFACTS_DIR / "sample_metadata.csv")
    test_metadata = metadata[metadata["split"] == "test"].reset_index(drop=True)

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test_raw.shape}")
    print(f"test metadata shape: {test_metadata.shape}")

    # ── 建立模型 ──────────────────────────────────────
    model = BiGRURegressor(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"✅ 已載入模型：{MODEL_PATH}")

    # ── 預測 ─────────────────────────────────────────
    X_tensor = torch.from_numpy(X_test).to(device)

    preds_scaled = []

    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(X_tensor), batch_size):
            xb = X_tensor[i:i + batch_size]
            pred = model(xb).cpu().numpy()
            preds_scaled.append(pred)

    y_pred_scaled = np.vstack(preds_scaled)

    # 還原成原始 target 尺度
    y_pred = target_scaler.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = y_test_raw.reshape(-1)

    # 避免負數預測
    y_pred = np.maximum(y_pred, 0)

    # ── 回歸指標 ─────────────────────────────────────
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mask = y_true > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    smape_value = smape(y_true, y_pred)

    print("📊 Regression Results")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.4f}")
    print(f"SMAPE: {smape_value:.4f}")

    # ── 輸出 predictions.csv ─────────────────────────
    predictions_df = pd.DataFrame({
        "user_id": test_metadata["user_id"],
        "date": test_metadata["date"],
        "y_true": y_true,
        "y_pred": y_pred,
    })

    predictions_path = OUTPUT_DIR / "predictions_bigru_ibm_finetuned.csv"
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    # ── 輸出簡單 metrics ─────────────────────────────
    metrics_df = pd.DataFrame([{
        "model": "bigru_ibm_pretrain_finetune",
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape_value,
    }])

    metrics_path = OUTPUT_DIR / "metrics_bigru_ibm_finetuned.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"💾 預測結果已輸出：{predictions_path}")
    print(f"💾 評估指標已輸出：{metrics_path}")
    print("🎉 [IBM BigRU Evaluate] 完成！")


if __name__ == "__main__":
    main()