import sys
from pathlib import Path

# 讓 `from ml.xxx import ...` 可以正常 import
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml.output_eval_utils import run_output_evaluation


# ── 1. 重新定義相同的模型架構 ──────────────────────────────────────────
class MyBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(MyBiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        predictions = self.fc(out)
        return predictions


def main():
    print("🚀 [Step 3] 開始進行 Bi-GRU 模型期末考 (最終評估)...")

    # ── 2. 路徑設定與載入測試資料 ────────────────────────────────────────
    ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

    X_test = np.load(ARTIFACTS_DIR / "my_X_test.npy").astype(np.float32)
    y_test_raw = np.load(ARTIFACTS_DIR / "my_y_test_raw.npy").astype(np.float32)

    with open(ARTIFACTS_DIR / "my_target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)

    metadata_path = ARTIFACTS_DIR / "sample_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到 sample metadata: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path)
    required_meta_cols = {"user_id", "date", "split"}
    if not required_meta_cols.issubset(metadata_df.columns):
        raise ValueError(f"sample_metadata.csv 缺少必要欄位: {required_meta_cols - set(metadata_df.columns)}")

    split_metadata_df = metadata_df[["user_id", "date", "split"]].copy()
    split_metadata_df["date"] = pd.to_datetime(split_metadata_df["date"]).dt.strftime("%Y-%m-%d")

    test_meta = split_metadata_df[split_metadata_df["split"] == "test"].reset_index(drop=True)

    # ── 3. 載入訓練好的模型權重 ──────────────────────────────────────────
    INPUT_SIZE = X_test.shape[2]
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyBiGRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)

    model_path = ARTIFACTS_DIR / "best_standard_bigru.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print(f"🔮 正在對 {len(X_test)} 筆未知的測試集進行預測...")

    # ── 4. 進行預測 ──────────────────────────────────────────────────────
    X_test_tensor = torch.tensor(X_test).to(device)
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).cpu().numpy()

    # ── 5. 反標準化 ──────────────────────────────────────────────────────
    predictions_real = target_scaler.inverse_transform(predictions_scaled)

    # flatten
    y_test_raw = y_test_raw.reshape(-1)
    predictions_real = predictions_real.reshape(-1)

    if len(test_meta) != len(y_test_raw):
        raise ValueError(
            f"test metadata 筆數 ({len(test_meta)}) 與 y_test_raw 筆數 ({len(y_test_raw)}) 不一致"
        )
    if len(test_meta) != len(predictions_real):
        raise ValueError(
            f"test metadata 筆數 ({len(test_meta)}) 與 predictions_real 筆數 ({len(predictions_real)}) 不一致"
        )

    # 支出不應為負值
    predictions_real = np.maximum(predictions_real, 0.0)

    # ── 6. 計算最終誤差 ──────────────────────────────────────────────────
    mae = mean_absolute_error(y_test_raw, predictions_real)
    rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_real))
    smape = np.mean(
        2.0 * np.abs(predictions_real - y_test_raw) /
        (np.abs(predictions_real) + np.abs(y_test_raw) + 1e-8)
    ) * 100

    mask = y_test_raw > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((predictions_real[mask] - y_test_raw[mask]) / y_test_raw[mask])) * 100
    else:
        mape = np.nan

    # ── 7. 組 spec 所需 dataframe ───────────────────────────────────────
    prediction_input_df = test_meta[["user_id", "date"]].copy()
    prediction_input_df["y_true"] = y_test_raw
    prediction_input_df["y_pred"] = predictions_real

    # ── 8. 本地輸出（debug 用） ─────────────────────────────────────────
    local_pred_path = ARTIFACTS_DIR / "bigru_prediction_input.csv"
    local_metrics_path = ARTIFACTS_DIR / "bigru_local_metrics.json"

    prediction_input_df.to_csv(local_pred_path, index=False, encoding="utf-8-sig")

    local_metrics = {
        "model_name": "bigru",
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": None if np.isnan(mape) else float(mape),
        "smape": float(smape),
        "rows": int(len(prediction_input_df)),
    }

    with open(local_metrics_path, "w", encoding="utf-8") as f:
        json.dump(local_metrics, f, indent=2, ensure_ascii=False)

    # ── 9. 舊 txt 報告（保留） ───────────────────────────────────────────
    report_content = f"""Bi-GRU Baseline Result
model_name: bigru
version: v2_with_spec
pretrained: False (Trained from scratch)
test_mae: {mae:.2f}
test_rmse: {rmse:.2f}
test_mape: {mape:.2f}%
test_smape: {smape:.2f}%

==============================================================
            My Standard Bi-GRU 獨立作戰結果 (越低越好)
==============================================================
  Test MAE   : {mae:.2f}
  Test RMSE  : {rmse:.2f}
  Test MAPE  : {mape:.2f}%
  Test sMAPE : {smape:.2f}%
==============================================================
"""

    print(report_content)

    report_path = ARTIFACTS_DIR / "standard_bigru_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"💾 本地 prediction_input 已儲存：{local_pred_path}")
    print(f"💾 本地 metrics 已儲存：{local_metrics_path}")
    print(f"💾 報告已成功儲存：{report_path}")

    # ── 10. 呼叫共用 evaluator（正式 spec） ─────────────────────────────
    run_output_evaluation(
        model_name="bigru",
        prediction_input_df=prediction_input_df,
        split_metadata_df=split_metadata_df,
    )

    print("✅ 已完成 shared evaluator 正式輸出 -> ml/model_outputs/bigru/")
    print("🎉 標準版 Bi-GRU baseline 評估完成。")


if __name__ == "__main__":
    main()