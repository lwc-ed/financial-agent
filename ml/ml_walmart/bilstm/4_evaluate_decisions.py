"""
Step 4：Bi-LSTM baseline 預測與評估（30 seeds）
"""

import pickle
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

MY_DIR        = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts"
PROJECT_ROOT  = MY_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from ml.output_eval_utils import run_output_evaluation, compute_per_seed_metrics

HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.3
OUTPUT_SIZE = 1

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


class MyBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc1     = nn.Linear(hidden_size * 2, 32)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(32, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)


def get_all_preds(X: np.ndarray, input_size: int) -> dict:
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    all_preds = {}
    for seed in SEEDS:
        ckpt  = torch.load(ARTIFACTS_DIR / f"bilstm_seed{seed}.pth", map_location=device)
        model = MyBiLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            all_preds[seed] = model(X_t).cpu().numpy()
    return all_preds


def main():
    print("🚀 [Step 4] Bi-LSTM baseline 評估（30 seeds）...")

    X_val        = np.load(ARTIFACTS_DIR / "my_X_val.npy").astype(np.float32)
    X_test       = np.load(ARTIFACTS_DIR / "my_X_test.npy").astype(np.float32)
    y_val_scaled = np.load(ARTIFACTS_DIR / "my_y_val_scaled.npy").astype(np.float32)
    y_test_raw   = np.load(ARTIFACTS_DIR / "my_y_test_raw.npy").astype(np.float32).ravel()
    input_size   = X_test.shape[2]

    with open(ARTIFACTS_DIR / "my_target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)

    metadata_df       = pd.read_csv(ARTIFACTS_DIR / "metadata.csv")
    split_metadata_df = metadata_df[["user_id", "date", "split"]].copy()
    split_metadata_df["date"] = pd.to_datetime(split_metadata_df["date"])
    test_meta = split_metadata_df[split_metadata_df["split"] == "test"].reset_index(drop=True)

    y_val_raw = target_scaler.inverse_transform(y_val_scaled)

    print("🔮 推論所有 seed...")
    val_preds_all  = get_all_preds(X_val,  input_size)
    test_preds_all = get_all_preds(X_test, input_size)

    print("🔍 貪婪搜尋最佳 seed 組合（依 val MAE）...")
    best_val_mae = float("inf")
    best_combo   = []
    remaining    = list(SEEDS)
    for _ in range(len(SEEDS)):
        best_new = None
        for cand in remaining:
            combo_try = best_combo + [cand]
            val_pred  = target_scaler.inverse_transform(
                np.mean([val_preds_all[s] for s in combo_try], axis=0))
            mae = float(np.mean(np.abs(y_val_raw - val_pred)))
            if mae < best_val_mae:
                best_val_mae = mae
                best_new     = cand
        if best_new is None:
            break
        best_combo.append(best_new)
        remaining.remove(best_new)
    print(f"  最佳 combo: seeds={best_combo}  val MAE={best_val_mae:.2f}")

    test_preds = target_scaler.inverse_transform(
        np.mean([test_preds_all[s] for s in best_combo], axis=0)).ravel()
    test_preds = np.maximum(test_preds, 0.0)

    mae  = float(np.mean(np.abs(y_test_raw - test_preds)))
    rmse = float(np.sqrt(np.mean((y_test_raw - test_preds) ** 2)))
    print(f"  Test MAE={mae:.2f}  RMSE={rmse:.2f}")

    prediction_input_df = pd.DataFrame({
        "user_id": test_meta["user_id"].values,
        "date":    pd.to_datetime(test_meta["date"].values),
        "y_true":  y_test_raw,
        "y_pred":  test_preds,
    })

    print("\n📊 呼叫共用評估器...")
    run_output_evaluation(
        model_name="bilstm",
        prediction_input_df=prediction_input_df,
        split_metadata_df=split_metadata_df,
    )

    print("\n📊 計算每個 seed 個別指標...")
    compute_per_seed_metrics(
        seed_preds_dict=test_preds_all,
        target_scaler=target_scaler,
        prediction_input_df=prediction_input_df,
        split_metadata_df=split_metadata_df,
        output_dir=MY_DIR.parent / "model_outputs" / "bilstm",
    )

    print("🎉 Bi-LSTM baseline 評估完成。")


if __name__ == "__main__":
    main()
