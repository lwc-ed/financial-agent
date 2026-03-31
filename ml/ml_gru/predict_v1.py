"""
版本 V1：預測 + 評估 + 報告輸出
================================
載入 V1 finetune 模型（GRUWithAttention），對驗證集與測試集評估，
並輸出 metrics_v1.json / predictions_v1_*.csv / training_summary_v1.txt
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import json
import os
from datetime import datetime

ARTIFACTS_DIR = "artificats"
VERSION       = "v1"
RESULT_PATH   = "result_v1.txt"


# ─────────────────────────────────────────
# 裝置設定
# ─────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ─────────────────────────────────────────
# V1 模型定義（與 finetune_gru_v1.py 完全一致）
# ─────────────────────────────────────────
class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        context    = self.layer_norm(context)
        out        = self.dropout(context)
        out        = self.relu(self.fc1(out))
        return self.fc2(out)


# ─────────────────────────────────────────
# 載入模型與 scaler
# ─────────────────────────────────────────
print(f"📦 載入 V1 模型與 scaler...")

checkpoint = torch.load(f"{ARTIFACTS_DIR}/finetune_gru_v1.pth", map_location=device)
hp         = checkpoint["hyperparams"]

model = GRUWithAttention(
    hp["input_size"], hp["hidden_size"],
    hp["num_layers"], hp["output_size"], hp["dropout"]
).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

print(f"  ✅ V1 模型載入完成（best epoch: {checkpoint['epoch']}）")
print(f"  架構：hidden={hp['hidden_size']}, layers={hp['num_layers']}, Attention=True")


# ─────────────────────────────────────────
# 載入資料
# ─────────────────────────────────────────
print("\n📂 載入 train / val / test...")

X_val        = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_val        = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")
X_test       = np.load(f"{ARTIFACTS_DIR}/personal_X_test.npy")
y_test       = np.load(f"{ARTIFACTS_DIR}/personal_y_test.npy")
X_train      = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
y_train      = np.load(f"{ARTIFACTS_DIR}/personal_y_train.npy")
val_user_ids  = np.load(f"{ARTIFACTS_DIR}/personal_val_user_ids.npy",  allow_pickle=True)
test_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_test_user_ids.npy", allow_pickle=True)

print(f"  訓練集 : {X_train.shape[0]} 筆")
print(f"  驗證集 : {X_val.shape[0]} 筆")
print(f"  測試集 : {X_test.shape[0]} 筆")


# ─────────────────────────────────────────
# 評估指標
# ─────────────────────────────────────────
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask  = denom > 0
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def per_user_nmae(y_true: np.ndarray, y_pred: np.ndarray, user_ids) -> float:
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    users = np.unique(user_ids)
    nmae_list = []
    for u in users:
        mask   = np.array(user_ids) == u
        mean_u = y_true[mask].mean()
        if mean_u > 0:
            nmae_list.append(
                np.mean(np.abs(y_pred[mask] - y_true[mask])) / mean_u * 100
            )
    return float(np.mean(nmae_list))


def evaluate_split(X_split, y_split, split_name):
    X_t = torch.tensor(X_split, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_t).cpu().numpy()
    y_pred_real = target_scaler.inverse_transform(y_pred_scaled)
    y_true_real = target_scaler.inverse_transform(y_split)
    mae  = float(np.mean(np.abs(y_pred_real - y_true_real)))
    rmse = float(np.sqrt(np.mean((y_pred_real - y_true_real) ** 2)))
    print(f"  {split_name} MAE  : {mae:,.2f} 元")
    print(f"  {split_name} RMSE : {rmse:,.2f} 元")
    return y_pred_real, y_true_real, mae, rmse


print("\n🔮 預測驗證集與測試集...")
y_val_pred_real,  y_val_true_real,  val_mae,  val_rmse  = evaluate_split(X_val,  y_val,  "Val")
y_test_pred_real, y_test_true_real, test_mae, test_rmse = evaluate_split(X_test, y_test, "Test")


# ─────────────────────────────────────────
# Baseline 計算
# ─────────────────────────────────────────
print("\n📊 計算 Baseline...")
EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]
df = pd.read_csv("features_all.csv")
df["date"] = pd.to_datetime(df["date"])
df = df[~df["user_id"].isin(EXCLUDE_USERS)].reset_index(drop=True)
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

df["naive_7d_sum"] = df.groupby("user_id")["daily_expense"].transform(
    lambda x: x.rolling(7, min_periods=1).sum()
)
naive_mae  = float((df["naive_7d_sum"] - df["future_expense_7d_sum"]).abs().mean())
naive_rmse = float(((df["naive_7d_sum"] - df["future_expense_7d_sum"]) ** 2).mean() ** 0.5)

df["moving_avg_30d_x7"] = df.groupby("user_id")["daily_expense"].transform(
    lambda x: x.rolling(30, min_periods=1).mean()
) * 7
ma_mae  = float((df["moving_avg_30d_x7"] - df["future_expense_7d_sum"]).abs().mean())
ma_rmse = float(((df["moving_avg_30d_x7"] - df["future_expense_7d_sum"]) ** 2).mean() ** 0.5)

beat_moving_avg = test_mae < ma_mae
print(f"  Naive 7d MAE       : {naive_mae:,.2f}")
print(f"  Moving Avg 30d MAE : {ma_mae:,.2f}")
print(f"  V1 GRU Test MAE    : {test_mae:,.2f}  {'✅ 贏過 baseline' if beat_moving_avg else '❌ 輸給 baseline'}")


# ─────────────────────────────────────────
# 相對誤差指標
# ─────────────────────────────────────────
print("\n📐 計算相對誤差指標...")
val_smape  = smape(y_val_true_real,  y_val_pred_real)
test_smape = smape(y_test_true_real, y_test_pred_real)
val_nmae   = per_user_nmae(y_val_true_real,  y_val_pred_real,  val_user_ids)
test_nmae  = per_user_nmae(y_test_true_real, y_test_pred_real, test_user_ids)

print(f"  Val  SMAPE         : {val_smape:.2f}%")
print(f"  Test SMAPE         : {test_smape:.2f}%")
print(f"  Val  Per-user NMAE : {val_nmae:.2f}%")
print(f"  Test Per-user NMAE : {test_nmae:.2f}%")


# ─────────────────────────────────────────
# 與 baseline v0 對比
# ─────────────────────────────────────────
# 從 metrics.json 讀取 v0 指標做對比
v0_metrics = {}
v0_path = f"{ARTIFACTS_DIR}/metrics.json"
if os.path.exists(v0_path):
    with open(v0_path) as f:
        v0_metrics = json.load(f)

print("\n📈 V0 vs V1 對比：")
if v0_metrics:
    def delta(v0, v1, label, lower_is_better=True):
        diff = v1 - v0
        sym  = "▼" if (diff < 0 and lower_is_better) else ("▲" if (diff > 0 and lower_is_better) else "—")
        print(f"  {label:25s}: V0={v0:>10.2f}  V1={v1:>10.2f}  {sym}{abs(diff):.2f}")
    delta(v0_metrics.get("test_mae", 0), test_mae, "Test MAE")
    delta(v0_metrics.get("test_rmse", 0), test_rmse, "Test RMSE")
    delta(v0_metrics.get("test_smape", 0), test_smape, "Test SMAPE (%)")
    delta(v0_metrics.get("test_per_user_nmae", 0), test_nmae, "Test Per-user NMAE (%)")
    delta(v0_metrics.get("val_mae", 0), val_mae, "Val MAE")
    delta(v0_metrics.get("val_smape", 0), val_smape, "Val SMAPE (%)")


# ─────────────────────────────────────────
# 儲存 predictions CSV
# ─────────────────────────────────────────
val_pred_df = pd.DataFrame({
    "y_true"    : y_val_true_real.flatten(),
    "y_pred"    : y_val_pred_real.flatten(),
    "error"     : (y_val_pred_real - y_val_true_real).flatten(),
    "abs_error" : np.abs(y_val_pred_real - y_val_true_real).flatten(),
})
val_pred_path = f"{ARTIFACTS_DIR}/predictions_val_v1.csv"
val_pred_df.to_csv(val_pred_path, index=False)

test_pred_df = pd.DataFrame({
    "y_true"    : y_test_true_real.flatten(),
    "y_pred"    : y_test_pred_real.flatten(),
    "error"     : (y_test_pred_real - y_test_true_real).flatten(),
    "abs_error" : np.abs(y_test_pred_real - y_test_true_real).flatten(),
})
test_pred_path = f"{ARTIFACTS_DIR}/predictions_test_v1.csv"
test_pred_df.to_csv(test_pred_path, index=False)


# ─────────────────────────────────────────
# 儲存 metrics JSON
# ─────────────────────────────────────────
metrics = {
    "model_name"          : "gru_transfer_v1_enhanced",
    "version"             : "v1",
    "architecture"        : "GRUWithAttention",
    "hidden_size"         : hp["hidden_size"],
    "val_mae"             : round(val_mae,       6),
    "val_rmse"            : round(val_rmse,      6),
    "val_smape"           : round(val_smape,     4),
    "val_per_user_nmae"   : round(val_nmae,      4),
    "test_mae"            : round(test_mae,      6),
    "test_rmse"           : round(test_rmse,     6),
    "test_smape"          : round(test_smape,    4),
    "test_per_user_nmae"  : round(test_nmae,     4),
    "naive_7d_mae"        : round(naive_mae,     6),
    "naive_7d_rmse"       : round(naive_rmse,    6),
    "moving_avg_30d_mae"  : round(ma_mae,        6),
    "moving_avg_30d_rmse" : round(ma_rmse,       6),
    "beat_moving_avg"     : bool(beat_moving_avg),
    "timestamp"           : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
metrics_path = f"{ARTIFACTS_DIR}/metrics_v1.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────
# Training Run Summary
# ─────────────────────────────────────────
FEATURE_COLS = [
    "daily_expense", "expense_7d_mean", "expense_30d_sum",
    "has_expense", "has_income", "net_30d_sum", "txn_30d_sum",
]

v0_test_mae  = v0_metrics.get("test_mae",  "N/A")
v0_test_smape = v0_metrics.get("test_smape", "N/A")

summary = f"""
{'=' * 60}
Training Run Summary — GRU Transfer Learning V1 Enhanced
{'=' * 60}
model_name        : gru_transfer_v1_enhanced
architecture      : GRUWithAttention (hidden=128, layers=2)
version           : v1
feature_set       : user_selected_v1
target_column     : future_expense_7d_sum
val_mae           : {val_mae:,.6f}
val_rmse          : {val_rmse:,.6f}
val_smape         : {val_smape:.2f}%
val_per_user_nmae : {val_nmae:.2f}%
test_mae          : {test_mae:,.6f}
test_rmse         : {test_rmse:,.6f}
test_smape        : {test_smape:.2f}%
test_per_user_nmae: {test_nmae:.2f}%

V0 vs V1 Comparison
  test_mae  V0={v0_test_mae}  →  V1={test_mae:,.2f}
  test_smape V0={v0_test_smape}%  →  V1={test_smape:.2f}%

Dataset Sizes
  train_rows    : {X_train.shape[0]}
  val_rows      : {X_val.shape[0]}
  test_rows     : {X_test.shape[0]}
  input_days    : {X_val.shape[1]}
  feature_count : {X_val.shape[2]}

V1 Architecture Changes
  - hidden_size     : 64 → 128
  - attention       : None → Temporal Attention (softmax over seq)
  - fc_head         : Linear(64,1) → Linear(128,64) + ReLU + Linear(64,1)
  - layer_norm      : added after attention pooling
  - loss_function   : MSELoss → HuberLoss(delta=1.0)
  - freeze_strategy : partial freeze → full fine-tuning
  - optimizer       : Adam → AdamW(weight_decay=1e-4)
  - lr_scheduler    : ReduceLROnPlateau → CosineAnnealingLR
  - learning_rate   : 5e-4 → 1e-4
  - epochs          : 150 → 200 (patience 20 → 30)

Selected Features
{''.join(f'  - {f}{chr(10)}' for f in FEATURE_COLS)}
Baselines
  naive_7d_sum mae         : {naive_mae:,.6f}
  naive_7d_sum rmse        : {naive_rmse:,.6f}
  moving_avg_30d_x7 mae    : {ma_mae:,.6f}
  moving_avg_30d_x7 rmse   : {ma_rmse:,.6f}
  beat_moving_avg_30d_x7   : {beat_moving_avg}

Artifacts
  pretrain_model  : {ARTIFACTS_DIR}/pretrain_gru_v1.pth
  finetune_model  : {ARTIFACTS_DIR}/finetune_gru_v1.pth
  metrics_json    : {metrics_path}
  val_predictions : {val_pred_path}
  test_predictions: {test_pred_path}
{'=' * 60}
"""

print(summary)

summary_path = f"{ARTIFACTS_DIR}/training_summary_v1.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary)

result_text = f"""GRU Predict Result
model_name: gru_transfer_v1_enhanced
version: {VERSION}
val_mae: {val_mae:,.6f}
val_rmse: {val_rmse:,.6f}
val_smape: {val_smape:.2f}%
val_per_user_nmae: {val_nmae:.2f}%
test_mae: {test_mae:,.6f}
test_rmse: {test_rmse:,.6f}
test_smape: {test_smape:.2f}%
test_per_user_nmae: {test_nmae:.2f}%
naive_7d_mae: {naive_mae:,.6f}
naive_7d_rmse: {naive_rmse:,.6f}
moving_avg_30d_mae: {ma_mae:,.6f}
moving_avg_30d_rmse: {ma_rmse:,.6f}
beat_moving_avg: {beat_moving_avg}
metrics_json: {metrics_path}
val_predictions_csv: {val_pred_path}
test_predictions_csv: {test_pred_path}
summary_txt: {summary_path}
"""
with open(RESULT_PATH, "w", encoding="utf-8") as f:
    f.write(result_text)

print(f"📄 V1 Summary 已儲存：{summary_path}")
print(f"📝 V1 Result 已儲存：{RESULT_PATH}")
print(f"📊 Metrics 已儲存：{metrics_path}")
