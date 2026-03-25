"""
步驟五：預測 + 風險評估 + Training Run Summary
================================
功能：
  1. 載入 finetune 好的模型，對驗證集與測試集做預測
  2. 計算 Val/Test MAE / RMSE，對比 baseline
  3. 示範單一使用者的風險評估
  4. 輸出 Training Run Summary（對齊 HGBR 格式）
  5. 儲存 predictions CSV 和 metrics JSON

風險等級（依預測消費佔餘額比例）：
  < 30%  → 🟢 安全
  < 50%  → 🟡 注意
  < 80%  → 🟠 警告
  ≥ 80%  → 🔴 危險
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

# ─────────────────────────────────────────
# 風險閾值設定
# ─────────────────────────────────────────
RISK_THRESHOLDS = {
    "安全": 0.30,   # 預測消費 < 餘額 30%
    "注意": 0.50,   # 預測消費 < 餘額 50%
    "警告": 0.80,   # 預測消費 < 餘額 80%
    "危險": 1.00,   # 預測消費 ≥ 餘額 80%
}


# ─────────────────────────────────────────
# 1. 裝置設定
# ─────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ─────────────────────────────────────────
# 2. 模型定義（跟 finetune 一樣）
# ─────────────────────────────────────────
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru     = nn.GRU(input_size, hidden_size, num_layers,
                              dropout=dropout if num_layers > 1 else 0,
                              batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)


# ─────────────────────────────────────────
# 3. 載入模型與 scaler
# ─────────────────────────────────────────
print("📦 載入模型與 scaler...")

checkpoint = torch.load(f"{ARTIFACTS_DIR}/finetune_gru.pth", map_location=device)
hp         = checkpoint["hyperparams"]

model = GRUModel(
    hp["input_size"], hp["hidden_size"],
    hp["num_layers"], hp["output_size"], hp["dropout"]
).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

print(f"  ✅ 模型載入完成（best epoch: {checkpoint['epoch']}）")


# ─────────────────────────────────────────
# 4. 載入資料
# ─────────────────────────────────────────
print("\n📂 載入 train / val / test...")

X_val = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_val = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")
X_test = np.load(f"{ARTIFACTS_DIR}/personal_X_test.npy")
y_test = np.load(f"{ARTIFACTS_DIR}/personal_y_test.npy")
X_train = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/personal_y_train.npy")

print(f"  訓練集 : {X_train.shape[0]} 筆")
print(f"  驗證集 : {X_val.shape[0]} 筆")
print(f"  測試集 : {X_test.shape[0]} 筆")


# ─────────────────────────────────────────
# 5. 對驗證集 / 測試集做預測
# ─────────────────────────────────────────
print("\n🔮 預測驗證集與測試集...")


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask  = denom > 0
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def per_user_nmae(y_true: np.ndarray, y_pred: np.ndarray, user_ids: list) -> float:
    """每個 user 的 MAE 除以該 user 的 y_true 平均，再對所有 user 取平均（%）"""
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    users = np.unique(user_ids)
    nmae_list = []
    for u in users:
        mask   = np.array(user_ids) == u
        mean_u = y_true[mask].mean()
        if mean_u > 0:
            nmae_list.append(np.mean(np.abs(y_pred[mask] - y_true[mask])) / mean_u * 100)
    return float(np.mean(nmae_list))


def evaluate_split(X_split: np.ndarray, y_split: np.ndarray, split_name: str):
    X_split_t = torch.tensor(X_split, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_scaled = model(X_split_t).cpu().numpy()

    y_pred_real = target_scaler.inverse_transform(y_pred_scaled)
    y_true_real = target_scaler.inverse_transform(y_split)
    mae  = np.mean(np.abs(y_pred_real - y_true_real))
    rmse = np.sqrt(np.mean((y_pred_real - y_true_real) ** 2))

    print(f"  {split_name} MAE  : {mae:,.2f} 元")
    print(f"  {split_name} RMSE : {rmse:,.2f} 元")
    return y_pred_real, y_true_real, mae, rmse


y_val_pred_real, y_val_true_real, val_mae, val_rmse = evaluate_split(X_val, y_val, "Val")
y_test_pred_real, y_test_true_real, test_mae, test_rmse = evaluate_split(X_test, y_test, "Test")


# ─────────────────────────────────────────
# 6. Baseline 計算
# ─────────────────────────────────────────
print("\n📊 計算 Baseline...")

df = pd.read_csv("features_all.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

# Naive：用過去 7 天加總當預測
df["naive_7d_sum"] = df.groupby("user_id")["daily_expense"].transform(
    lambda x: x.rolling(7, min_periods=1).sum()
)
naive_mae  = (df["naive_7d_sum"] - df["future_expense_7d_sum"]).abs().mean()
naive_rmse = ((df["naive_7d_sum"] - df["future_expense_7d_sum"]) ** 2).mean() ** 0.5

# Moving Average：過去 30 天均值 × 7
df["moving_avg_30d_x7"] = df.groupby("user_id")["daily_expense"].transform(
    lambda x: x.rolling(30, min_periods=1).mean()
) * 7
ma_mae  = (df["moving_avg_30d_x7"] - df["future_expense_7d_sum"]).abs().mean()
ma_rmse = ((df["moving_avg_30d_x7"] - df["future_expense_7d_sum"]) ** 2).mean() ** 0.5

beat_moving_avg = test_mae < ma_mae
print(f"  Naive 7d MAE       : {naive_mae:,.2f}")
print(f"  Moving Avg 30d MAE : {ma_mae:,.2f}")
print(f"  GRU Test MAE       : {test_mae:,.2f}  {'✅ 贏過 baseline' if beat_moving_avg else '❌ 輸給 baseline'}")


# ─────────────────────────────────────────
# 6b. 相對誤差指標（SMAPE、Per-user NMAE）
# ─────────────────────────────────────────
print("\n📐 計算相對誤差指標...")

INPUT_DAYS = 30
FEATURE_COLS_PREPROCESS = [
    "daily_expense", "expense_7d_mean", "expense_30d_sum",
    "has_expense", "has_income", "net_30d_sum", "txn_30d_sum",
]
TARGET_COL = "future_expense_7d_sum"

# 重建 val/test 各樣本對應的 user_id（與 preprocess_personal.py 切法一致）
val_user_ids, test_user_ids = [], []
for user_id in df["user_id"].unique():
    u = df[df["user_id"] == user_id].reset_index(drop=True)
    u = u.dropna(subset=[TARGET_COL])
    n_windows = len(u) - INPUT_DAYS
    if n_windows <= 0:
        continue
    t_end = int(n_windows * 0.70)
    v_end = int(n_windows * 0.85)
    val_user_ids.extend([user_id]  * (v_end - t_end))
    test_user_ids.extend([user_id] * (n_windows - v_end))

val_smape   = smape(y_val_true_real,  y_val_pred_real)
test_smape  = smape(y_test_true_real, y_test_pred_real)
val_nmae    = per_user_nmae(y_val_true_real,  y_val_pred_real,  val_user_ids)
test_nmae   = per_user_nmae(y_test_true_real, y_test_pred_real, test_user_ids)

print(f"  Val  SMAPE          : {val_smape:.2f}%")
print(f"  Test SMAPE          : {test_smape:.2f}%")
print(f"  Val  Per-user NMAE  : {val_nmae:.2f}%")
print(f"  Test Per-user NMAE  : {test_nmae:.2f}%")


# ─────────────────────────────────────────
# 7. 風險評估函數
# ─────────────────────────────────────────
def assess_risk(predicted_expense: float, current_balance: float) -> dict:
    """
    根據預測消費和目前餘額計算風險等級

    Args:
        predicted_expense: 預測未來 7 天總消費（元）
        current_balance:   使用者目前帳戶餘額（元）

    Returns:
        dict 包含風險等級、比例、建議
    """
    if current_balance <= 0:
        return {
            "risk_level"  : "危險",
            "emoji"       : "🔴",
            "ratio"       : float("inf"),
            "predicted"   : predicted_expense,
            "balance"     : current_balance,
            "advice"      : "餘額為零或負數，請立即注意收支狀況！"
        }

    ratio = predicted_expense / current_balance

    if ratio < RISK_THRESHOLDS["安全"]:
        level, emoji, advice = "安全", "🟢", "消費在合理範圍內，繼續保持！"
    elif ratio < RISK_THRESHOLDS["注意"]:
        level, emoji, advice = "注意", "🟡", "本週消費略高，建議減少非必要支出。"
    elif ratio < RISK_THRESHOLDS["警告"]:
        level, emoji, advice = "警告", "🟠", "本週消費偏高，請謹慎管控支出！"
    else:
        level, emoji, advice = "危險", "🔴", "預測消費超過餘額 80%，有透支風險，請立即調整！"

    return {
        "risk_level"  : level,
        "emoji"       : emoji,
        "ratio"       : round(ratio * 100, 1),   # 轉成百分比
        "predicted"   : round(predicted_expense, 0),
        "balance"     : current_balance,
        "advice"      : advice
    }


# ─────────────────────────────────────────
# 8. 示範：單一使用者風險評估
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("📱 風險評估示範")
print("=" * 55)

# 用驗證集第一筆做示範
sample_input = torch.tensor(X_test[0:1], dtype=torch.float32).to(device)
with torch.no_grad():
    sample_pred_scaled = model(sample_input).cpu().numpy()
sample_pred_real = float(target_scaler.inverse_transform(sample_pred_scaled)[0][0])

# 模擬使用者輸入餘額（實際使用時改成真實餘額）
demo_balance = 15000.0

result = assess_risk(sample_pred_real, demo_balance)

print(f"\n  使用者目前餘額     : NT$ {demo_balance:,.0f}")
print(f"  預測未來7天消費    : NT$ {result['predicted']:,.0f}")
print(f"  消費佔餘額比例     : {result['ratio']}%")
print(f"  風險等級           : {result['emoji']} {result['risk_level']}")
print(f"  建議               : {result['advice']}")


# ─────────────────────────────────────────
# 9. 儲存 predictions CSV
# ─────────────────────────────────────────
val_predictions_df = pd.DataFrame({
    "y_true": y_val_true_real.flatten(),
    "y_pred": y_val_pred_real.flatten(),
    "error": (y_val_pred_real - y_val_true_real).flatten(),
    "abs_error": np.abs(y_val_pred_real - y_val_true_real).flatten(),
})
val_predictions_path = f"{ARTIFACTS_DIR}/predictions_val.csv"
val_predictions_df.to_csv(val_predictions_path, index=False)

test_predictions_df = pd.DataFrame({
    "y_true": y_test_true_real.flatten(),
    "y_pred": y_test_pred_real.flatten(),
    "error": (y_test_pred_real - y_test_true_real).flatten(),
    "abs_error": np.abs(y_test_pred_real - y_test_true_real).flatten(),
})
test_predictions_path = f"{ARTIFACTS_DIR}/predictions_test.csv"
test_predictions_df.to_csv(test_predictions_path, index=False)


# ─────────────────────────────────────────
# 10. 儲存 metrics JSON
# ─────────────────────────────────────────
metrics = {
    "model_name"           : "gru_transfer_v1",
    "val_mae"              : round(float(val_mae), 6),
    "val_rmse"             : round(float(val_rmse), 6),
    "val_smape"            : round(val_smape, 4),
    "val_per_user_nmae"    : round(val_nmae, 4),
    "test_mae"             : round(float(test_mae), 6),
    "test_rmse"            : round(float(test_rmse), 6),
    "test_smape"           : round(test_smape, 4),
    "test_per_user_nmae"   : round(test_nmae, 4),
    "naive_7d_mae"         : round(float(naive_mae), 6),
    "naive_7d_rmse"        : round(float(naive_rmse), 6),
    "moving_avg_30d_mae"   : round(float(ma_mae), 6),
    "moving_avg_30d_rmse"  : round(float(ma_rmse), 6),
    "beat_moving_avg"      : bool(beat_moving_avg),
    "timestamp"            : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
metrics_path = f"{ARTIFACTS_DIR}/metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────
# 11. Training Run Summary（對齊 HGBR 格式）
# ─────────────────────────────────────────
FEATURE_COLS = [
    "daily_expense",
    "expense_7d_mean",
    "expense_30d_sum",
    "has_expense",
    "has_income",
    "net_30d_sum",
    "txn_30d_sum",
]

summary = f"""
{'=' * 55}
Training Run Summary — GRU Transfer Learning
{'=' * 55}
model_name       : gru_transfer_v1
feature_set      : user_selected_v1
target_column    : future_expense_7d_sum
val_mae          : {val_mae:,.6f}
val_rmse         : {val_rmse:,.6f}
val_smape        : {val_smape:.2f}%
val_per_user_nmae: {val_nmae:.2f}%
test_mae         : {test_mae:,.6f}
test_rmse        : {test_rmse:,.6f}
test_smape       : {test_smape:.2f}%
test_per_user_nmae: {test_nmae:.2f}%

Dataset Sizes
  train_rows     : {X_train.shape[0]}
  val_rows       : {X_val.shape[0]}
  test_rows      : {X_test.shape[0]}
  input_days     : {X_val.shape[1]}
  feature_count  : {X_val.shape[2]}

Selected Features
{''.join(f'  - {f}{chr(10)}' for f in FEATURE_COLS)}
Baselines
  naive_7d_sum mae          : {naive_mae:,.6f}
  naive_7d_sum rmse         : {naive_rmse:,.6f}
  moving_avg_30d_x7 mae     : {ma_mae:,.6f}
  moving_avg_30d_x7 rmse    : {ma_rmse:,.6f}
  beat_moving_avg_30d_x7    : {beat_moving_avg}

Risk Thresholds
  🟢 安全  : 預測消費 < 餘額 30%
  🟡 注意  : 預測消費 < 餘額 50%
  🟠 警告  : 預測消費 < 餘額 80%
  🔴 危險  : 預測消費 ≥ 餘額 80%

Artifacts
  finetune_model   : {ARTIFACTS_DIR}/finetune_gru.pth
  pretrain_model   : {ARTIFACTS_DIR}/pretrain_gru.pth
  feature_scaler   : {ARTIFACTS_DIR}/personal_feature_scaler.pkl
  target_scaler    : {ARTIFACTS_DIR}/personal_target_scaler.pkl
  val_predictions_csv : {val_predictions_path}
  test_predictions_csv: {test_predictions_path}
  metrics_json     : {metrics_path}
{'=' * 55}
"""

print(summary)

# 同時存成文字檔
summary_path = f"{ARTIFACTS_DIR}/training_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"📄 Summary 已儲存至：{summary_path}")
print(f"📊 Val Predictions 已儲存至：{val_predictions_path}")
print(f"📊 Test Predictions 已儲存至：{test_predictions_path}")
print(f"📈 Metrics 已儲存至：{metrics_path}")
