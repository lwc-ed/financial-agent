"""
步驟五：LSTM 預測 + 風險評估 + Training Run Summary
"""

import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from model import LSTMModel, get_device

ARTIFACTS_DIR = "artificats"

RISK_THRESHOLDS = {
    "安全": 0.30,
    "注意": 0.50,
    "警告": 0.80,
    "危險": 1.00,
}

device = get_device()


def assess_risk(predicted_expense: float, current_balance: float) -> dict:
    if current_balance <= 0:
        return {
            "risk_level": "危險",
            "emoji": "🔴",
            "ratio": float("inf"),
            "predicted": predicted_expense,
            "balance": current_balance,
            "advice": "餘額為零或負數，請立即注意收支狀況！",
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
        "risk_level": level,
        "emoji": emoji,
        "ratio": round(ratio * 100, 1),
        "predicted": round(predicted_expense, 0),
        "balance": current_balance,
        "advice": advice,
    }


print("📦 載入模型與 scaler...")

checkpoint = torch.load(f"{ARTIFACTS_DIR}/finetune_lstm.pth", map_location=device)
hp = checkpoint["hyperparams"]

model = LSTMModel(
    hp["input_size"],
    hp["hidden_size"],
    hp["num_layers"],
    hp["output_size"],
    hp["dropout"],
).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

print(f"  ✅ 模型載入完成（best epoch: {checkpoint['epoch']}）")

print("\n📂 載入驗證集...")

X_val = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_val = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")
X_train = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/personal_y_train.npy")

print(f"  訓練集 : {X_train.shape[0]} 筆")
print(f"  驗證集 : {X_val.shape[0]} 筆")

print("\n🔮 預測驗證集...")

X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred_scaled = model(X_val_t).cpu().numpy()

y_pred_real = target_scaler.inverse_transform(y_pred_scaled)
y_true_real = target_scaler.inverse_transform(y_val)

mae = np.mean(np.abs(y_pred_real - y_true_real))
rmse = np.sqrt(np.mean((y_pred_real - y_true_real) ** 2))

print(f"  MAE  : {mae:,.2f} 元")
print(f"  RMSE : {rmse:,.2f} 元")

print("\n📊 計算 Baseline...")

df = pd.read_csv("features_all.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

df["naive_7d_sum"] = df.groupby("user_id")["daily_expense"].transform(lambda x: x.rolling(7, min_periods=1).sum())
naive_mae = (df["naive_7d_sum"] - df["future_expense_7d_sum"]).abs().mean()
naive_rmse = ((df["naive_7d_sum"] - df["future_expense_7d_sum"]) ** 2).mean() ** 0.5

df["moving_avg_30d_x7"] = df.groupby("user_id")["daily_expense"].transform(
    lambda x: x.rolling(30, min_periods=1).mean()
) * 7
ma_mae = (df["moving_avg_30d_x7"] - df["future_expense_7d_sum"]).abs().mean()
ma_rmse = ((df["moving_avg_30d_x7"] - df["future_expense_7d_sum"]) ** 2).mean() ** 0.5

beat_moving_avg = mae < ma_mae
print(f"  Naive 7d MAE       : {naive_mae:,.2f}")
print(f"  Moving Avg 30d MAE : {ma_mae:,.2f}")
print(f"  LSTM MAE           : {mae:,.2f}  {'✅ 贏過 baseline' if beat_moving_avg else '❌ 輸給 baseline'}")

print("\n" + "=" * 55)
print("📱 風險評估示範")
print("=" * 55)

sample_input = X_val_t[0:1]
with torch.no_grad():
    sample_pred_scaled = model(sample_input).cpu().numpy()
sample_pred_real = float(target_scaler.inverse_transform(sample_pred_scaled)[0][0])

demo_balance = 15000.0
result = assess_risk(sample_pred_real, demo_balance)

print(f"\n  使用者目前餘額     : NT$ {demo_balance:,.0f}")
print(f"  預測未來7天消費    : NT$ {result['predicted']:,.0f}")
print(f"  消費佔餘額比例     : {result['ratio']}%")
print(f"  風險等級           : {result['emoji']} {result['risk_level']}")
print(f"  建議               : {result['advice']}")

predictions_df = pd.DataFrame(
    {
        "y_true": y_true_real.flatten(),
        "y_pred": y_pred_real.flatten(),
        "error": (y_pred_real - y_true_real).flatten(),
        "abs_error": np.abs(y_pred_real - y_true_real).flatten(),
    }
)
predictions_path = f"{ARTIFACTS_DIR}/predictions_val.csv"
predictions_df.to_csv(predictions_path, index=False)

metrics = {
    "model_name": "lstm_transfer_v1",
    "val_mae": round(float(mae), 6),
    "val_rmse": round(float(rmse), 6),
    "naive_7d_mae": round(float(naive_mae), 6),
    "naive_7d_rmse": round(float(naive_rmse), 6),
    "moving_avg_30d_mae": round(float(ma_mae), 6),
    "moving_avg_30d_rmse": round(float(ma_rmse), 6),
    "beat_moving_avg": bool(beat_moving_avg),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
metrics_path = f"{ARTIFACTS_DIR}/metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

feature_cols = [
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
Training Run Summary — LSTM Transfer Learning
{'=' * 55}
model_name       : lstm_transfer_v1
feature_set      : user_selected_v1
target_column    : future_expense_7d_sum
val_mae          : {mae:,.6f}
val_rmse         : {rmse:,.6f}

Dataset Sizes
  train_rows     : {X_train.shape[0]}
  val_rows       : {X_val.shape[0]}
  input_days     : {X_val.shape[1]}
  feature_count  : {X_val.shape[2]}

Selected Features
{''.join(f'  - {f}{chr(10)}' for f in feature_cols)}Baselines
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
  finetune_model   : {ARTIFACTS_DIR}/finetune_lstm.pth
  pretrain_model   : {ARTIFACTS_DIR}/pretrain_lstm.pth
  feature_scaler   : {ARTIFACTS_DIR}/personal_feature_scaler.pkl
  target_scaler    : {ARTIFACTS_DIR}/personal_target_scaler.pkl
  predictions_csv  : {predictions_path}
  metrics_json     : {metrics_path}
{'=' * 55}
"""

print(summary)

summary_path = f"{ARTIFACTS_DIR}/training_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"📄 Summary 已儲存至：{summary_path}")
print(f"📊 Predictions 已儲存至：{predictions_path}")
print(f"📈 Metrics 已儲存至：{metrics_path}")
