"""
步驟五：LSTM 預測 + 評估（Val + Test）+ Training Run Summary
"""

import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from model import LSTMModel, get_device

ARTIFACTS_DIR = "artificats"
EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]

RISK_THRESHOLDS = {
    "安全": 0.30,
    "注意": 0.50,
    "警告": 0.80,
    "危險": 1.00,
}

device = get_device()


# ── 輔助函數 ─────────────────────────────────────────────────────────────────

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)) * 100)


def per_user_nmae(y_true: np.ndarray, y_pred: np.ndarray, user_ids) -> float:
    """每個 user 的 MAE 除以該 user y_true 平均，再對所有 user 取平均（%）"""
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    users = np.unique(user_ids)
    nmae_list = []
    for u in users:
        mask = np.array(user_ids) == u
        mean_u = y_true[mask].mean()
        if mean_u > 0:
            nmae_list.append(np.mean(np.abs(y_pred[mask] - y_true[mask])) / mean_u * 100)
    return float(np.mean(nmae_list))


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


# ── 1. 載入模型 ──────────────────────────────────────────────────────────────
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

# ── 2. 載入資料 ──────────────────────────────────────────────────────────────
print("\n📂 載入 Train / Val / Test...")

X_train = np.load(f"{ARTIFACTS_DIR}/personal_X_train.npy")
y_train = np.load(f"{ARTIFACTS_DIR}/personal_y_train.npy")
X_val   = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_val   = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")
X_test  = np.load(f"{ARTIFACTS_DIR}/personal_X_test.npy")
y_test  = np.load(f"{ARTIFACTS_DIR}/personal_y_test.npy")

val_user_ids  = np.load(f"{ARTIFACTS_DIR}/personal_val_user_ids.npy",  allow_pickle=True)
test_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_test_user_ids.npy", allow_pickle=True)

print(f"  訓練集 : {X_train.shape[0]} 筆")
print(f"  驗證集 : {X_val.shape[0]} 筆")
print(f"  測試集 : {X_test.shape[0]} 筆")


# ── 3. 預測 Val ──────────────────────────────────────────────────────────────
print("\n🔮 預測 Val...")

X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
with torch.no_grad():
    y_val_pred_sc = model(X_val_t).cpu().numpy()

y_val_pred = target_scaler.inverse_transform(y_val_pred_sc)
y_val_true = target_scaler.inverse_transform(y_val)

val_mae   = float(np.mean(np.abs(y_val_pred - y_val_true)))
val_rmse  = float(np.sqrt(np.mean((y_val_pred - y_val_true) ** 2)))
val_smape = smape(y_val_true, y_val_pred)
val_nmae  = per_user_nmae(y_val_true, y_val_pred, val_user_ids)

print(f"  Val MAE        : {val_mae:,.2f}")
print(f"  Val RMSE       : {val_rmse:,.2f}")
print(f"  Val SMAPE      : {val_smape:.2f}%")
print(f"  Val NMAE       : {val_nmae:.2f}%")

# ── 4. 預測 Test ─────────────────────────────────────────────────────────────
print("\n🔮 預測 Test（最終評估）...")

X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    y_test_pred_sc = model(X_test_t).cpu().numpy()

y_test_pred = target_scaler.inverse_transform(y_test_pred_sc)
y_test_true = target_scaler.inverse_transform(y_test)

test_mae   = float(np.mean(np.abs(y_test_pred - y_test_true)))
test_rmse  = float(np.sqrt(np.mean((y_test_pred - y_test_true) ** 2)))
test_smape = smape(y_test_true, y_test_pred)
test_nmae  = per_user_nmae(y_test_true, y_test_pred, test_user_ids)

print(f"  Test MAE       : {test_mae:,.2f}")
print(f"  Test RMSE      : {test_rmse:,.2f}")
print(f"  Test SMAPE     : {test_smape:.2f}%")
print(f"  Test NMAE      : {test_nmae:.2f}%")

# ── 5. Baseline 計算（只用 12 人）────────────────────────────────────────────
print("\n📊 計算 Baseline...")

df_base = pd.read_csv("features_all.csv")
df_base["date"] = pd.to_datetime(df_base["date"])
df_base = df_base[~df_base["user_id"].isin(EXCLUDE_USERS)]
df_base = df_base.sort_values(["user_id", "date"]).reset_index(drop=True)

df_base["naive_7d_sum"] = df_base.groupby("user_id")["daily_expense"].transform(
    lambda x: x.rolling(7, min_periods=1).sum()
)
naive_mae  = float((df_base["naive_7d_sum"] - df_base["future_expense_7d_sum"]).abs().mean())
naive_rmse = float(((df_base["naive_7d_sum"] - df_base["future_expense_7d_sum"]) ** 2).mean() ** 0.5)

df_base["moving_avg_30d_x7"] = df_base.groupby("user_id")["daily_expense"].transform(
    lambda x: x.rolling(30, min_periods=1).mean()
) * 7
ma_mae  = float((df_base["moving_avg_30d_x7"] - df_base["future_expense_7d_sum"]).abs().mean())
ma_rmse = float(((df_base["moving_avg_30d_x7"] - df_base["future_expense_7d_sum"]) ** 2).mean() ** 0.5)

beat_moving_avg = test_mae < ma_mae
print(f"  Naive 7d MAE       : {naive_mae:,.2f}")
print(f"  Moving Avg 30d MAE : {ma_mae:,.2f}")
print(f"  LSTM Test MAE      : {test_mae:,.2f}  {'✅ 贏過 baseline' if beat_moving_avg else '❌ 輸給 baseline'}")

# ── 6. 風險評估示範 ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("📱 風險評估示範")
print("=" * 55)

sample_pred_real = float(target_scaler.inverse_transform(y_test_pred_sc[0:1])[0][0])
demo_balance = 15000.0
result = assess_risk(sample_pred_real, demo_balance)

print(f"\n  使用者目前餘額     : NT$ {demo_balance:,.0f}")
print(f"  預測未來7天消費    : NT$ {result['predicted']:,.0f}")
print(f"  消費佔餘額比例     : {result['ratio']}%")
print(f"  風險等級           : {result['emoji']} {result['risk_level']}")
print(f"  建議               : {result['advice']}")

# ── 7. 儲存 predictions CSV ──────────────────────────────────────────────────
test_pred_df = pd.DataFrame({
    "user_id":   test_user_ids,
    "y_true":    y_test_true.flatten(),
    "y_pred":    y_test_pred.flatten(),
    "error":     (y_test_pred - y_test_true).flatten(),
    "abs_error": np.abs(y_test_pred - y_test_true).flatten(),
})
predictions_path = f"{ARTIFACTS_DIR}/predictions_test.csv"
test_pred_df.to_csv(predictions_path, index=False)

val_pred_df = pd.DataFrame({
    "user_id":   val_user_ids,
    "y_true":    y_val_true.flatten(),
    "y_pred":    y_val_pred.flatten(),
    "error":     (y_val_pred - y_val_true).flatten(),
    "abs_error": np.abs(y_val_pred - y_val_true).flatten(),
})
val_predictions_path = f"{ARTIFACTS_DIR}/predictions_val.csv"
val_pred_df.to_csv(val_predictions_path, index=False)

# ── 8. metrics.json ──────────────────────────────────────────────────────────
metrics = {
    "model_name"          : "lstm_transfer_v1",
    "val_mae"             : round(val_mae,   6),
    "val_rmse"            : round(val_rmse,  6),
    "val_smape"           : round(val_smape, 4),
    "val_per_user_nmae"   : round(val_nmae,  4),
    "test_mae"            : round(test_mae,  6),
    "test_rmse"           : round(test_rmse, 6),
    "test_smape"          : round(test_smape,4),
    "test_per_user_nmae"  : round(test_nmae, 4),
    "naive_7d_mae"        : round(naive_mae, 6),
    "naive_7d_rmse"       : round(naive_rmse,6),
    "moving_avg_30d_mae"  : round(ma_mae,    6),
    "moving_avg_30d_rmse" : round(ma_rmse,   6),
    "beat_moving_avg"     : bool(beat_moving_avg),
    "timestamp"           : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
metrics_path = f"{ARTIFACTS_DIR}/metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# ── 9. training_summary.txt ──────────────────────────────────────────────────
feature_cols = [
    "daily_expense", "expense_7d_mean", "expense_30d_sum",
    "has_expense", "has_income", "net_30d_sum", "txn_30d_sum",
]

summary = f"""
{'=' * 60}
Training Run Summary — LSTM Transfer Learning
{'=' * 60}
model_name        : lstm_transfer_v1
target_column     : future_expense_7d_sum
data_split        : per-user 70 / 15 / 15（時序切割）
excluded_users    : {EXCLUDE_USERS}
timestamp         : {metrics['timestamp']}

{'─' * 60}
Validation Set
{'─' * 60}
  val_mae          : {val_mae:>12,.4f}
  val_rmse         : {val_rmse:>12,.4f}
  val_smape        : {val_smape:>12.2f} %
  val_per_user_nmae: {val_nmae:>12.2f} %

{'─' * 60}
Test Set（最終評估）
{'─' * 60}
  test_mae         : {test_mae:>12,.4f}
  test_rmse        : {test_rmse:>12,.4f}
  test_smape       : {test_smape:>12.2f} %
  test_per_user_nmae:{test_nmae:>11.2f} %

{'─' * 60}
Dataset Sizes
{'─' * 60}
  train_rows       : {X_train.shape[0]}
  val_rows         : {X_val.shape[0]}
  test_rows        : {X_test.shape[0]}
  input_days       : {X_test.shape[1]}
  feature_count    : {X_test.shape[2]}

{'─' * 60}
Selected Features
{'─' * 60}
{''.join(f'  - {f}{chr(10)}' for f in feature_cols)}
{'─' * 60}
Baselines
{'─' * 60}
  naive_7d_sum mae          : {naive_mae:>12,.4f}
  naive_7d_sum rmse         : {naive_rmse:>12,.4f}
  moving_avg_30d_x7 mae     : {ma_mae:>12,.4f}
  moving_avg_30d_x7 rmse    : {ma_rmse:>12,.4f}
  beat_moving_avg_30d_x7    : {beat_moving_avg}

{'─' * 60}
Risk Thresholds
{'─' * 60}
  🟢 安全  : 預測消費 < 餘額 30%
  🟡 注意  : 預測消費 < 餘額 50%
  🟠 警告  : 預測消費 < 餘額 80%
  🔴 危險  : 預測消費 ≥ 餘額 80%

{'─' * 60}
Artifacts
{'─' * 60}
  finetune_model    : {ARTIFACTS_DIR}/finetune_lstm.pth
  pretrain_model    : {ARTIFACTS_DIR}/pretrain_lstm.pth
  feature_scaler    : {ARTIFACTS_DIR}/personal_feature_scaler.pkl
  target_scaler     : {ARTIFACTS_DIR}/personal_target_scaler.pkl
  predictions_test  : {predictions_path}
  predictions_val   : {val_predictions_path}
  metrics_json      : {metrics_path}
{'=' * 60}
"""

print(summary)

summary_path = f"{ARTIFACTS_DIR}/training_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"📄 Summary 已儲存至：{summary_path}")
print(f"📊 Predictions 已儲存至：{predictions_path}")
print(f"📈 Metrics 已儲存至：{metrics_path}")
