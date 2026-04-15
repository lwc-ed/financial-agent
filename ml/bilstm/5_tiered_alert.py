"""
Step 5：三級警報評估（ckh BiLSTM）
=====================================
將二元預警擴展為三個等級：
  🟢 正常    ：預測值 < 個人基線 × 1.2
  🟡 低度警告：個人基線 × 1.2 ≤ 預測值 < 個人基線 × 1.8
  🔴 高度警告：預測值 ≥ 個人基線 × 1.8

注意：此腳本沿用 4_evaluate_decisions.py 的 baseline 計算方式
      baseline_7d = X_test[:, -1, 2] * 7（直接從 feature 取 roll_30d_mean）

輸出：artifacts/tiered_alert_evaluation.json
"""

import numpy as np
import torch
import torch.nn as nn
import pickle, json
import pandas as pd
from pathlib import Path

MY_DIR        = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts"
DATA_DIR      = MY_DIR.parent / "data"

LOW_RATIO   = 1.2
HIGH_RATIO  = 1.8
TIER_NAMES  = ["🟢 正常", "🟡 低度", "🔴 高度"]
TIER_LABELS = [0, 1, 2]

COST_MATRIX = np.array([
    [0.0,  0.5,  2.0],
    [2.0,  0.0,  1.0],
    [5.0,  3.0,  0.0],
])

SEQ_LEN      = 30
PREDICT_DAYS = 7
EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]
FEATURE_COLS  = ["daily_expense", "roll_7d_mean", "roll_30d_mean", "dow_sin", "dow_cos"]


class MyBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, bidirectional=True, dropout=dropout)
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


# ── 載入資料 ──────────────────────────────────────────────────────────────────
print("📂 載入 test 資料...")
X_test     = np.load(ARTIFACTS_DIR / "my_X_test.npy")
y_test_raw = np.load(ARTIFACTS_DIR / "my_y_test_raw.npy").ravel()

with open(ARTIFACTS_DIR / "my_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

# ── 重建 test_user_ids ────────────────────────────────────────────────────────
print("🔧 重建 test_user_ids...")
all_excel = sorted(DATA_DIR.glob("raw_transactions_*.xlsx"))
all_users_data = []
for file_path in sorted(all_excel):
    user_id = file_path.stem.replace("raw_transactions_", "")
    if user_id in EXCLUDE_USERS:
        continue
    df = pd.read_excel(file_path)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"]).dt.normalize()
    df = df[df["transaction_type"] == "Expense"].copy()
    daily = df.groupby("time_stamp")["amount"].sum().reset_index()
    daily.rename(columns={"time_stamp": "date", "amount": "daily_expense"}, inplace=True)
    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(date_range, fill_value=0).reset_index()
    daily.columns = ["date", "daily_expense"]
    daily["roll_7d_mean"]  = daily["daily_expense"].rolling(7,  min_periods=1).mean()
    daily["roll_30d_mean"] = daily["daily_expense"].rolling(30, min_periods=1).mean()
    dow = daily["date"].dt.dayofweek
    daily["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    daily["future_7d_sum"] = daily["daily_expense"].rolling(PREDICT_DAYS).sum().shift(-PREDICT_DAYS)
    daily = daily.dropna().reset_index(drop=True)
    daily["user_id"] = user_id
    all_users_data.append(daily)

full_df = pd.concat(all_users_data, ignore_index=True)
test_uid_list = []
for user_id in full_df["user_id"].unique():
    user_df   = full_df[full_df["user_id"] == user_id].reset_index(drop=True)
    n_samples = len(user_df) - SEQ_LEN
    if n_samples < 10:
        continue
    v_end = int(n_samples * 0.85)
    test_uid_list.extend([user_id] * (n_samples - v_end))

test_user_ids = np.array(test_uid_list)

# ── 個人基線（直接從 feature 取 roll_30d_mean × 7）───────────────────────────
baseline_7d = (X_test[:, -1, 2] * 7).astype(np.float32)

# ── 推論 ──────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = X_test.shape[2]
model = MyBiLSTM(INPUT_SIZE, 64, 2, 1).to(device)
model.load_state_dict(torch.load(ARTIFACTS_DIR / "best_lstm_model.pth",
                                  map_location=device, weights_only=True))
model.eval()
with torch.no_grad():
    preds_scaled = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

test_preds = target_scaler.inverse_transform(preds_scaled).ravel()


# ── 三級分類 ──────────────────────────────────────────────────────────────────
def to_tier(values, baseline):
    ratio = values / (baseline + 1e-8)
    tiers = np.zeros(len(values), dtype=int)
    tiers[ratio >= LOW_RATIO]  = 1
    tiers[ratio >= HIGH_RATIO] = 2
    return tiers

y_true_tier = to_tier(y_test_raw, baseline_7d)
y_pred_tier = to_tier(test_preds,  baseline_7d)

print(f"\n真實分佈：" + "  ".join(
    f"{TIER_NAMES[t]}={np.sum(y_true_tier==t)}({np.mean(y_true_tier==t)*100:.1f}%)"
    for t in TIER_LABELS))
print(f"預測分佈：" + "  ".join(
    f"{TIER_NAMES[t]}={np.sum(y_pred_tier==t)}({np.mean(y_pred_tier==t)*100:.1f}%)"
    for t in TIER_LABELS))


# ── 指標計算 ──────────────────────────────────────────────────────────────────
def tiered_metrics(y_true, y_pred):
    n  = len(y_true)
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    per_level = {}
    for lv in TIER_LABELS:
        tp   = cm[lv, lv]
        fp   = cm[:, lv].sum() - tp
        fn   = cm[lv, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        per_level[TIER_NAMES[lv]] = {
            "precision": round(float(prec), 4),
            "recall"   : round(float(rec),  4),
            "f1"       : round(float(f1),   4),
            "support"  : int(cm[lv].sum()),
        }

    exact_acc     = float(np.mean(y_true == y_pred))
    ordinal_acc   = float(np.mean(np.abs(y_true - y_pred) <= 1))
    severe_err    = float(np.mean(np.abs(y_true - y_pred) == 2))
    expected_cost = float(sum(COST_MATRIX[t, p] for t, p in zip(y_true, y_pred)) / n)

    high_mask = y_true == 2
    high_fnr  = float(np.sum((y_true == 2) & (y_pred != 2)) / high_mask.sum()) if high_mask.sum() > 0 else None

    return {
        "confusion_matrix" : cm.tolist(),
        "per_level"        : per_level,
        "exact_accuracy"   : round(exact_acc,    4),
        "ordinal_accuracy" : round(ordinal_acc,   4),
        "severe_error_rate": round(severe_err,    4),
        "expected_cost"    : round(expected_cost, 4),
        "high_alert_fnr"   : round(high_fnr, 4) if high_fnr is not None else None,
    }


global_tiered = tiered_metrics(y_true_tier, y_pred_tier)

per_user_tiered = {}
for uid in np.unique(test_user_ids):
    mask = test_user_ids == uid
    m    = tiered_metrics(y_true_tier[mask], y_pred_tier[mask])
    m["true_dist"] = {TIER_NAMES[t]: int(np.sum(y_true_tier[mask] == t)) for t in TIER_LABELS}
    m["pred_dist"] = {TIER_NAMES[t]: int(np.sum(y_pred_tier[mask] == t)) for t in TIER_LABELS}
    m["n_samples"] = int(mask.sum())
    per_user_tiered[uid] = m


# ── 列印 ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  三級警報評估（ckh BiLSTM，thresholds: {LOW_RATIO}× / {HIGH_RATIO}×）")
print(f"{'='*65}")
print(f"\n  混淆矩陣（行=真實，列=預測）：")
print(f"  {'':12s}  {'🟢預測正常':>10}  {'🟡預測低度':>10}  {'🔴預測高度':>10}")
for i, row in enumerate(global_tiered["confusion_matrix"]):
    print(f"  {TIER_NAMES[i]:12s}  {row[0]:>10}  {row[1]:>10}  {row[2]:>10}")

print(f"\n  Per-level 指標（one-vs-rest）：")
for name, m in global_tiered["per_level"].items():
    print(f"  {name}  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  (support={m['support']})")

print(f"\n  整體指標：")
print(f"  Exact Accuracy   ：{global_tiered['exact_accuracy']:.4f}")
print(f"  Ordinal Accuracy ：{global_tiered['ordinal_accuracy']:.4f}  （差 ≤ 1 級都算可接受）")
print(f"  Severe Error Rate：{global_tiered['severe_error_rate']:.4f}  （跨兩級的嚴重誤判）")
print(f"  Expected Cost    ：{global_tiered['expected_cost']:.4f}")
if global_tiered["high_alert_fnr"] is not None:
    print(f"  高度警告 FNR     ：{global_tiered['high_alert_fnr']:.4f}")

print(f"\n  Per-user 三級摘要：")
for uid, v in sorted(per_user_tiered.items()):
    high_fnr_str = f"HighFNR={v['high_alert_fnr']:.3f}" if v["high_alert_fnr"] is not None else "HighFNR=N/A"
    print(f"  {uid:10s}  n={v['n_samples']:3d}  "
          f"OrdAcc={v['ordinal_accuracy']:.3f}  Cost={v['expected_cost']:.3f}  {high_fnr_str}")
    print(f"    真實分佈：{v['true_dist']}  →  預測分佈：{v['pred_dist']}")


# ── 儲存 ──────────────────────────────────────────────────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

output = {
    "model"            : "ckh BiLSTM (single model)",
    "low_ratio"        : LOW_RATIO,
    "high_ratio"       : HIGH_RATIO,
    "tier_names"       : TIER_NAMES,
    "cost_matrix"      : COST_MATRIX.tolist(),
    "global_metrics"   : global_tiered,
    "true_distribution": {TIER_NAMES[t]: int(np.sum(y_true_tier==t)) for t in TIER_LABELS},
    "pred_distribution": {TIER_NAMES[t]: int(np.sum(y_pred_tier==t)) for t in TIER_LABELS},
    "per_user"         : per_user_tiered,
}
with open(ARTIFACTS_DIR / "tiered_alert_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, cls=NpEncoder)

print(f"\n✅ 儲存至 artifacts/tiered_alert_evaluation.json")
print("🎉 完成！")
