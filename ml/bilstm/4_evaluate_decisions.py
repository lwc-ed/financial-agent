"""
Step 4：決策層評估（同學 BiLSTM，raw features）
================================================
三層評估：
  Layer 1 - 回歸層  ：MAE, RMSE, sMAPE
  Layer 2 - 決策層  ：Precision, Recall, F1, FNR, FPR
  Layer 3 - 成本層  ：Expected Decision Cost（漏報成本 3×，誤報成本 1×）

預警定義：future_7d_sum > 個人近30日均值 × 1.5
注意：此腳本會自行重建 test_user_ids（原始 pipeline 未儲存）

⚠️  執行前請確認已跑完：
    1_prepare_data.py → 2_train_lstm.py
輸出：artifacts/decision_evaluation.json
"""

import numpy as np
import torch
import torch.nn as nn
import pickle, json
import pandas as pd
from pathlib import Path

MY_DIR        = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts"
# 直接指向 ml/data/，避免 rglob 找到 .claude/worktrees 的重複備份
DATA_DIR      = MY_DIR.parent / "data"

ALERT_RATIO         = 1.5
COST_FALSE_NEGATIVE = 3.0
COST_FALSE_POSITIVE = 1.0
SEQ_LEN             = 30
PREDICT_DAYS        = 7
EXCLUDE_USERS       = ["user4", "user5", "user6", "user14"]
FEATURE_COLS        = ["daily_expense", "roll_7d_mean", "roll_30d_mean", "dow_sin", "dow_cos"]

# ── 模型定義（與 2_train_lstm.py 完全相同）───────────────────────────────────
class MyBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc1     = nn.Linear(hidden_size * 2, 32)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(32, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]   # 只取最後一個時間步
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

# ── 載入已存的 test 資料 ──────────────────────────────────────────────────────
print("📂 載入 test 資料...")
X_test     = np.load(ARTIFACTS_DIR / "my_X_test.npy")
y_test_raw = np.load(ARTIFACTS_DIR / "my_y_test_raw.npy").ravel()

with open(ARTIFACTS_DIR / "my_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

# ── 重建 test_user_ids（完全照搬 1_prepare_data.py 的邏輯）──────────────────
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

# 按照 1_prepare_data.py 完全相同的迭代順序重建 test split
test_uid_list = []
for user_id in full_df["user_id"].unique():
    user_df   = full_df[full_df["user_id"] == user_id].reset_index(drop=True)
    n_samples = len(user_df) - SEQ_LEN
    if n_samples < 10:
        continue
    v_end = int(n_samples * 0.85)
    test_uid_list.extend([user_id] * (n_samples - v_end))

test_user_ids = np.array(test_uid_list)

# ── 個人基線：直接從 X_test 取最後一天的 roll_30d_mean（feature index=2）──────
# X_test shape: (n, seq_len, n_features)  FEATURE_COLS = [daily, roll7, roll30, sin, cos]
# X_test[:, -1, 2] = 最後一天的 roll_30d_mean（真實 TWD，未 scaled）
baseline_7d = (X_test[:, -1, 2] * 7).astype(np.float32)

alert_threshold = baseline_7d * ALERT_RATIO
y_true_alert    = (y_test_raw > alert_threshold).astype(int)
print(f"  test 樣本數：{len(y_test_raw)}  預警正例：{y_true_alert.mean()*100:.1f}%")

# ── 推論 ──────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔮 推論（single model）...")
INPUT_SIZE = X_test.shape[2]
model = MyBiLSTM(INPUT_SIZE, 64, 2, 1).to(device)
model.load_state_dict(torch.load(ARTIFACTS_DIR / "best_lstm_model.pth",
                                  map_location=device, weights_only=True))
model.eval()
X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    preds_scaled = model(X_t).cpu().numpy()

test_preds   = target_scaler.inverse_transform(preds_scaled).ravel()
y_pred_alert = (test_preds > alert_threshold).astype(int)


# ── 指標計算 ──────────────────────────────────────────────────────────────────
def decision_metrics(y_true, y_pred):
    TP = int(np.sum((y_true==1)&(y_pred==1)))
    TN = int(np.sum((y_true==0)&(y_pred==0)))
    FP = int(np.sum((y_true==0)&(y_pred==1)))
    FN = int(np.sum((y_true==1)&(y_pred==0)))
    precision = TP/(TP+FP+1e-8)
    recall    = TP/(TP+FN+1e-8)
    f1        = 2*precision*recall/(precision+recall+1e-8)
    fnr       = FN/(TP+FN+1e-8)
    fpr       = FP/(FP+TN+1e-8)
    cost      = (FN*COST_FALSE_NEGATIVE + FP*COST_FALSE_POSITIVE)/len(y_true)
    return {"TP":TP,"TN":TN,"FP":FP,"FN":FN,
            "precision":round(precision,4),"recall":round(recall,4),
            "f1":round(f1,4),"fnr":round(fnr,4),"fpr":round(fpr,4),
            "expected_cost":round(cost,4)}

mae   = float(np.mean(np.abs(y_test_raw - test_preds)))
rmse  = float(np.sqrt(np.mean((y_test_raw-test_preds)**2)))
smape = float(np.mean(np.abs(y_test_raw-test_preds)/((np.abs(y_test_raw)+np.abs(test_preds))/2+1e-8))*100)
global_dec = decision_metrics(y_true_alert, y_pred_alert)

per_user_results = {}
for uid in np.unique(test_user_ids):
    mask = test_user_ids == uid
    yt, yp = y_test_raw[mask], test_preds[mask]
    ya, ypa = y_true_alert[mask], y_pred_alert[mask]
    user_mae  = float(np.mean(np.abs(yt-yp)))
    user_nmae = user_mae/(np.mean(np.abs(yt))+1e-8)
    per_user_results[uid] = {"n_samples":int(mask.sum()),
                              "alert_ratio":float(ya.mean()),
                              "mae":round(user_mae,2),
                              "nmae":round(user_nmae,4),
                              **decision_metrics(ya,ypa)}

user_maes  = [v["mae"]           for v in per_user_results.values()]
user_f1s   = [v["f1"]            for v in per_user_results.values()]
user_fnrs  = [v["fnr"]           for v in per_user_results.values()]
user_costs = [v["expected_cost"] for v in per_user_results.values()]

print(f"\n{'='*62}")
print(f"  同學 BiLSTM（raw features，single model）")
print(f"{'='*62}")
print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  sMAPE={smape:.2f}%")
print(f"  Precision={global_dec['precision']:.4f}  Recall={global_dec['recall']:.4f}  F1={global_dec['f1']:.4f}")
print(f"  FNR={global_dec['fnr']:.4f}  FPR={global_dec['fpr']:.4f}  Cost={global_dec['expected_cost']:.4f}")
print(f"  Per-user MAE：avg={np.mean(user_maes):.2f}  max={max(user_maes):.2f}  std={np.std(user_maes):.2f}")
print(f"  Per-user F1 ：avg={np.mean(user_f1s):.4f}  min={min(user_f1s):.4f}")
print(f"  Per-user FNR：avg={np.mean(user_fnrs):.4f}  max={max(user_fnrs):.4f}")
print(f"\n  各 user 詳細：")
for uid, v in sorted(per_user_results.items()):
    print(f"    {uid:10s}  MAE={v['mae']:7.2f}  F1={v['f1']:.3f}  "
          f"Recall={v['recall']:.3f}  FNR={v['fnr']:.3f}  Cost={v['expected_cost']:.3f}")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

output = {
    "model"                   : "BiLSTM raw features (single model)",
    "alert_threshold_ratio"   : ALERT_RATIO,
    "cost_fn"                 : COST_FALSE_NEGATIVE,
    "cost_fp"                 : COST_FALSE_POSITIVE,
    "layer1_regression"       : {"mae":round(mae,2),"rmse":round(rmse,2),"smape":round(smape,2)},
    "layer2_decision_global"  : global_dec,
    "layer3_per_user_summary" : {
        "mae_mean":round(float(np.mean(user_maes)),2),
        "mae_max" :round(float(max(user_maes)),2),
        "mae_std" :round(float(np.std(user_maes)),2),
        "f1_mean" :round(float(np.mean(user_f1s)),4),
        "f1_min"  :round(float(min(user_f1s)),4),
        "fnr_mean":round(float(np.mean(user_fnrs)),4),
        "fnr_max" :round(float(max(user_fnrs)),4),
        "cost_mean":round(float(np.mean(user_costs)),4),
        "cost_max" :round(float(max(user_costs)),4),
        "worst_mae_user":max(per_user_results, key=lambda u:per_user_results[u]["mae"]),
        "worst_fnr_user":max(per_user_results, key=lambda u:per_user_results[u]["fnr"]),
    },
    "per_user_detail"         : per_user_results,
}
with open(ARTIFACTS_DIR/"decision_evaluation.json","w",encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, cls=NpEncoder)

print(f"\n✅ 儲存至 artifacts/decision_evaluation.json")
print("🎉 完成！")
