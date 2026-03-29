"""
ml_gru_nopretrain/predict.py
==============================
比較以下模型的結果：
  1. HGBR（22 個特徵，無 pretrain）
  2. GRU from scratch（無 Walmart pretrain）
  3. Naive baselines（naive 7-day, moving average 30-day）

參考基準（來自 ml_gru）：
  - 原版 HGBR Test MAE  : ~1,200
  - 原版最佳 GRU V4 Test MAE: 5,773
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import json
from datetime import datetime

ARTIFACTS_DIR = "artifacts"
DATA_PATH     = "../ml_gru/features_all.csv"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, num_layers,
                                 dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2        = nn.Linear(hidden_size // 2, output_size)
        self.relu       = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        context    = self.layer_norm(context)
        return self.fc2(self.relu(self.fc1(self.dropout(context))))


def smape(yt, yp):
    yt, yp = yt.flatten(), yp.flatten()
    d = (np.abs(yt) + np.abs(yp)) / 2
    m = d > 0
    return float(np.mean(np.abs(yp[m] - yt[m]) / d[m]) * 100)

def per_user_nmae(yt, yp, uids):
    yt, yp = yt.flatten(), yp.flatten()
    r = []
    for u in np.unique(uids):
        mask = np.array(uids) == u
        mu = yt[mask].mean()
        if mu > 0:
            r.append(np.mean(np.abs(yp[mask] - yt[mask])) / mu * 100)
    return float(np.mean(r))


# ─────────────────────────────────────────
# 載入資料
# ─────────────────────────────────────────
X_test_gru  = np.load(f"{ARTIFACTS_DIR}/gru_X_test.npy")
X_test_hgbr = np.load(f"{ARTIFACTS_DIR}/hgbr_X_test.npy")
y_test      = np.load(f"{ARTIFACTS_DIR}/y_test.npy").flatten()
X_val_gru   = np.load(f"{ARTIFACTS_DIR}/gru_X_val.npy")
X_val_hgbr  = np.load(f"{ARTIFACTS_DIR}/hgbr_X_val.npy")
y_val       = np.load(f"{ARTIFACTS_DIR}/y_val.npy").flatten()
val_uids    = np.load(f"{ARTIFACTS_DIR}/val_uids.npy",  allow_pickle=True)
test_uids   = np.load(f"{ARTIFACTS_DIR}/test_uids.npy", allow_pickle=True)

with open(f"{ARTIFACTS_DIR}/target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

# ─────────────────────────────────────────
# HGBR 預測
# ─────────────────────────────────────────
with open(f"{ARTIFACTS_DIR}/hgbr_model.pkl", "rb") as f:
    hgbr = pickle.load(f)
hgbr_val_pred  = hgbr.predict(X_val_hgbr)
hgbr_test_pred = hgbr.predict(X_test_hgbr)

# ─────────────────────────────────────────
# GRU scratch 預測
# ─────────────────────────────────────────
ckpt = torch.load(f"{ARTIFACTS_DIR}/gru_scratch.pth", map_location=device)
hp   = ckpt["hyperparams"]
gru  = GRUWithAttention(
    hp["input_size"], hp["hidden_size"], hp["num_layers"], hp["output_size"], hp["dropout"]
).to(device)
gru.load_state_dict(ckpt["model_state"])
gru.eval()

def predict_gru(X):
    with torch.no_grad():
        raw = gru(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    return target_scaler.inverse_transform(raw).flatten()

gru_val_pred  = predict_gru(X_val_gru)
gru_test_pred = predict_gru(X_test_gru)

# ─────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
df["naive_7d"]  = df.groupby("user_id")["daily_expense"].transform(lambda x: x.rolling(7,  min_periods=1).sum())
df["ma_30d_x7"] = df.groupby("user_id")["daily_expense"].transform(lambda x: x.rolling(30, min_periods=1).mean()) * 7
naive_mae = float((df["naive_7d"]  - df["future_expense_7d_sum"]).abs().mean())
ma_mae    = float((df["ma_30d_x7"] - df["future_expense_7d_sum"]).abs().mean())

# ─────────────────────────────────────────
# 印出比較表
# ─────────────────────────────────────────
print(f"\n{'='*25} 模型比較結果 {'='*25}")
print(f"  {'模型':<28} {'Test MAE':>10} {'Test RMSE':>11} {'SMAPE':>8} {'NMAE':>8}")
print(f"  {'-'*65}")
print(f"  {'Naive 7-day (baseline)':<28} {naive_mae:>10,.0f}")
print(f"  {'Moving Avg 30d (baseline)':<28} {ma_mae:>10,.0f}")
print(f"  {'原版 HGBR (7 feat, 參考)':<28} {'~1,200':>10}")
print(f"  {'原版 GRU V4 (pretrain, 參考)':<28} {'5,773':>10}")
print(f"  {'-'*65}")

models = {
    "HGBR (22 features)": (hgbr_val_pred,  hgbr_test_pred),
    "GRU from scratch":   (gru_val_pred,    gru_test_pred),
}
all_metrics = {}

for name, (vp, tp) in models.items():
    vm  = float(np.mean(np.abs(vp - y_val)))
    tm  = float(np.mean(np.abs(tp - y_test)))
    vr  = float(np.sqrt(np.mean((vp - y_val)**2)))
    tr  = float(np.sqrt(np.mean((tp - y_test)**2)))
    vs  = smape(y_val,  vp)
    ts  = smape(y_test, tp)
    vn  = per_user_nmae(y_val,  vp, val_uids)
    tn  = per_user_nmae(y_test, tp, test_uids)
    print(f"  {name:<28} {tm:>10,.0f} {tr:>11,.0f} {ts:>7.1f}% {tn:>7.1f}%")
    all_metrics[name] = {
        "val_mae":   round(vm, 2),  "val_rmse":  round(vr, 2),
        "val_smape": round(vs, 4),  "val_nmae":  round(vn, 4),
        "test_mae":  round(tm, 2),  "test_rmse": round(tr, 2),
        "test_smape":round(ts, 4),  "test_nmae": round(tn, 4),
    }

print(f"{'='*65}")
best_mae = min(all_metrics[k]["test_mae"] for k in all_metrics)
print(f"\n  最佳 Test MAE：{best_mae:,.0f}  |  原版 HGBR 參考：~1,200")

# ─────────────────────────────────────────
# 儲存結果
# ─────────────────────────────────────────
result = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "baselines": {
        "naive_7d_mae":    round(naive_mae, 2),
        "moving_avg_mae":  round(ma_mae,    2),
        "original_hgbr_test_mae":  1200,
        "original_gru_v4_test_mae": 5773,
    },
    "models": all_metrics,
}
with open(f"{ARTIFACTS_DIR}/comparison_results.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

pd.DataFrame({
    "y_true": y_test,
    "hgbr_pred":       hgbr_test_pred,
    "gru_scratch_pred": gru_test_pred,
    "user_id": test_uids,
}).to_csv(f"{ARTIFACTS_DIR}/predictions_test.csv", index=False)

print("\n✅ 結果已儲存至 artifacts/comparison_results.json")
