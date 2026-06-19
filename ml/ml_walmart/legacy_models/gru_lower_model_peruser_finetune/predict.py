"""
gru_lower_model_peruser_finetune/predict.py
============================================
評估三種設定的結果：
  1. 全域模型（global）：所有使用者共用一個模型
  2. Per-user fine-tune：每個使用者使用自己微調的模型
  3. Naive baselines

並輸出 per-user 詳細比較，讓你看出哪些使用者受益最大。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import json
import os
from pathlib import Path
from datetime import datetime

ARTIFACTS_DIR = "artifacts"
DATA_PATH     = Path(__file__).resolve().parents[2] / "processed_data" / "artifacts" / "features_all.csv"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SmallGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
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
    r = []
    for u in np.unique(uids):
        mask = np.array(uids) == u
        mu = yt[mask].mean()
        if mu > 0:
            r.append(np.mean(np.abs(yp[mask] - yt[mask])) / mu * 100)
    return float(np.mean(r))


# ─────────────────────────────────────────
# 載入
# ─────────────────────────────────────────
X_test      = np.load(f"{ARTIFACTS_DIR}/X_test.npy")
y_test_raw  = np.load(f"{ARTIFACTS_DIR}/y_test_raw.npy").flatten()
X_val       = np.load(f"{ARTIFACTS_DIR}/X_val.npy")
y_val_raw   = np.load(f"{ARTIFACTS_DIR}/y_val_raw.npy").flatten()
test_uids   = np.load(f"{ARTIFACTS_DIR}/test_uids.npy",  allow_pickle=True)
val_uids    = np.load(f"{ARTIFACTS_DIR}/val_uids.npy",   allow_pickle=True)

with open(f"{ARTIFACTS_DIR}/target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)
with open(f"{ARTIFACTS_DIR}/per_user_data.pkl", "rb") as f:
    per_user_data = pickle.load(f)

def load_model(path):
    ckpt  = torch.load(path, map_location=device)
    hp    = ckpt["hyperparams"]
    model = SmallGRU(hp["input_size"], hp["hidden_size"], hp["output_size"], hp["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def predict(model, X):
    with torch.no_grad():
        raw = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    return target_scaler.inverse_transform(raw).flatten()

# ─────────────────────────────────────────
# 全域模型預測
# ─────────────────────────────────────────
global_model    = load_model(f"{ARTIFACTS_DIR}/global_model.pth")
global_val_pred = predict(global_model, X_val)
global_te_pred  = predict(global_model, X_test)

# ─────────────────────────────────────────
# Per-user 模型預測（拼回全域 test 集）
# ─────────────────────────────────────────
peruser_te_pred  = np.zeros_like(y_test_raw)
peruser_val_pred = np.zeros_like(y_val_raw)

for uid in per_user_data.keys():
    model_path = f"{ARTIFACTS_DIR}/user_{uid}.pth"
    if not os.path.exists(model_path):
        continue
    u_model = load_model(model_path)

    # test
    te_mask = np.array(test_uids) == uid
    if te_mask.sum() > 0:
        peruser_te_pred[te_mask] = predict(u_model, X_test[te_mask])

    # val
    va_mask = np.array(val_uids) == uid
    if va_mask.sum() > 0:
        peruser_val_pred[va_mask] = predict(u_model, X_val[va_mask])

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
# 整體比較
# ─────────────────────────────────────────
models_result = {
    "Global SmallGRU":      (global_val_pred,  global_te_pred),
    "Per-user Fine-tuned":  (peruser_val_pred,  peruser_te_pred),
}

print(f"\n{'='*25} 整體結果比較 {'='*25}")
print(f"  {'模型':<25} {'Test MAE':>10} {'Test RMSE':>11} {'SMAPE':>8} {'NMAE':>8}")
print(f"  {'-'*62}")
print(f"  {'Naive 7-day':<25} {naive_mae:>10,.0f}")
print(f"  {'Moving Avg 30d':<25} {ma_mae:>10,.0f}")
print(f"  {'原版 GRU V4 (參考)':<25} {'5,773':>10}")
print(f"  {'-'*62}")

all_metrics = {}
for name, (vp, tp) in models_result.items():
    tm = float(np.mean(np.abs(tp - y_test_raw)))
    tr = float(np.sqrt(np.mean((tp - y_test_raw)**2)))
    ts = smape(y_test_raw, tp)
    tn = per_user_nmae(y_test_raw, tp, test_uids)
    print(f"  {name:<25} {tm:>10,.0f} {tr:>11,.0f} {ts:>7.1f}% {tn:>7.1f}%")
    all_metrics[name] = {"test_mae": round(tm,2), "test_rmse": round(tr,2),
                          "test_smape": round(ts,4), "test_nmae": round(tn,4)}

print(f"{'='*62}")

# ─────────────────────────────────────────
# Per-user 明細（global vs fine-tuned）
# ─────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"  Per-user 比較（Global vs Fine-tuned）")
print(f"  {'User':<12} {'n_test':>7} {'Global MAE':>12} {'Finetuned MAE':>14} {'改善':>8}")
print(f"  {'─'*55}")

per_user_detail = {}
for uid in sorted(per_user_data.keys(), key=str):
    te_mask = np.array(test_uids) == uid
    if te_mask.sum() == 0:
        continue
    yt = y_test_raw[te_mask]
    g_mae  = float(np.mean(np.abs(global_te_pred[te_mask] - yt)))
    fu_mae = float(np.mean(np.abs(peruser_te_pred[te_mask] - yt)))
    improve = g_mae - fu_mae
    flag = "✅" if improve > 0 else "❌"
    print(f"  {str(uid):<12} {te_mask.sum():>7} {g_mae:>12,.0f} {fu_mae:>14,.0f} {improve:>+8,.0f} {flag}")
    per_user_detail[str(uid)] = {"n_test": int(te_mask.sum()),
                                  "global_mae": round(g_mae, 2),
                                  "finetuned_mae": round(fu_mae, 2),
                                  "improvement": round(improve, 2)}

# ─────────────────────────────────────────
# 儲存
# ─────────────────────────────────────────
result = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "baselines": {"naive_7d_mae": round(naive_mae,2), "moving_avg_mae": round(ma_mae,2)},
    "overall_metrics": all_metrics,
    "per_user_detail": per_user_detail,
}
with open(f"{ARTIFACTS_DIR}/predict_results.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

pd.DataFrame({
    "y_true":        y_test_raw,
    "global_pred":   global_te_pred,
    "peruser_pred":  peruser_te_pred,
    "user_id":       test_uids,
}).to_csv(f"{ARTIFACTS_DIR}/predictions_test.csv", index=False)

print(f"\n✅ 結果已儲存至 artifacts/predict_results.json")
