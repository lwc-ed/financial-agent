"""
GRU 無 Pretrain 版本 — 預測評估
==================================
Ensemble 三模型 + Per-user 偏差修正
輸出 result_nopretrain.txt，與 V5（有 pretrain）直接對比。
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

ARTIFACTS_DIR  = "artificats"
VERSION        = "nopretrain"
ENSEMBLE_SEEDS = [42, 123, 777]
RESULT_PATH    = "result_nopretrain.txt"
MODEL_DIR = Path(__file__).resolve().parent
ML_ROOT = MODEL_DIR.parent.parent if MODEL_DIR.parent.name == "legacy_models" else MODEL_DIR.parent
FEATURES_PATH = ML_ROOT / "processed_data" / "artifacts" / "features_all.csv"

if torch.backends.mps.is_available():   device = torch.device("mps")
elif torch.cuda.is_available():         device = torch.device("cuda")
else:                                   device = torch.device("cpu")


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


# 載入三個 Ensemble 模型
print(f"📦 載入 {VERSION} Ensemble 模型（{len(ENSEMBLE_SEEDS)} seeds）...")
models = []
for seed in ENSEMBLE_SEEDS:
    ckpt = torch.load(f"{ARTIFACTS_DIR}/gru_{VERSION}_seed{seed}.pth", map_location=device)
    hp   = ckpt["hyperparams"]
    m    = GRUWithAttention(hp["input_size"], hp["hidden_size"], hp["num_layers"],
                            hp["output_size"], hp["dropout"]).to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    models.append(m)
    print(f"  ✅ Seed {seed} 載入完成（best val_loss={ckpt['val_loss']:.6f}, pretrained={ckpt['pretrained']}）")

# 載入 scalers
with open(f"{ARTIFACTS_DIR}/log_target_scaler_{VERSION}.pkl", "rb") as f:
    log_scaler = pickle.load(f)
with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "rb") as f:
    orig_scaler = pickle.load(f)

X_val         = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
y_val         = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")
X_test        = np.load(f"{ARTIFACTS_DIR}/personal_X_test.npy")
y_test        = np.load(f"{ARTIFACTS_DIR}/personal_y_test.npy")
val_user_ids  = np.load(f"{ARTIFACTS_DIR}/personal_val_user_ids.npy",  allow_pickle=True)
test_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_test_user_ids.npy", allow_pickle=True)


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
        mu   = yt[mask].mean()
        if mu > 0:
            r.append(np.mean(np.abs(yp[mask] - yt[mask])) / mu * 100)
    return float(np.mean(r))


def ensemble_predict(X):
    X_t   = torch.tensor(X, dtype=torch.float32).to(device)
    preds = []
    with torch.no_grad():
        for m in models:
            pred_sc  = m(X_t).cpu().numpy()
            pred_log = log_scaler.inverse_transform(pred_sc)
            pred_log = np.clip(pred_log, 0, None)
            preds.append(np.expm1(pred_log))
    return np.mean(preds, axis=0)


def get_true_real(y_scaled):
    return np.clip(orig_scaler.inverse_transform(y_scaled), 0, None)


print("\n🔮 Ensemble 預測（Val + Test）...")
y_val_pred_raw  = ensemble_predict(X_val)
y_val_true      = get_true_real(y_val)
y_test_pred_raw = ensemble_predict(X_test)
y_test_true     = get_true_real(y_test)


# ─────────────────────────────────────────
# Per-user 偏差修正
# ─────────────────────────────────────────
print("\n📐 Per-user 偏差修正（從 val set 估計偏差）...")
user_bias = {}
for u in np.unique(val_user_ids):
    mask = np.array(val_user_ids) == u
    bias = float(np.median(y_val_pred_raw[mask].flatten() - y_val_true[mask].flatten()))
    user_bias[str(u)] = bias
    print(f"  User {u:3} : val bias = {bias:+,.1f} 元")

y_test_pred = y_test_pred_raw.copy()
for u in np.unique(test_user_ids):
    mask = np.array(test_user_ids) == u
    y_test_pred[mask] -= user_bias.get(str(u), 0.0)

y_val_pred = y_val_pred_raw.copy()
for u in np.unique(val_user_ids):
    mask = np.array(val_user_ids) == u
    y_val_pred[mask] -= user_bias.get(str(u), 0.0)

y_test_pred = np.clip(y_test_pred, 0, None)
y_val_pred  = np.clip(y_val_pred,  0, None)


# 計算指標
val_mae    = float(np.mean(np.abs(y_val_pred  - y_val_true)))
val_rmse   = float(np.sqrt(np.mean((y_val_pred  - y_val_true)**2)))
val_medae  = float(np.median(np.abs(y_val_pred  - y_val_true)))
test_mae   = float(np.mean(np.abs(y_test_pred - y_test_true)))
test_rmse  = float(np.sqrt(np.mean((y_test_pred - y_test_true)**2)))
test_medae = float(np.median(np.abs(y_test_pred - y_test_true)))
val_smape  = smape(y_val_true,  y_val_pred)
test_smape = smape(y_test_true, y_test_pred)
val_nmae   = per_user_nmae(y_val_true,  y_val_pred,  val_user_ids)
test_nmae  = per_user_nmae(y_test_true, y_test_pred, test_user_ids)

test_mae_raw = float(np.mean(np.abs(y_test_pred_raw - y_test_true)))
print(f"\n  偏差修正效果：Test MAE {test_mae_raw:,.2f} → {test_mae:,.2f}（{'改善' if test_mae < test_mae_raw else '退步'}）")


# Baseline
EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]
df = pd.read_csv(FEATURES_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[~df["user_id"].isin(EXCLUDE_USERS)].reset_index(drop=True)
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
df["naive_7d"]  = df.groupby("user_id")["daily_expense"].transform(lambda x: x.rolling(7, min_periods=1).sum())
df["ma_30d_x7"] = df.groupby("user_id")["daily_expense"].transform(lambda x: x.rolling(30, min_periods=1).mean()) * 7
naive_mae  = float((df["naive_7d"]  - df["future_expense_7d_sum"]).abs().mean())
ma_mae     = float((df["ma_30d_x7"] - df["future_expense_7d_sum"]).abs().mean())
naive_rmse = float(((df["naive_7d"]  - df["future_expense_7d_sum"])**2).mean()**0.5)
ma_rmse    = float(((df["ma_30d_x7"] - df["future_expense_7d_sum"])**2).mean()**0.5)


print(f"\n{'='*50}  {VERSION} Results (Ensemble + Bias Correction)  {'='*50}")
print(f"  Val  MAE : {val_mae:,.2f}  RMSE: {val_rmse:,.2f}  MedAE: {val_medae:,.2f}  SMAPE: {val_smape:.2f}%  NMAE: {val_nmae:.2f}%")
print(f"  Test MAE : {test_mae:,.2f}  RMSE: {test_rmse:,.2f}  MedAE: {test_medae:,.2f}  SMAPE: {test_smape:.2f}%  NMAE: {test_nmae:.2f}%")
print(f"  💡 若 MAE >> MedAE，代表有少數極端誤差在拉高 MAE")
print(f"  Baseline Moving Avg MAE: {ma_mae:,.2f}")
print(f"  Beat baseline: {test_mae < ma_mae}")


# 對比其他版本
print("\n  === 全版本對比（含 pretrain 版）===")
compare_versions = [
    ("v0(orig)", f"{ARTIFACTS_DIR}/metrics.json"),
    ("v1",       f"{ARTIFACTS_DIR}/metrics_v1.json"),
    ("v2",       f"{ARTIFACTS_DIR}/metrics_v2.json"),
    ("v3",       f"{ARTIFACTS_DIR}/metrics_vv3.json"),
    ("v4",       f"{ARTIFACTS_DIR}/metrics_vv4.json"),
    ("v5",       f"{ARTIFACTS_DIR}/metrics_vv5.json"),
]
for name, path in compare_versions:
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f)
        print(f"  {name:12s}: Test MAE={prev.get('test_mae',0):>8,.2f}  SMAPE={prev.get('test_smape',0):>6.2f}%  NMAE={prev.get('test_per_user_nmae',0):>7.2f}%")
print(f"  {VERSION:12s}: Test MAE={test_mae:>8,.2f}  SMAPE={test_smape:>6.2f}%  NMAE={test_nmae:>7.2f}%")


# 儲存 metrics
metrics = {
    "model_name": f"gru_{VERSION}_ensemble_bias",
    "version": VERSION,
    "pretrained": False,
    "ensemble_seeds": ENSEMBLE_SEEDS,
    "val_mae": round(val_mae, 6),   "val_rmse": round(val_rmse, 6),
    "val_medae": round(val_medae, 6),
    "val_smape": round(val_smape, 4), "val_per_user_nmae": round(val_nmae, 4),
    "test_mae": round(test_mae, 6), "test_rmse": round(test_rmse, 6),
    "test_medae": round(test_medae, 6),
    "test_smape": round(test_smape, 4), "test_per_user_nmae": round(test_nmae, 4),
    "test_mae_before_bias_correction": round(test_mae_raw, 6),
    "naive_7d_mae": round(naive_mae, 6),       "naive_7d_rmse": round(naive_rmse, 6),
    "moving_avg_30d_mae": round(ma_mae, 6),    "moving_avg_30d_rmse": round(ma_rmse, 6),
    "beat_moving_avg": bool(test_mae < ma_mae),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
metrics_path = f"{ARTIFACTS_DIR}/metrics_{VERSION}.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

test_pred_path = f"{ARTIFACTS_DIR}/predictions_test_{VERSION}.csv"
pd.DataFrame({
    "y_true": y_test_true.flatten(),
    "y_pred": y_test_pred.flatten(),
    "abs_error": np.abs(y_test_pred - y_test_true).flatten(),
}).to_csv(test_pred_path, index=False)

# 輸出 result txt
result_text = f"""GRU Predict Result
model_name: gru_{VERSION}_ensemble_bias
version: {VERSION}
pretrained: False
ensemble_seeds: {ENSEMBLE_SEEDS}
val_mae: {val_mae:,.6f}
val_rmse: {val_rmse:,.6f}
val_smape: {val_smape:.2f}%
val_per_user_nmae: {val_nmae:.2f}%
test_mae: {test_mae:,.6f}
test_rmse: {test_rmse:,.6f}
test_smape: {test_smape:.2f}%
test_per_user_nmae: {test_nmae:.2f}%
test_mae_before_bias_correction: {test_mae_raw:,.6f}
naive_7d_mae: {naive_mae:,.6f}
naive_7d_rmse: {naive_rmse:,.6f}
moving_avg_30d_mae: {ma_mae:,.6f}
moving_avg_30d_rmse: {ma_rmse:,.6f}
beat_moving_avg: {test_mae < ma_mae}
metrics_json: {metrics_path}
test_predictions_csv: {test_pred_path}
"""
with open(RESULT_PATH, "w", encoding="utf-8") as f:
    f.write(result_text)

print(f"\n✅ metrics 已儲存：{metrics_path}")
print(f"📝 Result 已儲存：{RESULT_PATH}")
