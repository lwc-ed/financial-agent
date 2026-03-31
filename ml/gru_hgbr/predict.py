"""
gru_hgbr/predict.py
=====================
完整評估：GRU embedding + HGBR hybrid 模型 vs baselines。

同時輸出：
  - 整體指標比較（MAE, RMSE, SMAPE, NMAE）
  - Per-user 詳細比較（讓你知道哪個使用者受益最多）
  - 保存預測結果至 CSV 供後續分析
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import json
import os
from datetime import datetime

ARTIFACTS_DIR    = "artifacts"
ML_GRU_ARTIFACTS  = "../ml_gru/artificats"
GRU_MODEL_FILE    = "finetune_gru.pth"   # artifacts 裡實際的檔名
DATA_PATH        = "../ml_gru/features_all.csv"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class GRUModel(nn.Module):
    """與 finetune_gru.pth 實際架構一致：GRU + 單一 FC，無 Attention"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])


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
X_test_combined = np.load(f"{ARTIFACTS_DIR}/combined_X_test.npy")
X_test_flat     = np.load(f"{ARTIFACTS_DIR}/hgbr_X_test.npy")
y_test   = np.load(f"{ARTIFACTS_DIR}/y_test.npy").flatten()
X_val_combined  = np.load(f"{ARTIFACTS_DIR}/combined_X_val.npy")
X_val_flat      = np.load(f"{ARTIFACTS_DIR}/hgbr_X_val.npy")
y_val    = np.load(f"{ARTIFACTS_DIR}/y_val.npy").flatten()
test_uids = np.load(f"{ARTIFACTS_DIR}/test_uids.npy", allow_pickle=True)
val_uids  = np.load(f"{ARTIFACTS_DIR}/val_uids.npy",  allow_pickle=True)

# 純 GRU V4 預測（作對比）
ckpt_gru = torch.load(f"{ML_GRU_ARTIFACTS}/{GRU_MODEL_FILE}", map_location=device)
hp       = ckpt_gru["hyperparams"]
gru_v4   = GRUModel(
    hp["input_size"], hp["hidden_size"], hp["num_layers"], hp["output_size"], hp["dropout"]
).to(device)
gru_v4.load_state_dict(ckpt_gru["model_state"])
gru_v4.eval()

with open(f"{ML_GRU_ARTIFACTS}/personal_target_scaler.pkl", "rb") as f:
    gru_target_scaler = pickle.load(f)

gru_X_test = np.load(f"{ARTIFACTS_DIR}/gru_X_test.npy")
gru_X_val  = np.load(f"{ARTIFACTS_DIR}/gru_X_val.npy")

def predict_gru(X):
    with torch.no_grad():
        raw = gru_v4(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    return gru_target_scaler.inverse_transform(raw).flatten()

gru_val_pred  = predict_gru(gru_X_val)
gru_test_pred = predict_gru(gru_X_test)

# ─────────────────────────────────────────
# HGBR 預測
# ─────────────────────────────────────────
def hgbr_predict(model_name, X_va, X_te):
    path = f"{ARTIFACTS_DIR}/hgbr_{model_name}.pkl"
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        m = pickle.load(f)
    return m.predict(X_va), m.predict(X_te)

flat_vp, flat_tp = hgbr_predict("HGBR_flat_only",    X_val_flat,     X_test_flat)
combo_vp, combo_tp = hgbr_predict("HGBR_emb_and_flat", X_val_combined, X_test_combined)

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
models = {
    "GRU V4 (純 GRU)":       (gru_val_pred,  gru_test_pred),
    "HGBR (flat only)":      (flat_vp,        flat_tp),
    "HGBR (emb + flat)":     (combo_vp,       combo_tp),
}

print(f"\n{'='*25} 整體結果比較 {'='*25}")
print(f"  {'模型':<24} {'Test MAE':>10} {'Test RMSE':>11} {'SMAPE':>8} {'NMAE':>8}")
print(f"  {'-'*61}")
print(f"  {'Naive 7-day':<24} {naive_mae:>10,.0f}")
print(f"  {'Moving Avg 30d':<24} {ma_mae:>10,.0f}")
print(f"  {'-'*61}")

all_metrics = {}
for name, (vp, tp) in models.items():
    if tp is None:
        continue
    tm  = float(np.mean(np.abs(tp - y_test)))
    tr  = float(np.sqrt(np.mean((tp - y_test)**2)))
    ts  = smape(y_test, tp)
    tn  = per_user_nmae(y_test, tp, test_uids)
    vm  = float(np.mean(np.abs(vp - y_val)))
    print(f"  {name:<24} {tm:>10,.0f} {tr:>11,.0f} {ts:>7.1f}% {tn:>7.1f}%")
    all_metrics[name] = {"val_mae": round(vm,2),
                          "test_mae": round(tm,2), "test_rmse": round(tr,2),
                          "test_smape": round(ts,4), "test_nmae": round(tn,4)}

print(f"{'='*61}")

# ─────────────────────────────────────────
# Per-user 明細
# ─────────────────────────────────────────
if combo_tp is not None:
    print(f"\n  Per-user 比較（GRU V4 vs HGBR emb+flat）")
    print(f"  {'User':<12} {'n_test':>7} {'GRU V4 MAE':>12} {'HGBR hybrid':>13} {'改善':>8}")
    print(f"  {'─'*55}")
    for uid in sorted(set(test_uids), key=str):
        mask = np.array(test_uids) == uid
        yt   = y_test[mask]
        gm   = float(np.mean(np.abs(gru_test_pred[mask] - yt)))
        hm   = float(np.mean(np.abs(combo_tp[mask] - yt)))
        imp  = gm - hm
        flag = "✅" if imp > 0 else "❌"
        print(f"  {str(uid):<12} {mask.sum():>7} {gm:>12,.0f} {hm:>13,.0f} {imp:>+8,.0f} {flag}")

# ─────────────────────────────────────────
# 儲存
# ─────────────────────────────────────────
result = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "baselines": {"naive_7d_mae": round(naive_mae,2), "moving_avg_mae": round(ma_mae,2)},
    "models": all_metrics,
}
with open(f"{ARTIFACTS_DIR}/predict_results.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

out_df = pd.DataFrame({
    "y_true":           y_test,
    "gru_v4_pred":      gru_test_pred,
    "hgbr_flat_pred":   flat_tp   if flat_tp   is not None else np.nan,
    "hgbr_hybrid_pred": combo_tp  if combo_tp  is not None else np.nan,
    "user_id":          test_uids,
})
out_df.to_csv(f"{ARTIFACTS_DIR}/predictions_test.csv", index=False)

print(f"\n✅ 結果已儲存至 artifacts/predict_results.json")
