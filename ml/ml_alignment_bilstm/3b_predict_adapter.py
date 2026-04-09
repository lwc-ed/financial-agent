"""
Step 3b：Adapter 預測與五方比較
================================
比較：No Pretrain / Naive TL / GRU Aligned / BiLSTM Aligned / BiLSTM Adapter
輸出 : artifacts_bilstm/adapter_result.txt
"""

import numpy as np
import torch
import pickle, os, json, sys, glob, itertools

sys.path.insert(0, os.path.dirname(__file__))
from model_bilstm_adapter import BiLSTMWithAdapter

SAVE_DIR      = "artifacts_bilstm"
GRU_ARTIFACTS = "../ml_gru/artificats"
GRU_ALIGNED   = "../ml_alignment_lwc/artifacts_aligned"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ── 載入資料 ──────────────────────────────────────────────────────────────────
print("📂 載入測試資料...")
X_val      = np.load(f"{SAVE_DIR}/raw_X_val.npy")
X_test     = np.load(f"{SAVE_DIR}/raw_X_test.npy")
y_val_s    = np.load(f"{SAVE_DIR}/raw_y_val_s.npy")
y_test_raw = np.load(f"{SAVE_DIR}/raw_y_test_raw.npy")
val_uids   = np.load(f"{SAVE_DIR}/raw_val_uids.npy")
test_uids  = np.load(f"{SAVE_DIR}/raw_test_uids.npy")

with open(f"{SAVE_DIR}/raw_target_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
y_val_raw = scaler.inverse_transform(y_val_s)

# ── 自動偵測 seed，暴力搜尋最佳組合 ──────────────────────────────────────────
all_seeds = sorted([
    int(f.split("seed")[1].replace(".pth", ""))
    for f in glob.glob(f"{SAVE_DIR}/adapter_bilstm_seed*.pth")
])
print(f"🔍 偵測到 {len(all_seeds)} 個 seeds: {all_seeds}")

def predict_seeds(X, seed_list):
    preds = []
    for seed in seed_list:
        ckpt  = torch.load(f"{SAVE_DIR}/adapter_bilstm_seed{seed}.pth", map_location=device)
        model = BiLSTMWithAdapter(5, 7, 64, 2, 1, 0.4).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            p = scaler.inverse_transform(
                model(torch.tensor(X).to(device)).cpu().numpy()
            )
        preds.append(p)
    return np.mean(preds, axis=0)

# 找最佳 seed 組合
print("\n🔎 搜尋最佳 seed 組合...")
best_mae, best_seeds = float("inf"), all_seeds
for r in range(2, len(all_seeds) + 1):
    for combo in itertools.combinations(all_seeds, r):
        vp  = predict_seeds(X_val, list(combo))
        mae = float(np.mean(np.abs(vp - y_val_raw)))
        if mae < best_mae:
            best_mae   = mae
            best_seeds = list(combo)
print(f"  最佳組合（val MAE 最低）: {best_seeds}  val_mae={best_mae:.2f}")

# ── 用最佳組合推論 ────────────────────────────────────────────────────────────
print(f"\n🔮 用最佳 seeds {best_seeds} 推論...")
vp = predict_seeds(X_val,  best_seeds)
tp = predict_seeds(X_test, best_seeds)

# ── 指標 ─────────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, uids):
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    smape = float(np.mean(np.abs(y_true - y_pred) / denom) * 100)
    return {"mae": mae, "rmse": rmse, "smape": smape}

val_m  = metrics(y_val_raw,  vp, val_uids)
test_m = metrics(y_test_raw, tp, test_uids)

print(f"\n  Val  MAE : {val_m['mae']:.2f}")
print(f"  Test MAE : {test_m['mae']:.2f}")
print(f"  Test RMSE: {test_m['rmse']:.2f}")
print(f"  Test SMAPE: {test_m['smape']:.2f}%")

# Per-user MAE
print("\n  Per-user Test MAE:")
for uid in sorted(np.unique(test_uids)):
    m   = test_uids == uid
    mae = float(np.mean(np.abs(tp[m] - y_test_raw[m])))
    print(f"    {uid}: {mae:.1f}  (n={m.sum()})")

# ── 五方比較 ──────────────────────────────────────────────────────────────────
def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else {}

np_mae  = load_json(f"{GRU_ARTIFACTS}/metrics_nopretrain.json").get("test_mae", "N/A")
v5_mae  = load_json(f"{GRU_ARTIFACTS}/metrics_vv5.json").get("test_mae", "N/A")
gru_mae = load_json(f"{GRU_ALIGNED}/aligned_metrics.json").get("test_mae", "N/A")
bi_mae  = load_json(f"{SAVE_DIR}/bilstm_metrics.json").get("test_mae", "N/A")

def fmt(v): return f"{v:.2f}" if isinstance(v, float) else str(v)

report = f"""Bi-LSTM Adapter Pretrain Result
================================
model         : BiLSTMWithAdapter（Adapter: 5→7, BiLSTM body from pretrain）
best_seeds    : {best_seeds}
val_mae       : {val_m['mae']:.6f}
val_rmse      : {val_m['rmse']:.6f}
test_mae      : {test_m['mae']:.6f}
test_rmse     : {test_m['rmse']:.6f}
test_smape    : {test_m['smape']:.2f}%
input_features: daily_expense, roll_7d_mean, roll_30d_mean, dow_sin, dow_cos
adapter       : Linear(5→7, random init) + LayerNorm + ReLU
bilstm_body   : 繼承 Walmart aligned pretrain weights

{'='*62}
              五方比較（Test MAE，越低越好）
{'='*62}
  No Pretrain                 : {fmt(np_mae)}
  Naive TL (GRU v5)           : {fmt(v5_mae)}
  GRU Aligned Pretrain        : {fmt(gru_mae)}
  Bi-LSTM Aligned Pretrain    : {fmt(bi_mae)}
  Bi-LSTM Adapter (本模型)    : {test_m['mae']:.2f}
{'='*62}
  同學 Bi-LSTM（無 pretrain） : 743.51
  本模型                      : {test_m['mae']:.2f}
{'='*62}
"""
print(report)

with open(f"{SAVE_DIR}/adapter_result.txt", "w") as f:
    f.write(report)
with open(f"{SAVE_DIR}/adapter_metrics.json", "w") as f:
    json.dump({"val_mae": val_m["mae"], "test_mae": test_m["mae"],
               "test_rmse": test_m["rmse"], "best_seeds": best_seeds,
               "classmate_test_mae": 743.51}, f, indent=2)

print(f"✅ 結果儲存至 {SAVE_DIR}/adapter_result.txt")
