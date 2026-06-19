"""
Step 7：決策層評估（GRU Aligned，10 features）
================================================
三層評估：
  Layer 1 - 回歸層  ：MAE, RMSE, sMAPE
  Layer 2 - 決策層  ：Precision, Recall, F1, FNR, FPR
  Layer 3 - 成本層  ：Expected Decision Cost（漏報成本 3×，誤報成本 1×）

預警定義：future_7d_sum > 個人近30日均值 × 1.5
輸出：artifacts_aligned/decision_evaluation.json
"""

import numpy as np
import torch
import torch.nn as nn
import pickle, os, json, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS, load_personal_daily, INPUT_DAYS

ARTIFACTS_DIR = "artifacts_aligned"

ALERT_RATIO         = 1.5
COST_FALSE_NEGATIVE = 3.0
COST_FALSE_POSITIVE = 1.0

# 自動偵測所有 seed，使用全部（790.00 的最佳結果）
SEEDS = sorted([
    int(f.split("seed")[1].replace(".pth",""))
    for f in glob.glob(f"{ARTIFACTS_DIR}/finetune_aligned_gru_seed*.pth")
])
print(f"🔍 偵測到 {len(SEEDS)} 個 seeds: {SEEDS}")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

INPUT_SIZE  = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.4
OUTPUT_SIZE = 1


class GRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, num_layers,
                                 dropout=dropout if num_layers > 1 else 0,
                                 batch_first=True)
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
        out        = self.dropout(context)
        out        = self.relu(self.fc1(out))
        return self.fc2(out)


# ── 載入資料 ──────────────────────────────────────────────────────────────────
print("📂 載入資料...")
X_test        = np.load(f"{ARTIFACTS_DIR}/personal_aligned_X_test.npy")
y_test_raw    = np.load(f"{ARTIFACTS_DIR}/personal_aligned_y_test_raw.npy").ravel()
test_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_aligned_test_user_ids.npy")

with open(f"{ARTIFACTS_DIR}/personal_aligned_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

# ── 計算每個 test 樣本的個人基線（roll_30d_mean × 7）─────────────────────────
print("📊 計算個人30日基線...")
personal_daily = load_personal_daily()

baseline_7d_list = []
for user_id in sorted(personal_daily["user_id"].unique()):
    u = personal_daily[personal_daily["user_id"] == user_id].sort_values("date").reset_index(drop=True)
    u["roll30_mean"] = u["daily_expense"].rolling(30, min_periods=1).mean()
    u["baseline_7d"] = u["roll30_mean"] * 7
    future_col = u["daily_expense"].rolling(7).sum().shift(-7)
    valid_mask = future_col.notna()
    baseline_vals = u.loc[valid_mask, "baseline_7d"].values
    n = len(baseline_vals)
    if n <= INPUT_DAYS:
        continue
    windows_base = baseline_vals[INPUT_DAYS:]
    v_end = int(len(windows_base) * 0.85)
    baseline_7d_list.extend(windows_base[v_end:])

baseline_7d = np.array(baseline_7d_list[:len(y_test_raw)], dtype=np.float32)
alert_threshold = baseline_7d * ALERT_RATIO
y_true_alert    = (y_test_raw > alert_threshold).astype(int)
print(f"  預警正例：{y_true_alert.mean()*100:.1f}%  ({y_true_alert.sum()}/{len(y_true_alert)})")

# ── 推論 ──────────────────────────────────────────────────────────────────────
print(f"\n🔮 推論（seeds={SEEDS}）...")
preds_list = []
for seed in SEEDS:
    ckpt  = torch.load(f"{ARTIFACTS_DIR}/finetune_aligned_gru_seed{seed}.pth", map_location=device)
    model = GRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds_list.append(model(X_t).cpu().numpy())

test_preds    = target_scaler.inverse_transform(np.mean(preds_list, axis=0)).ravel()
y_pred_alert  = (test_preds > alert_threshold).astype(int)


# ── 指標計算 ──────────────────────────────────────────────────────────────────
def decision_metrics(y_true, y_pred):
    TP = int(np.sum((y_true==1)&(y_pred==1)))
    TN = int(np.sum((y_true==0)&(y_pred==0)))
    FP = int(np.sum((y_true==0)&(y_pred==1)))
    FN = int(np.sum((y_true==1)&(y_pred==0)))
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2*precision*recall / (precision+recall+1e-8)
    fnr       = FN / (TP + FN + 1e-8)
    fpr       = FP / (FP + TN + 1e-8)
    cost      = (FN*COST_FALSE_NEGATIVE + FP*COST_FALSE_POSITIVE) / len(y_true)
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
    user_nmae = user_mae / (np.mean(np.abs(yt))+1e-8)
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
print(f"  GRU Aligned（10 features，all seeds）")
print(f"{'='*62}")
print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  sMAPE={smape:.2f}%")
print(f"  Precision={global_dec['precision']:.4f}  Recall={global_dec['recall']:.4f}  F1={global_dec['f1']:.4f}")
print(f"  FNR={global_dec['fnr']:.4f}  FPR={global_dec['fpr']:.4f}  Cost={global_dec['expected_cost']:.4f}")
print(f"  Per-user MAE：avg={np.mean(user_maes):.2f}  max={max(user_maes):.2f}  std={np.std(user_maes):.2f}")
print(f"  Per-user F1 ：avg={np.mean(user_f1s):.4f}  min={min(user_f1s):.4f}")
print(f"  Per-user FNR：avg={np.mean(user_fnrs):.4f}  max={max(user_fnrs):.4f}")
print(f"  Per-user Cost：avg={np.mean(user_costs):.4f}  max={max(user_costs):.4f}")
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
    "model"                   : "GRU Aligned (10 features, all seeds)",
    "seeds_used"              : SEEDS,
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
with open(f"{ARTIFACTS_DIR}/decision_evaluation.json","w",encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, cls=NpEncoder)

print(f"\n✅ 儲存至 {ARTIFACTS_DIR}/decision_evaluation.json")
print("🎉 完成！")
