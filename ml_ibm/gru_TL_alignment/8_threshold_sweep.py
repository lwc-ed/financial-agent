"""
Step 8：Alert Threshold 掃描
============================
固定模型預測值不變，對 alert_ratio 做 grid search
找出在 F1 / Expected Cost 上最佳的 threshold 倍率

輸出：
  - artifacts_aligned/threshold_sweep.json
"""

import numpy as np
import torch
import torch.nn as nn
import pickle, os, json, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS, load_personal_daily, INPUT_DAYS

ARTIFACTS_DIR = "artifacts_aligned"
COST_FN = 3.0
COST_FP = 1.0

THRESHOLDS = [round(x * 0.1, 1) for x in range(8, 26)]  # 0.8 ~ 2.5

# ── 裝置 ──────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

INPUT_SIZE  = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.4


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
        self.fc2        = nn.Linear(hidden_size // 2, 1)
        self.relu       = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        context    = self.layer_norm(context)
        out        = self.dropout(context)
        out        = self.relu(self.fc1(out))
        return self.fc2(out)


# ── 自動偵測 seeds ────────────────────────────────────────────────────────────
SEEDS = sorted([
    int(f.split("seed")[1].replace(".pth", ""))
    for f in glob.glob(f"{ARTIFACTS_DIR}/finetune_aligned_gru_seed*.pth")
])
print(f"🔍 偵測到 {len(SEEDS)} 個 seeds: {SEEDS}")

# ── 載入資料 ──────────────────────────────────────────────────────────────────
print("📂 載入資料...")
X_test        = np.load(f"{ARTIFACTS_DIR}/personal_aligned_X_test.npy")
y_test_raw    = np.load(f"{ARTIFACTS_DIR}/personal_aligned_y_test_raw.npy").ravel()
test_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_aligned_test_user_ids.npy")

with open(f"{ARTIFACTS_DIR}/personal_aligned_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

# ── 推論（只跑一次）──────────────────────────────────────────────────────────
print("🔮 推論...")
preds_list = []
for seed in SEEDS:
    ckpt  = torch.load(f"{ARTIFACTS_DIR}/finetune_aligned_gru_seed{seed}.pth", map_location=device)
    model = GRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, 1, DROPOUT).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        preds_list.append(model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy())

test_preds = target_scaler.inverse_transform(np.mean(preds_list, axis=0)).ravel()

# ── 計算每個 test 樣本的個人基線（roll_30d_mean × 7）─────────────────────────
print("📊 計算個人30日基線...")
personal_daily = load_personal_daily()
baseline_7d_list = []
for user_id in sorted(personal_daily["user_id"].unique()):
    u = personal_daily[personal_daily["user_id"] == user_id].sort_values("date").reset_index(drop=True)
    u["roll30_mean"] = u["daily_expense"].rolling(30, min_periods=1).mean()
    u["baseline_7d"] = u["roll30_mean"] * 7
    future_col  = u["daily_expense"].rolling(7).sum().shift(-7)
    valid_mask  = future_col.notna()
    baseline_vals = u.loc[valid_mask, "baseline_7d"].values
    n = len(baseline_vals)
    if n <= INPUT_DAYS:
        continue
    windows_base = baseline_vals[INPUT_DAYS:]
    v_end = int(len(windows_base) * 0.85)
    baseline_7d_list.extend(windows_base[v_end:])

baseline_7d = np.array(baseline_7d_list[:len(y_test_raw)], dtype=np.float32)


# ── 指標計算 ──────────────────────────────────────────────────────────────────
def metrics_at_threshold(ratio):
    thresh       = baseline_7d * ratio
    y_true_alert = (y_test_raw > thresh).astype(int)
    y_pred_alert = (test_preds  > thresh).astype(int)
    TP = int(np.sum((y_true_alert == 1) & (y_pred_alert == 1)))
    TN = int(np.sum((y_true_alert == 0) & (y_pred_alert == 0)))
    FP = int(np.sum((y_true_alert == 0) & (y_pred_alert == 1)))
    FN = int(np.sum((y_true_alert == 1) & (y_pred_alert == 0)))
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    fnr       = FN / (TP + FN + 1e-8)
    fpr       = FP / (FP + TN + 1e-8)
    cost      = (FN * COST_FN + FP * COST_FP) / len(y_true_alert)
    alert_rate = y_true_alert.mean()
    return {
        "ratio"        : ratio,
        "alert_rate"   : round(float(alert_rate), 4),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "precision"    : round(precision, 4),
        "recall"       : round(recall,    4),
        "f1"           : round(f1,        4),
        "fnr"          : round(fnr,       4),
        "fpr"          : round(fpr,       4),
        "expected_cost": round(cost,      4),
    }


# ── 掃描所有 thresholds ───────────────────────────────────────────────────────
print(f"\n🔍 Threshold Sweep（{THRESHOLDS[0]}× ~ {THRESHOLDS[-1]}×）")
print("=" * 85)
print(f"  {'Ratio':>6}  {'Alert%':>7}  {'Precision':>10}  {'Recall':>7}  {'F1':>7}  {'FNR':>7}  {'FPR':>7}  {'Cost':>7}")
print("-" * 85)

results = []
for ratio in THRESHOLDS:
    m = metrics_at_threshold(ratio)
    results.append(m)
    print(f"  {ratio:>5.1f}×  {m['alert_rate']*100:>6.1f}%  {m['precision']:>10.4f}  "
          f"{m['recall']:>7.4f}  {m['f1']:>7.4f}  {m['fnr']:>7.4f}  {m['fpr']:>7.4f}  {m['expected_cost']:>7.4f}")

# ── 找最佳 ────────────────────────────────────────────────────────────────────
best_f1   = max(results, key=lambda x: x["f1"])
best_cost = min(results, key=lambda x: x["expected_cost"])

print("=" * 85)
print(f"\n  🏆 最佳 F1  ：ratio={best_f1['ratio']}×  F1={best_f1['f1']:.4f}  "
      f"Recall={best_f1['recall']:.4f}  Cost={best_f1['expected_cost']:.4f}")
print(f"  🏆 最低 Cost：ratio={best_cost['ratio']}×  Cost={best_cost['expected_cost']:.4f}  "
      f"F1={best_cost['f1']:.4f}  Recall={best_cost['recall']:.4f}")
print(f"\n  （原本 1.5× → F1={metrics_at_threshold(1.5)['f1']:.4f}，Cost={metrics_at_threshold(1.5)['expected_cost']:.4f}）")

# ── 儲存 ──────────────────────────────────────────────────────────────────────
output = {
    "model"           : "GRU Aligned (all seeds)",
    "sweep_range"     : f"{THRESHOLDS[0]}x ~ {THRESHOLDS[-1]}x",
    "cost_fn"         : COST_FN,
    "cost_fp"         : COST_FP,
    "best_f1"         : best_f1,
    "best_cost"       : best_cost,
    "baseline_1_5x"   : metrics_at_threshold(1.5),
    "all_results"     : results,
}
with open(f"{ARTIFACTS_DIR}/threshold_sweep.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ 儲存至 {ARTIFACTS_DIR}/threshold_sweep.json")
print("🎉 完成！")
