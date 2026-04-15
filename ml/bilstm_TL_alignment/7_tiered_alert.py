"""
Step 7：三級警報評估（BiLSTM v2）
=====================================
將二元預警擴展為三個等級：
  🟢 正常    ：預測值 < 個人基線 × 1.2
  🟡 低度警告：個人基線 × 1.2 ≤ 預測值 < 個人基線 × 1.8
  🔴 高度警告：預測值 ≥ 個人基線 × 1.8

輸出：artifacts_bilstm_v2/tiered_alert_evaluation.json
"""

import numpy as np
import torch
import torch.nn as nn
import pickle, os, json, sys
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS, load_personal_daily, INPUT_DAYS

ARTIFACTS_DIR = "artifacts_bilstm_v2"

LOW_RATIO   = 1.2
HIGH_RATIO  = 1.8
TIER_NAMES  = ["🟢 正常", "🟡 低度", "🔴 高度"]
TIER_LABELS = [0, 1, 2]

COST_MATRIX = np.array([
    [0.0,  0.5,  2.0],
    [2.0,  0.0,  1.0],
    [5.0,  3.0,  0.0],
])

BEST_COMBO = [789]

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


class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        bi_hidden       = hidden_size * 2
        self.attention  = nn.Linear(bi_hidden, 1)
        self.layer_norm = nn.LayerNorm(bi_hidden)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(bi_hidden, hidden_size)
        self.fc2        = nn.Linear(hidden_size, output_size)
        self.relu       = nn.ReLU()

    def forward(self, x):
        out, _  = self.lstm(x)
        attn_w  = torch.softmax(self.attention(out), dim=1)
        context = (out * attn_w).sum(dim=1)
        context = self.layer_norm(context)
        out     = self.dropout(context)
        out     = self.relu(self.fc1(out))
        return self.fc2(out)


# ── 載入資料 ──────────────────────────────────────────────────────────────────
X_test        = np.load(f"{ARTIFACTS_DIR}/personal_X_test.npy")
y_test_raw    = np.load(f"{ARTIFACTS_DIR}/personal_y_test_raw.npy").ravel()
test_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_test_user_ids.npy")

with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

# ── 建立個人基線 ──────────────────────────────────────────────────────────────
personal_daily = load_personal_daily()
baseline_7d_list = []
for user_id in sorted(personal_daily["user_id"].unique()):
    u = personal_daily[personal_daily["user_id"] == user_id].sort_values("date").reset_index(drop=True)
    u["roll30_mean"] = u["daily_expense"].rolling(30, min_periods=1).mean()
    u["baseline_7d"] = u["roll30_mean"] * 7
    feat_len = len(u) - 7
    if feat_len <= INPUT_DAYS:
        continue
    windows_baseline = [u["baseline_7d"].iloc[t] for t in range(INPUT_DAYS, feat_len)]
    n     = len(windows_baseline)
    v_end = int(n * 0.85)
    baseline_7d_list.extend(windows_baseline[v_end:])

baseline_7d = np.array(baseline_7d_list[:len(y_test_raw)], dtype=np.float32)

# ── 推論 ──────────────────────────────────────────────────────────────────────
preds_list = []
for seed in BEST_COMBO:
    ckpt  = torch.load(f"{ARTIFACTS_DIR}/finetune_bilstm_seed{seed}.pth", map_location=device)
    model = BiLSTMWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, 1, DROPOUT).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        preds_list.append(model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy())

test_preds = target_scaler.inverse_transform(np.mean(preds_list, axis=0)).ravel()


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

    exact_acc    = float(np.mean(y_true == y_pred))
    ordinal_acc  = float(np.mean(np.abs(y_true - y_pred) <= 1))
    severe_err   = float(np.mean(np.abs(y_true - y_pred) == 2))
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
print(f"  三級警報評估（BiLSTM v2，thresholds: {LOW_RATIO}× / {HIGH_RATIO}×）")
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
    "model"            : "BiLSTM v2 (seed=789)",
    "low_ratio"        : LOW_RATIO,
    "high_ratio"       : HIGH_RATIO,
    "tier_names"       : TIER_NAMES,
    "cost_matrix"      : COST_MATRIX.tolist(),
    "global_metrics"   : global_tiered,
    "true_distribution": {TIER_NAMES[t]: int(np.sum(y_true_tier==t)) for t in TIER_LABELS},
    "pred_distribution": {TIER_NAMES[t]: int(np.sum(y_pred_tier==t)) for t in TIER_LABELS},
    "per_user"         : per_user_tiered,
}
with open(f"{ARTIFACTS_DIR}/tiered_alert_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, cls=NpEncoder)

print(f"\n✅ 儲存至 {ARTIFACTS_DIR}/tiered_alert_evaluation.json")
print("🎉 完成！")
