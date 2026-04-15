"""
Step 9：三級警報評估（GRU Aligned）
=====================================
將二元預警擴展為三個等級：
  🟢 正常    ：預測值 < 個人基線 × LOW_RATIO
  🟡 低度警告：個人基線 × LOW_RATIO  ≤ 預測值 < 個人基線 × HIGH_RATIO
  🔴 高度警告：預測值 ≥ 個人基線 × HIGH_RATIO

評估指標：
  - 每個等級的 Precision / Recall / F1（one-vs-rest）
  - Ordinal Accuracy（預測等級與真實等級差 ≤ 1 視為可接受）
  - 三級成本矩陣（漏報高度警告懲罰最重）
  - Per-user 三級分佈

輸出：artifacts_aligned/tiered_alert_evaluation.json
"""

import numpy as np
import torch
import torch.nn as nn
import pickle, os, json, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS, load_personal_daily, INPUT_DAYS

ARTIFACTS_DIR = "artifacts_aligned"

# ── 三級門檻設定 ──────────────────────────────────────────────────────────────
LOW_RATIO  = 1.2   # 超過月均 20% → 低度警告
HIGH_RATIO = 1.8   # 超過月均 80% → 高度警告

TIER_NAMES  = ["🟢 正常", "🟡 低度", "🔴 高度"]
TIER_LABELS = [0, 1, 2]

# ── 三級成本矩陣（行=真實，列=預測）─────────────────────────────────────────
#              預測正常  預測低度  預測高度
COST_MATRIX = np.array([
    [0.0,   0.5,  2.0],   # 真實正常（誤報低度小罰，誤報高度大罰）
    [2.0,   0.0,  1.0],   # 真實低度（漏報罰2，誇大成高度罰1）
    [5.0,   3.0,  0.0],   # 真實高度（完全漏報最重罰5，降級罰3）
])

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


# ── 載入資料 ──────────────────────────────────────────────────────────────────
SEEDS = sorted([
    int(f.split("seed")[1].replace(".pth",""))
    for f in glob.glob(f"{ARTIFACTS_DIR}/finetune_aligned_gru_seed*.pth")
])
print(f"🔍 偵測到 {len(SEEDS)} 個 seeds: {SEEDS}")

X_test        = np.load(f"{ARTIFACTS_DIR}/personal_aligned_X_test.npy")
y_test_raw    = np.load(f"{ARTIFACTS_DIR}/personal_aligned_y_test_raw.npy").ravel()
test_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_aligned_test_user_ids.npy")

with open(f"{ARTIFACTS_DIR}/personal_aligned_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

# ── 建立個人基線 ──────────────────────────────────────────────────────────────
personal_daily = load_personal_daily()
baseline_7d_list = []
for user_id in sorted(personal_daily["user_id"].unique()):
    u = personal_daily[personal_daily["user_id"] == user_id].sort_values("date").reset_index(drop=True)
    u["roll30_mean"] = u["daily_expense"].rolling(30, min_periods=1).mean()
    u["baseline_7d"] = u["roll30_mean"] * 7
    future_col    = u["daily_expense"].rolling(7).sum().shift(-7)
    valid_mask    = future_col.notna()
    baseline_vals = u.loc[valid_mask, "baseline_7d"].values
    n = len(baseline_vals)
    if n <= INPUT_DAYS:
        continue
    windows_base = baseline_vals[INPUT_DAYS:]
    v_end = int(len(windows_base) * 0.85)
    baseline_7d_list.extend(windows_base[v_end:])

baseline_7d = np.array(baseline_7d_list[:len(y_test_raw)], dtype=np.float32)

# ── 推論 ──────────────────────────────────────────────────────────────────────
preds_list = []
for seed in SEEDS:
    ckpt  = torch.load(f"{ARTIFACTS_DIR}/finetune_aligned_gru_seed{seed}.pth", map_location=device)
    model = GRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, 1, DROPOUT).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        preds_list.append(model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy())

test_preds = target_scaler.inverse_transform(np.mean(preds_list, axis=0)).ravel()


# ── 三級分類函式 ──────────────────────────────────────────────────────────────
def to_tier(values, baseline):
    """回傳每個樣本的等級 array（0=正常, 1=低度, 2=高度）"""
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
    n = len(y_true)
    # 1. Confusion matrix (3×3)
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    # 2. Per-level Precision / Recall / F1（one-vs-rest）
    per_level = {}
    for lv in TIER_LABELS:
        tp = cm[lv, lv]
        fp = cm[:, lv].sum() - tp
        fn = cm[lv, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        per_level[TIER_NAMES[lv]] = {
            "precision": round(float(prec), 4),
            "recall"   : round(float(rec),  4),
            "f1"       : round(float(f1),   4),
            "support"  : int(cm[lv].sum()),
        }

    # 3. Ordinal Accuracy（差 ≤ 1 視為可接受）
    exact_acc   = float(np.mean(y_true == y_pred))
    ordinal_acc = float(np.mean(np.abs(y_true - y_pred) <= 1))

    # 4. 嚴重誤判率（高度→正常 或 正常→高度，差2級）
    severe_err = float(np.mean(np.abs(y_true - y_pred) == 2))

    # 5. 成本矩陣期望成本
    expected_cost = float(
        sum(COST_MATRIX[t, p] for t, p in zip(y_true, y_pred)) / n
    )

    # 6. 高度警告的漏報率（最重要）
    high_mask = y_true == 2
    if high_mask.sum() > 0:
        high_fnr = float(np.sum((y_true == 2) & (y_pred != 2)) / high_mask.sum())
    else:
        high_fnr = None

    return {
        "confusion_matrix" : cm.tolist(),
        "per_level"        : per_level,
        "exact_accuracy"   : round(exact_acc,   4),
        "ordinal_accuracy" : round(ordinal_acc,  4),
        "severe_error_rate": round(severe_err,   4),
        "expected_cost"    : round(expected_cost, 4),
        "high_alert_fnr"   : round(high_fnr, 4) if high_fnr is not None else None,
    }


global_tiered = tiered_metrics(y_true_tier, y_pred_tier)

# ── Per-user 三級指標 ─────────────────────────────────────────────────────────
per_user_tiered = {}
for uid in np.unique(test_user_ids):
    mask = test_user_ids == uid
    m    = tiered_metrics(y_true_tier[mask], y_pred_tier[mask])
    # 加上該 user 的三級分佈
    m["true_dist"] = {
        TIER_NAMES[t]: int(np.sum(y_true_tier[mask] == t)) for t in TIER_LABELS
    }
    m["pred_dist"] = {
        TIER_NAMES[t]: int(np.sum(y_pred_tier[mask] == t)) for t in TIER_LABELS
    }
    m["n_samples"] = int(mask.sum())
    per_user_tiered[uid] = m


# ── 列印 ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  三級警報評估（GRU Aligned，thresholds: {LOW_RATIO}× / {HIGH_RATIO}×）")
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
print(f"  Exact Accuracy   ：{global_tiered['exact_accuracy']:.4f}  （預測等級完全正確）")
print(f"  Ordinal Accuracy ：{global_tiered['ordinal_accuracy']:.4f}  （差 ≤ 1 級都算可接受）")
print(f"  Severe Error Rate：{global_tiered['severe_error_rate']:.4f}  （跨兩級的嚴重誤判）")
print(f"  Expected Cost    ：{global_tiered['expected_cost']:.4f}  （三級成本矩陣）")
if global_tiered["high_alert_fnr"] is not None:
    print(f"  高度警告 FNR     ：{global_tiered['high_alert_fnr']:.4f}  （高度未被偵測的比例）")

print(f"\n  成本矩陣說明：")
print(f"  {'':15s}  {'預測正常':>8}  {'預測低度':>8}  {'預測高度':>8}")
for i, row in enumerate(COST_MATRIX):
    print(f"  真實{TIER_NAMES[i]:10s}  {row[0]:>8.1f}  {row[1]:>8.1f}  {row[2]:>8.1f}")

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
    "model"            : "GRU Aligned (all seeds)",
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
