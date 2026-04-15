"""
Step 6：三級警報評估（GRU Classifier）
=========================================
分類模型的三級警報框架，與 regression 版本設計一致，方便跨模型比較

【核心設計說明】
  分類模型直接輸出 P(超標)（二元機率），沒有預測金額
  因此「預測等級」以機率區間定義：
    P < 0.35          → 🟢 正常（模型認為不太會超標）
    0.35 ≤ P < 0.65  → 🟡 低度（模型不確定，保守警示）
    P ≥ 0.65          → 🔴 高度（模型認為很可能超標）

  「真實等級」仍以實際金額 vs 個人基線定義（與 regression 版完全相同）：
    actual < 1.2× baseline → 🟢 正常
    1.2× ≤ actual < 1.8×  → 🟡 低度
    actual ≥ 1.8×          → 🔴 高度

【注意】
  raw future_7d_sum 未在 preprocess 時儲存，此腳本會重新從原始資料計算
  確保與 2_preprocess_personal_clf.py 完全相同的切分邏輯

輸出：artifacts_clf/tiered_alert_evaluation.json
"""

import numpy as np
import torch
import torch.nn as nn
import glob, os, sys, json
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import (
    ALIGNED_FEATURE_COLS, load_personal_daily,
    INPUT_DAYS, ALERT_RATIO
)

ARTIFACTS_DIR = "artifacts_clf"

# ── 三級門檻 ──────────────────────────────────────────────────────────────────
# 真實等級（金額比例）
TRUE_LOW_RATIO  = 1.2
TRUE_HIGH_RATIO = 1.8

# 預測等級（機率區間）
PROB_LOW_THRESH  = 0.35   # P < 0.35 → 正常；P ≥ 0.35 → 低度
PROB_HIGH_THRESH = 0.65   # P ≥ 0.65 → 高度

TIER_NAMES  = ["🟢 正常", "🟡 低度", "🔴 高度"]
TIER_LABELS = [0, 1, 2]

COST_MATRIX = np.array([
    [0.0,  0.5,  2.0],
    [2.0,  0.0,  1.0],
    [5.0,  3.0,  0.0],
])

INPUT_SIZE  = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.3

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
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


# ── 載入 test 資料 ────────────────────────────────────────────────────────────
print("📂 載入資料...")
X_test        = np.load(f"{ARTIFACTS_DIR}/personal_clf_X_test.npy")
test_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_clf_test_user_ids.npy")
baseline_7d   = np.load(f"{ARTIFACTS_DIR}/personal_clf_test_baseline_7d.npy")

# ── 重新計算 raw future_7d_sum（供三級真實標籤使用）─────────────────────────
# preprocess 只存了 binary label，需從原始資料重建真實金額
print("🔧 重建 raw future_7d_sum...")
personal_daily = load_personal_daily()
y_test_raw_list = []

for user_id in sorted(personal_daily["user_id"].unique()):
    u = personal_daily[personal_daily["user_id"] == user_id].sort_values("date").reset_index(drop=True)

    future_7d = u["daily_expense"].rolling(7).sum().shift(-7)
    u["future_7d_sum"] = future_7d

    # ⚠️  不 dropna：preprocess 裡 alert_label = (NaN > x) = False = 0.0，
    #     所以 dropna(subset=["alert_label"]) 不會丟掉最後 7 行。
    #     這裡填 0 讓 shape 完全對齊 preprocess 儲存的 baseline_7d。
    u["future_7d_sum"] = u["future_7d_sum"].fillna(0.0)

    if len(u) <= INPUT_DAYS + 5:
        continue

    future_arr = u["future_7d_sum"].values.astype(np.float32)

    windows_y = [future_arr[t] for t in range(INPUT_DAYS, len(u))]
    n     = len(windows_y)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)
    if t_end == 0:
        continue
    y_test_raw_list.extend(windows_y[v_end:])

y_test_raw = np.array(y_test_raw_list[:len(baseline_7d)], dtype=np.float32)
print(f"  重建完成：test_n={len(y_test_raw)}，baseline shape={baseline_7d.shape}")

# ── 推論：ensemble 機率 ───────────────────────────────────────────────────────
SEEDS = sorted([
    int(f.split("seed")[1].replace(".pth", ""))
    for f in glob.glob(f"{ARTIFACTS_DIR}/finetune_clf_seed*.pth")
])
print(f"🔮 推論（{len(SEEDS)} seeds ensemble）: {SEEDS}")

probs_list = []
for seed in SEEDS:
    ckpt  = torch.load(f"{ARTIFACTS_DIR}/finetune_clf_seed{seed}.pth", map_location=device)
    model = GRUClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().ravel()
        probs  = 1 / (1 + np.exp(-logits))
    probs_list.append(probs)

ensemble_probs = np.mean(probs_list, axis=0)   # shape: (n_test,)

# ── 真實等級（金額比例）───────────────────────────────────────────────────────
def actual_to_tier(actual, baseline):
    ratio = actual / (baseline + 1e-8)
    tiers = np.zeros(len(actual), dtype=int)
    tiers[ratio >= TRUE_LOW_RATIO]  = 1
    tiers[ratio >= TRUE_HIGH_RATIO] = 2
    return tiers

# ── 預測等級（機率區間）───────────────────────────────────────────────────────
def prob_to_tier(probs):
    tiers = np.zeros(len(probs), dtype=int)
    tiers[probs >= PROB_LOW_THRESH]  = 1
    tiers[probs >= PROB_HIGH_THRESH] = 2
    return tiers

y_true_tier = actual_to_tier(y_test_raw, baseline_7d)
y_pred_tier = prob_to_tier(ensemble_probs)

print(f"\n真實分佈：" + "  ".join(
    f"{TIER_NAMES[t]}={np.sum(y_true_tier==t)}({np.mean(y_true_tier==t)*100:.1f}%)"
    for t in TIER_LABELS))
print(f"預測分佈：" + "  ".join(
    f"{TIER_NAMES[t]}={np.sum(y_pred_tier==t)}({np.mean(y_pred_tier==t)*100:.1f}%)"
    for t in TIER_LABELS))
print(f"機率分佈：mean={ensemble_probs.mean():.3f}  "
      f"P<0.35: {(ensemble_probs<0.35).mean()*100:.1f}%  "
      f"0.35-0.65: {((ensemble_probs>=0.35)&(ensemble_probs<0.65)).mean()*100:.1f}%  "
      f"P≥0.65: {(ensemble_probs>=0.65).mean()*100:.1f}%")


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

    exact_acc     = float(np.mean(y_true == y_pred))
    ordinal_acc   = float(np.mean(np.abs(y_true - y_pred) <= 1))
    severe_err    = float(np.mean(np.abs(y_true - y_pred) == 2))
    expected_cost = float(sum(COST_MATRIX[t, p] for t, p in zip(y_true, y_pred)) / n)

    high_mask = y_true == 2
    high_fnr  = float(np.sum((y_true == 2) & (y_pred != 2)) / high_mask.sum()) \
                if high_mask.sum() > 0 else None

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
    m["true_dist"]      = {TIER_NAMES[t]: int(np.sum(y_true_tier[mask] == t)) for t in TIER_LABELS}
    m["pred_dist"]      = {TIER_NAMES[t]: int(np.sum(y_pred_tier[mask] == t)) for t in TIER_LABELS}
    m["n_samples"]      = int(mask.sum())
    m["prob_mean"]      = round(float(ensemble_probs[mask].mean()), 4)
    per_user_tiered[uid] = m


# ── 列印 ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  三級警報評估（GRU Classifier，prob thresholds: {PROB_LOW_THRESH} / {PROB_HIGH_THRESH}）")
print(f"  真實等級：< {TRUE_LOW_RATIO}× 正常 / {TRUE_LOW_RATIO}~{TRUE_HIGH_RATIO}× 低度 / ≥ {TRUE_HIGH_RATIO}× 高度")
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
print(f"  Expected Cost    ：{global_tiered['expected_cost']:.4f}  （三級成本矩陣）")
if global_tiered["high_alert_fnr"] is not None:
    print(f"  高度警告 FNR     ：{global_tiered['high_alert_fnr']:.4f}")

print(f"\n  【四模型比較（供參考）】")
print(f"  {'模型':20s}  {'OrdAcc':>8}  {'SevErr':>8}  {'Cost':>8}  {'HighFNR':>9}")
print(f"  {'ckh BiLSTM':20s}  {'0.7928':>8}  {'0.2072':>8}  {'0.9245':>8}  {'0.6636':>9}")
print(f"  {'GRU Aligned (lwc)':20s}  {'0.8193':>8}  {'0.1807':>8}  {'0.9650':>8}  {'0.7524':>9}")
print(f"  {'BiLSTM v2':20s}  {'0.8006':>8}  {'0.1994':>8}  {'0.9930':>8}  {'0.7905':>9}")
print(f"  {'GRU Classifier':20s}  {global_tiered['ordinal_accuracy']:>8.4f}  "
      f"{global_tiered['severe_error_rate']:>8.4f}  {global_tiered['expected_cost']:>8.4f}  "
      f"{str(global_tiered['high_alert_fnr']) if global_tiered['high_alert_fnr'] else 'N/A':>9}")

print(f"\n  Per-user 三級摘要：")
for uid, v in sorted(per_user_tiered.items()):
    high_fnr_str = f"HighFNR={v['high_alert_fnr']:.3f}" if v["high_alert_fnr"] is not None else "HighFNR=N/A"
    print(f"  {uid:10s}  n={v['n_samples']:3d}  P_mean={v['prob_mean']:.3f}  "
          f"OrdAcc={v['ordinal_accuracy']:.3f}  Cost={v['expected_cost']:.3f}  {high_fnr_str}")
    print(f"    真實分佈：{v['true_dist']}")
    print(f"    預測分佈：{v['pred_dist']}")


# ── 儲存 ──────────────────────────────────────────────────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

output = {
    "model"              : "GRU Classifier (12 features, Walmart TL, ensemble)",
    "seeds_used"         : SEEDS,
    "prob_low_thresh"    : PROB_LOW_THRESH,
    "prob_high_thresh"   : PROB_HIGH_THRESH,
    "true_low_ratio"     : TRUE_LOW_RATIO,
    "true_high_ratio"    : TRUE_HIGH_RATIO,
    "tier_names"         : TIER_NAMES,
    "cost_matrix"        : COST_MATRIX.tolist(),
    "global_metrics"     : global_tiered,
    "true_distribution"  : {TIER_NAMES[t]: int(np.sum(y_true_tier==t)) for t in TIER_LABELS},
    "pred_distribution"  : {TIER_NAMES[t]: int(np.sum(y_pred_tier==t)) for t in TIER_LABELS},
    "prob_stats"         : {
        "mean" : round(float(ensemble_probs.mean()), 4),
        "pct_normal": round(float((ensemble_probs < PROB_LOW_THRESH).mean()), 4),
        "pct_low"   : round(float(((ensemble_probs >= PROB_LOW_THRESH) & (ensemble_probs < PROB_HIGH_THRESH)).mean()), 4),
        "pct_high"  : round(float((ensemble_probs >= PROB_HIGH_THRESH).mean()), 4),
    },
    "per_user"           : per_user_tiered,
    "note": (
        f"預測等級以機率區間定義（P<{PROB_LOW_THRESH}→正常, "
        f"{PROB_LOW_THRESH}≤P<{PROB_HIGH_THRESH}→低度, P≥{PROB_HIGH_THRESH}→高度）"
        f"，與 regression 模型的「預測金額 vs 基線」方法不同，比較時需注意"
    ),
}
with open(f"{ARTIFACTS_DIR}/tiered_alert_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, cls=NpEncoder)

print(f"\n✅ 儲存至 {ARTIFACTS_DIR}/tiered_alert_evaluation.json")
print("🎉 完成！")
