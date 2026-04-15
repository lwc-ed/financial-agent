"""
Step 5：分類模型三層評估
==========================
三層評估框架（分類任務版）：
  Layer 1 - 分類層  ：Accuracy, Precision, Recall, F1, AUC-ROC
  Layer 2 - 決策層  ：TP/TN/FP/FN, FNR, FPR, Expected Cost
  Layer 3 - 公平性層：Per-user 指標（衡量模型對每位用戶是否公平）

預警策略：
  模型直接輸出 P(超標) > 0.5 → 預警
  （不需要先預測金額，訓練目標與評估目標完全一致）

輸出：artifacts_clf/clf_evaluation.json
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import glob, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS

ARTIFACTS_DIR     = "artifacts_clf"
COST_FALSE_NEGATIVE = 3.0
COST_FALSE_POSITIVE = 1.0
DECISION_THRESHOLD  = 0.5

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

    def encode(self, x):
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        return self.layer_norm(context)

    def forward(self, x):
        context = self.encode(x)
        out     = self.dropout(context)
        out     = self.relu(self.fc1(out))
        return self.fc2(out)


# ── 載入資料 ──────────────────────────────────────────────────────────────────
print("📂 載入測試資料...")
X_test        = np.load(f"{ARTIFACTS_DIR}/personal_clf_X_test.npy")
y_test        = np.load(f"{ARTIFACTS_DIR}/personal_clf_y_test.npy").ravel().astype(int)
test_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_clf_test_user_ids.npy")

print(f"  test_n={len(y_test)}  正例={y_test.sum()} ({y_test.mean()*100:.1f}%)")

# ── 自動偵測所有 seeds ────────────────────────────────────────────────────────
SEEDS = sorted([
    int(f.split("seed")[1].replace(".pth", ""))
    for f in glob.glob(f"{ARTIFACTS_DIR}/finetune_clf_seed*.pth")
])
print(f"🔍 偵測到 {len(SEEDS)} 個 seeds: {SEEDS}")

# ── 推論：所有 seeds 的機率平均（ensemble）────────────────────────────────────
print("\n🔮 推論（ensemble 平均機率）...")
probs_list = []
for seed in SEEDS:
    ckpt  = torch.load(f"{ARTIFACTS_DIR}/finetune_clf_seed{seed}.pth", map_location=device)
    model = GRUClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t).cpu().numpy().ravel()
        probs  = 1 / (1 + np.exp(-logits))   # sigmoid
    probs_list.append(probs)

ensemble_probs = np.mean(probs_list, axis=0)
y_pred         = (ensemble_probs > DECISION_THRESHOLD).astype(int)


# ── 指標函式 ──────────────────────────────────────────────────────────────────
def clf_metrics(y_true, y_pred, y_prob=None):
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc  = (TP + TN) / (len(y_true) + 1e-8)
    prec = TP / (TP + FP + 1e-8)
    rec  = TP / (TP + FN + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    fnr  = FN / (TP + FN + 1e-8)
    fpr  = FP / (FP + TN + 1e-8)
    cost = (FN * COST_FALSE_NEGATIVE + FP * COST_FALSE_POSITIVE) / len(y_true)
    auc  = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = None
    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy"     : round(float(acc),  4),
        "precision"    : round(float(prec), 4),
        "recall"       : round(float(rec),  4),
        "f1"           : round(float(f1),   4),
        "fnr"          : round(float(fnr),  4),
        "fpr"          : round(float(fpr),  4),
        "expected_cost": round(float(cost), 4),
        "auc_roc"      : round(auc, 4) if auc is not None else None,
    }


# ── Layer 1 & 2：全局指標 ─────────────────────────────────────────────────────
global_metrics = clf_metrics(y_test, y_pred, ensemble_probs)

print(f"\n{'='*60}")
print(f"  GRU Classifier（12 features，Walmart TL，ensemble）")
print(f"{'='*60}")
print(f"  [Layer 1 — 分類層]")
print(f"  Accuracy={global_metrics['accuracy']:.4f}  AUC-ROC={global_metrics['auc_roc']}")
print(f"  Precision={global_metrics['precision']:.4f}  Recall={global_metrics['recall']:.4f}  F1={global_metrics['f1']:.4f}")
print(f"\n  [Layer 2 — 決策層]")
print(f"  FNR={global_metrics['fnr']:.4f}  FPR={global_metrics['fpr']:.4f}  Cost={global_metrics['expected_cost']:.4f}")
print(f"  TP={global_metrics['TP']}  TN={global_metrics['TN']}  FP={global_metrics['FP']}  FN={global_metrics['FN']}")

# ── Layer 3：Per-user ────────────────────────────────────────────────────────
per_user_results = {}
for uid in np.unique(test_user_ids):
    mask    = test_user_ids == uid
    yt      = y_test[mask]
    yp      = y_pred[mask]
    yprob   = ensemble_probs[mask]
    m       = clf_metrics(yt, yp, yprob)
    m["n_samples"]   = int(mask.sum())
    m["alert_ratio"] = float(yt.mean())
    per_user_results[uid] = m

user_f1s   = [v["f1"]             for v in per_user_results.values()]
user_fnrs  = [v["fnr"]            for v in per_user_results.values()]
user_costs = [v["expected_cost"]  for v in per_user_results.values()]
user_aucs  = [v["auc_roc"] for v in per_user_results.values() if v["auc_roc"] is not None]

print(f"\n  [Layer 3 — Per-user 公平性]")
print(f"  Per-user F1  ：avg={np.mean(user_f1s):.4f}  min={min(user_f1s):.4f}")
print(f"  Per-user FNR ：avg={np.mean(user_fnrs):.4f}  max={max(user_fnrs):.4f}")
print(f"  Per-user Cost：avg={np.mean(user_costs):.4f}  max={max(user_costs):.4f}")
if user_aucs:
    print(f"  Per-user AUC ：avg={np.mean(user_aucs):.4f}  min={min(user_aucs):.4f}")

print(f"\n  各 user 詳細：")
for uid, v in sorted(per_user_results.items()):
    auc_str = f"AUC={v['auc_roc']:.3f}" if v["auc_roc"] is not None else "AUC=N/A"
    print(f"    {uid:10s}  n={v['n_samples']:3d}  Alert={v['alert_ratio']*100:.0f}%  "
          f"F1={v['f1']:.3f}  Recall={v['recall']:.3f}  FNR={v['fnr']:.3f}  "
          f"Cost={v['expected_cost']:.3f}  {auc_str}")

# ── 儲存 ──────────────────────────────────────────────────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

output = {
    "model"                  : "GRU Classifier (12 features, Walmart TL, ensemble)",
    "seeds_used"             : SEEDS,
    "decision_threshold"     : DECISION_THRESHOLD,
    "cost_fn"                : COST_FALSE_NEGATIVE,
    "cost_fp"                : COST_FALSE_POSITIVE,
    "layer1_classification"  : {
        "accuracy"    : global_metrics["accuracy"],
        "precision"   : global_metrics["precision"],
        "recall"      : global_metrics["recall"],
        "f1"          : global_metrics["f1"],
        "auc_roc"     : global_metrics["auc_roc"],
    },
    "layer2_decision_global" : global_metrics,
    "layer3_per_user_summary": {
        "f1_mean"   : round(float(np.mean(user_f1s)), 4),
        "f1_min"    : round(float(min(user_f1s)), 4),
        "fnr_mean"  : round(float(np.mean(user_fnrs)), 4),
        "fnr_max"   : round(float(max(user_fnrs)), 4),
        "cost_mean" : round(float(np.mean(user_costs)), 4),
        "cost_max"  : round(float(max(user_costs)), 4),
        "worst_fnr_user": max(per_user_results, key=lambda u: per_user_results[u]["fnr"]),
        "worst_f1_user" : min(per_user_results, key=lambda u: per_user_results[u]["f1"]),
    },
    "per_user_detail"        : per_user_results,
}
outpath = f"{ARTIFACTS_DIR}/clf_evaluation.json"
with open(outpath, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, cls=NpEncoder)

print(f"\n✅ 儲存至 {outpath}")
print("🎉 完成！")
