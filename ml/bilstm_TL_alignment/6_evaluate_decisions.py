"""
Step 6：決策層評估
===================
超越 MAE，從「能不能支持正確決策」來評估模型

三層評估：
  Layer 1 - 回歸層  ：MAE, RMSE, sMAPE（保留基本預測能力）
  Layer 2 - 決策層  ：Precision, Recall, F1, FNR, FPR（預警準確度）
  Layer 3 - 成本層  ：Expected Decision Cost（漏報 vs 誤報的不對稱成本）

預警定義：future_7d_sum > 個人近30天均值 × threshold_ratio
（相對門檻，對所有 user 公平）

個體公平性：每位 user 各算一份，報告平均/最差/標準差
"""

import numpy as np
import torch
import torch.nn as nn
import pickle, os, json, sys
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS, load_personal_daily, INPUT_DAYS

ARTIFACTS_DIR = "artifacts_bilstm_v2"

# ── 預警門檻設定 ───────────────────────────────────────────────────────────────
# 「未來7天花費 > 個人近30日均值 × 1.5」才算高消費預警
ALERT_RATIO = 1.5

# ── 決策成本設定（可調整）─────────────────────────────────────────────────────
# 漏報（該提醒沒提醒）的成本 vs 誤報（不該提醒卻提醒）的成本
# 漏報通常比誤報嚴重：錯過高消費預警 → 使用者超支
COST_FALSE_NEGATIVE = 3.0   # 漏報 1 次的成本
COST_FALSE_POSITIVE = 1.0   # 誤報 1 次的成本

# ── 最佳 combo（從 5_predict_bilstm.py 的結果）────────────────────────────────
BEST_COMBO = [789]   # ← 如果跑完 5_predict 有不同結果請更新這裡

# ── 設備 ──────────────────────────────────────────────────────────────────────
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
OUTPUT_SIZE = 1


class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        bi_hidden = hidden_size * 2
        self.attention  = nn.Linear(bi_hidden, 1)
        self.layer_norm = nn.LayerNorm(bi_hidden)
        self.dropout    = nn.Dropout(dropout)
        self.fc1 = nn.Linear(bi_hidden, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_w  = torch.softmax(self.attention(out), dim=1)
        context = (out * attn_w).sum(dim=1)
        context = self.layer_norm(context)
        out     = self.dropout(context)
        out     = self.relu(self.fc1(out))
        return self.fc2(out)


# ── 載入資料 ──────────────────────────────────────────────────────────────────
print("📂 載入資料...")
X_test        = np.load(f"{ARTIFACTS_DIR}/personal_X_test.npy")
y_test_raw    = np.load(f"{ARTIFACTS_DIR}/personal_y_test_raw.npy").ravel()
test_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_test_user_ids.npy")

with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

# ── 計算個人近30日均值（用來定義相對門檻）────────────────────────────────────
print("📊 計算每個樣本對應的個人30日均值（用於相對預警門檻）...")
personal_daily = load_personal_daily()

# 為每個 test 樣本計算對應的 rolling 30d mean
# 做法：為每位 user 重建 daily rolling mean，再 index 到 test 時間點
user_roll30_means = {}
for uid in personal_daily["user_id"].unique():
    u = personal_daily[personal_daily["user_id"] == uid].sort_values("date").reset_index(drop=True)
    u["roll30_mean"] = u["daily_expense"].rolling(30, min_periods=1).mean()
    # 未來 7 天加總的「個人基線」= roll30_mean × 7
    u["baseline_7d"] = u["roll30_mean"] * 7
    user_roll30_means[uid] = u["baseline_7d"].values

# 對每個 test 樣本，找對應 user 的 baseline
# 先重建 test 的時間 index
baseline_7d_list = []
for uid in personal_daily["user_id"].unique():
    u = personal_daily[personal_daily["user_id"] == uid].sort_values("date").reset_index(drop=True)
    u["roll30_mean"] = u["daily_expense"].rolling(30, min_periods=1).mean()
    u["baseline_7d"] = u["roll30_mean"] * 7
    feat_len = len(u) - 7  # 扣掉 target NaN
    if feat_len <= INPUT_DAYS:
        continue
    windows_baseline = []
    for t in range(INPUT_DAYS, feat_len):
        windows_baseline.append(u["baseline_7d"].iloc[t])
    n = len(windows_baseline)
    v_end = int(n * 0.85)
    baseline_7d_list.extend(windows_baseline[v_end:])

baseline_7d = np.array(baseline_7d_list[:len(y_test_raw)], dtype=np.float32)

# 預警門檻（每個樣本各自的）
alert_threshold = baseline_7d * ALERT_RATIO
# 真實標籤：實際是否超出門檻
y_true_alert = (y_test_raw > alert_threshold).astype(int)
print(f"  預警正例比例：{y_true_alert.mean()*100:.1f}%  "
      f"（{y_true_alert.sum()} / {len(y_true_alert)} 筆）")

# ── 推論 ──────────────────────────────────────────────────────────────────────
print(f"\n🔮 推論（best combo={BEST_COMBO}）...")
preds_list = []
for seed in BEST_COMBO:
    ckpt  = torch.load(f"{ARTIFACTS_DIR}/finetune_bilstm_seed{seed}.pth", map_location=device)
    model = BiLSTMWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()
    preds_list.append(preds)

test_preds = target_scaler.inverse_transform(np.mean(preds_list, axis=0)).ravel()

# 預測預警：預測值是否超出門檻
y_pred_alert = (test_preds > alert_threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1：回歸層
# ─────────────────────────────────────────────────────────────────────────────
mae   = float(np.mean(np.abs(y_test_raw - test_preds)))
rmse  = float(np.sqrt(np.mean((y_test_raw - test_preds) ** 2)))
denom = (np.abs(y_test_raw) + np.abs(test_preds)) / 2 + 1e-8
smape = float(np.mean(np.abs(y_test_raw - test_preds) / denom) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2：決策層（全體）
# ─────────────────────────────────────────────────────────────────────────────
def decision_metrics(y_true, y_pred):
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    fnr       = FN / (TP + FN + 1e-8)   # 漏報率
    fpr       = FP / (FP + TN + 1e-8)   # 誤報率

    # 決策成本
    expected_cost = (FN * COST_FALSE_NEGATIVE + FP * COST_FALSE_POSITIVE) / len(y_true)

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "precision"      : round(precision,     4),
        "recall"         : round(recall,         4),
        "f1"             : round(f1,             4),
        "fnr"            : round(fnr,            4),
        "fpr"            : round(fpr,            4),
        "expected_cost"  : round(expected_cost,  4),
    }

global_decision = decision_metrics(y_true_alert, y_pred_alert)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3：個體公平性（per-user）
# ─────────────────────────────────────────────────────────────────────────────
print("\n👤 計算 per-user 指標...")
per_user_results = {}

for uid in np.unique(test_user_ids):
    mask = test_user_ids == uid
    yt   = y_test_raw[mask]
    yp   = test_preds[mask]
    ya   = y_true_alert[mask]
    ypa  = y_pred_alert[mask]
    base = baseline_7d[mask]

    user_mae   = float(np.mean(np.abs(yt - yp)))
    user_nmae  = user_mae / (np.mean(np.abs(yt)) + 1e-8)
    user_dec   = decision_metrics(ya, ypa)

    per_user_results[uid] = {
        "n_samples"   : int(mask.sum()),
        "alert_ratio" : float(ya.mean()),
        "mae"         : round(user_mae, 2),
        "nmae"        : round(user_nmae, 4),
        **user_dec,
    }

# 統計：平均、最差、標準差
user_maes  = [v["mae"]           for v in per_user_results.values()]
user_f1s   = [v["f1"]            for v in per_user_results.values()]
user_fnrs  = [v["fnr"]           for v in per_user_results.values()]
user_costs = [v["expected_cost"] for v in per_user_results.values()]

worst_mae_user  = max(per_user_results, key=lambda u: per_user_results[u]["mae"])
worst_fnr_user  = max(per_user_results, key=lambda u: per_user_results[u]["fnr"])


# ─────────────────────────────────────────────────────────────────────────────
# 列印結果
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  Layer 1：回歸層")
print(f"{'='*62}")
print(f"  MAE   : {mae:.2f}   （同學基準：743.51）")
print(f"  RMSE  : {rmse:.2f}")
print(f"  sMAPE : {smape:.2f}%")

print(f"\n{'='*62}")
print(f"  Layer 2：決策層（預警門檻 = 個人30日均值 × {ALERT_RATIO}）")
print(f"{'='*62}")
print(f"  Precision（準確率） : {global_decision['precision']:.4f}  "
      f"（預測有高消費，真的有的比例）")
print(f"  Recall   （召回率） : {global_decision['recall']:.4f}  "
      f"（真正高消費，有被提醒的比例）")
print(f"  F1 Score            : {global_decision['f1']:.4f}")
print(f"  FNR（漏報率）       : {global_decision['fnr']:.4f}  "
      f"（該提醒沒提醒的比例，越低越好）")
print(f"  FPR（誤報率）       : {global_decision['fpr']:.4f}  "
      f"（不該提醒卻提醒的比例）")
print(f"  TP={global_decision['TP']}  TN={global_decision['TN']}  "
      f"FP={global_decision['FP']}  FN={global_decision['FN']}")

print(f"\n{'='*62}")
print(f"  Layer 3：決策成本（漏報成本={COST_FALSE_NEGATIVE}×，誤報成本={COST_FALSE_POSITIVE}×）")
print(f"{'='*62}")
print(f"  Expected Cost / 樣本 : {global_decision['expected_cost']:.4f}")

print(f"\n{'='*62}")
print(f"  個體公平性（Per-User）")
print(f"{'='*62}")
print(f"  MAE    - 平均: {np.mean(user_maes):.2f}  最差: {max(user_maes):.2f}  "
      f"Std: {np.std(user_maes):.2f}  （最差: {worst_mae_user}）")
print(f"  F1     - 平均: {np.mean(user_f1s):.4f}  最差: {min(user_f1s):.4f}  "
      f"Std: {np.std(user_f1s):.4f}")
print(f"  FNR    - 平均: {np.mean(user_fnrs):.4f}  最差: {max(user_fnrs):.4f}  "
      f"（最差: {worst_fnr_user}）")
print(f"  Cost   - 平均: {np.mean(user_costs):.4f}  最差: {max(user_costs):.4f}  "
      f"Std: {np.std(user_costs):.4f}")

print(f"\n  各 user 詳細：")
for uid, v in sorted(per_user_results.items()):
    print(f"    {uid:10s}  MAE={v['mae']:7.2f}  F1={v['f1']:.3f}  "
          f"Recall={v['recall']:.3f}  FNR={v['fnr']:.3f}  Cost={v['expected_cost']:.3f}  "
          f"(n={v['n_samples']}, alert={v['alert_ratio']*100:.0f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 儲存
# ─────────────────────────────────────────────────────────────────────────────
output = {
    "alert_threshold_ratio"   : ALERT_RATIO,
    "cost_false_negative"     : COST_FALSE_NEGATIVE,
    "cost_false_positive"     : COST_FALSE_POSITIVE,
    "best_combo"              : BEST_COMBO,
    "layer1_regression": {
        "mae"  : round(mae,   2),
        "rmse" : round(rmse,  2),
        "smape": round(smape, 2),
    },
    "layer2_decision_global"  : global_decision,
    "layer3_per_user_summary" : {
        "mae_mean" : round(float(np.mean(user_maes)),  2),
        "mae_max"  : round(float(max(user_maes)),      2),
        "mae_std"  : round(float(np.std(user_maes)),   2),
        "f1_mean"  : round(float(np.mean(user_f1s)),   4),
        "f1_min"   : round(float(min(user_f1s)),       4),
        "fnr_mean" : round(float(np.mean(user_fnrs)),  4),
        "fnr_max"  : round(float(max(user_fnrs)),      4),
        "cost_mean": round(float(np.mean(user_costs)), 4),
        "cost_max" : round(float(max(user_costs)),     4),
        "worst_mae_user": worst_mae_user,
        "worst_fnr_user": worst_fnr_user,
    },
    "per_user_detail"         : per_user_results,
}

class NpEncoder(json.JSONEncoder):
    """讓 numpy int/float 可以被 json.dump 序列化"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(f"{ARTIFACTS_DIR}/decision_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, cls=NpEncoder)

print(f"\n✅ 結果儲存至 {ARTIFACTS_DIR}/decision_evaluation.json")
print(f"\n🎉 完成！")
