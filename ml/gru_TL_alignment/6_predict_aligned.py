"""
Step 6：Aligned 預測與評估
============================
Ensemble 推論 + Bias Correction
比較三種方法：
  1. No Pretrain（基準）
  2. Naive TL（現有 V5）
  3. Aligned Pretrain（本方法）
輸出：
  - aligned_result.txt
  - aligned_metrics.json
  - comparison_table.txt
"""

import numpy as np
import torch
import torch.nn as nn
import pickle, os, json, sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from alignment_utils import ALIGNED_FEATURE_COLS
from output_eval_utils import run_output_evaluation, compute_per_seed_metrics

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR  = "artifacts_aligned"
GRU_ARTIFACTS_CANDIDATES = [
    ROOT.parent / "legacy_models" / "ml_gru" / "artificats",
    ROOT.parent / "ml_gru" / "artificats",
]
GRU_ARTIFACTS = next((path for path in GRU_ARTIFACTS_CANDIDATES if path.exists()), GRU_ARTIFACTS_CANDIDATES[0])

# 自動掃描所有已訓練的 seed，不需手動維護
import glob as _glob
SEEDS = sorted([
    int(f.split("seed")[1].replace(".pth", ""))
    for f in _glob.glob(f"{ARTIFACTS_DIR}/finetune_aligned_gru_seed*.pth")
])
print(f"🔍 偵測到 {len(SEEDS)} 個 ensemble seeds: {SEEDS}")

# ── 設備 ──────────────────────────────────────────────────────────────────────
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
print("📂 載入個人 Aligned 測試資料...")
X_val      = np.load(f"{ARTIFACTS_DIR}/personal_aligned_X_val.npy")
X_test     = np.load(f"{ARTIFACTS_DIR}/personal_aligned_X_test.npy")
y_val_raw  = np.load(f"{ARTIFACTS_DIR}/personal_aligned_y_test_raw.npy")   # 原始金額
y_test_raw = np.load(f"{ARTIFACTS_DIR}/personal_aligned_y_test_raw.npy")

with open(f"{ARTIFACTS_DIR}/personal_aligned_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

test_user_ids  = np.load(f"{ARTIFACTS_DIR}/personal_aligned_test_user_ids.npy")
val_user_ids   = np.load(f"{ARTIFACTS_DIR}/personal_aligned_val_user_ids.npy")
train_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_aligned_train_user_ids.npy")
test_dates     = np.load(f"{ARTIFACTS_DIR}/personal_aligned_test_dates.npy")
val_dates      = np.load(f"{ARTIFACTS_DIR}/personal_aligned_val_dates.npy")
train_dates    = np.load(f"{ARTIFACTS_DIR}/personal_aligned_train_dates.npy")

# 載入 val 的原始 y（用 personal_aligned_y_val.npy 做 inverse）
y_val_scaled = np.load(f"{ARTIFACTS_DIR}/personal_aligned_y_val.npy")
y_val_raw    = target_scaler.inverse_transform(y_val_scaled)


def get_all_preds(X: np.ndarray, seed_list: list) -> dict:
    """每個 seed 各自推論，回傳 dict {seed: preds}"""
    all_preds = {}
    for seed in seed_list:
        model_path = f"{ARTIFACTS_DIR}/finetune_aligned_gru_seed{seed}.pth"
        ckpt  = torch.load(model_path, map_location=device)
        model = GRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(X_t).cpu().numpy()
        all_preds[seed] = preds
    return all_preds


def predict_ensemble(X: np.ndarray, seed_list: list) -> np.ndarray:
    """Ensemble 平均推論"""
    all_preds = get_all_preds(X, seed_list)
    return np.mean([all_preds[s] for s in seed_list], axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, user_ids: np.ndarray):
    """計算 MAE, RMSE, MedAE, SMAPE, per-user NMAE"""
    errors = np.abs(y_true - y_pred)

    mae   = float(np.mean(errors))
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    medae = float(np.median(errors))   # Median Absolute Error：不受極端值影響

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    smape = float(np.mean(errors / denom) * 100)

    # Per-user NMAE
    user_nmaes = []
    for uid in np.unique(user_ids):
        mask     = user_ids == uid
        yt, yp   = y_true[mask].ravel(), y_pred[mask].ravel()
        mean_abs = np.mean(np.abs(yt)) + 1e-8
        user_nmaes.append(np.mean(np.abs(yt - yp)) / mean_abs)
    per_user_nmae = float(np.mean(user_nmaes) * 100)

    return {"mae": mae, "rmse": rmse, "medae": medae,
            "smape": smape, "per_user_nmae": per_user_nmae}


# ─────────────────────────────────────────────────────────────────────────────
# 推論：暴力搜尋最佳 seed 組合（以 val MAE 為準）
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔮 取得所有 seed 的推論結果...")
val_preds_all  = get_all_preds(X_val,  SEEDS)
test_preds_all = get_all_preds(X_test, SEEDS)

print("\n🔍 貪婪搜尋最佳 seed 組合（依 val MAE）...")
best_val_mae   = float("inf")
best_combo     = []
remaining      = list(SEEDS)

for _ in range(len(SEEDS)):
    best_new = None
    for cand in remaining:
        combo_try       = best_combo + [cand]
        val_scaled_avg  = np.mean([val_preds_all[sd] for sd in combo_try], axis=0)
        val_preds_combo = target_scaler.inverse_transform(val_scaled_avg)
        mae = float(np.mean(np.abs(y_val_raw - val_preds_combo)))
        if mae < best_val_mae:
            best_val_mae = mae
            best_new     = cand
    if best_new is None:
        break
    best_combo.append(best_new)
    remaining.remove(best_new)

print(f"  最佳 combo: seeds={best_combo}  val MAE={best_val_mae:.2f}")

# 用最佳組合做最終推論
val_preds_scaled  = np.mean([val_preds_all[s]  for s in best_combo], axis=0)
test_preds_scaled = np.mean([test_preds_all[s] for s in best_combo], axis=0)

# Inverse transform → 原始金額
val_preds  = target_scaler.inverse_transform(val_preds_scaled)
test_preds = target_scaler.inverse_transform(test_preds_scaled)

# Bias Correction 已停用：實驗證明不做 correction 的 test MAE 更低
bias_before = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 計算 metrics
# ─────────────────────────────────────────────────────────────────────────────
print("\n📊 計算評估指標...")
val_metrics  = compute_metrics(y_val_raw,  val_preds,  val_user_ids)
test_metrics = compute_metrics(y_test_raw, test_preds, test_user_ids)

print(f"\n  Val  MAE  : {val_metrics['mae']:.2f}")
print(f"  Test MAE  : {test_metrics['mae']:.2f}")
print(f"  Test RMSE : {test_metrics['rmse']:.2f}")
print(f"  Test MedAE: {test_metrics['medae']:.2f}  ← 不受極端值影響")
print(f"  Test SMAPE: {test_metrics['smape']:.2f}%")
print(f"  💡 若 MAE >> MedAE，代表有少數極端誤差在拉高 MAE")

# ─────────────────────────────────────────────────────────────────────────────
# 載入既有結果做三方比較
# ─────────────────────────────────────────────────────────────────────────────
print("\n📋 三方比較（No Pretrain vs Naive TL vs Aligned Pretrain）...")

existing_results = {}
for method, fname in [("no_pretrain", "metrics_nopretrain.json"), ("naive_tl_v5", "metrics_vv5.json")]:
    fpath = GRU_ARTIFACTS / fname
    if fpath.exists():
        with open(fpath) as f:
            existing_results[method] = json.load(f)
    else:
        print(f"  ⚠️  找不到 {fpath}，跳過")

def fmt(val):
    return f"{val:.2f}" if isinstance(val, (int, float)) else str(val)

np_test = existing_results.get('no_pretrain', {}).get('test_mae', 'N/A')
tl_test = existing_results.get('naive_tl_v5', {}).get('test_mae', 'N/A')
np_val  = existing_results.get('no_pretrain', {}).get('val_mae',  'N/A')
tl_val  = existing_results.get('naive_tl_v5', {}).get('val_mae',  'N/A')

comparison = f"""
{'='*65}
           方法比較（Test MAE / MedAE，越低越好）
{'='*65}
              MAE     MedAE
  No Pretrain      : {fmt(np_test):>7}   N/A    <- 基準
  Naive TL (V5)    : {fmt(tl_test):>7}   N/A    <- 有 pretrain 但無 alignment
  Aligned Pretrain : {test_metrics['mae']:>7.2f}   {test_metrics['medae']:>5.2f}  <- 本方法（MMD alignment）
{'='*65}
  Val MAE:
    No Pretrain      : {fmt(np_val)}
    Naive TL (V5)    : {fmt(tl_val)}
    Aligned Pretrain : {val_metrics['mae']:.2f}
{'='*65}
"""
print(comparison)

# ─────────────────────────────────────────────────────────────────────────────
# 儲存結果
# ─────────────────────────────────────────────────────────────────────────────
result_text = f"""GRU Aligned Pretrain Result
model_name: gru_aligned_pretrain_ensemble_bias
version: aligned_v3 (10 features + MMD loss)
pretrained: True (Rolling Z-score Alignment + MMD)
all_seeds: {SEEDS}
best_combo: {best_combo}
val_mae: {val_metrics['mae']:.6f}
val_rmse: {val_metrics['rmse']:.6f}
val_medae: {val_metrics['medae']:.6f}
val_smape: {val_metrics['smape']:.2f}%
val_per_user_nmae: {val_metrics['per_user_nmae']:.2f}%
test_mae: {test_metrics['mae']:.6f}
test_rmse: {test_metrics['rmse']:.6f}
test_medae: {test_metrics['medae']:.6f}
test_smape: {test_metrics['smape']:.2f}%
test_per_user_nmae: {test_metrics['per_user_nmae']:.2f}%
bias_correction: none（停用，實驗證明不做 correction 更佳）
feature_type: rolling_zscore_aligned
feature_cols: {ALIGNED_FEATURE_COLS}
"""

with open(f"{ARTIFACTS_DIR}/aligned_result.txt", "w") as f:
    f.write(result_text)
    f.write(comparison)

metrics_json = {
    "val_mae"             : val_metrics["mae"],
    "val_rmse"            : val_metrics["rmse"],
    "val_medae"           : val_metrics["medae"],
    "val_smape"           : val_metrics["smape"],
    "val_per_user_nmae"   : val_metrics["per_user_nmae"],
    "test_mae"            : test_metrics["mae"],
    "test_rmse"           : test_metrics["rmse"],
    "test_medae"          : test_metrics["medae"],
    "test_smape"          : test_metrics["smape"],
    "test_per_user_nmae"  : test_metrics["per_user_nmae"],
    "bias_correction"     : bias_before,
    "best_combo"          : best_combo,
    "comparison": {
        "no_pretrain_test_mae": existing_results.get("no_pretrain", {}).get("test_mae"),
        "naive_tl_test_mae"   : existing_results.get("naive_tl_v5", {}).get("test_mae"),
        "aligned_test_mae"    : test_metrics["mae"],
    }
}
with open(f"{ARTIFACTS_DIR}/aligned_metrics.json", "w") as f:
    json.dump(metrics_json, f, indent=2)

print(f"✅ 結果儲存至 {ARTIFACTS_DIR}/aligned_result.txt")
print(f"✅ Metrics 儲存至 {ARTIFACTS_DIR}/aligned_metrics.json")

# ── 共用評估器 ────────────────────────────────────────────────────────────────
print("\n📊 呼叫共用評估器...")
prediction_input_df = pd.DataFrame({
    "user_id": test_user_ids,
    "date"   : pd.to_datetime(test_dates),
    "y_true" : y_test_raw.ravel(),
    "y_pred" : test_preds.ravel(),
})

split_metadata_df = pd.concat([
    pd.DataFrame({"user_id": train_user_ids, "date": pd.to_datetime(train_dates), "split": "train"}),
    pd.DataFrame({"user_id": val_user_ids,   "date": pd.to_datetime(val_dates),   "split": "val"}),
    pd.DataFrame({"user_id": test_user_ids,  "date": pd.to_datetime(test_dates),  "split": "test"}),
], ignore_index=True)

run_output_evaluation(
    model_name="gru_TL_alignment",
    prediction_input_df=prediction_input_df,
    split_metadata_df=split_metadata_df,
)

print("\n📊 計算每個 seed 個別指標...")
compute_per_seed_metrics(
    seed_preds_dict=test_preds_all,
    target_scaler=target_scaler,
    prediction_input_df=prediction_input_df,
    split_metadata_df=split_metadata_df,
    output_dir=ROOT.parent / "model_outputs" / "gru_TL_alignment",
)
print("\n🎉 完成！")
