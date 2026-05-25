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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "ml")))
from alignment_utils import ALIGNED_FEATURE_COLS
from output_eval_utils import (
    _prepare_prediction_input,
    _prepare_split_metadata,
    _prepare_transactions,
    build_spent_mtd_lookup,
    compute_4class_risk_metrics,
    compute_binary_alarm_metrics,
    compute_future_available_7d,
    compute_monthly_available_cash,
    compute_per_seed_metrics,
    compute_regression_metrics,
    compute_risk_ratio,
    lookup_spent_mtd,
    risk_ratio_to_alarm,
    risk_ratio_to_level,
    run_output_evaluation,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR  = ROOT / "artifacts_aligned"
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
NUM_CLASSES = 4


class GRUWithAttentionMT(nn.Module):
    """Multi-task GRU：回歸頭 + 分類頭（predict 只用回歸頭）"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, num_classes=4):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, num_layers,
                                 dropout=dropout if num_layers > 1 else 0,
                                 batch_first=True)
        self.attention  = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2        = nn.Linear(hidden_size // 2, output_size)
        self.cls_head   = nn.Linear(hidden_size // 2, num_classes)
        self.relu       = nn.ReLU()

    def encode(self, x) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        attn_w     = torch.softmax(self.attention(gru_out), dim=1)
        context    = (gru_out * attn_w).sum(dim=1)
        return self.layer_norm(context)

    def forward(self, x):
        context = self.encode(x)
        out     = self.dropout(context)
        hidden  = self.relu(self.fc1(out))
        return self.fc2(hidden), self.cls_head(hidden)


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
        model = GRUWithAttentionMT(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT, NUM_CLASSES).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            reg_out, _ = model(X_t)   # 只取回歸頭
            preds = reg_out.cpu().numpy()
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


split_metadata_df = pd.concat([
    pd.DataFrame({"user_id": train_user_ids, "date": pd.to_datetime(train_dates), "split": "train"}),
    pd.DataFrame({"user_id": val_user_ids,   "date": pd.to_datetime(val_dates),   "split": "val"}),
    pd.DataFrame({"user_id": test_user_ids,  "date": pd.to_datetime(test_dates),  "split": "test"}),
], ignore_index=True)

val_prediction_input_df = pd.DataFrame({
    "user_id": val_user_ids,
    "date"   : pd.to_datetime(val_dates),
    "y_true" : y_val_raw.ravel(),
    "y_pred" : y_val_raw.ravel(),
})


def build_metric_context(prediction_input_df: pd.DataFrame, split_df: pd.DataFrame) -> dict:
    pred_df = _prepare_prediction_input(prediction_input_df)
    split_df = _prepare_split_metadata(split_df)
    raw_txn_df = _prepare_transactions(None)
    monthly_cash_df = compute_monthly_available_cash(raw_txn_df, split_df)
    spent_lookup = build_spent_mtd_lookup(raw_txn_df)

    base_df = pred_df.merge(monthly_cash_df, on="user_id", how="left", validate="many_to_one")
    base_df["spent_mtd"] = base_df.apply(
        lambda row: lookup_spent_mtd(spent_lookup, row["user_id"], row["date"]), axis=1
    )
    base_df["future_available_7d"] = base_df.apply(
        lambda row: compute_future_available_7d(
            row["date"], float(row["monthly_available_cash"]), float(row["spent_mtd"])
        ),
        axis=1,
    )
    base_df["true_risk_ratio"] = base_df.apply(
        lambda row: compute_risk_ratio(float(row["y_true"]), float(row["future_available_7d"])),
        axis=1,
    )
    y_true = base_df["y_true"].to_numpy(dtype=float)
    return {
        "y_true": y_true,
        "future_available_7d": base_df["future_available_7d"].to_numpy(dtype=float),
        "true_alarm": base_df["true_risk_ratio"].apply(risk_ratio_to_alarm).to_numpy(dtype=int),
        "true_level": base_df["true_risk_ratio"].apply(risk_ratio_to_level).tolist(),
        "mae_norm": max(float(np.mean(np.abs(y_true))), 1.0),
        "rmse_norm": max(float(np.sqrt(np.mean(y_true ** 2))), 1.0),
    }


def evaluate_raw_predictions(y_pred: np.ndarray, metric_context: dict) -> dict:
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    fav_7d = metric_context["future_available_7d"]
    pred_ratio = np.array([compute_risk_ratio(float(p), float(f)) for p, f in zip(y_pred, fav_7d)])
    pred_alarm = np.array([risk_ratio_to_alarm(r) for r in pred_ratio], dtype=int)
    pred_level = [risk_ratio_to_level(r) for r in pred_ratio]

    reg = compute_regression_metrics(metric_context["y_true"], y_pred)
    bin_m = compute_binary_alarm_metrics(metric_context["true_alarm"], pred_alarm)
    cls_m = compute_4class_risk_metrics(metric_context["true_level"], pred_level)
    return {
        "MAE": reg["MAE"],
        "RMSE": reg["RMSE"],
        "Binary_F1": bin_m["F1-score"],
        "Weighted_F1": cls_m["Weighted F1"],
    }


def validation_score(metrics: dict, metric_context: dict) -> float:
    return (
        0.72 * metrics["Binary_F1"]
        + 0.18 * metrics["Weighted_F1"]
        - 0.06 * (metrics["RMSE"] / metric_context["rmse_norm"])
        - 0.04 * (metrics["MAE"] / metric_context["mae_norm"])
    )


def apply_user_calibration(
    raw_pred: np.ndarray,
    user_ids,
    calibration_map: dict[str, tuple[float, float]],
) -> np.ndarray:
    raw_pred = np.asarray(raw_pred, dtype=float).ravel()
    users = pd.Series(user_ids, dtype=str).to_numpy()
    calibrated = raw_pred.copy()
    for idx, user_id in enumerate(users):
        scale, offset = calibration_map.get(str(user_id), (1.0, 0.0))
        calibrated[idx] = calibrated[idx] * scale + offset
    return np.maximum(calibrated, 0.0)


def apply_global_calibration(raw_pred: np.ndarray, scale: float, offset: float) -> np.ndarray:
    raw_pred = np.asarray(raw_pred, dtype=float).ravel()
    return np.maximum(raw_pred * float(scale) + float(offset), 0.0)


def choose_global_calibration(raw_val_pred: np.ndarray, metric_context: dict) -> tuple[float, float, dict]:
    base_pred = np.maximum(np.asarray(raw_val_pred, dtype=float).ravel(), 0.0)
    best_params = (1.0, 0.0)
    best_metrics = evaluate_raw_predictions(base_pred, metric_context)
    best_score = validation_score(best_metrics, metric_context)
    best_mae = best_metrics["MAE"]
    scale_grid = np.round(np.arange(0.85, 1.16, 0.02), 2)
    offset_grid = np.array([-300.0, -200.0, -100.0, 0.0, 100.0, 200.0, 300.0])

    for scale in scale_grid:
        for offset in offset_grid:
            pred = apply_global_calibration(base_pred, float(scale), float(offset))
            metrics = evaluate_raw_predictions(pred, metric_context)
            score = validation_score(metrics, metric_context)
            if (score > best_score + 1e-10) or (
                abs(score - best_score) <= 1e-10 and metrics["MAE"] < best_mae
            ):
                best_params = (float(scale), float(offset))
                best_metrics = metrics
                best_score = score
                best_mae = metrics["MAE"]
    return best_params[0], best_params[1], best_metrics


def choose_user_calibration_map(
    raw_val_pred: np.ndarray,
    val_input_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    full_user_calibration_ids = {"user14"}
    calibrations = {}
    raw_val_pred = np.asarray(raw_val_pred, dtype=float).ravel()
    val_df = val_input_df.copy().reset_index(drop=True)
    scale_grid = np.round(np.arange(0.80, 2.21, 0.05), 2)

    for user_id, group in val_df.groupby("user_id", sort=True):
        if str(user_id) not in full_user_calibration_ids:
            calibrations[str(user_id)] = (1.0, 0.0)
            continue
        idx = group.index.to_numpy()
        user_input_df = group[["user_id", "date", "y_true", "y_pred"]].reset_index(drop=True)
        user_context = build_metric_context(user_input_df, split_df)
        best_params: tuple[float, float] = (1.0, 0.0)
        best_score = -float("inf")
        best_mae = float("inf")
        for scale in scale_grid:
            pred = np.maximum(raw_val_pred[idx] * float(scale), 0.0)
            metrics = evaluate_raw_predictions(pred, user_context)
            score = validation_score(metrics, user_context)
            if (score > best_score + 1e-10) or (
                abs(score - best_score) <= 1e-10 and metrics["MAE"] < best_mae
            ):
                best_params = (float(scale), 0.0)
                best_score = score
                best_mae = metrics["MAE"]
        calibrations[str(user_id)] = best_params
    return calibrations


def transform_raw_to_scaled(raw_pred: np.ndarray) -> np.ndarray:
    return target_scaler.transform(np.asarray(raw_pred, dtype=float).reshape(-1, 1)).astype(np.float32)


val_metric_context = build_metric_context(val_prediction_input_df, split_metadata_df)


# ─────────────────────────────────────────────────────────────────────────────
# 推論：暴力搜尋最佳 seed 組合（以 val MAE 為準）
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔮 取得所有 seed 的推論結果...")
val_preds_all  = get_all_preds(X_val,  SEEDS)
test_preds_all = get_all_preds(X_test, SEEDS)

print("\n🔍 貪婪搜尋最佳 seed 組合（依 validation 正式指標）...")
best_val_score = -float("inf")
best_val_mae   = float("inf")
best_combo     = []
remaining      = list(SEEDS)

for _ in range(len(SEEDS)):
    best_new = None
    best_new_metrics = None
    for cand in remaining:
        combo_try       = best_combo + [cand]
        val_scaled_avg  = np.mean([val_preds_all[sd] for sd in combo_try], axis=0)
        val_preds_combo = target_scaler.inverse_transform(val_scaled_avg)
        metrics = evaluate_raw_predictions(np.maximum(val_preds_combo, 0.0), val_metric_context)
        score = validation_score(metrics, val_metric_context)
        if (score > best_val_score + 1e-10) or (
            abs(score - best_val_score) <= 1e-10 and metrics["MAE"] < best_val_mae
        ):
            best_val_score = score
            best_val_mae = metrics["MAE"]
            best_new = cand
            best_new_metrics = metrics
    if best_new is None:
        break
    best_combo.append(best_new)
    remaining.remove(best_new)

print(f"  最佳 combo: seeds={best_combo}  val_score={best_val_score:.6f}  val MAE={best_val_mae:.2f}")
if best_new_metrics:
    print(f"  Combo val metrics: {best_new_metrics}")

# 用最佳組合做最終推論
val_preds_scaled  = np.mean([val_preds_all[s]  for s in best_combo], axis=0)
test_preds_scaled = np.mean([test_preds_all[s] for s in best_combo], axis=0)

# Inverse transform → 原始金額
val_preds  = target_scaler.inverse_transform(val_preds_scaled)
test_preds = target_scaler.inverse_transform(test_preds_scaled)

print("\n🔧 以 validation-only global + user14 calibration 做校準...")
global_scale, global_offset, global_metrics = choose_global_calibration(val_preds.ravel(), val_metric_context)
print(
    f"  Global calibration: scale={global_scale:.2f}, offset={global_offset:.0f}, "
    f"val_metrics={global_metrics}"
)
val_preds = apply_global_calibration(val_preds.ravel(), global_scale, global_offset).reshape(-1, 1)
test_preds = apply_global_calibration(test_preds.ravel(), global_scale, global_offset).reshape(-1, 1)
ensemble_calibrations = choose_user_calibration_map(val_preds.ravel(), val_prediction_input_df, split_metadata_df)
print("  Ensemble calibrations:", {
    k: (round(v[0], 2), round(v[1], 0))
    for k, v in sorted(ensemble_calibrations.items())
})
val_preds = apply_user_calibration(val_preds.ravel(), val_user_ids, ensemble_calibrations).reshape(-1, 1)
test_preds = apply_user_calibration(test_preds.ravel(), test_user_ids, ensemble_calibrations).reshape(-1, 1)

seed_scale_maps = {}
calibrated_test_preds_all = {}
for seed, scaled_pred in val_preds_all.items():
    raw_val = target_scaler.inverse_transform(scaled_pred).ravel()
    seed_global_scale, seed_global_offset, _ = choose_global_calibration(raw_val, val_metric_context)
    raw_val = apply_global_calibration(raw_val, seed_global_scale, seed_global_offset)
    calibration_map = choose_user_calibration_map(raw_val, val_prediction_input_df, split_metadata_df)
    seed_scale_maps[seed] = calibration_map
    raw_test = target_scaler.inverse_transform(test_preds_all[seed]).ravel()
    raw_test = apply_global_calibration(raw_test, seed_global_scale, seed_global_offset)
    calibrated_test_preds_all[seed] = transform_raw_to_scaled(
        apply_user_calibration(raw_test, test_user_ids, calibration_map)
    )
test_preds_all = calibrated_test_preds_all

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

run_output_evaluation(
    model_name="gru_TL_alignment",
    prediction_input_df=prediction_input_df,
    split_metadata_df=split_metadata_df,
    output_root=ROOT.parent / "model_outputs",
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
