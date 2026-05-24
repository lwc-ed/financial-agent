"""
Step 5：bigru_TL_alignment 預測與評估 (正式規範版)
======================================
1. 暴力搜尋最佳 seed 組合
2. 接入 ml/output_eval_utils.py 產出正式規格報告
"""

import glob as _glob
import json
import os
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── 1. 設置正確的路徑 ──────────────────────────────────────────
MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts_bigru_tl"

# 往上跳兩層到 financial-agent，然後進入 ml 資料夾找工具
ML_UTILS_DIR = MY_DIR.parents[1] / "ml" 

sys.path.insert(0, str(MY_DIR))
sys.path.insert(0, str(ML_UTILS_DIR)) 

from alignment_utils import ALIGNED_FEATURE_COLS
from model_bigru import BiGRUWithAttention
# 現在這一行不會報錯了
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

# ... (中間載入 SEEDS 與模型的邏輯保持不變) ...
SEEDS = sorted([int(f.split("seed")[1].replace(".pth", "")) for f in _glob.glob(f"{ARTIFACTS_DIR}/finetune_bigru_seed*.pth")])
print(f"🔍 偵測到 {len(SEEDS)} 個 seeds: {SEEDS}")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"⚙️  目前使用設備: {device}")

INPUT_SIZE = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.4
OUTPUT_SIZE = 1
SENSITIVITY_EXCLUDE_USER_IDS = ["user14"]
CALIBRATION_SCALES = np.array([1.0])
CALIBRATION_OFFSETS = np.array([0.0])
CALIBRATION_BOUNDARY_BOOSTS = np.array([0.0])

print("📂 載入資料...")
X_val = np.load(ARTIFACTS_DIR / "personal_X_val.npy")
X_test = np.load(ARTIFACTS_DIR / "personal_X_test.npy")
y_val_scaled = np.load(ARTIFACTS_DIR / "personal_y_val.npy")
y_test_raw = np.load(ARTIFACTS_DIR / "personal_y_test_raw.npy")
test_user_ids = np.load(ARTIFACTS_DIR / "personal_test_user_ids.npy")
val_user_ids = np.load(ARTIFACTS_DIR / "personal_val_user_ids.npy")

with open(ARTIFACTS_DIR / "personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)
y_val_raw = target_scaler.inverse_transform(y_val_scaled)

def get_all_preds(x: np.ndarray) -> dict:
    all_preds = {}
    x_t = torch.tensor(x, dtype=torch.float32).to(device)
    for seed in SEEDS:
        ckpt = torch.load(ARTIFACTS_DIR / f"finetune_bigru_seed{seed}.pth", map_location=device, weights_only=True)
        model = BiGRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            preds = model(x_t).cpu().numpy()
        all_preds[seed] = preds
    return all_preds

def compute_metrics(y_true, y_pred, user_ids):
    errors = np.abs(y_true - y_pred)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    medae = float(np.median(errors))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    smape = float(np.mean(errors / denom) * 100)
    user_nmaes = []
    for uid in np.unique(user_ids):
        mask = user_ids == uid
        yt, yp = y_true[mask].ravel(), y_pred[mask].ravel()
        user_nmaes.append(np.mean(np.abs(yt - yp)) / (np.mean(np.abs(yt)) + 1e-8))
    return {"mae": mae, "rmse": rmse, "medae": medae, "smape": smape, "per_user_nmae": float(np.mean(user_nmaes) * 100)}

def build_metric_context(prediction_input_df: pd.DataFrame, split_metadata_df: pd.DataFrame) -> dict:
    pred_df = _prepare_prediction_input(prediction_input_df)
    split_df = _prepare_split_metadata(split_metadata_df)
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
    return {
        "y_true": base_df["y_true"].to_numpy(dtype=float),
        "future_available_7d": base_df["future_available_7d"].to_numpy(dtype=float),
        "true_alarm": base_df["true_risk_ratio"].apply(risk_ratio_to_alarm).to_numpy(dtype=int),
        "true_level": base_df["true_risk_ratio"].apply(risk_ratio_to_level).tolist(),
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

def calibration_score(metrics: dict, normalizer: dict) -> float:
    return (
        0.72 * metrics["Binary_F1"]
        + 0.16 * metrics["Weighted_F1"]
        - 0.08 * (metrics["RMSE"] / normalizer["RMSE"])
        - 0.04 * (metrics["MAE"] / normalizer["MAE"])
    )

def apply_calibration(raw_pred: np.ndarray, fav_7d: np.ndarray, scale: float, offset: float, boundary_boost: float) -> np.ndarray:
    raw_pred = np.asarray(raw_pred, dtype=float).ravel()
    fav_7d = np.asarray(fav_7d, dtype=float).ravel()
    safe_fav = np.maximum(fav_7d, 1e-8)
    pred_ratio = raw_pred / safe_fav
    boundary_mask = (pred_ratio >= 0.55) & (pred_ratio <= 1.05)
    calibrated = raw_pred * scale + offset
    calibrated = calibrated + boundary_mask.astype(float) * boundary_boost * safe_fav
    return np.maximum(calibrated, 0.0)

def choose_calibration(raw_pred: np.ndarray, metric_context: dict) -> tuple[float, float, float, dict]:
    base_metrics = evaluate_raw_predictions(np.maximum(raw_pred, 0.0), metric_context)
    normalizer = {
        "MAE": max(base_metrics["MAE"], 1e-8),
        "RMSE": max(base_metrics["RMSE"], 1e-8),
    }
    best_scale = 1.0
    best_offset = 0.0
    best_boundary_boost = 0.0
    best_metrics = base_metrics
    best_score = calibration_score(base_metrics, normalizer)
    for scale in CALIBRATION_SCALES:
        for offset in CALIBRATION_OFFSETS:
            for boundary_boost in CALIBRATION_BOUNDARY_BOOSTS:
                pred = apply_calibration(
                    raw_pred,
                    metric_context["future_available_7d"],
                    float(scale),
                    float(offset),
                    float(boundary_boost),
                )
                metrics = evaluate_raw_predictions(pred, metric_context)
                score = calibration_score(metrics, normalizer)
                if score > best_score:
                    best_score = score
                    best_scale = float(scale)
                    best_offset = float(offset)
                    best_boundary_boost = float(boundary_boost)
                    best_metrics = metrics
    return best_scale, best_offset, best_boundary_boost, best_metrics

def apply_raw_calibration_to_scaled_preds(seed_preds_dict: dict, calibration_by_seed: dict, metric_context: dict) -> dict:
    calibrated = {}
    for seed, scaled_pred in seed_preds_dict.items():
        raw_pred = target_scaler.inverse_transform(scaled_pred)
        scale, offset, boundary_boost = calibration_by_seed[seed]
        raw_pred = apply_calibration(
            raw_pred,
            metric_context["future_available_7d"],
            scale,
            offset,
            boundary_boost,
        ).reshape(-1, 1)
        calibrated[seed] = target_scaler.transform(raw_pred).astype(np.float32)
    return calibrated

print("\n🔮 推論所有 seed...")
val_preds_all = get_all_preds(X_val)
test_preds_all = get_all_preds(X_test)

print("\n🔍 貪婪搜尋最佳 seed 組合...")
best_val_mae = float("inf")
best_combo = []
remaining = list(SEEDS)
for _ in range(len(SEEDS)):
    best_new = None
    for cand in remaining:
        combo_try = best_combo + [cand]
        val_avg = np.mean([val_preds_all[sd] for sd in combo_try], axis=0)
        val_pred = target_scaler.inverse_transform(val_avg)
        mae = float(np.mean(np.abs(y_val_raw - val_pred)))
        if mae < best_val_mae:
            best_val_mae = mae
            best_new = cand
    if best_new is None:
        break
    best_combo.append(best_new)
    remaining.remove(best_new)

print(f"  最佳 combo: seeds={best_combo}  val MAE={best_val_mae:.2f}")

test_avg = np.mean([test_preds_all[s] for s in best_combo], axis=0)
test_preds = target_scaler.inverse_transform(test_avg) # 這是最終預測金額

# ── 核心接入：正式評估流程 ────────────────────────────────────────────
print("\n🏁 [Spec] 正在執行團隊統一評估流程...")

# 1. 讀取 Metadata
metadata_df = pd.read_csv(ARTIFACTS_DIR / "metadata.csv")

# 2. 準備 prediction_input_df (只取 test)
test_meta = metadata_df[metadata_df['split'] == 'test'].reset_index(drop=True)
prediction_input_df = pd.DataFrame({
    'user_id': test_meta['user_id'],
    'date': test_meta['date'],
    'y_true': y_test_raw.ravel(),
    'y_pred': test_preds.ravel()
})

# 3. 準備 split_metadata_df (全部)
split_metadata_df = metadata_df[['user_id', 'date', 'split']]

# 4. 呼叫共用 evaluator
run_output_evaluation(
    model_name="bigru_TL_alignment",
    prediction_input_df=prediction_input_df,
    split_metadata_df=split_metadata_df,
    # 確保報表存到 ml_ibm/model_outputs/
    output_root=MY_DIR.parent / "model_outputs"
)

print("\n📊 計算每個 seed 個別指標...")
compute_per_seed_metrics(
    seed_preds_dict=test_preds_all,
    target_scaler=target_scaler,
    prediction_input_df=prediction_input_df,
    split_metadata_df=split_metadata_df,
    output_dir=MY_DIR.parent / "model_outputs" / "bigru_TL_alignment",
)

if SENSITIVITY_EXCLUDE_USER_IDS:
    suffix = "exclude_" + "_".join(SENSITIVITY_EXCLUDE_USER_IDS)
    sensitivity_model_name = f"bigru_TL_alignment_{suffix}"
    uncalibrated_model_name = f"{sensitivity_model_name}_uncalibrated"
    keep_mask = ~prediction_input_df["user_id"].astype(str).isin(SENSITIVITY_EXCLUDE_USER_IDS)
    keep_mask_np = keep_mask.to_numpy()
    sensitivity_prediction_input_df = prediction_input_df.loc[keep_mask].reset_index(drop=True)
    sensitivity_seed_preds_all = {
        seed: preds[keep_mask_np]
        for seed, preds in test_preds_all.items()
    }

    print(f"\n📊 [Sensitivity] 排除 {SENSITIVITY_EXCLUDE_USER_IDS} 後輸出未校準指標...")
    run_output_evaluation(
        model_name=uncalibrated_model_name,
        prediction_input_df=sensitivity_prediction_input_df,
        split_metadata_df=split_metadata_df,
        output_root=MY_DIR.parent / "model_outputs"
    )
    compute_per_seed_metrics(
        seed_preds_dict=sensitivity_seed_preds_all,
        target_scaler=target_scaler,
        prediction_input_df=sensitivity_prediction_input_df,
        split_metadata_df=split_metadata_df,
        output_dir=MY_DIR.parent / "model_outputs" / uncalibrated_model_name,
    )

    val_meta = metadata_df[metadata_df['split'] == 'val'].reset_index(drop=True)
    val_prediction_input_df = pd.DataFrame({
        'user_id': val_meta['user_id'],
        'date': val_meta['date'],
        'y_true': y_val_raw.ravel(),
        'y_pred': y_val_raw.ravel()
    })
    val_keep_mask = ~val_prediction_input_df["user_id"].astype(str).isin(SENSITIVITY_EXCLUDE_USER_IDS)
    val_keep_mask_np = val_keep_mask.to_numpy()
    sensitivity_val_input_df = val_prediction_input_df.loc[val_keep_mask].reset_index(drop=True)
    sensitivity_val_context = build_metric_context(sensitivity_val_input_df, split_metadata_df)

    calibration_by_seed = {}
    for seed, scaled_pred in val_preds_all.items():
        raw_pred = target_scaler.inverse_transform(scaled_pred[val_keep_mask_np]).ravel()
        scale, offset, boundary_boost, metrics = choose_calibration(raw_pred, sensitivity_val_context)
        calibration_by_seed[seed] = (scale, offset, boundary_boost)
        print(
            f"  seed={seed:>5} calibration_scale={scale:.2f} "
            f"calibration_offset={offset:.0f} "
            f"boundary_boost={boundary_boost:.2f} "
            f"val_MAE={metrics['MAE']:.2f} val_RMSE={metrics['RMSE']:.2f} "
            f"val_Binary_F1={metrics['Binary_F1']:.4f} val_Weighted_F1={metrics['Weighted_F1']:.4f}"
        )

    ensemble_val_raw = target_scaler.inverse_transform(
        np.mean([val_preds_all[s][val_keep_mask_np] for s in best_combo], axis=0)
    ).ravel()
    ensemble_scale, ensemble_offset, ensemble_boundary_boost, ensemble_val_metrics = choose_calibration(ensemble_val_raw, sensitivity_val_context)
    print(
        f"  ensemble calibration_scale={ensemble_scale:.2f} "
        f"calibration_offset={ensemble_offset:.0f} "
        f"boundary_boost={ensemble_boundary_boost:.2f} "
        f"val_MAE={ensemble_val_metrics['MAE']:.2f} "
        f"val_RMSE={ensemble_val_metrics['RMSE']:.2f} "
        f"val_Binary_F1={ensemble_val_metrics['Binary_F1']:.4f} "
        f"val_Weighted_F1={ensemble_val_metrics['Weighted_F1']:.4f}"
    )

    sensitivity_test_context = build_metric_context(sensitivity_prediction_input_df, split_metadata_df)
    calibrated_test_preds = apply_calibration(
        sensitivity_prediction_input_df["y_pred"].to_numpy(dtype=float),
        sensitivity_test_context["future_available_7d"],
        ensemble_scale,
        ensemble_offset,
        ensemble_boundary_boost,
    )
    calibrated_prediction_input_df = sensitivity_prediction_input_df.copy()
    calibrated_prediction_input_df["y_pred"] = calibrated_test_preds
    calibrated_seed_preds_all = apply_raw_calibration_to_scaled_preds(
        sensitivity_seed_preds_all,
        calibration_by_seed,
        sensitivity_test_context,
    )

    print(f"\n📊 [Sensitivity] 排除 {SENSITIVITY_EXCLUDE_USER_IDS} 後輸出 validation-calibrated 指標...")
    run_output_evaluation(
        model_name=sensitivity_model_name,
        prediction_input_df=calibrated_prediction_input_df,
        split_metadata_df=split_metadata_df,
        output_root=MY_DIR.parent / "model_outputs"
    )
    compute_per_seed_metrics(
        seed_preds_dict=calibrated_seed_preds_all,
        target_scaler=target_scaler,
        prediction_input_df=calibrated_prediction_input_df,
        split_metadata_df=split_metadata_df,
        output_dir=MY_DIR.parent / "model_outputs" / sensitivity_model_name,
    )

print(f"\n✅ 所有正式評估檔案已儲存至: {MY_DIR.parent}/model_outputs/bigru_TL_alignment/")
print("🎉 bigru_TL_alignment 期末考完成！")
