"""
Step 5：bigru_TL_alignment 預測與評估
======================================
暴力搜尋最佳 seed 組合（依 val MAE）→ 最終 test 評估
"""

import glob as _glob
import json
import os
import pickle
import sys
from itertools import combinations

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import ALIGNED_FEATURE_COLS
from model_bigru import BiGRUWithAttention

ARTIFACTS_DIR = "artifacts_bigru_tl"

SEEDS = sorted([int(f.split("seed")[1].replace(".pth", "")) for f in _glob.glob(f"{ARTIFACTS_DIR}/finetune_bigru_seed*.pth")])
print(f"🔍 偵測到 {len(SEEDS)} 個 seeds: {SEEDS}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

INPUT_SIZE = len(ALIGNED_FEATURE_COLS)
HIDDEN_SIZE = 48
NUM_LAYERS = 2
DROPOUT = 0.4
OUTPUT_SIZE = 1

print("📂 載入資料...")
X_val = np.load(f"{ARTIFACTS_DIR}/personal_X_val.npy")
X_test = np.load(f"{ARTIFACTS_DIR}/personal_X_test.npy")
y_val_scaled = np.load(f"{ARTIFACTS_DIR}/personal_y_val.npy")
y_test_raw = np.load(f"{ARTIFACTS_DIR}/personal_y_test_raw.npy")
test_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_test_user_ids.npy")
val_user_ids = np.load(f"{ARTIFACTS_DIR}/personal_val_user_ids.npy")

with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)
y_val_raw = target_scaler.inverse_transform(y_val_scaled)


def get_all_preds(x: np.ndarray) -> dict:
    all_preds = {}
    x_t = torch.tensor(x, dtype=torch.float32).to(device)
    for seed in SEEDS:
        ckpt = torch.load(f"{ARTIFACTS_DIR}/finetune_bigru_seed{seed}.pth", map_location=device)
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
    per_user_nmae = float(np.mean(user_nmaes) * 100)

    return {
        "mae": mae,
        "rmse": rmse,
        "medae": medae,
        "smape": smape,
        "per_user_nmae": per_user_nmae,
    }


print("\n🔮 推論所有 seed...")
val_preds_all = get_all_preds(X_val)
test_preds_all = get_all_preds(X_test)

print("\n🔍 暴力搜尋最佳 seed 組合（依 val MAE）...")
best_val_mae = float("inf")
best_combo = SEEDS

for r in range(1, len(SEEDS) + 1):
    for combo in combinations(SEEDS, r):
        combo = list(combo)
        val_avg = np.mean([val_preds_all[s] for s in combo], axis=0)
        val_pred = target_scaler.inverse_transform(val_avg)
        mae = float(np.mean(np.abs(y_val_raw - val_pred)))
        if mae < best_val_mae:
            best_val_mae = mae
            best_combo = combo

print(f"  最佳 combo: seeds={best_combo}  val MAE={best_val_mae:.2f}")

val_avg = np.mean([val_preds_all[s] for s in best_combo], axis=0)
test_avg = np.mean([test_preds_all[s] for s in best_combo], axis=0)
val_preds = target_scaler.inverse_transform(val_avg)
test_preds = target_scaler.inverse_transform(test_avg)

val_metrics = compute_metrics(y_val_raw, val_preds, val_user_ids)
test_metrics = compute_metrics(y_test_raw, test_preds, test_user_ids)

all_val_avg = np.mean(list(val_preds_all.values()), axis=0)
all_test_avg = np.mean(list(test_preds_all.values()), axis=0)
all_test_preds = target_scaler.inverse_transform(all_test_avg)
all_test_metrics = compute_metrics(y_test_raw, all_test_preds, test_user_ids)

print(f"\n{'=' * 65}")
print("  BiGRU TL Alignment 結果")
print(f"{'=' * 65}")
print(f"  最佳 combo {best_combo}")
print(f"    Val  MAE  : {val_metrics['mae']:.2f}")
print(f"    Test MAE  : {test_metrics['mae']:.2f}")
print(f"    Test RMSE : {test_metrics['rmse']:.2f}")
print(f"    Test MedAE: {test_metrics['medae']:.2f}")
print(f"    SMAPE     : {test_metrics['smape']:.2f}%")
print("  全 seeds")
print(f"    Test MAE  : {all_test_metrics['mae']:.2f}")
print(f"{'=' * 65}")
print("  💡 若 MAE >> MedAE，代表有少數極端誤差在拉高 MAE")
print("\n  BiGRU baseline（無 TL）: 801.00")
print(f"  差距：{test_metrics['mae'] - 801.00:+.2f}")

result_text = f"""BiGRU TL Alignment Result
========================
model       : BiGRUWithAttention + MMD loss
features    : {len(ALIGNED_FEATURE_COLS)} aligned features
feature_cols: {ALIGNED_FEATURE_COLS}
all_seeds   : {SEEDS}
best_combo  : {best_combo}

val_mae           : {val_metrics['mae']:.4f}
val_rmse          : {val_metrics['rmse']:.4f}
val_medae         : {val_metrics['medae']:.4f}
test_mae          : {test_metrics['mae']:.4f}
test_rmse         : {test_metrics['rmse']:.4f}
test_medae        : {test_metrics['medae']:.4f}
test_smape        : {test_metrics['smape']:.2f}%
test_per_user_nmae: {test_metrics['per_user_nmae']:.2f}%

all_seeds_test_mae: {all_test_metrics['mae']:.4f}

baseline_bigru    : 801.00
gap               : {test_metrics['mae'] - 801.00:+.2f}
"""

with open(f"{ARTIFACTS_DIR}/result.txt", "w", encoding="utf-8") as f:
    f.write(result_text)

with open(f"{ARTIFACTS_DIR}/metrics.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "best_combo": best_combo,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_medae": val_metrics["medae"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_medae": test_metrics["medae"],
            "test_smape": test_metrics["smape"],
            "test_per_user_nmae": test_metrics["per_user_nmae"],
            "all_seeds_test_mae": all_test_metrics["mae"],
            "baseline_bigru_mae": 801.00,
            "gap": round(test_metrics["mae"] - 801.00, 2),
        },
        f,
        indent=2,
    )

print(f"\n✅ 結果儲存至 {ARTIFACTS_DIR}/result.txt")
print(f"✅ Metrics 儲存至 {ARTIFACTS_DIR}/metrics.json")
print("\n🎉 完成！")
