"""
Step 5：Transformer 預測與評估
======================================
1. 暴力搜尋最佳 seed 組合
2. 接入 ml/output_eval_utils.py 產出正式規格報告
"""

import glob as _glob
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts_transformer_tl"
ML_ROOT = MY_DIR.parent

sys.path.insert(0, str(MY_DIR))
sys.path.insert(0, str(ML_ROOT))

from alignment_utils import ALIGNED_FEATURE_COLS
from model_transformer import TransformerModel
from output_eval_utils import run_output_evaluation

SEEDS = sorted([int(f.split("seed")[1].replace(".pth", "")) for f in _glob.glob(f"{ARTIFACTS_DIR}/finetune_transformer_seed*.pth")])
print(f"🔍 偵測到 {len(SEEDS)} 個 seeds: {SEEDS}")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"⚙️  目前使用設備: {device}")

INPUT_SIZE  = len(ALIGNED_FEATURE_COLS)
D_MODEL     = 64
NHEAD       = 4
NUM_LAYERS  = 2
DROPOUT     = 0.4
OUTPUT_SIZE = 1

print("📂 載入資料...")
X_val         = np.load(ARTIFACTS_DIR / "personal_X_val.npy")
X_test        = np.load(ARTIFACTS_DIR / "personal_X_test.npy")
y_val_scaled  = np.load(ARTIFACTS_DIR / "personal_y_val.npy")
y_test_raw    = np.load(ARTIFACTS_DIR / "personal_y_test_raw.npy")
test_user_ids = np.load(ARTIFACTS_DIR / "personal_test_user_ids.npy")
val_user_ids  = np.load(ARTIFACTS_DIR / "personal_val_user_ids.npy")

with open(ARTIFACTS_DIR / "personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)
y_val_raw = target_scaler.inverse_transform(y_val_scaled)


def get_all_preds(x: np.ndarray) -> dict:
    all_preds = {}
    x_t = torch.tensor(x, dtype=torch.float32).to(device)
    for seed in SEEDS:
        ckpt = torch.load(ARTIFACTS_DIR / f"finetune_transformer_seed{seed}.pth", map_location=device, weights_only=True)
        model = TransformerModel(INPUT_SIZE, D_MODEL, NHEAD, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            preds = model(x_t).cpu().numpy()
        all_preds[seed] = preds
    return all_preds


print("\n🔮 推論所有 seed...")
val_preds_all  = get_all_preds(X_val)
test_preds_all = get_all_preds(X_test)

print("\n🔍 暴力搜尋最佳 seed 組合...")
best_val_mae = float("inf")
best_combo = SEEDS
for r in range(1, len(SEEDS) + 1):
    for combo in combinations(SEEDS, r):
        val_avg  = np.mean([val_preds_all[s] for s in combo], axis=0)
        val_pred = target_scaler.inverse_transform(val_avg)
        mae = float(np.mean(np.abs(y_val_raw - val_pred)))
        if mae < best_val_mae:
            best_val_mae = mae
            best_combo   = list(combo)

print(f"  最佳 combo: seeds={best_combo}  val MAE={best_val_mae:.2f}")

test_avg   = np.mean([test_preds_all[s] for s in best_combo], axis=0)
test_preds = target_scaler.inverse_transform(test_avg)

print("\n🏁 [Spec] 正在執行團隊統一評估流程...")

metadata_df = pd.read_csv(ARTIFACTS_DIR / "metadata.csv")
test_meta   = metadata_df[metadata_df['split'] == 'test'].reset_index(drop=True)
prediction_input_df = pd.DataFrame({
    'user_id': test_meta['user_id'],
    'date':    test_meta['date'],
    'y_true':  y_test_raw.ravel(),
    'y_pred':  test_preds.ravel()
})
split_metadata_df = metadata_df[['user_id', 'date', 'split']]

run_output_evaluation(
    model_name="transformer_TL_alignment",
    prediction_input_df=prediction_input_df,
    split_metadata_df=split_metadata_df,
)

print(f"\n✅ 所有正式評估檔案已儲存至: {ML_ROOT}/model_outputs/transformer_TL_alignment/")
print("🎉 transformer_TL_alignment 評估完成！")
