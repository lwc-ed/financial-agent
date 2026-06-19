"""
[NO-ALIGN pipeline] Step 5：預測與評估（30-seed ensemble）
==========================================================
重用 ml_walmart/output_eval_utils 的正式評估器（只讀 import），對 raw（無對齊）
模型算出與 aligned 版相同規格的 MAE / RMSE / Binary F1 / Weighted F1，
方便直接和 aligned 結果對照。

說明：本步驟採「全 seed ensemble + per-seed mean±std」，不含原 5_predict_bigru.py
的 calibration / 最佳 seed 組合搜尋（那些是 aligned 專屬的後處理微調，
對照基準採乾淨 ensemble 較公平）。

輸出 → ml_temp/model_outputs/bigru_noalign/
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ORIG = HERE.parent / "ml_ibm" / "bigru_TL_alignment"
EVAL = HERE.parent / "ml_walmart"
for p in (HERE, ORIG, EVAL):
    sys.path.insert(0, str(p))

from model_bigru import BiGRUWithAttention  # noqa: E402
from no_alignment_utils import RAW_FEATURE_COLS  # noqa: E402
from output_eval_utils import compute_per_seed_metrics, run_output_evaluation  # noqa: E402

ART = HERE / "artifacts_noalign"
OUT_ROOT = HERE / "model_outputs"
MODEL_NAME = "bigru_noalign"

INPUT_SIZE = len(RAW_FEATURE_COLS)
HIDDEN_SIZE, NUM_LAYERS, DROPOUT, OUTPUT_SIZE = 48, 2, 0.4, 1

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚙️  設備: {device}")

seed_paths = sorted(ART.glob("finetune_bigru_seed*.pth"))
SEEDS = [int(p.stem.split("seed")[1]) for p in seed_paths]
print(f"🔍 偵測到 {len(SEEDS)} 個 seeds")

X_test = np.load(ART / "personal_X_test.npy")
y_test_raw = np.load(ART / "personal_y_test_raw.npy").ravel()
with open(ART / "personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

metadata_df = pd.read_csv(ART / "metadata.csv")
split_metadata_df = metadata_df[["user_id", "date", "split"]]
test_meta = metadata_df[metadata_df["split"] == "test"].reset_index(drop=True)
assert len(test_meta) == len(y_test_raw), f"test meta {len(test_meta)} != y_test {len(y_test_raw)}"

x_t = torch.tensor(X_test, dtype=torch.float32).to(device)
seed_preds_scaled = {}
for seed, path in zip(SEEDS, seed_paths):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = BiGRUWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        seed_preds_scaled[seed] = model(x_t).cpu().numpy()

# 全 seed ensemble（在 raw 空間平均）
ensemble_raw = np.mean(
    [target_scaler.inverse_transform(p).ravel() for p in seed_preds_scaled.values()], axis=0
)
ensemble_raw = np.maximum(ensemble_raw, 0.0)

prediction_input_df = pd.DataFrame({
    "user_id": test_meta["user_id"],
    "date": test_meta["date"],
    "y_true": y_test_raw,
    "y_pred": ensemble_raw,
})

print("\n📊 產出正式規格報告（與 aligned 同一評估器）...")
run_output_evaluation(
    model_name=MODEL_NAME,
    prediction_input_df=prediction_input_df,
    split_metadata_df=split_metadata_df,
    output_root=OUT_ROOT,
)

print("\n📊 計算每個 seed 個別指標（mean±std）...")
compute_per_seed_metrics(
    seed_preds_dict=seed_preds_scaled,
    target_scaler=target_scaler,
    prediction_input_df=prediction_input_df,
    split_metadata_df=split_metadata_df,
    output_dir=OUT_ROOT / MODEL_NAME,
)

print(f"\n✅ 無對齊評估完成，報告在：{OUT_ROOT / MODEL_NAME}")
print("   ⇄ 對照 aligned 版：ml_ibm/model_outputs/bigru_TL_alignment/")
