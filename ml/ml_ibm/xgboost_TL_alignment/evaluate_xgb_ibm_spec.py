import sys
from pathlib import Path

import pandas as pd

# 讓 ml_ibm/xgboost_TL_alignment 可以 import 專案根目錄下的 ml/output_eval_utils.py
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from ml.output_eval_utils import run_output_evaluation


THIS_DIR = Path(__file__).resolve().parent

PRED_PATH = THIS_DIR / "results" / "own_valid_predictions_finetune.csv"
OWN_ALIGNED_PATH = THIS_DIR / "data" / "processed" / "own_processed_aligned.csv"
OUTPUT_ROOT = THIS_DIR.parent / "model_outputs"


def main():
    print("🚀 [XGBoost IBM Spec Eval] 開始產生正式 spec 輸出...")

    if not PRED_PATH.exists():
        raise FileNotFoundError(f"找不到預測檔案：{PRED_PATH}")

    if not OWN_ALIGNED_PATH.exists():
        raise FileNotFoundError(f"找不到 own aligned data：{OWN_ALIGNED_PATH}")

    pred_df = pd.read_csv(PRED_PATH)
    own_df = pd.read_csv(OWN_ALIGNED_PATH)

    pred_df["date"] = pd.to_datetime(pred_df["date"])
    own_df["date"] = pd.to_datetime(own_df["date"])

    print("[INFO] prediction columns:", list(pred_df.columns))
    print("[INFO] own aligned columns:", list(own_df.columns))

    # 嘗試自動判斷 prediction 檔案欄位
    if "y_true" not in pred_df.columns:
        if "target" in pred_df.columns:
            pred_df = pred_df.rename(columns={"target": "y_true"})
        elif "true" in pred_df.columns:
            pred_df = pred_df.rename(columns={"true": "y_true"})
        else:
            raise ValueError("預測檔缺少 y_true 欄位，請檢查 own_valid_predictions_finetune.csv")

    if "y_pred" not in pred_df.columns:
        if "pred" in pred_df.columns:
            pred_df = pred_df.rename(columns={"pred": "y_pred"})
        elif "prediction" in pred_df.columns:
            pred_df = pred_df.rename(columns={"prediction": "y_pred"})
        else:
            raise ValueError("預測檔缺少 y_pred 欄位，請檢查 own_valid_predictions_finetune.csv")

    prediction_input_df = pred_df[["user_id", "date", "y_true", "y_pred"]].copy()

    # 建立 split_metadata_df
    # 若 own_processed_aligned.csv 已經有 split 欄位，直接使用
    if "split" in own_df.columns:
        split_metadata_df = own_df[["user_id", "date", "split"]].copy()
    else:
        print("[INFO] own data 沒有 split 欄位，依 per-user 70/15/15 時間序重新建立 split_metadata_df")

        metadata_rows = []

        for user_id, user_df in own_df.groupby("user_id"):
            user_df = user_df.sort_values("date").reset_index(drop=True)
            n = len(user_df)

            train_end = int(n * 0.70)
            valid_end = int(n * 0.85)

            for i, row in user_df.iterrows():
                if i < train_end:
                    split = "train"
                elif i < valid_end:
                    split = "valid"
                else:
                    split = "test"

                metadata_rows.append({
                    "user_id": row["user_id"],
                    "date": row["date"],
                    "split": split,
                })

        split_metadata_df = pd.DataFrame(metadata_rows)

    print("[INFO] prediction_input_df shape:", prediction_input_df.shape)
    print("[INFO] split_metadata_df shape:", split_metadata_df.shape)
    print("[INFO] split counts:")
    print(split_metadata_df["split"].value_counts())

    run_output_evaluation(
        model_name="xgboost_TL_alignment",
        prediction_input_df=prediction_input_df,
        split_metadata_df=split_metadata_df,
        output_root=OUTPUT_ROOT,
    )

    print("🎉 [XGBoost IBM Spec Eval] 完成！")
    print(f"輸出位置：{OUTPUT_ROOT / 'xgboost_TL_alignment'}")


if __name__ == "__main__":
    main()