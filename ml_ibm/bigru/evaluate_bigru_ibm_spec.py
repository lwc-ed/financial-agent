import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from ml.output_eval_utils import run_output_evaluation


THIS_DIR = Path(__file__).resolve().parent

PRED_PATH = THIS_DIR.parent / "model_outputs" / "bigru" / "predictions_bigru_ibm_finetuned.csv"
METADATA_PATH = THIS_DIR / "artifacts" / "sample_metadata.csv"
OUTPUT_ROOT = THIS_DIR.parent / "model_outputs"


def main():
    print("🚀 [BigRU IBM Spec Eval] 開始產生正式 spec 輸出...")

    if not PRED_PATH.exists():
        raise FileNotFoundError(f"找不到 BigRU 預測檔案：{PRED_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"找不到 sample metadata：{METADATA_PATH}")

    pred_df = pd.read_csv(PRED_PATH)
    metadata_df = pd.read_csv(METADATA_PATH)

    pred_df["date"] = pd.to_datetime(pred_df["date"])
    metadata_df["date"] = pd.to_datetime(metadata_df["date"])

    print("[INFO] prediction columns:", list(pred_df.columns))
    print("[INFO] metadata columns:", list(metadata_df.columns))

    required_pred_cols = ["user_id", "date", "y_true", "y_pred"]
    missing_pred = [c for c in required_pred_cols if c not in pred_df.columns]
    if missing_pred:
        raise ValueError(f"BigRU 預測檔缺少欄位：{missing_pred}")

    required_meta_cols = ["user_id", "date", "split"]
    missing_meta = [c for c in required_meta_cols if c not in metadata_df.columns]
    if missing_meta:
        raise ValueError(f"sample_metadata.csv 缺少欄位：{missing_meta}")

    prediction_input_df = pred_df[["user_id", "date", "y_true", "y_pred"]].copy()
    split_metadata_df = metadata_df[["user_id", "date", "split"]].copy()

    print("[INFO] prediction_input_df shape:", prediction_input_df.shape)
    print("[INFO] split_metadata_df shape:", split_metadata_df.shape)
    print("[INFO] split counts:")
    print(split_metadata_df["split"].value_counts())

    run_output_evaluation(
        model_name="bigru",
        prediction_input_df=prediction_input_df,
        split_metadata_df=split_metadata_df,
        output_root=OUTPUT_ROOT,
    )

    print("🎉 [BigRU IBM Spec Eval] 完成！")
    print(f"輸出位置：{OUTPUT_ROOT / 'bigru'}")


if __name__ == "__main__":
    main()