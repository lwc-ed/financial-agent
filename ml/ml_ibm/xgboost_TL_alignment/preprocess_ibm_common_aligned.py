import json
from pathlib import Path
import pandas as pd



ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    input_path = ROOT.parent / "processed_data" / "artifacts" / "ibm_daily.csv"
    output_path = DATA_DIR / "ibm_processed_aligned.csv"

    print(f"[INFO] loading IBM daily data from: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"找不到 IBM daily 檔案：{input_path}")

    df = pd.read_csv(input_path, nrows=300)

    required_cols = [
        "user_id",
        "date",
        "daily_expense",
        "daily_income",
        "txn_count",
        "daily_net",
        "dow",
        "is_weekend",
        "day",
        "month",
        "expense_7d_sum",
        "expense_7d_mean",
        "expense_30d_sum",
        "expense_30d_mean",
        "zscore_7d",
        "zscore_14d",
        "zscore_30d",
        "target",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"ibm_daily.csv 缺少必要欄位: {missing}")

    df["date"] = pd.to_datetime(df["date"])

    # IBM 已經在 build_ibm_daily.py 做好 aligned 特徵
    # 這裡統一輸出給 xgboost_TL_alignment 使用
    feature_cols = [
        "daily_expense",
        "daily_income",
        "txn_count",
        "daily_net",
        "dow",
        "is_weekend",
        "day",
        "month",
        "expense_7d_sum",
        "expense_7d_mean",
        "expense_30d_sum",
        "expense_30d_mean",
    ]

    schema_path = DATA_DIR / "common_features.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved feature schema -> {schema_path}")

    keep_cols = ["user_id", "date"] + feature_cols + ["target"]
    df = df[keep_cols].copy()

    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[OK] saved aligned IBM data -> {output_path}")
    print(f"[INFO] shape = {df.shape}")
    print(f"[INFO] users = {df['user_id'].nunique()}")


if __name__ == "__main__":
    main()