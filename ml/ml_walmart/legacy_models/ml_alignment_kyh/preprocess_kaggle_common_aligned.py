from pathlib import Path
import pandas as pd

from alignment_utils import (
    DATA_DIR,
    COMMON_FEATURES,
    ensure_datetime,
    fill_missing_values,
    save_feature_schema,
)

ROOT = Path(__file__).resolve().parent


def main():
    input_path = ROOT.parent.parent / "processed_data" / "artifacts" / "kaggle_processed_common.csv"
    output_path = DATA_DIR / "kaggle_processed_aligned.csv"
    schema_path = DATA_DIR / "common_features.json"

    df = pd.read_csv(input_path)

    required_cols = ["user_id", "date", "target"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"kaggle_processed_common.csv 缺少必要欄位: {missing}")

    df = ensure_datetime(df, "date")

    # 確保共同特徵都存在
    missing_features = [c for c in COMMON_FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(f"kaggle_processed_common.csv 缺少共同特徵欄位: {missing_features}")

    # label 直接用現成 target
    df["label"] = df["target"]

    # 補缺值
    df = fill_missing_values(df)

    # 保留欄位
    final_cols = ["user_id", "date"] + COMMON_FEATURES + ["label"]
    df = df[final_cols].copy()

    # 去除 label 缺值
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    save_feature_schema(COMMON_FEATURES, schema_path)

    print(f"[OK] saved aligned kaggle data -> {output_path}")
    print(f"[OK] saved feature schema -> {schema_path}")
    print(df.head())
    print(df[COMMON_FEATURES + ['label']].describe())


if __name__ == "__main__":
    main()
