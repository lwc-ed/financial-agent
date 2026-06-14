from pathlib import Path
import pandas as pd

from alignment_utils import (
    DATA_DIR,
    load_feature_schema,
    ensure_datetime,
    fill_missing_values,
)

ROOT = Path(__file__).resolve().parent


def main():
    input_path = ROOT.parent / "ml_XGboost" / "data" / "processed" / "own_processed_common.csv"
    output_path = DATA_DIR / "own_processed_aligned.csv"
    schema_path = DATA_DIR / "common_features.json"

    df = pd.read_csv(input_path)

    required_cols = ["user_id", "date", "target"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"own_processed_common.csv 缺少必要欄位: {missing}")

    df = ensure_datetime(df, "date")

    common_features = load_feature_schema(schema_path)

    missing_features = [c for c in common_features if c not in df.columns]
    if missing_features:
        raise ValueError(f"own_processed_common.csv 缺少共同特徵欄位: {missing_features}")

    df = fill_missing_values(df)

    df["target"] = df["target"].clip(lower=0)
    df["label"] = df["target"]

    final_cols = ["user_id", "date"] + common_features + ["target", "label"]
    df = df[final_cols].copy()
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[OK] saved aligned own data -> {output_path}")


if __name__ == "__main__":
    main()
