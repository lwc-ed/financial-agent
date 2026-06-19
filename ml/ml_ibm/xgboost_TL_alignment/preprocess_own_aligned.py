from pathlib import Path
import numpy as np
import pandas as pd

from alignment_utils import (
    DATA_DIR,
    load_feature_schema,
    ensure_datetime,
    fill_missing_values,
)

ROOT = Path(__file__).resolve().parent


def main():
    # own data 來源改成 features_all.csv
    input_path = ROOT.parents[1] / "ml" / "processed_data" / "artifacts" / "features_all.csv"
    output_path = DATA_DIR / "own_processed_aligned.csv"
    schema_path = DATA_DIR / "common_features.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到 own data 來源檔案: {input_path}")

    print(f"[INFO] OWN DATA SOURCE: {input_path}")

    df = pd.read_csv(input_path)

    # -------------------------------------------------
    # 0. 排除與 DL 模型一致的使用者
    # -------------------------------------------------
    EXCLUDE_USERS = ["user4", "user5", "user6"]
    before = len(df)
    df = df[~df["user_id"].astype(str).isin(EXCLUDE_USERS)].reset_index(drop=True)
    print(f"[INFO] 排除 {EXCLUDE_USERS}：{before} → {len(df)} 筆")

    # -------------------------------------------------
    # 1. 欄位對應：future_expense_7d_sum 當成 target
    # -------------------------------------------------
    if "future_expense_7d_sum" not in df.columns:
        raise ValueError(
            "features_all.csv 缺少 future_expense_7d_sum 欄位，"
            "無法建立 target。"
        )

    df = df.rename(columns={"future_expense_7d_sum": "target"})

    # -------------------------------------------------
    # 2. 檢查必要欄位
    # -------------------------------------------------
    required_cols = ["user_id", "date", "target"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"features_all.csv 缺少必要欄位: {missing_required}")

    # -------------------------------------------------
    # 3. 日期轉換
    # -------------------------------------------------
    df = ensure_datetime(df, "date")

    # -------------------------------------------------
    # 4. 讀取 kaggle 對齊特徵
    # -------------------------------------------------
    if not schema_path.exists():
        raise FileNotFoundError(
            f"找不到 feature schema: {schema_path}\n"
            "請先跑 preprocess_kaggle_common_aligned.py"
        )

    common_features = load_feature_schema(schema_path)

    # -------------------------------------------------
    # 5. 檢查 features_all.csv 是否包含 v2 需要的共同特徵
    # -------------------------------------------------
    missing_features = [c for c in common_features if c not in df.columns]
    if missing_features:
        raise ValueError(
            "features_all.csv 缺少共同特徵欄位: "
            f"{missing_features}\n"
            "請確認 features_all.csv 欄位名稱是否和 v2 一致。"
        )

    # -------------------------------------------------
    # 6. 補缺值
    # -------------------------------------------------
    df = fill_missing_values(df)

    # -------------------------------------------------
    # 7. 建立 label（沿用 v2：log1p）
    # -------------------------------------------------
    df["target"] = df["target"].clip(lower=0)
    df["label"] = np.log1p(df["target"])

    # -------------------------------------------------
    # 8. 只保留 v2 需要的欄位
    # -------------------------------------------------
    final_cols = ["user_id", "date"] + common_features + ["target", "label"]
    df = df[final_cols].copy()

    # 去除 label 空值
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    # -------------------------------------------------
    # 9. 存檔
    # -------------------------------------------------
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # -------------------------------------------------
    # 10. Debug 輸出
    # -------------------------------------------------
    print(f"[OK] saved aligned own data -> {output_path}")
    print("\n[HEAD]")
    print(df.head())

    print("\n[STATS]")
    print(df[common_features + ["target"]].describe())


if __name__ == "__main__":
    main()
