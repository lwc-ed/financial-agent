import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from alignment_utils import (
    DATA_DIR,
    RESULT_DIR,
    MODEL_DIR,
    load_feature_schema,
)

ROOT = Path(__file__).resolve().parent


def evaluate_model(model_path: Path, df: pd.DataFrame, feature_cols, name: str):
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型檔案: {model_path}")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"{name} 評估資料缺少特徵欄位: {missing_features}")

    if "label" not in df.columns:
        raise ValueError(f"{name} 評估資料缺少 label 欄位")

    X = df[feature_cols].copy()
    y = df["label"].copy()

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds) ** 0.5

    # 存 prediction 結果
    pred_df = pd.DataFrame()
    if "user_id" in df.columns:
        pred_df["user_id"] = df["user_id"].values
    if "date" in df.columns:
        pred_df["date"] = df["date"].values

    pred_df["y_true"] = y.values
    pred_df["y_pred"] = preds
    pred_df.to_csv(
        RESULT_DIR / f"predictions_{name}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    metrics = {
        "model_name": name,
        "mae": float(mae),
        "rmse": float(rmse),
        "rows": int(len(df)),
        "feature_count": int(len(feature_cols)),
    }
    return metrics


def main():
    own_data_path = DATA_DIR / "own_processed_aligned.csv"
    schema_path = DATA_DIR / "common_features.json"

    base_model_path = MODEL_DIR / "xgb_kaggle_aligned.json"
    finetuned_model_path = MODEL_DIR / "xgb_finetuned_own_aligned.json"

    if not own_data_path.exists():
        raise FileNotFoundError(f"找不到資料檔案: {own_data_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"找不到 schema 檔案: {schema_path}")

    df = pd.read_csv(own_data_path)
    feature_cols = load_feature_schema(schema_path)

    # 後 20% 當測試集
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    if len(test_df) == 0:
        raise ValueError("測試集為空，請檢查資料量")

    print("========== TEST INFO ==========")
    print(f"Test size: {test_df.shape}")
    print("Feature columns:")
    print(feature_cols)
    print("Test label summary:")
    print(test_df["label"].describe())

    results = []

    if base_model_path.exists():
        results.append(
            evaluate_model(base_model_path, test_df, feature_cols, "base_aligned")
        )
    else:
        print(f"[WARN] base model not found: {base_model_path}")

    if finetuned_model_path.exists():
        results.append(
            evaluate_model(finetuned_model_path, test_df, feature_cols, "finetuned_aligned")
        )
    else:
        print(f"[WARN] finetuned model not found: {finetuned_model_path}")

    output_json = RESULT_DIR / "transfer_test_metrics.json"
    output_txt = RESULT_DIR / "aligned_results.txt"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_txt, "w", encoding="utf-8") as f:
        for item in results:
            f.write(
                f"Model: {item['model_name']}\n"
                f"Rows: {item['rows']}\n"
                f"Feature count: {item['feature_count']}\n"
                f"MAE: {item['mae']:.4f}\n"
                f"RMSE: {item['rmse']:.4f}\n\n"
            )

    print(f"[OK] saved evaluation -> {output_json}")
    print(f"[OK] saved summary -> {output_txt}")
    print(f"[OK] saved predictions -> {RESULT_DIR / 'predictions_base_aligned.csv'}")
    print(f"[OK] saved predictions -> {RESULT_DIR / 'predictions_finetuned_aligned.csv'}")
    print(results)


if __name__ == "__main__":
    main()