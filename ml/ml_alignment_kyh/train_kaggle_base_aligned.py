import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from alignment_utils import (
    DATA_DIR,
    MODEL_DIR,
    RESULT_DIR,
    load_feature_schema,
    split_xy,
    train_valid_split_by_time,
)

ROOT = Path(__file__).resolve().parent


def main():
    data_path = DATA_DIR / "kaggle_processed_aligned.csv"
    model_path = MODEL_DIR / "xgb_kaggle_aligned.json"
    metric_path = RESULT_DIR / "kaggle_base_metrics.json"

    if not data_path.exists():
        raise FileNotFoundError(f"找不到資料檔案: {data_path}")

    df = pd.read_csv(data_path)

    # 只使用 schema 裡定義的共同特徵
    feature_cols = load_feature_schema(DATA_DIR / "common_features.json")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"資料缺少特徵欄位: {missing_features}")

    if "label" not in df.columns:
        raise ValueError("資料缺少 label 欄位")

    # 依時間切 train / valid
    train_df, valid_df = train_valid_split_by_time(df, valid_ratio=0.2)

    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError("train/valid 切分後資料為空，請檢查資料量")

    X_train, y_train = split_xy(train_df, feature_cols, "label")
    X_valid, y_valid = split_xy(valid_df, feature_cols, "label")

    print("========== TRAIN INFO ==========")
    print(f"Train size: {X_train.shape}, Valid size: {X_valid.shape}")
    print("Feature columns:")
    print(feature_cols)
    print("Train label summary:")
    print(y_train.describe())
    print("Valid label summary:")
    print(y_valid.describe())

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    preds = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, preds)
    rmse = mean_squared_error(y_valid, preds) ** 0.5

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "feature_count": int(len(feature_cols)),
    }

    # 存模型
    model.save_model(model_path)

    # 存 metrics
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 存 validation 預測結果，方便檢查
    pred_df = valid_df[["user_id", "date"]].copy() if {"user_id", "date"}.issubset(valid_df.columns) else pd.DataFrame()
    pred_df["y_true"] = y_valid.values
    pred_df["y_pred"] = preds
    pred_df.to_csv(
        RESULT_DIR / "kaggle_valid_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"[OK] saved model -> {model_path}")
    print(f"[OK] saved metrics -> {metric_path}")
    print(f"[OK] saved valid predictions -> {RESULT_DIR / 'kaggle_valid_predictions.csv'}")
    print(metrics)


if __name__ == "__main__":
    main()