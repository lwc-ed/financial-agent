import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from alignment_utils import (
    DATA_DIR,
    MODEL_DIR,
    RESULT_DIR,
    load_feature_schema,
    load_pickle,
    split_xy,
    train_valid_test_split_by_time,
)

ROOT = Path(__file__).resolve().parent


def main():
    base_model_path = MODEL_DIR / "xgb_kaggle_aligned_v2.json"
    scaler_path = MODEL_DIR / "feature_scaler_v2.pkl"
    own_data_path = DATA_DIR / "own_processed_aligned.csv"
    finetuned_model_path = MODEL_DIR / "xgb_finetuned_own_aligned_v2.json"
    metric_path = RESULT_DIR / "finetune_own_metrics_v2.json"

    if not own_data_path.exists():
        raise FileNotFoundError(f"找不到資料檔案: {own_data_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"找不到 scaler: {scaler_path}")

    df = pd.read_csv(own_data_path)
    feature_cols = load_feature_schema(DATA_DIR / "common_features.json")

    train_df, valid_df, test_df = train_valid_test_split_by_time(
        df, train_ratio=0.70, valid_ratio=0.15
    )
    print(f"[INFO] split sizes — train: {len(train_df)}, valid: {len(valid_df)}, test: {len(test_df)}")

    X_train, y_train = split_xy(train_df, feature_cols, "label")
    X_valid, y_valid = split_xy(valid_df, feature_cols, "label")
    X_test, y_test = split_xy(test_df, feature_cols, "label")

    scaler = load_pickle(scaler_path)
    X_train_sc = scaler.transform(X_train)
    X_valid_sc = scaler.transform(X_valid)
    X_test_sc = scaler.transform(X_test)

    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )

    if base_model_path.exists():
        model.fit(
            X_train_sc,
            y_train,
            xgb_model=str(base_model_path),
            eval_set=[(X_valid_sc, y_valid)],
            verbose=False
        )
    else:
        model.fit(
            X_train_sc,
            y_train,
            eval_set=[(X_valid_sc, y_valid)],
            verbose=False
        )

    preds_log = model.predict(X_test_sc)

    y_test_raw = np.expm1(y_test)
    preds_raw = np.expm1(preds_log)

    metrics = {
        "mae": float(mean_absolute_error(y_test_raw, preds_raw)),
        "rmse": float(mean_squared_error(y_test_raw, preds_raw) ** 0.5),
    }

    model.save_model(finetuned_model_path)
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pred_df = test_df[["user_id", "date", "target"]].copy()
    pred_df["y_true"] = y_test_raw
    pred_df["y_pred"] = preds_raw
    pred_df.to_csv(
        RESULT_DIR / "own_valid_predictions_finetune.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"[OK] saved finetuned model -> {finetuned_model_path}")
    print(f"[OK] saved metrics -> {metric_path}")
    print(f"[OK] saved test predictions -> {RESULT_DIR / 'own_valid_predictions_finetune.csv'}")
    print(metrics)


if __name__ == "__main__":
    main()