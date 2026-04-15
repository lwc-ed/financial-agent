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
    base_model_path = MODEL_DIR / "xgb_kaggle_aligned.json"
    own_data_path = DATA_DIR / "own_processed_aligned.csv"
    finetuned_model_path = MODEL_DIR / "xgb_finetuned_own_aligned.json"
    metric_path = RESULT_DIR / "finetune_own_metrics.json"

    df = pd.read_csv(own_data_path)
    feature_cols = load_feature_schema(DATA_DIR / "common_features.json")

    train_df, valid_df = train_valid_split_by_time(df, valid_ratio=0.2)
    X_train, y_train = split_xy(train_df, feature_cols, "label")
    X_valid, y_valid = split_xy(valid_df, feature_cols, "label")

    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )

    # 用 source model 的 booster 做 warm start
    if base_model_path.exists():
        model.fit(
            X_train,
            y_train,
            xgb_model=str(base_model_path),
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
    else:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )

    preds = model.predict(X_valid)

    metrics = {
        "mae": float(mean_absolute_error(y_valid, preds)),
        "rmse": float(mean_squared_error(y_valid, preds) ** 0.5),
    }

    model.save_model(finetuned_model_path)
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[OK] saved finetuned model -> {finetuned_model_path}")
    print(f"[OK] saved metrics -> {metric_path}")
    print(metrics)


if __name__ == "__main__":
    main()