import json
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
    train_valid_split_by_time,
)


def main():
    own_data_path = DATA_DIR / "own_processed_aligned.csv"
    base_model_path = MODEL_DIR / "xgb_kaggle_aligned_v3_huber_raw.json"
    scaler_path = MODEL_DIR / "feature_scaler_v3_huber_raw.pkl"
    finetuned_model_path = MODEL_DIR / "xgb_finetuned_own_aligned_v3_huber_raw.json"
    metric_path = RESULT_DIR / "finetune_own_metrics_v3_huber_raw.json"
    pred_path = RESULT_DIR / "own_valid_predictions_finetune_v3_huber_raw.csv"

    df = pd.read_csv(own_data_path)
    features = load_feature_schema(DATA_DIR / "common_features.json")

    train_df, valid_df = train_valid_split_by_time(df, ratio=0.2)

    X_train, y_train = split_xy(train_df, features, "label")
    X_valid, y_valid = split_xy(valid_df, features, "label")

    scaler = load_pickle(scaler_path)
    X_train_sc = scaler.transform(X_train)
    X_valid_sc = scaler.transform(X_valid)

    model = xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=5.0,
        reg_lambda=10.0,
        min_child_weight=10,
        gamma=1.0,
        random_state=42,
    )

    if base_model_path.exists():
        model.fit(
            X_train_sc,
            y_train,
            xgb_model=str(base_model_path),
            eval_set=[(X_valid_sc, y_valid)],
            verbose=False,
        )
    else:
        model.fit(
            X_train_sc,
            y_train,
            eval_set=[(X_valid_sc, y_valid)],
            verbose=False,
        )

    preds = model.predict(X_valid_sc)
    preds = preds.clip(min=0)

    metrics = {
        "mae": float(mean_absolute_error(y_valid, preds)),
        "rmse": float(mean_squared_error(y_valid, preds) ** 0.5),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "feature_count": int(len(features)),
        "objective": "reg:pseudohubererror",
        "target_space": "raw",
    }

    model.save_model(finetuned_model_path)

    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pred_df = valid_df[["user_id", "date", "target"]].copy()
    pred_df["y_true"] = y_valid.values
    pred_df["y_pred"] = preds
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print(f"[OK] saved finetuned model -> {finetuned_model_path}")
    print(f"[OK] saved metrics -> {metric_path}")
    print(f"[OK] saved valid predictions -> {pred_path}")
    print(metrics)


if __name__ == "__main__":
    main()