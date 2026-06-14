import json
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from alignment_utils import (
    DATA_DIR,
    MODEL_DIR,
    RESULT_DIR,
    load_feature_schema,
    save_pickle,
    split_xy,
    train_valid_split_by_time,
)


def main():
    data_path = DATA_DIR / "kaggle_processed_aligned.csv"
    model_path = MODEL_DIR / "xgb_kaggle_aligned_v3_huber_raw.json"
    scaler_path = MODEL_DIR / "feature_scaler_v3_huber_raw.pkl"
    metric_path = RESULT_DIR / "kaggle_base_metrics_v3_huber_raw.json"
    pred_path = RESULT_DIR / "kaggle_valid_predictions_v3_huber_raw.csv"

    df = pd.read_csv(data_path)
    features = load_feature_schema(DATA_DIR / "common_features.json")

    train_df, valid_df = train_valid_split_by_time(df, ratio=0.2)

    X_train, y_train = split_xy(train_df, features, "label")
    X_valid, y_valid = split_xy(valid_df, features, "label")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_valid_sc = scaler.transform(X_valid)
    save_pickle(scaler, scaler_path)

    print("========== TRAIN INFO ==========")
    print(f"Train size: {X_train.shape}, Valid size: {X_valid.shape}")
    print("Feature columns:")
    print(features)
    print("Train label summary (raw space):")
    print(y_train.describe())
    print("Valid label summary (raw space):")
    print(y_valid.describe())

    model = xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=200,
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

    model.fit(
        X_train_sc,
        y_train,
        eval_set=[(X_valid_sc, y_valid)],
        verbose=False,
    )

    preds = model.predict(X_valid_sc)

    # 原始 target 空間下，限制極端值
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

    model.save_model(model_path)

    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pred_df = valid_df[["user_id", "date", "target"]].copy()
    pred_df["y_true"] = y_valid.values
    pred_df["y_pred"] = preds
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print(f"[OK] saved model -> {model_path}")
    print(f"[OK] saved scaler -> {scaler_path}")
    print(f"[OK] saved metrics -> {metric_path}")
    print(f"[OK] saved valid predictions -> {pred_path}")
    print(metrics)


if __name__ == "__main__":
    main()