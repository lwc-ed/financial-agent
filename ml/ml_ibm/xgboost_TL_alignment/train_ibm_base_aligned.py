import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLS = [
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

TARGET_COL = "target"


def split_per_user_time_order(df):
    train_parts = []
    valid_parts = []

    for user_id, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("date").reset_index(drop=True)

        n = len(user_df)
        if n < 10:
            continue

        split_idx = int(n * 0.85)

        train_parts.append(user_df.iloc[:split_idx])
        valid_parts.append(user_df.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True)
    valid_df = pd.concat(valid_parts, ignore_index=True)

    return train_df, valid_df


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mask = y_true > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = None

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": None if mape is None else float(mape),
    }


def main():
    data_path = DATA_DIR / "ibm_processed_aligned.csv"
    model_path = MODEL_DIR / "xgb_ibm_aligned_v2.json"
    scaler_path = MODEL_DIR / "xgb_ibm_feature_scaler.pkl"
    metric_path = RESULT_DIR / "ibm_base_metrics_v2.json"
    pred_path = RESULT_DIR / "ibm_valid_predictions.csv"

    print(f"[INFO] loading IBM aligned data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"找不到 IBM aligned data：{data_path}")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"ibm_processed_aligned.csv 缺少欄位: {missing}")

    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    train_df, valid_df = split_per_user_time_order(df)

    print(f"[INFO] train shape = {train_df.shape}")
    print(f"[INFO] valid shape = {valid_df.shape}")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values

    X_valid = valid_df[FEATURE_COLS].values
    y_valid = valid_df[TARGET_COL].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_valid_sc = scaler.transform(X_valid)

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    print("[INFO] training IBM XGBoost base model...")

    model.fit(
        X_train_sc,
        y_train,
        eval_set=[(X_valid_sc, y_valid)],
        verbose=50,
    )

    preds = model.predict(X_valid_sc)
    preds = np.maximum(preds, 0)

    metrics = regression_metrics(y_valid, preds)

    print("[INFO] validation metrics:")
    print(metrics)

    model.save_model(model_path)

    import pickle
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_df = valid_df[["user_id", "date"]].copy()
    pred_df["y_true"] = y_valid
    pred_df["y_pred"] = preds
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print(f"[OK] saved IBM model -> {model_path}")
    print(f"[OK] saved scaler -> {scaler_path}")
    print(f"[OK] saved metrics -> {metric_path}")
    print(f"[OK] saved valid predictions -> {pred_path}")


if __name__ == "__main__":
    main()