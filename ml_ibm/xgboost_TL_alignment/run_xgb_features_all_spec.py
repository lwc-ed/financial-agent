import sys
from pathlib import Path

# 讓 `from ml.xxx import ...` 可以正常 import
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml.output_eval_utils import run_output_evaluation


MODEL_NAME = "xgboost_TL_alignment"
DATA_PATH = PROJECT_ROOT / "ml" / "processed_data" / "artifacts" / "features_all.csv"
LOCAL_RESULT_DIR = PROJECT_ROOT / "ml" / "xgboost_TL_alignment" / "results"
LOCAL_MODEL_DIR = PROJECT_ROOT / "ml" / "xgboost_TL_alignment" / "models"


def ensure_dirs():
    LOCAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到資料檔案: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_cols = ["user_id", "date", "future_expense_7d_sum"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"features_all.csv 缺少必要欄位: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    return df


def choose_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    盡量使用 README 中的共用特徵。
    若某些欄位不存在，就自動跳過。
    """
    candidate_features = [
        "daily_income",
        "daily_expense",
        "daily_net",
        "txn_count",
        "has_income",
        "has_expense",
        "dow",
        "is_weekend",
        "day",
        "month",
        "expense_7d_sum",
        "expense_7d_mean",
        "expense_30d_sum",
        "expense_30d_mean",
        "net_7d_sum",
        "net_30d_sum",
        "txn_7d_sum",
        "txn_30d_sum",
        "expense_7d_30d_ratio",
        "expense_trend",
        "days_to_end_of_month",
    ]

    forbidden = {
        "future_expense_7d_sum",  # target
        "target",
        "label",
        "date",
        "user_id",
        "split",
    }

    feature_cols = [c for c in candidate_features if c in df.columns and c not in forbidden]

    if not feature_cols:
        raise ValueError("找不到可用特徵欄位，請檢查 features_all.csv 欄位名稱")

    return feature_cols


def add_split_per_user(df: pd.DataFrame) -> pd.DataFrame:
    """
    依每個 user 的時間順序切分，避免時序洩漏。
    預設邏輯：
    - n >= 10: 70% train / 15% valid / 15% test
    - 3 <= n < 10: 至少保留 1 筆 valid、1 筆 test
    - n == 2: 1 train / 1 test
    - n == 1: 全 train
    """
    parts = []

    for user_id, g in df.groupby("user_id", sort=False):
        g = g.sort_values("date").copy()
        n = len(g)

        if n == 1:
            g["split"] = "train"
        elif n == 2:
            g["split"] = ["train", "test"]
        elif 3 <= n < 10:
            train_n = n - 2
            valid_n = 1
            test_n = 1
            g["split"] = ["train"] * train_n + ["valid"] * valid_n + ["test"] * test_n
        else:
            train_n = int(n * 0.7)
            valid_n = int(n * 0.15)
            test_n = n - train_n - valid_n

            # 防呆：確保三個 split 都盡量有資料
            if valid_n == 0:
                valid_n = 1
                train_n -= 1
            if test_n == 0:
                test_n = 1
                train_n -= 1
            if train_n <= 0:
                raise ValueError(f"user {user_id} 切分後 train 筆數 <= 0，請檢查資料量")

            g["split"] = (
                ["train"] * train_n +
                ["valid"] * valid_n +
                ["test"] * test_n
            )

        parts.append(g)

    out = pd.concat(parts, axis=0).reset_index(drop=True)
    return out


def prepare_xy(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    target_col = "future_expense_7d_sum"

    for name, part in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        if len(part) == 0:
            print(f"[WARN] {name} split 為空")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[target_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    # 確保數值型態
    for X in [X_train, X_valid, X_test]:
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # 用 train median 補缺值
    train_medians = X_train.median(numeric_only=True)

    X_train = X_train.fillna(train_medians)
    X_valid = X_valid.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    if len(X_valid) > 0:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, verbose=False)

    return model


def save_local_outputs(
    model: xgb.XGBRegressor,
    feature_cols: List[str],
    prediction_input_df: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
):
    model_path = LOCAL_MODEL_DIR / "xgb_features_all_spec.json"
    model.save_model(model_path)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    mask = y_test > 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_pred[mask] - y_test[mask]) / y_test[mask])) * 100)
    else:
        mape = None

    local_metrics = {
        "model_name": MODEL_NAME,
        "data_path": str(DATA_PATH),
        "rows_test": int(len(prediction_input_df)),
        "feature_count": int(len(feature_cols)),
        "features": feature_cols,
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": mape,
    }

    with open(LOCAL_RESULT_DIR / "xgb_features_all_metrics.json", "w", encoding="utf-8") as f:
        json.dump(local_metrics, f, indent=2, ensure_ascii=False)

    prediction_input_df.to_csv(
        LOCAL_RESULT_DIR / "xgb_features_all_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[OK] saved local model -> {model_path}")
    print(f"[OK] saved local metrics -> {LOCAL_RESULT_DIR / 'xgb_features_all_metrics.json'}")
    print(f"[OK] saved local predictions -> {LOCAL_RESULT_DIR / 'xgb_features_all_predictions.csv'}")


def main():
    ensure_dirs()

    print("========== LOAD DATA ==========")
    df = load_data()
    print(f"Full data shape: {df.shape}")
    print(f"Users: {df['user_id'].nunique()}")
    print(f"Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")

    print("\n========== CHOOSE FEATURES ==========")
    feature_cols = choose_feature_columns(df)
    print(f"Feature count: {len(feature_cols)}")
    print(feature_cols)

    print("\n========== BUILD SPLITS ==========")
    df = add_split_per_user(df)

    split_counts = df["split"].value_counts(dropna=False).to_dict()
    print("Split counts:", split_counts)

    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df = df[df["split"] == "test"].copy()

    if len(train_df) == 0:
        raise ValueError("train_df 為空，無法訓練模型")
    if len(test_df) == 0:
        raise ValueError("test_df 為空，無法做正式評估")

    X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_xy(
        train_df, valid_df, test_df, feature_cols
    )

    print("\n========== TRAIN MODEL ==========")
    model = train_xgboost(X_train, y_train, X_valid, y_valid)

    print("\n========== PREDICT ==========")
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0.0)  # 支出預測不允許負數

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"Test MAE : {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    print("\n========== BUILD EVALUATOR INPUT ==========")
    prediction_input_df = test_df[["user_id", "date"]].copy()
    prediction_input_df["date"] = pd.to_datetime(prediction_input_df["date"]).dt.strftime("%Y-%m-%d")
    prediction_input_df["y_true"] = y_test.values
    prediction_input_df["y_pred"] = y_pred

    split_metadata_df = df[["user_id", "date", "split"]].copy()
    split_metadata_df["date"] = pd.to_datetime(split_metadata_df["date"]).dt.strftime("%Y-%m-%d")

    # 檢查必要欄位
    pred_required = ["user_id", "date", "y_true", "y_pred"]
    split_required = ["user_id", "date", "split"]

    for c in pred_required:
        if c not in prediction_input_df.columns:
            raise ValueError(f"prediction_input_df 缺少必要欄位: {c}")

    for c in split_required:
        if c not in split_metadata_df.columns:
            raise ValueError(f"split_metadata_df 缺少必要欄位: {c}")

    if "train" not in set(split_metadata_df["split"].unique()):
        raise ValueError("split_metadata_df 中沒有 train，無法給 evaluator 計算 monthly_available_cash")

    save_local_outputs(model, feature_cols, prediction_input_df, y_test, y_pred)

    print("\n========== RUN SHARED EVALUATOR ==========")
    run_output_evaluation(
        model_name=MODEL_NAME,
        prediction_input_df=prediction_input_df,
        split_metadata_df=split_metadata_df,
    )

    print(f"[OK] shared evaluator done -> ml/model_outputs/{MODEL_NAME}/")
    print("[DONE] XGBoost on features_all.csv completed successfully.")


if __name__ == "__main__":
    main()