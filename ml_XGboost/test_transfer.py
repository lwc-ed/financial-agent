import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

KAGGLE_PATH = "ml_XGboost/data/processed/kaggle_processed_common.csv"
OWN_PATH = "ml_XGboost/data/processed/own_processed_common.csv"
MODEL_DIR = "ml_XGboost/models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_kaggle_common.json")

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

def main():
    # 1. 讀資料
    kaggle_df = pd.read_csv(KAGGLE_PATH)
    own_df = pd.read_csv(OWN_PATH)

    kaggle_df["date"] = pd.to_datetime(kaggle_df["date"])
    own_df["date"] = pd.to_datetime(own_df["date"])

    print("Kaggle shape:", kaggle_df.shape)
    print("Own shape   :", own_df.shape)

    # 2. Kaggle 訓練集 / 驗證集
    split_date = pd.Timestamp("2012-01-01")
    kaggle_train = kaggle_df[kaggle_df["date"] < split_date].copy()
    kaggle_valid = kaggle_df[kaggle_df["date"] >= split_date].copy()

    print("Kaggle train shape:", kaggle_train.shape)
    print("Kaggle valid shape:", kaggle_valid.shape)

    X_train = kaggle_train[FEATURE_COLS]
    y_train = kaggle_train[TARGET_COL]

    X_valid = kaggle_valid[FEATURE_COLS]
    y_valid = kaggle_valid[TARGET_COL]

    # 3. 訓練 Kaggle model
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    # 4. 先看 Kaggle 驗證表現
    kaggle_pred = model.predict(X_valid)
    kaggle_mae = mean_absolute_error(y_valid, kaggle_pred)
    kaggle_rmse = np.sqrt(mean_squared_error(y_valid, kaggle_pred))

    print("\n=== Kaggle validation result ===")
    print(f"MAE  : {kaggle_mae:.4f}")
    print(f"RMSE : {kaggle_rmse:.4f}")

    # 5. 存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"\n模型已儲存到: {MODEL_PATH}")

    # 6. 直接在 own data 上測試
    X_own = own_df[FEATURE_COLS]
    y_own = own_df[TARGET_COL]

    own_pred = model.predict(X_own)
    own_mae = mean_absolute_error(y_own, own_pred)
    own_rmse = np.sqrt(mean_squared_error(y_own, own_pred))

    print("\n=== Direct Transfer Result (Kaggle -> Own) ===")
    print(f"MAE  : {own_mae:.4f}")
    print(f"RMSE : {own_rmse:.4f}")

    # 7. 看幾筆預測結果
    result_df = own_df[["user_id", "date", TARGET_COL]].copy()
    result_df["pred"] = own_pred

    print("\nOwn prediction sample:")
    print(result_df.head(10))

if __name__ == "__main__":
    main()