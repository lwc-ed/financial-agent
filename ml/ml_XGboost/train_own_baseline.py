import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

OWN_PATH = "ml_XGboost/data/processed/own_processed_common.csv"

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
    df = pd.read_csv(OWN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    print("Own data shape:", df.shape)
    print("Date range:", df["date"].min(), "->", df["date"].max())

    # 2. 切分資料：70% train / 10% valid / 20% test
    n = len(df)
    train_end = int(n * 0.7)
    valid_end = int(n * 0.8)

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()

    print("Train shape:", train_df.shape)
    print("Valid shape:", valid_df.shape)
    print("Test shape :", test_df.shape)

    # 3. 切 X / y
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    X_valid = valid_df[FEATURE_COLS]
    y_valid = valid_df[TARGET_COL]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    # 4. 建立 baseline model
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

    # 5. 訓練模型（含 validation）
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    # 6. 在 test set 上評估
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print("\n=== Own Baseline Result ===")
    print(f"Test MAE  : {mae:.4f}")
    print(f"Test RMSE : {rmse:.4f}")

    # 7. 顯示部分預測結果
    result_df = test_df[["user_id", "date", TARGET_COL]].copy()
    result_df["pred"] = pred

    print("\nPrediction sample:")
    print(result_df.head(10))

if __name__ == "__main__":
    main()