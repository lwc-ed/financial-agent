import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROCESSED_PATH = "ml_XGboost/data/processed/kaggle_processed.csv"
MODEL_DIR = "ml_XGboost/models"

def main():
    # 1. 讀資料
    df = pd.read_csv(PROCESSED_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    print("資料 shape:", df.shape)
    print("日期範圍:", df["Date"].min(), "->", df["Date"].max())

    # 2. 時間切分
    # 這裡先用簡單版本：2012-01-01 當切點
    split_date = pd.Timestamp("2012-01-01")

    train_df = df[df["Date"] < split_date].copy()
    valid_df = df[df["Date"] >= split_date].copy()

    print("train shape:", train_df.shape)
    print("valid shape:", valid_df.shape)

    # 3. 選特徵欄位
    feature_cols = [
        "IsHoliday",
        "Temperature",
        "Fuel_Price",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
        "CPI",
        "Unemployment",
        "Type",
        "Size",
        "year",
        "month",
        "week",
        "dayofweek",
        "expense_4w_sum",
        "expense_4w_mean",
        "expense_12w_sum",
        "expense_12w_mean",
    ]

    target_col = "target"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col]

    # 4. 建立模型
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

    # 5. 訓練模型
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=True
    )

    # 6. 預測
    pred = model.predict(X_valid)

    # 7. 評估
    mae = mean_absolute_error(y_valid, pred)
    rmse = np.sqrt(mean_squared_error(y_valid, pred))

    print("\n=== Kaggle Base Model Result ===")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    # 8. 建立模型資料夾
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 9. 存模型
    model_path = os.path.join(MODEL_DIR, "xgb_kaggle_base.json")
    model.save_model(model_path)
    print(f"模型已儲存到: {model_path}")

if __name__ == "__main__":
    main()