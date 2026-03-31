import os
import pandas as pd

RAW_DIR = "ml_XGboost/data/raw"
PROCESSED_DIR = "ml_XGboost/data/processed"
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "kaggle_processed_common.csv")

def main():
    train = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
    features = pd.read_csv(os.path.join(RAW_DIR, "features.csv"))
    stores = pd.read_csv(os.path.join(RAW_DIR, "stores.csv"))

    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")

    df["Date"] = pd.to_datetime(df["Date"])
    df["user_id"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)

    df = df.sort_values(["user_id", "Date"]).reset_index(drop=True)

    # 對齊 own data 欄位命名
    df["date"] = df["Date"]
    df["daily_expense"] = df["Weekly_Sales"]
    df["daily_income"] = 0
    df["txn_count"] = 1
    df["daily_net"] = -df["Weekly_Sales"]

    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month

    df["expense_7d_sum"] = (
        df.groupby("user_id")["daily_expense"]
        .transform(lambda x: x.rolling(7).sum())
    )
    df["expense_7d_mean"] = (
        df.groupby("user_id")["daily_expense"]
        .transform(lambda x: x.rolling(7).mean())
    )
    df["expense_30d_sum"] = (
        df.groupby("user_id")["daily_expense"]
        .transform(lambda x: x.rolling(30).sum())
    )
    df["expense_30d_mean"] = (
        df.groupby("user_id")["daily_expense"]
        .transform(lambda x: x.rolling(30).mean())
    )

    # 對齊 own data target 名稱
    df["target"] = (
        df.groupby("user_id")["daily_expense"]
        .transform(lambda x: x.shift(-1).rolling(7).sum())
    )

    keep_cols = [
        "user_id", "date",
        "daily_expense", "daily_income", "txn_count", "daily_net",
        "dow", "is_weekend", "day", "month",
        "expense_7d_sum", "expense_7d_mean",
        "expense_30d_sum", "expense_30d_mean",
        "target"
    ]

    df = df[keep_cols].dropna().reset_index(drop=True)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("處理完成")
    print("輸出:", OUTPUT_PATH)
    print("shape:", df.shape)
    print(df.head())

if __name__ == "__main__":
    main()