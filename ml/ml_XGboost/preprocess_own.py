import os
import glob
import pandas as pd

RAW_DIR = "data"
PROCESSED_DIR = "ml_XGboost/data/processed"
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "own_processed_common.csv")

def main():
    file_paths = glob.glob(os.path.join(RAW_DIR, "*.xlsx"))

    if not file_paths:
        raise FileNotFoundError(f"在 {RAW_DIR} 找不到任何 xlsx 檔案")

    df_list = []

    for path in file_paths:
        df = pd.read_excel(path)
        df.columns = df.columns.str.lower()

        # 從檔名抓 user_id，例如 raw_transactions_user14.xlsx -> 14
        filename = os.path.basename(path)
        user_id = int(filename.split("user")[-1].split(".")[0])
        df["user_id"] = user_id

        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)

    print("原始交易資料 shape:", df_all.shape)
    print(df_all.head())

    # 1. 處理時間欄位
    df_all["time_stamp"] = pd.to_datetime(df_all["time_stamp"])
    df_all["date"] = df_all["time_stamp"].dt.date

    # 2. 統一 transaction_type 格式
    df_all["transaction_type"] = df_all["transaction_type"].astype(str).str.lower().str.strip()

    # 3. 建立收入 / 支出欄位
    df_all["expense"] = df_all.apply(
        lambda row: row["amount"] if row["transaction_type"] == "expense" else 0,
        axis=1
    )

    df_all["income"] = df_all.apply(
        lambda row: row["amount"] if row["transaction_type"] == "income" else 0,
        axis=1
    )

    # 4. daily aggregation
    df_daily = df_all.groupby(["user_id", "date"]).agg(
        daily_expense=("expense", "sum"),
        daily_income=("income", "sum"),
        txn_count=("amount", "count"),
        daily_net=("net_cash_flow", "sum")
    ).reset_index()

    # 5. 日期與排序
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.sort_values(["user_id", "date"]).reset_index(drop=True)

    # 6. 時間特徵
    df_daily["dow"] = df_daily["date"].dt.dayofweek
    df_daily["is_weekend"] = df_daily["dow"].isin([5, 6]).astype(int)
    df_daily["day"] = df_daily["date"].dt.day
    df_daily["month"] = df_daily["date"].dt.month

    # 7. rolling features
    # 重點：加 shift(1)，只使用「昨天以前」的資料，避免看到當天或未來資訊
    df_daily["expense_7d_sum"] = df_daily.groupby("user_id")["daily_expense"].transform(
        lambda x: x.shift(1).rolling(7).sum()
    )

    df_daily["expense_7d_mean"] = df_daily.groupby("user_id")["daily_expense"].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )

    df_daily["expense_30d_sum"] = df_daily.groupby("user_id")["daily_expense"].transform(
        lambda x: x.shift(1).rolling(30).sum()
    )

    df_daily["expense_30d_mean"] = df_daily.groupby("user_id")["daily_expense"].transform(
        lambda x: x.shift(1).rolling(30).mean()
    )

    # 8. target：未來 7 天總支出
    # 這裡也要小心，只能是未來資料
    df_daily["target"] = df_daily.groupby("user_id")["daily_expense"].transform(
        lambda x: x.shift(-1).rolling(7).sum()
    )

    # 9. 去除 rolling 與 target 造成的缺值
    df_daily = df_daily.dropna().reset_index(drop=True)

    # 10. 只保留和 Kaggle common 版一致的欄位
    keep_cols = [
        "user_id",
        "date",
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
        "target"
    ]

    df_daily = df_daily[keep_cols]

    # 11. 輸出
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_daily.to_csv(OUTPUT_PATH, index=False)

    print("\n處理完成")
    print("輸出:", OUTPUT_PATH)
    print("shape:", df_daily.shape)
    print(df_daily.head())

if __name__ == "__main__":
    main()