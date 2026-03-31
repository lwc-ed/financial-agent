import os
import pandas as pd

RAW_DIR = "ml_XGboost/data/raw"
PROCESSED_DIR = "ml_XGboost/data/processed"

def main():
    # 1. 讀取資料
    train = pd.read_csv(os.path.join(RAW_DIR, "train.csv"))
    features = pd.read_csv(os.path.join(RAW_DIR, "features.csv"))
    stores = pd.read_csv(os.path.join(RAW_DIR, "stores.csv"))

    print("原始 train shape:", train.shape)
    print("原始 features shape:", features.shape)
    print("原始 stores shape:", stores.shape)

    # 2. 合併資料
    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")

    print("合併後 shape:", df.shape)

    # 3. 日期轉型
    df["Date"] = pd.to_datetime(df["Date"])

    # 4. 建立 user_id
    df["user_id"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)

    # 5. 欄位重新命名
    df = df.rename(columns={"Weekly_Sales": "weekly_expense"})

    # 6. 排序
    df = df.sort_values(["user_id", "Date"]).reset_index(drop=True)

    # 7. 基本時間特徵
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["dayofweek"] = df["Date"].dt.dayofweek

    # 8. 布林轉數值
    df["IsHoliday"] = df["IsHoliday"].astype(int)

    # 9. Type 類別轉數值
    if "Type" in df.columns:
        df["Type"] = df["Type"].astype("category").cat.codes

    # 10. MarkDown 缺失值補 0
    markdown_cols = [col for col in df.columns if col.startswith("MarkDown")]
    for col in markdown_cols:
        df[col] = df[col].fillna(0)

    # 11. 其他常見欄位缺失值處理
    fill_zero_cols = ["CPI", "Unemployment", "Fuel_Price", "Temperature"]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 12. Rolling features
    df["expense_4w_sum"] = (
        df.groupby("user_id")["weekly_expense"]
        .transform(lambda x: x.rolling(4).sum())
    )

    df["expense_4w_mean"] = (
        df.groupby("user_id")["weekly_expense"]
        .transform(lambda x: x.rolling(4).mean())
    )

    df["expense_12w_sum"] = (
        df.groupby("user_id")["weekly_expense"]
        .transform(lambda x: x.rolling(12).sum())
    )

    df["expense_12w_mean"] = (
        df.groupby("user_id")["weekly_expense"]
        .transform(lambda x: x.rolling(12).mean())
    )

    # 13. 建立 target：預測未來 4 週總額
    df["target"] = (
        df.groupby("user_id")["weekly_expense"]
        .transform(lambda x: x.shift(-1).rolling(4).sum())
    )

    # 14. 去除因 rolling / target 產生的 NA
    df = df.dropna().reset_index(drop=True)

    # 15. 建立輸出資料夾
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 16. 存檔
    output_path = os.path.join(PROCESSED_DIR, "kaggle_processed.csv")
    df.to_csv(output_path, index=False)

    print("處理完成")
    print("輸出路徑:", output_path)
    print("處理後 shape:", df.shape)
    print("欄位如下：")
    print(df.columns.tolist())
    print(df.head())

if __name__ == "__main__":
    main()