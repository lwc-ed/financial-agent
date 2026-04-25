"""
build_ibm_daily.py
==================
將 IBM Credit Card Transactions (ealtman2019/credit-card-transactions)
從 transaction-level 聚合成 per-user 日粒度資料。

輸入：ml_ibm/processed_data/raw/transactions.csv
輸出：ml_ibm/processed_data/artifacts/ibm_daily.csv

欄位說明（輸出）：
  user_id        int    IBM 原始 User 欄位（0~1999）
  date           date   交易日期
  daily_expense  float  當日所有有效支出總和（負值代表退款，clip at 0）
  daily_refund   float  當日退款總額（abs）
  txn_count      int    當日有效交易筆數（不含 fraud、不含 error）
  raw_txn_count  int    當日全部交易筆數（含 error，不含 fraud）

處理規則：
  - Is Fraud? == "Yes" → 一律排除
  - Errors? != "" → 排除（declined / glitch 交易）
  - Amount 去掉 $ 轉 float
  - Amount < 0 → 退款，計入 daily_refund，daily_expense 不計負值
  - 補齊每位 user 的連續日期（missing date 填 0）
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = RAW_DIR / "transactions.csv"
OUTPUT_PATH = ARTIFACTS_DIR / "ibm_daily.csv"


def parse_amount(s: pd.Series) -> pd.Series:
    """將 '$134.09' / '$-99.00' 轉換為 float。"""
    return s.str.replace("$", "", regex=False).astype(float)


def load_raw(path: Path) -> pd.DataFrame:
    print(f"[INFO] 讀取原始資料: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"[INFO] 原始筆數: {len(df):,}，用戶數: {df['User'].nunique():,}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 建立 date 欄位
    df = df.copy()
    df["date"] = pd.to_datetime(
        df["Year"].astype(str) + "-"
        + df["Month"].astype(str).str.zfill(2) + "-"
        + df["Day"].astype(str).str.zfill(2)
    )

    # 2. 解析金額
    df["amount"] = parse_amount(df["Amount"])

    # 3. 排除詐騙交易
    before = len(df)
    df = df[df["Is Fraud?"].str.strip().str.upper() == "NO"].copy()
    print(f"[INFO] 排除詐騙：{before - len(df):,} 筆 → 剩 {len(df):,} 筆")

    # 4. 排除有錯誤的交易（Insufficient Balance / Technical Glitch 等）
    before = len(df)
    df = df[df["Errors?"].isna() | (df["Errors?"].str.strip() == "")].copy()
    print(f"[INFO] 排除 error 交易：{before - len(df):,} 筆 → 剩 {len(df):,} 筆")

    # 5. 重新命名 user_id
    df = df.rename(columns={"User": "user_id"})

    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] 開始日粒度聚合...")

    # 正值 = 支出；負值 = 退款
    df["expense"] = df["amount"].clip(lower=0)
    df["refund"] = (-df["amount"]).clip(lower=0)

    daily = (
        df.groupby(["user_id", "date"])
        .agg(
            daily_expense=("expense", "sum"),
            daily_refund=("refund", "sum"),
            txn_count=("expense", "count"),      # 有效（非負）交易筆數
            raw_txn_count=("amount", "count"),   # 所有交易筆數
        )
        .reset_index()
    )

    # 退款抵扣（但不讓 daily_expense 變負）
    daily["daily_expense"] = (daily["daily_expense"] - daily["daily_refund"]).clip(lower=0)

    return daily


def fill_date_gaps(daily: pd.DataFrame) -> pd.DataFrame:
    """補齊每位 user 的連續日期，missing date 填 0。"""
    print("[INFO] 補齊連續日期...")
    parts = []
    for uid, grp in daily.groupby("user_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        date_range = pd.date_range(grp["date"].min(), grp["date"].max(), freq="D")
        grp = (
            grp.set_index("date")
            .reindex(date_range, fill_value=0)
            .reset_index()
            .rename(columns={"index": "date"})
        )
        grp["user_id"] = uid
        parts.append(grp)

    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(["user_id", "date"]).reset_index(drop=True)
    return result


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"找不到原始資料：{INPUT_PATH}\n"
            "請先下載 Kaggle ealtman2019/credit-card-transactions\n"
            "並將 transactions.csv 放到 ml_ibm/processed_data/raw/ 資料夾"
        )

    df_raw = load_raw(INPUT_PATH)
    df = preprocess(df_raw)
    daily = aggregate_daily(df)
    daily = fill_date_gaps(daily)

    daily.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[OK] 儲存至 {OUTPUT_PATH}")
    print(f"[STATS] 總筆數: {len(daily):,}，用戶數: {daily['user_id'].nunique():,}")
    print(f"[STATS] 日期範圍: {daily['date'].min().date()} ~ {daily['date'].max().date()}")
    print(f"[STATS] daily_expense 描述:")
    print(daily["daily_expense"].describe())


if __name__ == "__main__":
    main()
