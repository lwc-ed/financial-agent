"""
build_ibm_daily.py
==================
將 IBM Credit Card Transactions 從 transaction-level 聚合成 per-user 日粒度資料。

輸入：ml_ibm/ibm_data/credit_card_transactions-ibm_v2.csv
輸出：ml_ibm/processed_data/artifacts/ibm_daily.csv

輸出欄位：
  user_id          int    IBM 原始 User 欄位（0~1999）
  date             date   交易日期
  daily_expense    float  當日支出總和（raw USD）
  daily_income     float  固定為 0（IBM 無收入資料）
  txn_count        int    當日交易筆數
  daily_net        float  = -daily_expense（income=0）
  dow              int    星期幾（0=週一）
  is_weekend       int    0 或 1
  day              int    幾號（1~31）
  month            int    幾月（1~12）
  expense_7d_sum   float  7 日支出總和
  expense_7d_mean  float  7 日支出平均
  expense_30d_sum  float  30 日支出總和
  expense_30d_mean float  30 日支出平均
  zscore_7d        float  7 日 z-score（幣值對齊用）
  zscore_14d       float  14 日 z-score
  zscore_30d       float  30 日 z-score
  target           float  未來 7 天支出總和（raw USD）
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
IBM_DATA_DIR = ROOT.parent / "ibm_data"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = IBM_DATA_DIR / "credit_card_transactions-ibm_v2.csv"
OUTPUT_PATH = ARTIFACTS_DIR / "ibm_daily.csv"

EPS = 1e-6


def parse_amount(s: pd.Series) -> pd.Series:
    """'$134.09' / '$-99.00' → float"""
    return s.str.replace("$", "", regex=False).astype(float)


def compute_zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=1).mean()
    std  = s.rolling(window, min_periods=2).std().fillna(0)
    return ((s - mean) / (std + EPS)).clip(-5, 5).fillna(0)


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """只讀需要的欄位，解析 Amount，建立 date，聚合成日粒度。"""
    chunk["amount"] = parse_amount(chunk["Amount"])
    chunk["date"] = pd.to_datetime(
        chunk["Year"].astype(str) + "-"
        + chunk["Month"].astype(str).str.zfill(2) + "-"
        + chunk["Day"].astype(str).str.zfill(2)
    )
    # clip 負值（退款）不讓日支出變負
    chunk["expense"] = chunk["amount"].clip(lower=0)

    daily = (
        chunk.groupby(["User", "date"])
        .agg(
            daily_expense=("expense", "sum"),
            txn_count=("expense", "count"),
        )
        .reset_index()
        .rename(columns={"User": "user_id"})
    )
    return daily


def fill_date_gaps(daily: pd.DataFrame) -> pd.DataFrame:
    """補齊每位 user 的連續日期，missing date 填 0。"""
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
    return pd.concat(parts, ignore_index=True).sort_values(["user_id", "date"]).reset_index(drop=True)


def compute_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-user 計算所有衍生特徵。"""
    parts = []
    for uid, grp in daily.groupby("user_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        s = grp["daily_expense"]

        # Rolling 特徵（raw 金額空間）
        grp["expense_7d_sum"]  = s.rolling(7,  min_periods=1).sum()
        grp["expense_7d_mean"] = s.rolling(7,  min_periods=1).mean()
        grp["expense_30d_sum"] = s.rolling(30, min_periods=1).sum()
        grp["expense_30d_mean"]= s.rolling(30, min_periods=1).mean()

        # Z-score（抹除 USD/TWD 幣值差異，用 raw 金額計算）
        grp["zscore_7d"]  = compute_zscore(s, 7)
        grp["zscore_14d"] = compute_zscore(s, 14)
        grp["zscore_30d"] = compute_zscore(s, 30)

        # 日期特徵
        grp["dow"]        = grp["date"].dt.dayofweek
        grp["is_weekend"] = (grp["dow"] >= 5).astype(int)
        grp["day"]        = grp["date"].dt.day
        grp["month"]      = grp["date"].dt.month

        # 填 0 的欄位
        grp["daily_income"] = 0.0
        grp["daily_net"]    = -s

        # daily_expense 保持 raw 金額
        grp["daily_expense"] = s

        # Target：未來 7 天支出總和（raw 金額空間）
        grp["target"] = s.rolling(7).sum().shift(-7)

        parts.append(grp)

    return pd.concat(parts, ignore_index=True)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"找不到原始資料：{INPUT_PATH}\n"
            "請將 credit_card_transactions-ibm_v2.csv 放到 ml_ibm/ibm_data/"
        )

    print(f"[INFO] 開始讀取：{INPUT_PATH}")
    print("[INFO] 使用 chunking 讀取（每批 500,000 筆）...")

    USECOLS = ["User", "Year", "Month", "Day", "Amount"]
    chunks = []
    for i, chunk in enumerate(pd.read_csv(INPUT_PATH, usecols=USECOLS, chunksize=500_000)):
        daily_chunk = process_chunk(chunk)
        chunks.append(daily_chunk)
        print(f"  chunk {i+1} 處理完畢，累積 {sum(len(c) for c in chunks):,} 筆日資料")

    print("[INFO] 合併所有 chunk...")
    daily = (
        pd.concat(chunks, ignore_index=True)
        .groupby(["user_id", "date"])
        .agg(daily_expense=("daily_expense", "sum"),
             txn_count=("txn_count", "sum"))
        .reset_index()
    )
    print(f"[INFO] 聚合後：{len(daily):,} 筆，{daily['user_id'].nunique():,} 位用戶")

    print("[INFO] 補齊連續日期...")
    daily = fill_date_gaps(daily)

    print("[INFO] 計算衍生特徵...")
    daily = compute_features(daily)

    # 去除 target 為 NaN 的列（每位 user 最後 6 天）
    daily = daily.dropna(subset=["target"]).reset_index(drop=True)

    # 整理欄位順序
    cols = [
        "user_id", "date",
        "daily_expense", "daily_income", "txn_count", "daily_net",
        "dow", "is_weekend", "day", "month",
        "expense_7d_sum", "expense_7d_mean", "expense_30d_sum", "expense_30d_mean",
        "zscore_7d", "zscore_14d", "zscore_30d",
        "target",
    ]
    daily = daily[cols]

    daily.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[OK] 儲存至 {OUTPUT_PATH}")
    print(f"[STATS] 總筆數：{len(daily):,}")
    print(f"[STATS] 用戶數：{daily['user_id'].nunique():,}")
    print(f"[STATS] 日期範圍：{daily['date'].min()} ~ {daily['date'].max()}")
    print(f"\n[STATS] daily_expense（raw USD）描述：")
    print(daily["daily_expense"].describe())
    print(f"\n[STATS] target（raw USD）描述：")
    print(daily["target"].describe())


if __name__ == "__main__":
    main()
