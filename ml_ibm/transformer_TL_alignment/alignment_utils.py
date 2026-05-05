"""
alignment_utils.py
==================
共用工具函式（BiGRU TL：10 domain-invariant aligned features）
  - compute_aligned_features : 計算 10 個 aligned 特徵
  - load_walmart_daily       : 讀取 Walmart 展開成日資料
  - load_personal_daily      : 讀取個人資料彙整成日資料
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
WALMART_DIR = BASE_DIR.parent / "walmart"
PERSONAL_DIR = BASE_DIR.parent / "data"
EXCLUDE_USERS = ["user4", "user5", "user6"]

ALIGNED_FEATURE_COLS = [
    "zscore_7d",
    "zscore_14d",
    "zscore_30d",
    "pct_change_norm",
    "volatility_7d",
    "is_above_mean_30d",
    "pct_rank_7d",
    "pct_rank_30d",
    "dow_sin",
    "dow_cos",
]
TARGET_COL = "future_expense_7d_sum"
INPUT_DAYS = 30


def compute_aligned_features(series: pd.Series, dates: pd.Series) -> pd.DataFrame:
    eps = 1e-6
    s = series.reset_index(drop=True).astype(float)
    d = pd.to_datetime(dates).reset_index(drop=True)

    roll7_mean = s.rolling(7, min_periods=1).mean()
    roll7_std = s.rolling(7, min_periods=2).std().fillna(0)
    roll14_mean = s.rolling(14, min_periods=1).mean()
    roll14_std = s.rolling(14, min_periods=2).std().fillna(0)
    roll30_mean = s.rolling(30, min_periods=1).mean()
    roll30_std = s.rolling(30, min_periods=2).std().fillna(0)

    zscore_7d = ((s - roll7_mean) / (roll7_std + eps)).clip(-5, 5).fillna(0)
    zscore_14d = ((s - roll14_mean) / (roll14_std + eps)).clip(-5, 5).fillna(0)
    zscore_30d = ((s - roll30_mean) / (roll30_std + eps)).clip(-5, 5).fillna(0)
    pct_change_norm = (s.diff().fillna(0) / (roll30_mean + eps)).clip(-3, 3)
    volatility_7d = (roll7_std / (roll7_mean + eps)).clip(0, 5)
    is_above_mean_30d = (s > roll30_mean).astype(float)
    pct_rank_7d = s.rolling(7, min_periods=1).rank(pct=True)
    pct_rank_30d = s.rolling(30, min_periods=1).rank(pct=True)

    dow = d.dt.dayofweek
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    return pd.DataFrame(
        {
            "zscore_7d": zscore_7d.values,
            "zscore_14d": zscore_14d.values,
            "zscore_30d": zscore_30d.values,
            "pct_change_norm": pct_change_norm.values,
            "volatility_7d": volatility_7d.values,
            "is_above_mean_30d": is_above_mean_30d.values,
            "pct_rank_7d": pct_rank_7d.values,
            "pct_rank_30d": pct_rank_30d.values,
            "dow_sin": dow_sin.values,
            "dow_cos": dow_cos.values,
        }
    )


def load_walmart_daily() -> pd.DataFrame:
    train = pd.read_csv(WALMART_DIR / "train.csv")
    train["Date"] = pd.to_datetime(train["Date"])

    store_weekly = (
        train.groupby(["Store", "Date"])["Weekly_Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Store": "store_id", "Date": "week_start", "Weekly_Sales": "weekly_sales"})
    )
    store_weekly = store_weekly.sort_values(["store_id", "week_start"]).reset_index(drop=True)

    rows = []
    for _, row in store_weekly.iterrows():
        daily_sales = row["weekly_sales"] / 7.0
        for offset in range(7):
            rows.append(
                {
                    "store_id": row["store_id"],
                    "date": row["week_start"] + pd.Timedelta(days=offset),
                    "daily_expense": daily_sales,
                }
            )

    return pd.DataFrame(rows).sort_values(["store_id", "date"]).reset_index(drop=True)


def load_personal_daily() -> pd.DataFrame:
    all_rows = []
    for file_path in sorted(PERSONAL_DIR.glob("raw_transactions_*.xlsx")):
        user_id = file_path.stem.replace("raw_transactions_", "")
        if user_id in EXCLUDE_USERS:
            continue

        df = pd.read_excel(file_path)
        df["time_stamp"] = pd.to_datetime(df["time_stamp"]).dt.normalize()
        df = df[df["transaction_type"] == "Expense"].copy()

        daily = (
            df.groupby("time_stamp")["amount"]
            .sum()
            .reset_index()
            .rename(columns={"time_stamp": "date", "amount": "daily_expense"})
        )

        date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = daily.set_index("date").reindex(date_range, fill_value=0).reset_index()
        daily.columns = ["date", "daily_expense"]
        daily["user_id"] = user_id
        all_rows.append(daily)

    if not all_rows:
        raise FileNotFoundError(f"找不到可用的個人交易資料：{PERSONAL_DIR}")

    return pd.concat(all_rows, ignore_index=True).sort_values(["user_id", "date"]).reset_index(drop=True)
