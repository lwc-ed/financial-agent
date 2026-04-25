"""
alignment_utils.py
==================
共用工具函式（BiLSTM v2：10 domain-invariant aligned features）
  - compute_aligned_features : 計算 10 個 rolling z-score + percentile 特徵
  - load_walmart_daily        : 讀取 Walmart 展開成日資料
  - load_personal_daily       : 讀取個人資料彙整成日資料
"""

import numpy as np
import pandas as pd
import os

# ── 路徑設定（相對於 ml/ 目錄執行）──────────────────────────────────────────
WALMART_DIR   = "../walmart"
PERSONAL_DIR  = "../data"
EXCLUDE_USERS = ["user4", "user5", "user6"]

# ── 共用特徵欄位名稱（10 個）────────────────────────────────────────────────
ALIGNED_FEATURE_COLS = [
    "zscore_7d",
    "zscore_14d",           # 補中間尺度（7d 與 30d 之間）
    "zscore_30d",
    "pct_change_norm",
    "volatility_7d",
    "is_above_mean_30d",
    "pct_rank_7d",          # 近 7 天百分位排名（0~1，天然 domain-invariant）
    "pct_rank_30d",         # 近 30 天百分位排名（0~1，天然 domain-invariant）
    "dow_sin",
    "dow_cos",
]
TARGET_COL  = "future_expense_7d_sum"
INPUT_DAYS  = 30


# ─────────────────────────────────────────────────────────────────────────────
# 核心特徵函式：Walmart 與個人資料共用同一份邏輯
# ─────────────────────────────────────────────────────────────────────────────
def compute_aligned_features(series: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """
    輸入：任意消費時間序列（已按時間排序）
    輸出：10 個 domain-invariant 特徵

    zscore / pct_rank 都是相對統計量，不受幣別或絕對金額影響，
    使 Walmart 與個人資料在特徵空間中更接近（低 MMD）。
    """
    eps = 1e-6
    s   = series.reset_index(drop=True).astype(float)
    d   = pd.to_datetime(dates).reset_index(drop=True)

    # ── 滾動統計 ──────────────────────────────────────────────────────────────
    roll7_mean  = s.rolling(7,  min_periods=1).mean()
    roll7_std   = s.rolling(7,  min_periods=2).std().fillna(0)
    roll14_mean = s.rolling(14, min_periods=1).mean()
    roll14_std  = s.rolling(14, min_periods=2).std().fillna(0)
    roll30_mean = s.rolling(30, min_periods=1).mean()
    roll30_std  = s.rolling(30, min_periods=2).std().fillna(0)

    # 1. 7日滾動 Z-score
    zscore_7d  = ((s - roll7_mean)  / (roll7_std  + eps)).clip(-5, 5).fillna(0)

    # 2. 14日滾動 Z-score（中間尺度）
    zscore_14d = ((s - roll14_mean) / (roll14_std + eps)).clip(-5, 5).fillna(0)

    # 3. 30日滾動 Z-score
    zscore_30d = ((s - roll30_mean) / (roll30_std + eps)).clip(-5, 5).fillna(0)

    # 4. 變化率（正規化）
    pct_change_norm = (s.diff().fillna(0) / (roll30_mean + eps)).clip(-3, 3)

    # 5. 7日波動率（變異係數）
    volatility_7d = (roll7_std / (roll7_mean + eps)).clip(0, 5)

    # 6. 是否高於30日均值（binary）
    is_above_mean_30d = (s > roll30_mean).astype(float)

    # 7. 近 7 天百分位排名（0~1）
    pct_rank_7d  = s.rolling(7,  min_periods=1).rank(pct=True)

    # 8. 近 30 天百分位排名（0~1）
    pct_rank_30d = s.rolling(30, min_periods=1).rank(pct=True)

    # 9 & 10. 星期幾週期編碼
    dow     = d.dt.dayofweek
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    return pd.DataFrame({
        "zscore_7d"        : zscore_7d.values,
        "zscore_14d"       : zscore_14d.values,
        "zscore_30d"       : zscore_30d.values,
        "pct_change_norm"  : pct_change_norm.values,
        "volatility_7d"    : volatility_7d.values,
        "is_above_mean_30d": is_above_mean_30d.values,
        "pct_rank_7d"      : pct_rank_7d.values,
        "pct_rank_30d"     : pct_rank_30d.values,
        "dow_sin"          : dow_sin.values,
        "dow_cos"          : dow_cos.values,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 載入 Walmart 日資料
# ─────────────────────────────────────────────────────────────────────────────
def load_walmart_daily() -> pd.DataFrame:
    train = pd.read_csv(f"{WALMART_DIR}/train.csv")
    train["Date"] = pd.to_datetime(train["Date"])

    store_weekly = (
        train.groupby(["Store", "Date"])["Weekly_Sales"]
        .sum().reset_index()
        .rename(columns={"Store": "store_id", "Date": "week_start", "Weekly_Sales": "weekly_sales"})
    )
    store_weekly = store_weekly.sort_values(["store_id", "week_start"]).reset_index(drop=True)

    rows = []
    for _, row in store_weekly.iterrows():
        daily = row["weekly_sales"] / 7.0
        for d in range(7):
            rows.append({
                "store_id"     : row["store_id"],
                "date"         : row["week_start"] + pd.Timedelta(days=d),
                "daily_expense": daily,
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["store_id", "date"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 載入個人日資料
# ─────────────────────────────────────────────────────────────────────────────
def load_personal_daily() -> pd.DataFrame:
    all_rows = []
    for fname in sorted(os.listdir(PERSONAL_DIR)):
        if not fname.endswith(".xlsx"):
            continue
        user_id = fname.replace("raw_transactions_", "").replace(".xlsx", "")
        if user_id in EXCLUDE_USERS:
            continue

        df = pd.read_excel(f"{PERSONAL_DIR}/{fname}")
        df["time_stamp"] = pd.to_datetime(df["time_stamp"]).dt.normalize()
        df = df[df["transaction_type"] == "Expense"].copy()

        daily = (
            df.groupby("time_stamp")["amount"]
            .sum().reset_index()
            .rename(columns={"time_stamp": "date", "amount": "daily_expense"})
        )

        date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = daily.set_index("date").reindex(date_range, fill_value=0).reset_index()
        daily.columns = ["date", "daily_expense"]
        daily["user_id"] = user_id
        all_rows.append(daily)

    df = pd.concat(all_rows).reset_index(drop=True)
    return df.sort_values(["user_id", "date"]).reset_index(drop=True)
