"""
alignment_utils.py
==================
兩個 dataset 共用的工具函式：
  - compute_aligned_features : 計算 domain-invariant rolling z-score 特徵
  - load_walmart_daily        : 讀取 Walmart 並展開成日資料
  - load_personal_daily       : 讀取個人資料並彙整成日資料
"""

import numpy as np
import pandas as pd
import os

# ── 路徑設定（相對於 ml/ 目錄執行）──────────────────────────────────────────
WALMART_DIR  = "../walmart"
PERSONAL_DIR = "../data"
EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]

# ── 共用特徵欄位名稱 ─────────────────────────────────────────────────────────
ALIGNED_FEATURE_COLS = [
    "zscore_7d",
    "zscore_30d",
    "pct_change_norm",
    "volatility_7d",
    "is_above_mean_30d",
    "dow_sin",
    "dow_cos",
]
TARGET_COL   = "future_expense_7d_sum"
INPUT_DAYS   = 30


# ─────────────────────────────────────────────────────────────────────────────
# 核心特徵函式：兩個 dataset 共用同一份邏輯
# ─────────────────────────────────────────────────────────────────────────────
def compute_aligned_features(series: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """
    輸入：任意消費時間序列（已按時間排序）
    輸出：7個 domain-invariant 特徵

    這個函式對 Walmart 和個人資料完全一樣，
    讓 GRU 在兩個 domain 學到「相對消費動態」而非「絕對數值」。
    """
    eps = 1e-6
    s   = series.reset_index(drop=True).astype(float)
    d   = pd.to_datetime(dates).reset_index(drop=True)

    # ── 滾動統計 ──────────────────────────────────────────────────────────────
    roll7_mean  = s.rolling(7,  min_periods=1).mean()
    roll7_std   = s.rolling(7,  min_periods=2).std().fillna(0)
    roll30_mean = s.rolling(30, min_periods=1).mean()
    roll30_std  = s.rolling(30, min_periods=2).std().fillna(0)

    # 1. 7日滾動 Z-score：「今天比最近一週的均值高/低幾個標準差」
    zscore_7d = ((s - roll7_mean) / (roll7_std + eps)).clip(-5, 5).fillna(0)

    # 2. 30日滾動 Z-score：「今天比最近一個月的均值高/低幾個標準差」
    zscore_30d = ((s - roll30_mean) / (roll30_std + eps)).clip(-5, 5).fillna(0)

    # 3. 變化率（正規化）：用30日均值當分母，消除尺度影響
    raw_diff = s.diff().fillna(0)
    pct_change_norm = (raw_diff / (roll30_mean + eps)).clip(-3, 3)

    # 4. 7日波動率（變異係數）：捕捉消費穩定/不穩定程度
    volatility_7d = (roll7_std / (roll7_mean + eps)).clip(0, 5)

    # 5. 是否高於30日均值（binary）
    is_above_mean_30d = (s > roll30_mean).astype(float)

    # 6 & 7. 星期幾週期編碼（Walmart 週資料展開後也保留此資訊）
    dow     = d.dt.dayofweek  # 0=Mon, 6=Sun
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    return pd.DataFrame({
        "zscore_7d"       : zscore_7d.values,
        "zscore_30d"      : zscore_30d.values,
        "pct_change_norm" : pct_change_norm.values,
        "volatility_7d"   : volatility_7d.values,
        "is_above_mean_30d": is_above_mean_30d.values,
        "dow_sin"         : dow_sin.values,
        "dow_cos"         : dow_cos.values,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 載入 Walmart 日資料
# ─────────────────────────────────────────────────────────────────────────────
def load_walmart_daily() -> pd.DataFrame:
    """
    讀取 Walmart train.csv，彙整成 store-level 日資料（weekly ÷ 7）
    回傳 columns: [store_id, date, daily_expense]
    """
    train = pd.read_csv(f"{WALMART_DIR}/train.csv")
    train["Date"] = pd.to_datetime(train["Date"])

    # 彙整：每家店每週的總銷售額
    store_weekly = (
        train.groupby(["Store", "Date"])["Weekly_Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Store": "store_id", "Date": "week_start", "Weekly_Sales": "weekly_sales"})
    )
    store_weekly = store_weekly.sort_values(["store_id", "week_start"]).reset_index(drop=True)

    # 展開：每週 → 7天（每天 = 週銷售 ÷ 7）
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
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 載入個人日資料
# ─────────────────────────────────────────────────────────────────────────────
def load_personal_daily() -> pd.DataFrame:
    """
    讀取所有用戶的 Excel，彙整成 user-level 日資料
    回傳 columns: [user_id, date, daily_expense]
    """
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

        # 彙整成日級別
        daily = (
            df.groupby("time_stamp")["amount"]
            .sum()
            .reset_index()
            .rename(columns={"time_stamp": "date", "amount": "daily_expense"})
        )

        # 補齊缺失的日期（沒有消費的天 = 0）
        date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = daily.set_index("date").reindex(date_range, fill_value=0).reset_index()
        daily.columns = ["date", "daily_expense"]
        daily["user_id"] = user_id

        all_rows.append(daily)

    df = pd.concat(all_rows).reset_index(drop=True)
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    return df
