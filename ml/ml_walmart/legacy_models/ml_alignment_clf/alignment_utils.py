"""
alignment_utils.py  （分類任務版）
====================================
12 個 domain-invariant 特徵：
  - 10 個原始 aligned 特徵（與 lwc/v2 相同）
  - 2 個新增 spike 偵測特徵：
      spike_ratio   : max_7d / mean_7d（峰均比，捕捉短期尖峰）
      max_ratio_30d : max_7d / max_30d（近期最高點相對30天最高點的比例）

任務目標：binary classification
  label = (future_7d_sum > roll_30d_mean * 7 * ALERT_RATIO)
"""

import numpy as np
import pandas as pd
import os

WALMART_DIR   = "../walmart"
PERSONAL_DIR  = "../data"
EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]

ALERT_RATIO = 1.5      # 預警倍率（與 regression 版相同，方便比較）
INPUT_DAYS  = 30

ALIGNED_FEATURE_COLS = [
    # ── 原本 10 個 aligned 特徵 ──────────────────────────────────────────────
    "zscore_7d",
    "zscore_14d",
    "zscore_30d",
    "pct_change_norm",
    "volatility_7d",
    "is_above_mean_30d",
    "pct_rank_7d",
    "pct_rank_30d",
    # ── 新增：spike 偵測特徵 ─────────────────────────────────────────────────
    "spike_ratio",      # max_7d / mean_7d：峰均比，值 > 1 代表有尖峰
    "max_ratio_30d",    # max_7d / max_30d：近期最高點佔30日最高點的比例
    # ── 時間週期 ─────────────────────────────────────────────────────────────
    "dow_sin",
    "dow_cos",
]
TARGET_COL = "alert_label"   # 分類標籤（0/1）


def compute_aligned_features(series: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """
    計算 12 個 domain-invariant 特徵（含 2 個 spike 偵測特徵）
    輸入可以是 Walmart 或個人資料，邏輯完全相同。
    """
    eps = 1e-6
    s   = series.reset_index(drop=True).astype(float)
    d   = pd.to_datetime(dates).reset_index(drop=True)

    # ── 滾動統計 ──────────────────────────────────────────────────────────────
    roll7_mean  = s.rolling(7,  min_periods=1).mean()
    roll7_std   = s.rolling(7,  min_periods=2).std().fillna(0)
    roll7_max   = s.rolling(7,  min_periods=1).max()
    roll14_mean = s.rolling(14, min_periods=1).mean()
    roll14_std  = s.rolling(14, min_periods=2).std().fillna(0)
    roll30_mean = s.rolling(30, min_periods=1).mean()
    roll30_std  = s.rolling(30, min_periods=2).std().fillna(0)
    roll30_max  = s.rolling(30, min_periods=1).max()

    # 1. 7日 Z-score
    zscore_7d  = ((s - roll7_mean)  / (roll7_std  + eps)).clip(-5, 5).fillna(0)
    # 2. 14日 Z-score
    zscore_14d = ((s - roll14_mean) / (roll14_std + eps)).clip(-5, 5).fillna(0)
    # 3. 30日 Z-score
    zscore_30d = ((s - roll30_mean) / (roll30_std + eps)).clip(-5, 5).fillna(0)
    # 4. 正規化日環比
    pct_change_norm = (s.diff().fillna(0) / (roll30_mean + eps)).clip(-3, 3)
    # 5. 7日波動率
    volatility_7d   = (roll7_std / (roll7_mean + eps)).clip(0, 5)
    # 6. 是否高於30日均值
    is_above_mean_30d = (s > roll30_mean).astype(float)
    # 7. 近 7 天百分位排名
    pct_rank_7d  = s.rolling(7,  min_periods=1).rank(pct=True)
    # 8. 近 30 天百分位排名
    pct_rank_30d = s.rolling(30, min_periods=1).rank(pct=True)

    # ── 新增 spike 特徵 ───────────────────────────────────────────────────────
    # 9. 峰均比：近7天最高點 / 近7天均值（domain-invariant：比值無單位）
    #    spike_ratio > 1 → 該週有尖峰消費；= 1 → 消費平穩
    spike_ratio   = (roll7_max / (roll7_mean + eps)).clip(1, 5)

    # 10. 近期高點比例：近7天最高點 / 近30天最高點
    #     值接近 1 → 最近出現了30天以來的最高點（預警信號！）
    max_ratio_30d = (roll7_max / (roll30_max + eps)).clip(0, 1)

    # ── 時間週期 ──────────────────────────────────────────────────────────────
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
        "spike_ratio"      : spike_ratio.values,
        "max_ratio_30d"    : max_ratio_30d.values,
        "dow_sin"          : dow_sin.values,
        "dow_cos"          : dow_cos.values,
    })


def compute_alert_label(series: pd.Series, alert_ratio: float = ALERT_RATIO) -> pd.Series:
    """
    計算分類標籤：未來 7 天加總 > 30日均值 × 7 × alert_ratio → 1，否則 → 0
    """
    eps        = 1e-6
    s          = series.reset_index(drop=True).astype(float)
    roll30_mean = s.rolling(30, min_periods=1).mean()
    baseline_7d = roll30_mean * 7
    future_7d   = s.rolling(7).sum().shift(-7)   # 未來 7 天加總
    alert       = (future_7d > baseline_7d * alert_ratio).astype(float)
    return alert, future_7d, baseline_7d


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
    df = pd.DataFrame(rows).sort_values(["store_id", "date"]).reset_index(drop=True)
    return df


def load_personal_daily() -> pd.DataFrame:
    all_rows = []
    for fname in sorted(os.listdir(PERSONAL_DIR)):
        if not fname.endswith(".xlsx"):
            continue
        user_id = fname.replace("raw_transactions_", "").replace(".xlsx", "")
        if user_id in EXCLUDE_USERS:
            continue
        df_raw = pd.read_excel(f"{PERSONAL_DIR}/{fname}")
        df_raw["time_stamp"] = pd.to_datetime(df_raw["time_stamp"]).dt.normalize()
        df_raw = df_raw[df_raw["transaction_type"] == "Expense"].copy()
        daily = (
            df_raw.groupby("time_stamp")["amount"].sum().reset_index()
            .rename(columns={"time_stamp": "date", "amount": "daily_expense"})
        )
        date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = daily.set_index("date").reindex(date_range, fill_value=0).reset_index()
        daily.columns = ["date", "daily_expense"]
        daily["user_id"] = user_id
        all_rows.append(daily)
    df = pd.concat(all_rows).reset_index(drop=True)
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    return df
