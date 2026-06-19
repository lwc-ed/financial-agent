"""
no_alignment_utils.py
=====================
對照組：完全沒有 Domain Alignment 的「原始絕對尺度」特徵。
與 bigru_TL_alignment/alignment_utils.py 的 compute_aligned_features 平行對照：
  - aligned 版：z-score / 百分位 / sin-cos → domain-invariant（抹掉絕對金額）
  - raw 版（本檔）：保留每日支出的「絕對金額」與其滾動統計 → domain gap 會顯現

用途：產生 source(IBM) 與 target(個人) 在「對齊前 vs 對齊後」的特徵分布，
      證明三層 Domain Alignment 確實縮小了 source/target 分布差異。
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ALIGN_DIR = BASE_DIR.parent / "ml_ibm" / "bigru_TL_alignment"
IBM_DAILY_PATH = BASE_DIR.parent / "ml_ibm" / "processed_data" / "artifacts" / "ibm_daily.csv"

INPUT_DAYS = 30
TARGET_COL = "future_expense_7d_sum"

# 原始（未對齊）特徵：刻意保留絕對金額尺度，與 aligned 的 10 維一一對照
RAW_FEATURE_COLS = [
    "daily_expense",   # 對照 zscore_7d   ：直接看絕對金額，不做標準化
    "roll7_mean",      # 對照 zscore_14d  ：7 天滾動平均（絕對值）
    "roll14_mean",     # 對照 zscore_30d  ：14 天滾動平均（絕對值）
    "roll30_mean",     # 對照 pct_rank_30d：30 天滾動平均（絕對值）
    "roll7_std",       # 對照 volatility_7d：7 天滾動標準差（絕對值，未除以 mean）
    "roll30_std",      # 對照 pct_rank_7d ：30 天滾動標準差（絕對值）
    "diff_1d",         # 對照 pct_change_norm：日變化量（絕對值，未除以 mean）
    "roll7_sum",       # 對照 is_above_mean_30d：7 天滾動總和（絕對值）
    "dow_sin",         # 與 aligned 相同（時間特徵本就 domain-invariant）
    "dow_cos",         # 與 aligned 相同
]


def compute_raw_features(series: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """完全不做 Domain Alignment：直接用絕對金額與其滾動統計。"""
    s = series.reset_index(drop=True).astype(float)
    d = pd.to_datetime(dates).reset_index(drop=True)

    roll7_mean = s.rolling(7, min_periods=1).mean()
    roll14_mean = s.rolling(14, min_periods=1).mean()
    roll30_mean = s.rolling(30, min_periods=1).mean()
    roll7_std = s.rolling(7, min_periods=2).std().fillna(0)
    roll30_std = s.rolling(30, min_periods=2).std().fillna(0)
    diff_1d = s.diff().fillna(0)
    roll7_sum = s.rolling(7, min_periods=1).sum()

    dow = d.dt.dayofweek
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    return pd.DataFrame(
        {
            "daily_expense": s.values,
            "roll7_mean": roll7_mean.values,
            "roll14_mean": roll14_mean.values,
            "roll30_mean": roll30_mean.values,
            "roll7_std": roll7_std.values,
            "roll30_std": roll30_std.values,
            "diff_1d": diff_1d.values,
            "roll7_sum": roll7_sum.values,
            "dow_sin": dow_sin.values,
            "dow_cos": dow_cos.values,
        }
    )


def load_ibm_daily_sample(n_users: int = 200) -> pd.DataFrame:
    """分塊讀取 1.6GB 的 ibm_daily.csv，只取前 n_users 位（足以呈現分布差異）。"""
    if not IBM_DAILY_PATH.exists():
        raise FileNotFoundError(f"找不到 ibm_daily.csv：{IBM_DAILY_PATH}")

    collected, seen_users = [], set()
    for chunk in pd.read_csv(IBM_DAILY_PATH, parse_dates=["date"], chunksize=500_000):
        for uid, grp in chunk.groupby("user_id"):
            if uid not in seen_users and len(seen_users) >= n_users:
                continue
            seen_users.add(uid)
            collected.append(grp)
        if len(seen_users) >= n_users:
            break

    df = pd.concat(collected, ignore_index=True)
    df = df[df["user_id"].isin(sorted(seen_users)[:n_users])]
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    print(f"[INFO] IBM 抽樣：{df['user_id'].nunique()} 位用戶、{len(df):,} 筆日資料")
    return df
