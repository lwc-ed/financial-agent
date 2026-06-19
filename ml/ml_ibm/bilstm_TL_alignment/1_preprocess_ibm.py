"""
Step 1：IBM Credit Card Transactions 前處理（取代 1_preprocess_walmart.py）
=========================================================================
讀取 ibm_daily.csv → 組合 10 個 aligned 特徵 → 建立 3D 滑動視窗 → 儲存 .npy

輸入：ml_ibm/processed_data/artifacts/ibm_daily.csv
輸出：ml_ibm/bilstm_TL_alignment/artifacts_bilstm_v2/
  ibm_X_train.npy  ibm_y_train.npy
  ibm_X_val.npy    ibm_y_val.npy
  ibm_X_test.npy   ibm_y_test.npy
  ibm_target_scaler.pkl

特徵來源（10 個，對應 ALIGNED_FEATURE_COLS）：
  直接從 ibm_daily.csv 拿：zscore_7d / zscore_14d / zscore_30d
  從 dow 欄位計算：       dow_sin / dow_cos
  從 daily_expense 計算： pct_change_norm / volatility_7d / is_above_mean_30d
                          pct_rank_7d / pct_rank_30d
  （後 5 個為相對特徵，在 log1p 空間計算不影響 domain-invariant 性質）

切分：全體樣本依時間排序後 70/15/15（非 per-user，pretrain 不需要）
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts_bilstm_v2"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MY_DIR))
from alignment_utils import ALIGNED_FEATURE_COLS, INPUT_DAYS, TARGET_COL

IBM_DAILY_PATH = MY_DIR.parent / "processed_data" / "artifacts" / "ibm_daily.csv"

N_SOURCE_USERS = 1000
MIN_DAYS_PER_USER = INPUT_DAYS + 7 + 10  # 30 + 7 + 10 緩衝


def load_ibm_daily() -> pd.DataFrame:
    if not IBM_DAILY_PATH.exists():
        raise FileNotFoundError(
            f"找不到 ibm_daily.csv：{IBM_DAILY_PATH}\n"
            "請先執行 ml_ibm/processed_data/build_ibm_daily.py"
        )
    df = pd.read_csv(IBM_DAILY_PATH, parse_dates=["date"])
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    all_users = sorted(df["user_id"].unique())
    selected_users = all_users[:N_SOURCE_USERS]
    df = df[df["user_id"].isin(selected_users)].reset_index(drop=True)

    print(f"[INFO] ibm_daily 載入：{len(df):,} 筆，使用 {df['user_id'].nunique():,} 位用戶（共 {len(all_users):,} 位）")
    return df


def build_features(grp: pd.DataFrame) -> pd.DataFrame:
    """組合 10 個 aligned 特徵，保證欄位順序與 ALIGNED_FEATURE_COLS 一致。"""
    eps = 1e-6
    s = grp["daily_expense"].reset_index(drop=True)  # log1p 空間

    roll7_mean  = s.rolling(7,  min_periods=1).mean()
    roll7_std   = s.rolling(7,  min_periods=2).std().fillna(0)
    roll30_mean = s.rolling(30, min_periods=1).mean()

    dow = grp["dow"].reset_index(drop=True)

    return pd.DataFrame({
        # 直接從 ibm_daily.csv 拿
        "zscore_7d"        : grp["zscore_7d"].values,
        "zscore_14d"       : grp["zscore_14d"].values,
        "zscore_30d"       : grp["zscore_30d"].values,
        # 從 daily_expense（log1p）計算，均為相對特徵
        "pct_change_norm"  : (s.diff().fillna(0) / (roll30_mean + eps)).clip(-3, 3).values,
        "volatility_7d"    : (roll7_std / (roll7_mean + eps)).clip(0, 5).values,
        "is_above_mean_30d": (s > roll30_mean).astype(float).values,
        "pct_rank_7d"      : s.rolling(7,  min_periods=1).rank(pct=True).values,
        "pct_rank_30d"     : s.rolling(30, min_periods=1).rank(pct=True).values,
        # 從 dow 計算
        "dow_sin"          : np.sin(2 * np.pi * dow / 7).values,
        "dow_cos"          : np.cos(2 * np.pi * dow / 7).values,
    })[ALIGNED_FEATURE_COLS]


def build_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    skipped = 0

    for uid, grp in df.groupby("user_id"):
        grp = grp.sort_values("date").reset_index(drop=True)

        if len(grp) < MIN_DAYS_PER_USER:
            skipped += 1
            continue

        feats      = build_features(grp)
        feat_arr   = feats.values.astype(np.float32)
        target_arr = grp["target"].values.astype(np.float32)

        for t in range(INPUT_DAYS, len(grp)):
            if np.isnan(target_arr[t]):
                continue
            X_list.append(feat_arr[t - INPUT_DAYS : t])
            y_list.append([target_arr[t]])

    print(f"[INFO] 跳過資料不足的用戶：{skipped} 位")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def main():
    print(f"📂 工作目錄：{MY_DIR}")

    df = load_ibm_daily()

    print(f"\n📊 組合 {len(ALIGNED_FEATURE_COLS)} 個 aligned 特徵並建立滑動視窗...")
    X, y = build_windows(df)
    print(f"[INFO] 全部視窗數：{len(X):,}，shape: {X.shape}")

    train_end = int(len(X) * 0.70)
    val_end   = int(len(X) * 0.85)
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    print(f"[INFO] Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    print("\n📐 標準化 IBM Target...")
    target_scaler = StandardScaler()
    target_scaler.fit(y_train)
    y_train_s = target_scaler.transform(y_train).astype(np.float32)
    y_val_s   = target_scaler.transform(y_val).astype(np.float32)
    y_test_s  = target_scaler.transform(y_test).astype(np.float32)

    print(f"\n💾 儲存至 {ARTIFACTS_DIR}...")
    np.save(ARTIFACTS_DIR / "ibm_X_train.npy", X_train)
    np.save(ARTIFACTS_DIR / "ibm_y_train.npy", y_train_s)
    np.save(ARTIFACTS_DIR / "ibm_X_val.npy",   X_val)
    np.save(ARTIFACTS_DIR / "ibm_y_val.npy",   y_val_s)
    np.save(ARTIFACTS_DIR / "ibm_X_test.npy",  X_test)
    np.save(ARTIFACTS_DIR / "ibm_y_test.npy",  y_test_s)
    with open(ARTIFACTS_DIR / "ibm_target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    print("\n✅ IBM 前處理完成！")
    print(f"   y_train mean={y_train.mean():.4f}  std={y_train.std():.4f}")


if __name__ == "__main__":
    main()
