"""
Step 1：IBM Credit Card Transactions 前處理
=========================================================================
讀取 ibm_daily.csv → 計算 10 個 aligned 特徵 → 建立 3D 滑動視窗 → 儲存 .npy

輸入：ml_ibm/processed_data/artifacts/ibm_daily.csv
輸出：ml_ibm/transformer_TL_alignment/artifacts_transformer_tl/
  ibm_X_train.npy  ibm_y_train.npy
  ibm_X_val.npy    ibm_y_val.npy
  ibm_X_test.npy   ibm_y_test.npy
  ibm_target_scaler.pkl

切分：全體樣本依時間排序後 70/15/15（非 per-user，pretrain 不需要）
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts_transformer_tl"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MY_DIR))
from alignment_utils import ALIGNED_FEATURE_COLS, INPUT_DAYS, TARGET_COL, compute_aligned_features

IBM_DAILY_PATH = MY_DIR.parent / "processed_data" / "artifacts" / "ibm_daily.csv"

N_SOURCE_USERS = 2000       # 先用 100 人測試，最多可到 2000
MIN_DAYS_PER_USER = INPUT_DAYS + 7 + 10


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
    print(f"[INFO] ibm_daily 載入：{len(df):,} 筆，使用 {df['user_id'].nunique():,} 位用戶（共 {len(all_users):,} 位，N_SOURCE_USERS={N_SOURCE_USERS}）")
    return df


def build_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    skipped = 0

    for uid, grp in df.groupby("user_id"):
        grp = grp.sort_values("date").reset_index(drop=True)

        if len(grp) < MIN_DAYS_PER_USER:
            skipped += 1
            continue

        feats = compute_aligned_features(grp["daily_expense"], grp["date"])
        target = grp["daily_expense"].rolling(7).sum().shift(-7)
        feats[TARGET_COL] = target.values

        valid_mask = feats[TARGET_COL].notna()
        feats = feats[valid_mask].reset_index(drop=True)

        feat_arr = feats[ALIGNED_FEATURE_COLS].values.astype(np.float32)
        target_arr = feats[TARGET_COL].values.astype(np.float32)

        for t in range(INPUT_DAYS, len(feats)):
            X_list.append(feat_arr[t - INPUT_DAYS : t])
            y_list.append([target_arr[t]])

    print(f"[INFO] 跳過資料不足的用戶：{skipped} 位")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def main():
    print(f"📂 工作目錄：{MY_DIR}")

    df = load_ibm_daily()

    print(f"\n📊 計算 {len(ALIGNED_FEATURE_COLS)} 個 aligned 特徵並建立滑動視窗...")
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
    print(f"   y_train mean={y_train.mean():.2f}  std={y_train.std():.2f}")


if __name__ == "__main__":
    main()
