"""
Step 1：IBM Credit Card Transactions 前處理（取代 1_preprocess_walmart.py）
=========================================================================
讀取 ibm_daily.csv → 計算 10 個 aligned 特徵 → 建立 3D 滑動視窗 → 儲存 .npy

輸入：ml_ibm/processed_data/artifacts/ibm_daily.csv
輸出：ml_ibm/bigru_TL_alignment/artifacts_bigru_tl/
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
ARTIFACTS_DIR = MY_DIR / "artifacts_bigru_tl"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MY_DIR))
from alignment_utils import ALIGNED_FEATURE_COLS, INPUT_DAYS, TARGET_COL, compute_aligned_features

IBM_DAILY_PATH = MY_DIR.parent / "processed_data" / "artifacts" / "ibm_daily.csv"

# 每位 user 最少需要的有效日數（避免滑動視窗太短）
MIN_DAYS_PER_USER = INPUT_DAYS + 7 + 10   # 30 + 7 + 10 緩衝


def load_ibm_daily() -> pd.DataFrame:
    if not IBM_DAILY_PATH.exists():
        raise FileNotFoundError(
            f"找不到 ibm_daily.csv：{IBM_DAILY_PATH}\n"
            "請先執行 ml_ibm/processed_data/build_ibm_daily.py"
        )
    df = pd.read_csv(IBM_DAILY_PATH, parse_dates=["date"])
    
# --- 關鍵修改：從 2000 人降為 1000 人 ---
    target_users = sorted(df['user_id'].unique())[:1000] 
    df = df[df['user_id'].isin(target_users)].reset_index(drop=True)
    print(f"⚠️  防爆記憶體模式：載入 1000 位用戶資料")
    # --------------------------------------------

    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    print(f"[INFO] ibm_daily 載入：{len(df):,} 筆，{df['user_id'].nunique():,} 位用戶")
    return df


def build_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """對每位 user 計算 aligned 特徵並建立滑動視窗。"""
    X_list, y_list = [], []
    skipped = 0

    for uid, grp in df.groupby("user_id"):
        grp = grp.sort_values("date").reset_index(drop=True)

        if len(grp) < MIN_DAYS_PER_USER:
            skipped += 1
            continue

        # 計算 aligned 特徵
        feats = compute_aligned_features(grp["daily_expense"], grp["date"])

        # 計算 target：未來 7 天支出總和
        target = grp["daily_expense"].rolling(7).sum().shift(-7)
        feats[TARGET_COL] = target.values

        # 去除 target 為 NaN 的列（最後 6 行）
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

    # 依時間順序切 70/15/15（pretrain 不需要 per-user）
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
