"""
Step 1：Walmart 前處理 (路徑鎖定版)
=======================
使用 10 個 aligned 特徵對 Walmart 資料做前處理，並鎖定輸出路徑。
"""

import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── 1. 路徑鎖定 ──────────────────────────────────────────────────────────
# 抓取這支程式所在的絕對路徑
MY_DIR = Path(__file__).resolve().parent
# 鎖定存檔資料夾
ARTIFACTS_DIR = MY_DIR / "artifacts_bigru_tl"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 確保能 import 同資料夾下的 alignment_utils
sys.path.insert(0, str(MY_DIR))
from alignment_utils import ALIGNED_FEATURE_COLS, INPUT_DAYS, TARGET_COL, compute_aligned_features, load_walmart_daily

def main():
    print(f"📂 工作目錄鎖定為: {MY_DIR}")
    print("📂 載入 Walmart 原始資料...")
    
    # 這裡會呼叫 alignment_utils 裡的載入函式
    df = load_walmart_daily()
    print(f"  共 {len(df):,} 筆 | {df['store_id'].nunique()} 家店")

    print(f"\n📊 計算 {len(ALIGNED_FEATURE_COLS)} 個 aligned 特徵 (Z-score)...")
    result_list = []
    for store_id in sorted(df["store_id"].unique()):
        s = df[df["store_id"] == store_id].sort_values("date").reset_index(drop=True)
        # 計算對齊特徵
        feats = compute_aligned_features(s["daily_expense"], s["date"])
        # 計算未來 7 天支出總和 (Walmart 端的 Target)
        s[TARGET_COL] = s["daily_expense"].rolling(7).sum().shift(-7)
        feats[TARGET_COL] = s[TARGET_COL].values
        feats["store_id"] = store_id
        result_list.append(feats)

    daily = pd.concat(result_list, ignore_index=True)
    daily = daily.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"  有效筆數：{len(daily):,}")

    print(f"\n🪟 建立 3D 滑動視窗（INPUT_DAYS={INPUT_DAYS}）...")
    X_list, y_list = [], []
    for store_id in sorted(daily["store_id"].unique()):
        s = daily[daily["store_id"] == store_id].reset_index(drop=True)
        feat_arr = s[ALIGNED_FEATURE_COLS].values.astype(np.float32)
        target_arr = s[TARGET_COL].values.astype(np.float32)
        
        for t in range(INPUT_DAYS, len(s)):
            X_list.append(feat_arr[t - INPUT_DAYS : t])
            y_list.append([target_arr[t]])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # 按照 70/15/15 比例切分 Walmart 資料集
    train_end = int(len(X) * 0.70)
    val_end = int(len(X) * 0.85)
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    print(f"  Walmart Train: {X_train.shape}  Val: {X_val.shape}")

    print("\n📐 標準化 Walmart Target...")
    target_scaler = StandardScaler()
    target_scaler.fit(y_train)
    y_train_s = target_scaler.transform(y_train).astype(np.float32)
    y_val_s = target_scaler.transform(y_val).astype(np.float32)
    y_test_s = target_scaler.transform(y_test).astype(np.float32)

    print(f"\n💾 儲存 Walmart 彈藥至 {ARTIFACTS_DIR}...")
    np.save(ARTIFACTS_DIR / "walmart_X_train.npy", X_train)
    np.save(ARTIFACTS_DIR / "walmart_y_train.npy", y_train_s)
    np.save(ARTIFACTS_DIR / "walmart_X_val.npy", X_val)
    np.save(ARTIFACTS_DIR / "walmart_y_val.npy", y_val_s)
    np.save(ARTIFACTS_DIR / "walmart_X_test.npy", X_test)
    np.save(ARTIFACTS_DIR / "walmart_y_test.npy", y_test_s)
    
    with open(ARTIFACTS_DIR / "walmart_target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    print("\n✅ Walmart 前處理完成！地基已打好。")

if __name__ == "__main__":
    main()