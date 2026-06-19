"""
Step 1：Walmart 前處理
=======================
使用 10 個 Aligned 特徵對 Walmart 資料做前處理
輸出：
  - walmart_X_train/val/test.npy
  - walmart_y_train/val/test.npy
  - walmart_target_scaler.pkl
"""

import numpy as np
import pandas as pd
import pickle, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.preprocessing import StandardScaler
from alignment_utils import (
    compute_aligned_features,
    load_walmart_daily,
    ALIGNED_FEATURE_COLS,
    TARGET_COL,
    INPUT_DAYS,
)

ARTIFACTS_DIR = "artifacts_bilstm_v2"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("📂 載入 Walmart 資料...")
df = load_walmart_daily()
print(f"  共 {len(df):,} 筆 | {df['store_id'].nunique()} 家店")

print(f"\n📊 計算 {len(ALIGNED_FEATURE_COLS)} 個 Aligned 特徵...")
result_list = []

for store_id in sorted(df["store_id"].unique()):
    s = df[df["store_id"] == store_id].sort_values("date").reset_index(drop=True)
    feats = compute_aligned_features(s["daily_expense"], s["date"])
    s["future_expense_7d_sum"] = s["daily_expense"].rolling(7).sum().shift(-7)
    feats["future_expense_7d_sum"] = s["future_expense_7d_sum"].values
    feats["store_id"] = store_id
    result_list.append(feats)

daily = pd.concat(result_list).reset_index(drop=True)
daily = daily.dropna(subset=["future_expense_7d_sum"]).reset_index(drop=True)
print(f"  有效筆數：{len(daily):,}")

print(f"\n🪟 建立滑動視窗（INPUT_DAYS={INPUT_DAYS}）...")
X_list, y_list = [], []

for store_id in sorted(daily["store_id"].unique()):
    s = daily[daily["store_id"] == store_id].reset_index(drop=True)
    feat_arr   = s[ALIGNED_FEATURE_COLS].values.astype(np.float32)
    target_arr = s[TARGET_COL].values.astype(np.float32)
    for t in range(INPUT_DAYS, len(s)):
        X_list.append(feat_arr[t - INPUT_DAYS : t])
        y_list.append([target_arr[t]])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
print(f"  X: {X.shape}  y: {y.shape}")

train_end = int(len(X) * 0.70)
val_end   = int(len(X) * 0.85)
X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

print("\n📐 標準化 Target...")
target_scaler = StandardScaler()
target_scaler.fit(y_train)
y_train = target_scaler.transform(y_train).astype(np.float32)
y_val   = target_scaler.transform(y_val).astype(np.float32)
y_test  = target_scaler.transform(y_test).astype(np.float32)

print(f"\n💾 儲存至 {ARTIFACTS_DIR}/...")
np.save(f"{ARTIFACTS_DIR}/walmart_X_train.npy", X_train)
np.save(f"{ARTIFACTS_DIR}/walmart_y_train.npy", y_train)
np.save(f"{ARTIFACTS_DIR}/walmart_X_val.npy",   X_val)
np.save(f"{ARTIFACTS_DIR}/walmart_y_val.npy",   y_val)
np.save(f"{ARTIFACTS_DIR}/walmart_X_test.npy",  X_test)
np.save(f"{ARTIFACTS_DIR}/walmart_y_test.npy",  y_test)
with open(f"{ARTIFACTS_DIR}/walmart_target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)

print("  ✅ 完成！下一步：2_preprocess_personal.py")
