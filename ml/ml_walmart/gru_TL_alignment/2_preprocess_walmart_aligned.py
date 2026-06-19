"""
Step 2：Walmart Aligned 前處理
================================
使用 Rolling Z-score aligned 特徵對 Walmart 資料做前處理
輸出：
  - walmart_aligned_X_train/val/test.npy
  - walmart_aligned_y_train/val/test.npy
  - walmart_aligned_target_scaler.pkl
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

ARTIFACTS_DIR = "artifacts_aligned"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 載入 Walmart 日資料
# ─────────────────────────────────────────────────────────────────────────────
print("📂 載入 Walmart 資料並展開為日資料...")
df = load_walmart_daily()
print(f"  共 {len(df):,} 筆 | {df['store_id'].nunique()} 家店")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 計算 Aligned 特徵 + Target（每家店獨立計算）
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n📊 計算 Aligned 特徵與 Target（future_7d_sum）...")
result_list = []

for store_id in sorted(df["store_id"].unique()):
    s = df[df["store_id"] == store_id].sort_values("date").reset_index(drop=True)

    # 計算 aligned 特徵
    feats = compute_aligned_features(s["daily_expense"], s["date"])

    # Target：未來 7 天的日消費加總
    s["future_expense_7d_sum"] = (
        s["daily_expense"].rolling(7).sum().shift(-7)
    )

    # 合併
    feats["future_expense_7d_sum"] = s["future_expense_7d_sum"].values
    feats["store_id"]              = store_id
    result_list.append(feats)

daily = pd.concat(result_list).reset_index(drop=True)
before = len(daily)
daily  = daily.dropna(subset=["future_expense_7d_sum"]).reset_index(drop=True)
print(f"  計算完後：{before:,} → {len(daily):,} 筆（移除 target NaN）")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 建立滑動視窗（INPUT_DAYS 天 → 預測下 7 天）
# ─────────────────────────────────────────────────────────────────────────────
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
print(f"  X shape: {X.shape}  （樣本, {INPUT_DAYS}天, {len(ALIGNED_FEATURE_COLS)}特徵）")
print(f"  y shape: {y.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 切分（70 / 15 / 15）
# ─────────────────────────────────────────────────────────────────────────────
train_end = int(len(X) * 0.70)
val_end   = int(len(X) * 0.85)
X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

print(f"\n✂️  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 特徵：不做額外 scaler（rolling z-score 本身已經是 [-5,5] 的標準空間）
#    Target：用 Walmart train 的 StandardScaler 標準化，方便 GRU 訓練
# ─────────────────────────────────────────────────────────────────────────────
print("\n📐 標準化 Target（fit on Walmart train only）...")
target_scaler = StandardScaler()
target_scaler.fit(y_train)
y_train = target_scaler.transform(y_train).astype(np.float32)
y_val   = target_scaler.transform(y_val  ).astype(np.float32)
y_test  = target_scaler.transform(y_test ).astype(np.float32)

print(f"  Target scaler mean={target_scaler.mean_[0]:.2f}, std={target_scaler.scale_[0]:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 儲存
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n💾 儲存至 {ARTIFACTS_DIR}/...")
np.save(f"{ARTIFACTS_DIR}/walmart_aligned_X_train.npy", X_train)
np.save(f"{ARTIFACTS_DIR}/walmart_aligned_y_train.npy", y_train)
np.save(f"{ARTIFACTS_DIR}/walmart_aligned_X_val.npy",   X_val)
np.save(f"{ARTIFACTS_DIR}/walmart_aligned_y_val.npy",   y_val)
np.save(f"{ARTIFACTS_DIR}/walmart_aligned_X_test.npy",  X_test)
np.save(f"{ARTIFACTS_DIR}/walmart_aligned_y_test.npy",  y_test)

with open(f"{ARTIFACTS_DIR}/walmart_aligned_target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)

print("  ✅ walmart_aligned_X_train/val/test.npy")
print("  ✅ walmart_aligned_y_train/val/test.npy")
print("  ✅ walmart_aligned_target_scaler.pkl")
print(f"\n🎉 完成！下一步：執行 3_preprocess_personal_aligned.py")
