"""
Step 3：個人資料 Aligned 前處理
================================
使用與 Walmart 完全相同的 Rolling Z-score pipeline
輸出：
  - personal_aligned_X_train/val/test.npy
  - personal_aligned_y_train/val/test.npy
  - personal_aligned_target_scaler.pkl（用於 inverse transform 還原預測值）
  - personal_aligned_train/val/test_user_ids.npy
"""

import numpy as np
import pandas as pd
import pickle, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.preprocessing import StandardScaler
from alignment_utils import (
    compute_aligned_features,
    load_personal_daily,
    ALIGNED_FEATURE_COLS,
    TARGET_COL,
    INPUT_DAYS,
)

ARTIFACTS_DIR = "artifacts_aligned"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 載入個人日資料
# ─────────────────────────────────────────────────────────────────────────────
print("📂 載入個人資料...")
df = load_personal_daily()
print(f"  共 {len(df):,} 筆 | {df['user_id'].nunique()} 位用戶")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 計算 Aligned 特徵 + Target（每位用戶獨立計算）
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n📊 計算 Aligned 特徵與 Target...")
result_list = []

for user_id in sorted(df["user_id"].unique()):
    u = df[df["user_id"] == user_id].sort_values("date").reset_index(drop=True)

    feats = compute_aligned_features(u["daily_expense"], u["date"])

    # Target：未來 7 天日消費加總
    u["future_expense_7d_sum"] = (
        u["daily_expense"].rolling(7).sum().shift(-7)
    )

    feats["future_expense_7d_sum"] = u["future_expense_7d_sum"].values
    feats["user_id"]               = user_id
    feats["date"]                  = u["date"].values
    result_list.append(feats)

daily = pd.concat(result_list).reset_index(drop=True)
before = len(daily)
daily  = daily.dropna(subset=["future_expense_7d_sum"]).reset_index(drop=True)
print(f"  計算完後：{before:,} → {len(daily):,} 筆")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 滑動視窗 + per-user 70/15/15 切分
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n🪟 建立滑動視窗（INPUT_DAYS={INPUT_DAYS}，per-user 70/15/15）...")
X_train_list, y_train_list, train_uid, train_dates = [], [], [], []
X_val_list,   y_val_list,   val_uid,   val_dates   = [], [], [], []
X_test_list,  y_test_list,  test_uid,  test_dates  = [], [], [], []

for user_id in sorted(daily["user_id"].unique()):
    u          = daily[daily["user_id"] == user_id].reset_index(drop=True)
    feat_arr   = u[ALIGNED_FEATURE_COLS].values.astype(np.float32)
    target_arr = u[TARGET_COL].values.astype(np.float32)

    date_arr = u["date"].values
    windows_X, windows_y, windows_dates = [], [], []
    for t in range(INPUT_DAYS, len(u)):
        windows_X.append(feat_arr[t - INPUT_DAYS : t])
        windows_y.append([target_arr[t]])
        windows_dates.append(date_arr[t])

    n = len(windows_X)
    if n < 5:
        print(f"  ⚠️  {user_id} 資料不足（{n} 筆），跳過")
        continue

    t_end = int(n * 0.70)
    v_end = int(n * 0.85)
    if t_end == 0:
        continue

    X_tr = np.array(windows_X[:t_end],   dtype=np.float32)
    y_tr = np.array(windows_y[:t_end],   dtype=np.float32)
    X_vl = np.array(windows_X[t_end:v_end], dtype=np.float32)
    y_vl = np.array(windows_y[t_end:v_end], dtype=np.float32)
    X_te = np.array(windows_X[v_end:],   dtype=np.float32)
    y_te = np.array(windows_y[v_end:],   dtype=np.float32)

    X_train_list.extend(X_tr); y_train_list.extend(y_tr); train_uid.extend([user_id]*len(X_tr))
    X_val_list.extend(X_vl);   y_val_list.extend(y_vl);   val_uid.extend([user_id]*len(X_vl))
    X_test_list.extend(X_te);  y_test_list.extend(y_te);  test_uid.extend([user_id]*len(X_te))
    train_dates.extend(windows_dates[:t_end])
    val_dates.extend(windows_dates[t_end:v_end])
    test_dates.extend(windows_dates[v_end:])

X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)
X_val   = np.array(X_val_list,   dtype=np.float32)
y_val   = np.array(y_val_list,   dtype=np.float32)
X_test  = np.array(X_test_list,  dtype=np.float32)
y_test  = np.array(y_test_list,  dtype=np.float32)

print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 標準化 Target（fit on personal train only）
#    → 儲存此 scaler 供 predict 還原真實金額
# ─────────────────────────────────────────────────────────────────────────────
print("\n📐 標準化 Target（fit on personal train only）...")
target_scaler = StandardScaler()
target_scaler.fit(y_train)

y_train_scaled = target_scaler.transform(y_train).astype(np.float32)
y_val_scaled   = target_scaler.transform(y_val  ).astype(np.float32)
y_test_scaled  = target_scaler.transform(y_test ).astype(np.float32)

print(f"  Target scaler mean={target_scaler.mean_[0]:.2f}, std={target_scaler.scale_[0]:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 儲存
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n💾 儲存至 {ARTIFACTS_DIR}/...")

np.save(f"{ARTIFACTS_DIR}/personal_aligned_X_train.npy", X_train)
np.save(f"{ARTIFACTS_DIR}/personal_aligned_y_train.npy", y_train_scaled)
np.save(f"{ARTIFACTS_DIR}/personal_aligned_X_val.npy",   X_val)
np.save(f"{ARTIFACTS_DIR}/personal_aligned_y_val.npy",   y_val_scaled)
np.save(f"{ARTIFACTS_DIR}/personal_aligned_X_test.npy",  X_test)
np.save(f"{ARTIFACTS_DIR}/personal_aligned_y_test.npy",  y_test_scaled)

# 保留原始 y_test（未 scale）供最終 MAE 計算
np.save(f"{ARTIFACTS_DIR}/personal_aligned_y_test_raw.npy", y_test)

np.save(f"{ARTIFACTS_DIR}/personal_aligned_train_user_ids.npy", np.array(train_uid))
np.save(f"{ARTIFACTS_DIR}/personal_aligned_val_user_ids.npy",   np.array(val_uid))
np.save(f"{ARTIFACTS_DIR}/personal_aligned_test_user_ids.npy",  np.array(test_uid))
np.save(f"{ARTIFACTS_DIR}/personal_aligned_train_dates.npy", np.array(train_dates, dtype="datetime64[D]"))
np.save(f"{ARTIFACTS_DIR}/personal_aligned_val_dates.npy",   np.array(val_dates,   dtype="datetime64[D]"))
np.save(f"{ARTIFACTS_DIR}/personal_aligned_test_dates.npy",  np.array(test_dates,  dtype="datetime64[D]"))

with open(f"{ARTIFACTS_DIR}/personal_aligned_target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)

print("  ✅ personal_aligned_X_train/val/test.npy")
print("  ✅ personal_aligned_y_train/val/test.npy（scaled）")
print("  ✅ personal_aligned_y_test_raw.npy（原始金額，供 MAE 計算）")
print("  ✅ personal_aligned_train/val/test_user_ids.npy")
print("  ✅ personal_aligned_target_scaler.pkl")
print(f"\n🎉 完成！下一步：執行 4_pretrain_aligned.py")
