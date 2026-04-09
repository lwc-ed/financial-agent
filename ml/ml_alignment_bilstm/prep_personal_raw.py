"""
prep_personal_raw.py
====================
為 Adapter finetune 準備個人資料（raw features，不做 z-score）
特徵設計與同學的 Bi-LSTM 一致：
  daily_expense, roll_7d_mean, roll_30d_mean, dow_sin, dow_cos
直接從 features_all.csv 讀取，不重新計算
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = "../ml_gru/features_all.csv"
SAVE_DIR      = "artifacts_bilstm"
EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]
INPUT_DAYS    = 30
TARGET_COL    = "future_expense_7d_sum"

RAW_FEATURE_COLS = [
    "daily_expense",
    "roll_7d_mean",
    "roll_30d_mean",
    "dow_sin",
    "dow_cos",
]

print("📂 讀取 features_all.csv...")
df = pd.read_csv(FEATURES_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[~df["user_id"].isin(EXCLUDE_USERS)].reset_index(drop=True)
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

# 計算 raw rolling features（從 expense_7d_mean, expense_30d_mean 直接取）
df["roll_7d_mean"]  = df["expense_7d_mean"]
df["roll_30d_mean"] = df["expense_30d_mean"]

# 星期幾週期編碼
df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

print(f"  總筆數 : {len(df)}  用戶數 : {df['user_id'].nunique()}")
print(f"  缺值  : {df[RAW_FEATURE_COLS + [TARGET_COL]].isnull().sum().sum()}")

# ── 滑動視窗 + Per-user 70/15/15 ─────────────────────────────────────────────
print(f"\n🪟 建立滑動視窗（INPUT_DAYS={INPUT_DAYS}）...")

X_train_l, y_train_l = [], []
X_val_l,   y_val_l   = [], []
X_test_l,  y_test_l  = [], []
train_uids, val_uids, test_uids = [], [], []

for uid in df["user_id"].unique():
    u       = df[df["user_id"] == uid].reset_index(drop=True)
    feat    = u[RAW_FEATURE_COLS].values.astype(np.float32)
    target  = u[TARGET_COL].values.astype(np.float32)

    windows_X, windows_y = [], []
    for i in range(INPUT_DAYS, len(u)):
        windows_X.append(feat[i - INPUT_DAYS: i])
        windows_y.append([target[i]])

    n = len(windows_X)
    if n < 10:
        continue

    t_end = int(n * 0.70)
    v_end = int(n * 0.85)

    X_train_l.extend(windows_X[:t_end]);      y_train_l.extend(windows_y[:t_end])
    X_val_l.extend(windows_X[t_end:v_end]);   y_val_l.extend(windows_y[t_end:v_end])
    X_test_l.extend(windows_X[v_end:]);       y_test_l.extend(windows_y[v_end:])
    train_uids.extend([uid] * len(windows_X[:t_end]))
    val_uids.extend([uid]   * len(windows_X[t_end:v_end]))
    test_uids.extend([uid]  * len(windows_X[v_end:]))

X_train = np.array(X_train_l, dtype=np.float32)
y_train = np.array(y_train_l, dtype=np.float32)
X_val   = np.array(X_val_l,   dtype=np.float32)
y_val   = np.array(y_val_l,   dtype=np.float32)
X_test  = np.array(X_test_l,  dtype=np.float32)
y_test  = np.array(y_test_l,  dtype=np.float32)

print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

# ── Target 標準化（只對 y，X 保持 raw）────────────────────────────────────────
print("\n📏 標準化 Target（X 保持原始金額，不做標準化）...")
target_scaler = StandardScaler()
y_train_s = target_scaler.fit_transform(y_train).astype(np.float32)
y_val_s   = target_scaler.transform(y_val).astype(np.float32)
y_test_s  = target_scaler.transform(y_test).astype(np.float32)

# ── 儲存 ──────────────────────────────────────────────────────────────────────
print(f"\n💾 儲存至 {SAVE_DIR}/raw_*...")
np.save(f"{SAVE_DIR}/raw_X_train.npy",    X_train)
np.save(f"{SAVE_DIR}/raw_y_train_s.npy",  y_train_s)
np.save(f"{SAVE_DIR}/raw_X_val.npy",      X_val)
np.save(f"{SAVE_DIR}/raw_y_val_s.npy",    y_val_s)
np.save(f"{SAVE_DIR}/raw_X_test.npy",     X_test)
np.save(f"{SAVE_DIR}/raw_y_test_s.npy",   y_test_s)
np.save(f"{SAVE_DIR}/raw_y_test_raw.npy", y_test)
np.save(f"{SAVE_DIR}/raw_train_uids.npy", np.array(train_uids))
np.save(f"{SAVE_DIR}/raw_val_uids.npy",   np.array(val_uids))
np.save(f"{SAVE_DIR}/raw_test_uids.npy",  np.array(test_uids))

with open(f"{SAVE_DIR}/raw_target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)

print("  ✅ 完成！")
print(f"\n下一步：執行 2b_finetune_adapter.py")
