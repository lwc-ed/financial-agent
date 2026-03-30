"""
步驟二：個人資料前處理
切割方式：per-user 70 / 15 / 15（時序切割，無資料洩漏）
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = "artificats"
FEATURES_PATH = "features_all.csv"
INPUT_DAYS    = 30
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
# TEST_RATIO  = 0.15（剩餘全部）

FEATURE_COLS = [
    "daily_expense",
    "expense_7d_mean",
    "expense_30d_sum",
    "has_expense",
    "has_income",
    "net_30d_sum",
    "txn_30d_sum",
]
TARGET_COL = "future_expense_7d_sum"

# ── 1. 讀取資料 ──────────────────────────────────────────────────────────────
print("📂 讀取 features_all.csv...")
EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]

df = pd.read_csv(FEATURES_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[~df["user_id"].isin(EXCLUDE_USERS)].reset_index(drop=True)
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

print(f"  總筆數   : {len(df)}")
print(f"  使用者數 : {df['user_id'].nunique()}  (排除 {EXCLUDE_USERS})")
print(f"  缺值     : {df[FEATURE_COLS + [TARGET_COL]].isnull().sum().sum()}")

# ── 2. Per-user P99 Winsorization ────────────────────────────────────────────
print("\n✂️  Per-user Winsorization（P99）...")
user_clip_values = {}
for uid in df["user_id"].unique():
    mask = df["user_id"] == uid
    clips = {}
    for col in FEATURE_COLS + [TARGET_COL]:
        p99 = df.loc[mask, col].quantile(0.99)
        df.loc[mask, col] = df.loc[mask, col].clip(upper=p99)
        clips[col] = float(p99)
    user_clip_values[uid] = clips
print(f"  完成（共 {len(user_clip_values)} 位使用者）")

# ── 3. 標準化（用訓練集 fit）────────────────────────────────────────────────
print("\n📐 標準化（以 train 部分 fit）...")

train_rows = []
for uid in df["user_id"].unique():
    u = df[df["user_id"] == uid].reset_index(drop=True)
    n_train = int(len(u) * TRAIN_RATIO)
    train_rows.append(u.iloc[:n_train])

train_df = pd.concat(train_rows, ignore_index=True)
feature_scaler = StandardScaler().fit(train_df[FEATURE_COLS])
target_scaler  = StandardScaler().fit(train_df[[TARGET_COL]])

df[FEATURE_COLS] = feature_scaler.transform(df[FEATURE_COLS])
df[[TARGET_COL]] = target_scaler.transform(df[[TARGET_COL]])
print("  完成")

# ── 4. Per-user 70/15/15 滑動視窗 ────────────────────────────────────────────
print(f"\n🪟 建立 per-user 70/15/15 滑動視窗（INPUT_DAYS={INPUT_DAYS}）...")

X_train_list, y_train_list, train_uid_list = [], [], []
X_val_list,   y_val_list,   val_uid_list   = [], [], []
X_test_list,  y_test_list,  test_uid_list  = [], [], []

for uid in df["user_id"].unique():
    u = df[df["user_id"] == uid].reset_index(drop=True)
    n       = len(u)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    feats  = u[FEATURE_COLS].values
    target = u[TARGET_COL].values

    # Train: rows 0 ~ n_train
    for t in range(INPUT_DAYS, n_train):
        X_train_list.append(feats[t - INPUT_DAYS : t])
        y_train_list.append([target[t]])
        train_uid_list.append(uid)

    # Val: 以 n_train 前 INPUT_DAYS 做 context
    val_end = n_train + n_val
    for t in range(n_train, val_end):
        if t - INPUT_DAYS < 0:
            continue
        X_val_list.append(feats[t - INPUT_DAYS : t])
        y_val_list.append([target[t]])
        val_uid_list.append(uid)

    # Test: 以 val_end 前 INPUT_DAYS 做 context
    for t in range(val_end, n):
        if t - INPUT_DAYS < 0:
            continue
        X_test_list.append(feats[t - INPUT_DAYS : t])
        y_test_list.append([target[t]])
        test_uid_list.append(uid)

X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)
X_val   = np.array(X_val_list,   dtype=np.float32)
y_val   = np.array(y_val_list,   dtype=np.float32)
X_test  = np.array(X_test_list,  dtype=np.float32)
y_test  = np.array(y_test_list,  dtype=np.float32)

print(f"  Train : {X_train.shape[0]} 筆  |  Val : {X_val.shape[0]} 筆  |  Test : {X_test.shape[0]} 筆")
print(f"  X shape : (N, {INPUT_DAYS}, {len(FEATURE_COLS)})")

# ── 5. 儲存 ─────────────────────────────────────────────────────────────────
print(f"\n💾 儲存至 {ARTIFACTS_DIR}/...")

np.save(f"{ARTIFACTS_DIR}/personal_X_train.npy",        X_train)
np.save(f"{ARTIFACTS_DIR}/personal_y_train.npy",        y_train)
np.save(f"{ARTIFACTS_DIR}/personal_X_val.npy",          X_val)
np.save(f"{ARTIFACTS_DIR}/personal_y_val.npy",          y_val)
np.save(f"{ARTIFACTS_DIR}/personal_X_test.npy",         X_test)
np.save(f"{ARTIFACTS_DIR}/personal_y_test.npy",         y_test)
np.save(f"{ARTIFACTS_DIR}/personal_train_user_ids.npy", np.array(train_uid_list))
np.save(f"{ARTIFACTS_DIR}/personal_val_user_ids.npy",   np.array(val_uid_list))
np.save(f"{ARTIFACTS_DIR}/personal_test_user_ids.npy",  np.array(test_uid_list))

with open(f"{ARTIFACTS_DIR}/personal_feature_scaler.pkl", "wb") as f:
    pickle.dump(feature_scaler, f)
with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)
with open(f"{ARTIFACTS_DIR}/personal_user_clip_values.pkl", "wb") as f:
    pickle.dump(user_clip_values, f)

print("  ✅ X_train / y_train / X_val / y_val / X_test / y_test")
print("  ✅ train / val / test user_ids.npy")
print("  ✅ personal_feature_scaler.pkl / personal_target_scaler.pkl")
print("  ✅ personal_user_clip_values.pkl")
print("\n🎉 完成！下一步：執行 finetune_lstm.py")
