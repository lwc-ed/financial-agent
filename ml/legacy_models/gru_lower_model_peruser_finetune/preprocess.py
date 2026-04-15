"""
gru_lower_model_peruser_finetune/preprocess.py
================================================
資料前處理：
  1. 建立全域 (global) 訓練資料 → 供 train_global.py 使用
  2. 儲存 per-user 資料字典    → 供 finetune_peruser.py 使用

特徵：與 ml_gru 相同的 7 個核心特徵
目標：future_expense_7d_sum

執行順序：
  python preprocess.py → python train_global.py → python finetune_peruser.py → python predict.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

os.makedirs("artifacts", exist_ok=True)

DATA_PATH            = Path(__file__).resolve().parents[2] / "processed_data" / "artifacts" / "features_all.csv"
INPUT_DAYS           = 30
USER_CLIP_PERCENTILE = 99

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

# ─────────────────────────────────────────
# 1. 讀取
# ─────────────────────────────────────────
print("📂 讀取 features_all.csv...")
EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[~df["user_id"].isin(EXCLUDE_USERS)].reset_index(drop=True)
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
print(f"  總筆數: {len(df)}  使用者數: {df['user_id'].nunique()}  (排除 {EXCLUDE_USERS})")

# ─────────────────────────────────────────
# 2. Per-user 滑動視窗 + 70/15/15 切分
# ─────────────────────────────────────────
print(f"\n🪟 建立滑動視窗（INPUT_DAYS={INPUT_DAYS}）...")

X_train_list, y_train_list = [], []
X_val_list,   y_val_list   = [], []
X_test_list,  y_test_list  = [], []
train_uids, val_uids, test_uids = [], [], []
user_clip_map   = {}
per_user_data   = {}   # {user_id: {X_train, y_train, X_val, y_val, X_test, y_test}}

for user_id in df["user_id"].unique():
    u         = df[df["user_id"] == user_id].reset_index(drop=True)
    feats     = u[FEATURE_COLS].values.astype(np.float32)
    target    = u[TARGET_COL].values.astype(np.float32)

    wins_X, wins_y = [], []
    for t in range(INPUT_DAYS, len(u)):
        wins_X.append(feats[t - INPUT_DAYS:t])
        wins_y.append([target[t]])

    if len(wins_X) == 0:
        continue

    n     = len(wins_X)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)
    if t_end == 0:
        continue

    X_arr = np.array(wins_X, dtype=np.float32)
    y_arr = np.array(wins_y, dtype=np.float32)

    # Per-user P99 clipping（僅用 train split）
    clip_vals = {
        col: float(np.percentile(X_arr[:t_end, :, i], USER_CLIP_PERCENTILE))
        for i, col in enumerate(FEATURE_COLS)
    }
    user_clip_map[str(user_id)] = clip_vals

    def clip(arr):
        out = arr.copy()
        for i, col in enumerate(FEATURE_COLS):
            out[:, :, i] = np.clip(out[:, :, i], None, clip_vals[col])
        return out

    u_train = clip(X_arr[:t_end]);  u_y_tr = y_arr[:t_end]
    u_val   = clip(X_arr[t_end:v_end]); u_y_va = y_arr[t_end:v_end]
    u_test  = clip(X_arr[v_end:]);  u_y_te = y_arr[v_end:]

    per_user_data[user_id] = {
        "X_train": u_train, "y_train": u_y_tr,
        "X_val":   u_val,   "y_val":   u_y_va,
        "X_test":  u_test,  "y_test":  u_y_te,
    }

    X_train_list.extend(u_train); y_train_list.extend(u_y_tr)
    X_val_list.extend(u_val);     y_val_list.extend(u_y_va)
    X_test_list.extend(u_test);   y_test_list.extend(u_y_te)
    train_uids.extend([user_id] * len(u_train))
    val_uids.extend([user_id]   * len(u_val))
    test_uids.extend([user_id]  * len(u_test))

X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)
X_val   = np.array(X_val_list,   dtype=np.float32)
y_val   = np.array(y_val_list,   dtype=np.float32)
X_test  = np.array(X_test_list,  dtype=np.float32)
y_test  = np.array(y_test_list,  dtype=np.float32)

print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ─────────────────────────────────────────
# 3. 全域標準化（fit on global train only）
# ─────────────────────────────────────────
print("\n📐 全域標準化（fit on train only）...")
feat_scaler   = StandardScaler()
target_scaler = StandardScaler()

feat_scaler.fit(X_train.reshape(-1, len(FEATURE_COLS)))
target_scaler.fit(y_train)

def scale(X, y):
    Xs = feat_scaler.transform(X.reshape(-1, len(FEATURE_COLS))).reshape(X.shape).astype(np.float32)
    ys = target_scaler.transform(y).astype(np.float32)
    return Xs, ys

X_train_sc, y_train_sc = scale(X_train, y_train)
X_val_sc,   y_val_sc   = scale(X_val,   y_val)
X_test_sc,  y_test_sc  = scale(X_test,  y_test)

# 對 per_user_data 也做標準化
for uid in per_user_data:
    d = per_user_data[uid]
    d["X_train_sc"], d["y_train_sc"] = scale(d["X_train"], d["y_train"])
    d["X_val_sc"],   d["y_val_sc"]   = scale(d["X_val"],   d["y_val"])
    d["X_test_sc"],  d["y_test_sc"]  = scale(d["X_test"],  d["y_test"])

# ─────────────────────────────────────────
# 4. 儲存
# ─────────────────────────────────────────
print("\n💾 儲存至 artifacts/...")
np.save("artifacts/X_train.npy",    X_train_sc)
np.save("artifacts/y_train.npy",    y_train_sc)
np.save("artifacts/X_val.npy",      X_val_sc)
np.save("artifacts/y_val.npy",      y_val_sc)
np.save("artifacts/X_test.npy",     X_test_sc)
np.save("artifacts/y_test.npy",     y_test_sc)
np.save("artifacts/y_test_raw.npy", y_test)    # 原始值，用於最終評估
np.save("artifacts/y_val_raw.npy",  y_val)
np.save("artifacts/train_uids.npy", np.array(train_uids))
np.save("artifacts/val_uids.npy",   np.array(val_uids))
np.save("artifacts/test_uids.npy",  np.array(test_uids))

with open("artifacts/feat_scaler.pkl",       "wb") as f: pickle.dump(feat_scaler,    f)
with open("artifacts/target_scaler.pkl",     "wb") as f: pickle.dump(target_scaler,  f)
with open("artifacts/user_clip_values.pkl",  "wb") as f: pickle.dump(user_clip_map,  f)
with open("artifacts/per_user_data.pkl",     "wb") as f: pickle.dump(per_user_data,  f)

print(f"  ✅ 全域資料（X_train / X_val / X_test）")
print(f"  ✅ per_user_data.pkl（{len(per_user_data)} 個使用者）")
print(f"\n✅ 完成！下一步：python train_global.py")
