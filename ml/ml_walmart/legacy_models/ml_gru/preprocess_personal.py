"""
步驟三（更新版）：個人資料前處理
================================
直接使用已有的 features_all.csv（包含 7 個最佳特徵）
日級別，INPUT_DAYS=30
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = "artificats"
MODEL_DIR = Path(__file__).resolve().parent
ML_ROOT = MODEL_DIR.parent.parent if MODEL_DIR.parent.name == "legacy_models" else MODEL_DIR.parent
FEATURES_PATH = ML_ROOT / "processed_data" / "artifacts" / "features_all.csv"
INPUT_DAYS    = 30
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


def fit_user_clip_values(X_user_train: np.ndarray, y_user_train: np.ndarray):
    clip_values = {
        col: float(np.percentile(X_user_train[:, :, idx], USER_CLIP_PERCENTILE))
        for idx, col in enumerate(FEATURE_COLS)
    }
    return clip_values


def apply_user_clipping(X_split: np.ndarray, y_split: np.ndarray, clip_values: dict):
    X_out = X_split.copy()
    for idx, col in enumerate(FEATURE_COLS):
        X_out[:, :, idx] = np.clip(X_out[:, :, idx], None, clip_values[col])
    return X_out.astype(np.float32), y_split.astype(np.float32)


def standardize_split(
    X_split: np.ndarray,
    y_split: np.ndarray,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
):
    X_out = feature_scaler.transform(X_split.reshape(-1, len(FEATURE_COLS))).reshape(X_split.shape)
    y_out = target_scaler.transform(y_split)
    return X_out.astype(np.float32), y_out.astype(np.float32)

# ─────────────────────────────────────────
# 1. 讀取
# ─────────────────────────────────────────
print("📂 讀取 features_all.csv...")
EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]

df = pd.read_csv(FEATURES_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df[~df["user_id"].isin(EXCLUDE_USERS)].reset_index(drop=True)
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

print(f"  總筆數   : {len(df)}")
print(f"  使用者數 : {df['user_id'].nunique()}  (排除 {EXCLUDE_USERS})")
print(f"  缺值     : {df[FEATURE_COLS + [TARGET_COL]].isnull().sum().sum()}")


# ─────────────────────────────────────────
# 2. 滑動視窗 + Per-user 70/15/15 時間切分
# ─────────────────────────────────────────
print(f"\n🪟 建立滑動視窗（INPUT_DAYS={INPUT_DAYS}，per-user 70/15/15）...")
X_train_list, y_train_list = [], []
X_val_list,   y_val_list   = [], []
X_test_list,  y_test_list  = [], []
train_user_ids, val_user_ids, test_user_ids = [], [], []
user_clip_map = {}
clip_counts = {col: 0 for col in FEATURE_COLS}

for user_id in df["user_id"].unique():
    u            = df[df["user_id"] == user_id].reset_index(drop=True)
    features_arr = u[FEATURE_COLS].values
    target_arr   = u[TARGET_COL].values

    windows_X, windows_y = [], []
    for t in range(INPUT_DAYS, len(u)):
        windows_X.append(features_arr[t - INPUT_DAYS : t])
        windows_y.append([target_arr[t]])

    n         = len(windows_X)
    if n == 0:
        continue
    t_end     = int(n * 0.70)
    v_end     = int(n * 0.85)
    if t_end == 0:
        continue

    X_train_user = np.array(windows_X[:t_end], dtype=np.float32)
    y_train_user = np.array(windows_y[:t_end], dtype=np.float32)
    X_val_user   = np.array(windows_X[t_end:v_end], dtype=np.float32)
    y_val_user   = np.array(windows_y[t_end:v_end], dtype=np.float32)
    X_test_user  = np.array(windows_X[v_end:], dtype=np.float32)
    y_test_user  = np.array(windows_y[v_end:], dtype=np.float32)

    clip_values = fit_user_clip_values(X_train_user, y_train_user)
    user_clip_map[str(user_id)] = clip_values

    for idx, col in enumerate(FEATURE_COLS):
        clip_counts[col] += int((X_train_user[:, :, idx] > clip_values[col]).sum())
        if len(X_val_user) > 0:
            clip_counts[col] += int((X_val_user[:, :, idx] > clip_values[col]).sum())
        if len(X_test_user) > 0:
            clip_counts[col] += int((X_test_user[:, :, idx] > clip_values[col]).sum())
    X_train_user, y_train_user = apply_user_clipping(X_train_user, y_train_user, clip_values)
    X_val_user, y_val_user     = apply_user_clipping(X_val_user, y_val_user, clip_values)
    X_test_user, y_test_user   = apply_user_clipping(X_test_user, y_test_user, clip_values)

    X_train_list.extend(X_train_user)
    y_train_list.extend(y_train_user)
    train_user_ids.extend([user_id] * len(X_train_user))

    X_val_list.extend(X_val_user)
    y_val_list.extend(y_val_user)
    val_user_ids.extend([user_id] * len(X_val_user))

    X_test_list.extend(X_test_user)
    y_test_list.extend(y_test_user)
    test_user_ids.extend([user_id] * len(X_test_user))

X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)
X_val   = np.array(X_val_list, dtype=np.float32)
y_val   = np.array(y_val_list, dtype=np.float32)
X_test  = np.array(X_test_list, dtype=np.float32)
y_test  = np.array(y_test_list, dtype=np.float32)
train_user_ids = np.array(train_user_ids)
val_user_ids   = np.array(val_user_ids)
test_user_ids  = np.array(test_user_ids)

print(
    f"\n✂️  訓練集 : {X_train.shape}"
    f"  驗證集 : {X_val.shape}"
    f"  測試集 : {X_test.shape}"
)


# ─────────────────────────────────────────
# 4. Per-user Winsorization + global scaler
# ─────────────────────────────────────────
print(f"\n✂️  Per-user Winsorization（P{USER_CLIP_PERCENTILE}, train only）...")
print(f"  套用 user 門檻數 : {len(user_clip_map)}")
for col in FEATURE_COLS:
    print(f"  {col}: 被截斷值數量 {clip_counts[col]:,}")

print("\n📐 標準化（fit on train only）...")
feature_scaler = StandardScaler()
target_scaler = StandardScaler()
feature_scaler.fit(X_train.reshape(-1, len(FEATURE_COLS)))
target_scaler.fit(y_train)

X_train, y_train = standardize_split(X_train, y_train, feature_scaler, target_scaler)
X_val, y_val     = standardize_split(X_val, y_val, feature_scaler, target_scaler)
X_test, y_test   = standardize_split(X_test, y_test, feature_scaler, target_scaler)
print("  完成")


# ─────────────────────────────────────────
# 5. 儲存
# ─────────────────────────────────────────
print(f"\n💾 儲存至 {ARTIFACTS_DIR}/...")

np.save(f"{ARTIFACTS_DIR}/personal_X_train.npy", X_train)
np.save(f"{ARTIFACTS_DIR}/personal_y_train.npy", y_train)
np.save(f"{ARTIFACTS_DIR}/personal_X_val.npy",   X_val)
np.save(f"{ARTIFACTS_DIR}/personal_y_val.npy",   y_val)
np.save(f"{ARTIFACTS_DIR}/personal_X_test.npy",  X_test)
np.save(f"{ARTIFACTS_DIR}/personal_y_test.npy",  y_test)
np.save(f"{ARTIFACTS_DIR}/personal_train_user_ids.npy", train_user_ids)
np.save(f"{ARTIFACTS_DIR}/personal_val_user_ids.npy",   val_user_ids)
np.save(f"{ARTIFACTS_DIR}/personal_test_user_ids.npy",  test_user_ids)

with open(f"{ARTIFACTS_DIR}/personal_feature_scaler.pkl", "wb") as f:
    pickle.dump(feature_scaler, f)
with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)
with open(f"{ARTIFACTS_DIR}/personal_user_clip_values.pkl", "wb") as f:
    pickle.dump(user_clip_map, f)

print("  ✅ personal_X_train / y_train / X_val / y_val / X_test / y_test")
print("  ✅ personal_train_user_ids / val_user_ids / test_user_ids")
print("  ✅ personal_feature_scaler.pkl")
print("  ✅ personal_target_scaler.pkl")
print("  ✅ personal_user_clip_values.pkl")
print(f"\n🎉 完成！下一步：執行 finetune_gru.py")
