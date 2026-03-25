"""
步驟三（更新版）：個人資料前處理
================================
直接使用已有的 features_all.csv（包含 7 個最佳特徵）
日級別，INPUT_DAYS=30
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = "artificats"
FEATURES_PATH = "features_all.csv"   # 你已經做好的特徵檔
INPUT_DAYS    = 30

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


def fit_clipping_and_scalers(X_train: np.ndarray, y_train: np.ndarray):
    clip_values = {
        col: float(np.percentile(X_train[:, :, idx], 95))
        for idx, col in enumerate(FEATURE_COLS)
    }
    clip_values[TARGET_COL] = float(np.percentile(y_train[:, 0], 95))

    X_train_clipped = X_train.copy()
    for idx, col in enumerate(FEATURE_COLS):
        X_train_clipped[:, :, idx] = np.clip(X_train_clipped[:, :, idx], None, clip_values[col])
    y_train_clipped = np.clip(y_train.copy(), None, clip_values[TARGET_COL])

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(X_train_clipped.reshape(-1, len(FEATURE_COLS)))
    target_scaler.fit(y_train_clipped)

    return clip_values, feature_scaler, target_scaler


def transform_split(
    X_split: np.ndarray,
    y_split: np.ndarray,
    clip_values: dict,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
):
    X_out = X_split.copy()
    for idx, col in enumerate(FEATURE_COLS):
        X_out[:, :, idx] = np.clip(X_out[:, :, idx], None, clip_values[col])
    y_out = np.clip(y_split.copy(), None, clip_values[TARGET_COL])

    X_out = feature_scaler.transform(X_out.reshape(-1, len(FEATURE_COLS))).reshape(X_out.shape)
    y_out = target_scaler.transform(y_out)
    return X_out.astype(np.float32), y_out.astype(np.float32)

# ─────────────────────────────────────────
# 1. 讀取
# ─────────────────────────────────────────
print("📂 讀取 features_all.csv...")
df = pd.read_csv(FEATURES_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

print(f"  總筆數   : {len(df)}")
print(f"  使用者數 : {df['user_id'].nunique()}")
print(f"  缺值     : {df[FEATURE_COLS + [TARGET_COL]].isnull().sum().sum()}")


# ─────────────────────────────────────────
# 2. 滑動視窗 + Per-user 70/15/15 時間切分
# ─────────────────────────────────────────
print(f"\n🪟 建立滑動視窗（INPUT_DAYS={INPUT_DAYS}，per-user 70/15/15）...")
X_train_list, y_train_list = [], []
X_val_list,   y_val_list   = [], []
X_test_list,  y_test_list  = [], []

for user_id in df["user_id"].unique():
    u            = df[df["user_id"] == user_id].reset_index(drop=True)
    features_arr = u[FEATURE_COLS].values
    target_arr   = u[TARGET_COL].values

    windows_X, windows_y = [], []
    for t in range(INPUT_DAYS, len(u)):
        windows_X.append(features_arr[t - INPUT_DAYS : t])
        windows_y.append([target_arr[t]])

    n         = len(windows_X)
    t_end     = int(n * 0.70)
    v_end     = int(n * 0.85)

    X_train_list.extend(windows_X[:t_end])
    y_train_list.extend(windows_y[:t_end])
    X_val_list.extend(windows_X[t_end:v_end])
    y_val_list.extend(windows_y[t_end:v_end])
    X_test_list.extend(windows_X[v_end:])
    y_test_list.extend(windows_y[v_end:])

X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)
X_val   = np.array(X_val_list,   dtype=np.float32)
y_val   = np.array(y_val_list,   dtype=np.float32)
X_test  = np.array(X_test_list,  dtype=np.float32)
y_test  = np.array(y_test_list,  dtype=np.float32)

print(
    f"\n✂️  訓練集 : {X_train.shape}"
    f"  驗證集 : {X_val.shape}"
    f"  測試集 : {X_test.shape}"
)


# ─────────────────────────────────────────
# 4. 只用 train 擬合 Winsorization / Scaler，並套到 val/test
# ─────────────────────────────────────────
print("\n✂️  Winsorization（P95, train only）...")
clip_values, feature_scaler, target_scaler = fit_clipping_and_scalers(X_train, y_train)
for col in FEATURE_COLS + [TARGET_COL]:
    print(f"  {col}: 上限 {clip_values[col]:,.4f}")

print("\n📐 標準化（fit on train only）...")
X_train, y_train = transform_split(X_train, y_train, clip_values, feature_scaler, target_scaler)
X_val, y_val = transform_split(X_val, y_val, clip_values, feature_scaler, target_scaler)
X_test, y_test = transform_split(X_test, y_test, clip_values, feature_scaler, target_scaler)
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

with open(f"{ARTIFACTS_DIR}/personal_feature_scaler.pkl", "wb") as f:
    pickle.dump(feature_scaler, f)
with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)

print("  ✅ personal_X_train / y_train / X_val / y_val / X_test / y_test")
print("  ✅ personal_feature_scaler.pkl")
print("  ✅ personal_target_scaler.pkl")
print(f"\n🎉 完成！下一步：執行 finetune_gru.py")
