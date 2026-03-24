"""
步驟二：個人資料前處理
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = "artificats"
FEATURES_PATH = "features_all.csv"
INPUT_DAYS = 30

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

print("📂 讀取 features_all.csv...")
df = pd.read_csv(FEATURES_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

print(f"  總筆數   : {len(df)}")
print(f"  使用者數 : {df['user_id'].nunique()}")
print(f"  缺值     : {df[FEATURE_COLS + [TARGET_COL]].isnull().sum().sum()}")

print("\n✂️  Winsorization（P95）...")
for col in FEATURE_COLS + [TARGET_COL]:
    p95 = df[col].quantile(0.95)
    clipped = (df[col] > p95).sum()
    df[col] = df[col].clip(upper=p95)
    if clipped > 0:
        print(f"  {col}: 壓縮 {clipped} 筆（上限 {p95:,.0f}）")

print("\n📐 標準化...")
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

df[FEATURE_COLS] = feature_scaler.fit_transform(df[FEATURE_COLS])
df[[TARGET_COL]] = target_scaler.fit_transform(df[[TARGET_COL]])
print("  完成")

print(f"\n🪟 建立滑動視窗（INPUT_DAYS={INPUT_DAYS}）...")
X_list, y_list = [], []

for user_id in df["user_id"].unique():
    u = df[df["user_id"] == user_id].reset_index(drop=True)
    features_arr = u[FEATURE_COLS].values
    target_arr = u[TARGET_COL].values

    for t in range(INPUT_DAYS, len(u)):
        X_list.append(features_arr[t - INPUT_DAYS : t])
        y_list.append([target_arr[t]])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
print(f"  X shape : {X.shape} -> (樣本數, {INPUT_DAYS}天, {len(FEATURE_COLS)}特徵)")
print(f"  y shape : {y.shape}")

split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]
print(f"\n✂️  訓練集 : {X_train.shape}  驗證集 : {X_val.shape}")

print(f"\n💾 儲存至 {ARTIFACTS_DIR}/...")

np.save(f"{ARTIFACTS_DIR}/personal_X_train.npy", X_train)
np.save(f"{ARTIFACTS_DIR}/personal_y_train.npy", y_train)
np.save(f"{ARTIFACTS_DIR}/personal_X_val.npy", X_val)
np.save(f"{ARTIFACTS_DIR}/personal_y_val.npy", y_val)

with open(f"{ARTIFACTS_DIR}/personal_feature_scaler.pkl", "wb") as f:
    pickle.dump(feature_scaler, f)
with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)

print("  ✅ personal_X_train / y_train / X_val / y_val")
print("  ✅ personal_feature_scaler.pkl")
print("  ✅ personal_target_scaler.pkl")
print("\n🎉 完成！下一步：執行 finetune_lstm.py")
