"""
步驟一（更新版）：Walmart 資料前處理 → 日級別，7個特徵
================================
特徵對齊個人記帳資料的最佳組合：
  daily_expense   → Walmart 每日銷售額（週銷售 ÷ 7）
  expense_7d_mean → 過去 7 天平均銷售
  expense_30d_sum → 過去 30 天銷售加總
  has_expense     → 當日是否有銷售（幾乎恆為 1）
  has_income      → Walmart 無收入資料，全填 0
  net_30d_sum     → 過去 30 天淨值（無收入，等於 -expense_30d_sum）
  txn_30d_sum     → 過去 30 天部門數加總（代理交易筆數）
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# 0. 設定
# ─────────────────────────────────────────
DATA_DIR      = "wallmart"
ARTIFACTS_DIR = "artificats"
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

# ─────────────────────────────────────────
# 1. 讀取並彙整 Walmart 週資料
# ─────────────────────────────────────────
print("📂 讀取 Walmart 資料...")

train    = pd.read_csv(f"{DATA_DIR}/train.csv")
features = pd.read_csv(f"{DATA_DIR}/features.csv")
stores   = pd.read_csv(f"{DATA_DIR}/stores.csv")

df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
df = df.merge(stores, on="Store", how="left")
df["Date"] = pd.to_datetime(df["Date"])

store_weekly = df.groupby(["Store", "Date"]).agg(
    Weekly_Sales = ("Weekly_Sales", "sum"),
    Dept_Count   = ("Dept",         "nunique"),
    IsHoliday    = ("IsHoliday",    "first"),
    CPI          = ("CPI",          "first"),
    Unemployment = ("Unemployment", "first"),
).reset_index()
store_weekly = store_weekly.sort_values(["Store", "Date"]).reset_index(drop=True)

for col in ["CPI", "Unemployment"]:
    store_weekly[col] = (
        store_weekly.groupby("Store")[col]
        .transform(lambda x: x.ffill().bfill())
    )

print(f"  週彙整後 shape : {store_weekly.shape}")


# ─────────────────────────────────────────
# 2. 展開週 → 日
# ─────────────────────────────────────────
print("\n📅 展開週資料 → 日資料...")

rows = []
for _, row in store_weekly.iterrows():
    for d in range(7):
        day = row["Date"] + pd.Timedelta(days=d)
        rows.append({
            "Store"         : row["Store"],
            "Date"          : day,
            "daily_expense" : row["Weekly_Sales"] / 7,
            "has_expense"   : 1,
            "has_income"    : 0,
            "dept_count"    : row["Dept_Count"],
        })

daily = pd.DataFrame(rows)
daily = daily.sort_values(["Store", "Date"]).reset_index(drop=True)
print(f"  展開後 shape : {daily.shape}")


# ─────────────────────────────────────────
# 3. 計算滾動視窗特徵
# ─────────────────────────────────────────
print("\n📊 計算滾動視窗特徵...")

result_list = []
for store_id in sorted(daily["Store"].unique()):
    s = daily[daily["Store"] == store_id].copy().reset_index(drop=True)

    s["expense_7d_mean"] = s["daily_expense"].rolling(7,  min_periods=1).mean()
    s["expense_30d_sum"] = s["daily_expense"].rolling(30, min_periods=1).sum()
    s["net_30d_sum"]     = -s["expense_30d_sum"]
    s["txn_30d_sum"]     = s["dept_count"].rolling(30, min_periods=1).sum()
    s["future_expense_7d_sum"] = s["daily_expense"].rolling(7).sum().shift(-7)

    result_list.append(s)

daily  = pd.concat(result_list).reset_index(drop=True)
before = len(daily)
daily  = daily.dropna(subset=["future_expense_7d_sum"]).reset_index(drop=True)
print(f"  計算完後：{before} → {len(daily)} 筆")


# ─────────────────────────────────────────
# 4. Winsorization（P95）
# ─────────────────────────────────────────
print("\n✂️  Winsorization（P95）...")
for col in FEATURE_COLS + [TARGET_COL]:
    p95     = daily[col].quantile(0.95)
    clipped = (daily[col] > p95).sum()
    daily[col] = daily[col].clip(upper=p95)
    if clipped > 0:
        print(f"  {col}: 壓縮 {clipped} 筆（上限 {p95:,.0f}）")


# ─────────────────────────────────────────
# 5. 標準化
# ─────────────────────────────────────────
print("\n📐 標準化...")
feature_scaler = StandardScaler()
target_scaler  = StandardScaler()
daily[FEATURE_COLS]  = feature_scaler.fit_transform(daily[FEATURE_COLS])
daily[[TARGET_COL]]  = target_scaler.fit_transform(daily[[TARGET_COL]])


# ─────────────────────────────────────────
# 6. 滑動視窗
# ─────────────────────────────────────────
print(f"\n🪟 建立滑動視窗（INPUT_DAYS={INPUT_DAYS}）...")
X_list, y_list = [], []

for store_id in sorted(daily["Store"].unique()):
    s            = daily[daily["Store"] == store_id].reset_index(drop=True)
    features_arr = s[FEATURE_COLS].values
    target_arr   = s[TARGET_COL].values

    for t in range(INPUT_DAYS, len(s)):
        X_list.append(features_arr[t - INPUT_DAYS : t])
        y_list.append([target_arr[t]])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
print(f"  X shape : {X.shape}  → (樣本數, {INPUT_DAYS}天, {len(FEATURE_COLS)}特徵)")
print(f"  y shape : {y.shape}")


# ─────────────────────────────────────────
# 7. 切割
# ─────────────────────────────────────────
split_idx      = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]
print(f"\n✂️  訓練集 : {X_train.shape}  驗證集 : {X_val.shape}")


# ─────────────────────────────────────────
# 8. 儲存
# ─────────────────────────────────────────
print(f"\n💾 儲存至 {ARTIFACTS_DIR}/...")

np.save(f"{ARTIFACTS_DIR}/walmart_X_train.npy", X_train)
np.save(f"{ARTIFACTS_DIR}/walmart_y_train.npy", y_train)
np.save(f"{ARTIFACTS_DIR}/walmart_X_val.npy",   X_val)
np.save(f"{ARTIFACTS_DIR}/walmart_y_val.npy",   y_val)

with open(f"{ARTIFACTS_DIR}/walmart_feature_scaler.pkl", "wb") as f:
    pickle.dump(feature_scaler, f)
with open(f"{ARTIFACTS_DIR}/walmart_target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)

daily.to_csv(f"{ARTIFACTS_DIR}/walmart_daily_processed.csv", index=False)

print("  ✅ walmart_X_train / y_train / X_val / y_val")
print("  ✅ walmart_feature_scaler.pkl")
print("  ✅ walmart_target_scaler.pkl")
print("  ✅ walmart_daily_processed.csv")
print(f"\n🎉 完成！下一步：執行 pretrain_gru.py")