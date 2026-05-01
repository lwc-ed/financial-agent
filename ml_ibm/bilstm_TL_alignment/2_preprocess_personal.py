"""
Step 2：個人資料前處理
=======================
使用與 IBM 完全相同的 10 個 Aligned 特徵
輸出：
  - personal_X_train/val/test.npy
  - personal_y_train/val/test.npy（scaled）
  - personal_y_test_raw.npy（原始金額）
  - personal_target_scaler.pkl
  - personal_train/val/test_user_ids.npy
  - personal_y_train_risk_labels.npy  ← 新增：4-class risk label（用於 MT finetune）
  - personal_y_val_risk_labels.npy    ← 新增
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
    PERSONAL_DIR,
)

ARTIFACTS_DIR = "artifacts_bilstm_v2"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def _load_income_monthly(user_id: str, train_start, train_end) -> float:
    """訓練期間 monthly_available_cash = 總收入 / 月數"""
    fname = os.path.join(PERSONAL_DIR, f"raw_transactions_{user_id}.xlsx")
    if not os.path.exists(fname):
        return 0.0
    df = pd.read_excel(fname)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"]).dt.normalize()
    income = df[df["transaction_type"] == "Income"].copy()
    ts, te = pd.Timestamp(train_start), pd.Timestamp(train_end)
    income = income[(income["time_stamp"] >= ts) & (income["time_stamp"] <= te)]
    total  = float(income["amount"].sum())
    months = max((te - ts).days / 30.0, 1.0)
    return total / months


def _risk_label(expense_7d: float, monthly_cash: float) -> int:
    """future_available_7d ≈ monthly_cash * 7/30，依 spec 切 4 class"""
    avail = monthly_cash * 7.0 / 30.0
    if avail <= 0:
        return 0
    r = expense_7d / avail
    if r < 0.8:  return 0
    if r < 1.0:  return 1
    if r < 1.2:  return 2
    return 3


print("📂 載入個人資料...")
df = load_personal_daily()
print(f"  共 {len(df):,} 筆 | {df['user_id'].nunique()} 位用戶")

print(f"\n📊 計算 {len(ALIGNED_FEATURE_COLS)} 個 Aligned 特徵...")
result_list = []

for user_id in sorted(df["user_id"].unique()):
    u = df[df["user_id"] == user_id].sort_values("date").reset_index(drop=True)
    feats = compute_aligned_features(u["daily_expense"], u["date"])
    u["future_expense_7d_sum"] = u["daily_expense"].rolling(7).sum().shift(-7)
    feats["future_expense_7d_sum"] = u["future_expense_7d_sum"].values
    feats["user_id"] = user_id
    feats["date"] = u["date"].values
    result_list.append(feats)

daily = pd.concat(result_list).reset_index(drop=True)
daily = daily.dropna(subset=["future_expense_7d_sum"]).reset_index(drop=True)
print(f"  有效筆數：{len(daily):,}")

print(f"\n🪟 滑動視窗 + per-user 70/15/15 切分...")
X_train_list, y_train_list, train_uid, train_dates = [], [], [], []
X_val_list,   y_val_list,   val_uid,   val_dates   = [], [], [], []
X_test_list,  y_test_list,  test_uid,  test_dates  = [], [], [], []
train_risk_list, val_risk_list = [], []

for user_id in sorted(daily["user_id"].unique()):
    u          = daily[daily["user_id"] == user_id].reset_index(drop=True)
    feat_arr   = u[ALIGNED_FEATURE_COLS].values.astype(np.float32)
    target_arr = u[TARGET_COL].values.astype(np.float32)
    date_arr   = u["date"].values

    windows_X, windows_y, windows_dates = [], [], []
    for t in range(INPUT_DAYS, len(u)):
        windows_X.append(feat_arr[t - INPUT_DAYS : t])
        windows_y.append([target_arr[t]])
        windows_dates.append(date_arr[t])

    n = len(windows_X)
    if n < 5:
        print(f"  ⚠️  {user_id} 資料不足（{n}），跳過")
        continue

    t_end = int(n * 0.70)
    v_end = int(n * 0.85)
    if t_end == 0:
        continue

    # ── 計算 risk labels（僅用訓練期間收入算 monthly_cash）─────────────────────
    train_w_dates = windows_dates[:t_end]
    monthly_cash  = _load_income_monthly(
        user_id,
        train_w_dates[0] if train_w_dates else date_arr[0],
        train_w_dates[-1] if train_w_dates else date_arr[-1],
    )
    tr_risk = [_risk_label(y[0], monthly_cash) for y in windows_y[:t_end]]
    vl_risk = [_risk_label(y[0], monthly_cash) for y in windows_y[t_end:v_end]]

    X_train_list.extend(windows_X[:t_end]);    y_train_list.extend(windows_y[:t_end])
    X_val_list.extend(windows_X[t_end:v_end]); y_val_list.extend(windows_y[t_end:v_end])
    X_test_list.extend(windows_X[v_end:]);     y_test_list.extend(windows_y[v_end:])
    train_uid.extend([user_id] * t_end)
    val_uid.extend([user_id]   * (v_end - t_end))
    test_uid.extend([user_id]  * (n - v_end))
    train_dates.extend(windows_dates[:t_end])
    val_dates.extend(windows_dates[t_end:v_end])
    test_dates.extend(windows_dates[v_end:])
    train_risk_list.extend(tr_risk)
    val_risk_list.extend(vl_risk)

X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)
X_val   = np.array(X_val_list,   dtype=np.float32)
y_val   = np.array(y_val_list,   dtype=np.float32)
X_test  = np.array(X_test_list,  dtype=np.float32)
y_test  = np.array(y_test_list,  dtype=np.float32)
print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

train_risk = np.array(train_risk_list, dtype=np.int64)
val_risk   = np.array(val_risk_list,   dtype=np.int64)
from collections import Counter
print(f"  Train risk 分佈: {dict(sorted(Counter(train_risk.tolist()).items()))}")

print("\n📐 標準化 Target（fit on personal train）...")
target_scaler = StandardScaler()
target_scaler.fit(y_train)
y_train_s = target_scaler.transform(y_train).astype(np.float32)
y_val_s   = target_scaler.transform(y_val).astype(np.float32)
y_test_s  = target_scaler.transform(y_test).astype(np.float32)
print(f"  mean={target_scaler.mean_[0]:.2f}  std={target_scaler.scale_[0]:.2f}")

print(f"\n💾 儲存至 {ARTIFACTS_DIR}/...")
np.save(f"{ARTIFACTS_DIR}/personal_X_train.npy",    X_train)
np.save(f"{ARTIFACTS_DIR}/personal_y_train.npy",    y_train_s)
np.save(f"{ARTIFACTS_DIR}/personal_X_val.npy",      X_val)
np.save(f"{ARTIFACTS_DIR}/personal_y_val.npy",      y_val_s)
np.save(f"{ARTIFACTS_DIR}/personal_X_test.npy",     X_test)
np.save(f"{ARTIFACTS_DIR}/personal_y_test.npy",     y_test_s)
np.save(f"{ARTIFACTS_DIR}/personal_y_test_raw.npy", y_test)
np.save(f"{ARTIFACTS_DIR}/personal_train_user_ids.npy", np.array(train_uid))
np.save(f"{ARTIFACTS_DIR}/personal_val_user_ids.npy",   np.array(val_uid))
np.save(f"{ARTIFACTS_DIR}/personal_test_user_ids.npy",  np.array(test_uid))
np.save(f"{ARTIFACTS_DIR}/personal_train_dates.npy", np.array(train_dates, dtype="datetime64[D]"))
np.save(f"{ARTIFACTS_DIR}/personal_val_dates.npy",   np.array(val_dates,   dtype="datetime64[D]"))
np.save(f"{ARTIFACTS_DIR}/personal_test_dates.npy",  np.array(test_dates,  dtype="datetime64[D]"))
np.save(f"{ARTIFACTS_DIR}/personal_y_train_risk_labels.npy", train_risk)
np.save(f"{ARTIFACTS_DIR}/personal_y_val_risk_labels.npy",   val_risk)
with open(f"{ARTIFACTS_DIR}/personal_target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)

print("  ✅ 完成！下一步：3_pretrain_bilstm.py")
