"""
ml_gru_nopretrain/preprocess.py
================================
同時準備兩個模型所需的資料：
  1. HGBR 扁平特徵（22 個特徵，包含日曆、滾動統計）
  2. GRU from scratch 序列資料（30 天視窗，7 個特徵）

執行順序：
  python preprocess.py → python train_hgbr.py → python train_gru_scratch.py → python predict.py
"""

import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

os.makedirs("artifacts", exist_ok=True)

DATA_PATH            = Path(__file__).resolve().parents[2] / "processed_data" / "artifacts" / "features_all.csv"
INPUT_DAYS           = 30
USER_CLIP_PERCENTILE = 99

# GRU 使用的 7 個核心特徵（與原版 ml_gru 一致）
GRU_FEATURES = [
    "daily_expense",
    "expense_7d_mean",
    "expense_30d_sum",
    "has_expense",
    "has_income",
    "net_30d_sum",
    "txn_30d_sum",
]

# HGBR 使用豐富特徵集（22 個）：原版 GRU 沒用到的日曆與統計特徵
HGBR_FEATURES = [
    "daily_expense", "daily_income", "daily_net",
    "has_expense", "has_income",
    "dow", "is_weekend", "day", "month",
    "is_summer_vacation", "is_winter_vacation", "days_to_end_of_month",
    "expense_7d_sum", "expense_7d_mean", "net_7d_sum", "txn_7d_sum",
    "expense_30d_sum", "expense_30d_mean", "net_30d_sum", "txn_30d_sum",
    "expense_7d_30d_ratio", "expense_trend",
]

TARGET_COL = "future_expense_7d_sum"

# ─────────────────────────────────────────
# 1. 讀取資料
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
print(f"\n🪟 建立滑動視窗（INPUT_DAYS={INPUT_DAYS}，per-user 70/15/15）...")

gru_X_tr, gru_X_va, gru_X_te     = [], [], []
hgbr_X_tr, hgbr_X_va, hgbr_X_te = [], [], []
y_tr, y_va, y_te                  = [], [], []
tr_uids, va_uids, te_uids         = [], [], []
user_clip_map = {}

for user_id in df["user_id"].unique():
    u          = df[df["user_id"] == user_id].reset_index(drop=True)
    gru_feats  = u[GRU_FEATURES].values.astype(np.float32)
    hgbr_feats = u[HGBR_FEATURES].values.astype(np.float32)
    target     = u[TARGET_COL].values.astype(np.float32)

    gru_wins, hgbr_rows, y_vals = [], [], []
    for t in range(INPUT_DAYS, len(u)):
        gru_wins.append(gru_feats[t - INPUT_DAYS:t])   # shape (30, 7)
        hgbr_rows.append(hgbr_feats[t])                 # shape (22,) 當前時間點的所有特徵
        y_vals.append(target[t])

    if len(gru_wins) == 0:
        continue

    n     = len(gru_wins)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)
    if t_end == 0:
        continue

    gru_w  = np.array(gru_wins,  dtype=np.float32)
    hgbr_f = np.array(hgbr_rows, dtype=np.float32)
    y_arr  = np.array(y_vals,    dtype=np.float32)

    # Per-user P99 clipping（只在 GRU 序列特徵上做）
    clip_vals = {
        col: float(np.percentile(gru_w[:t_end, :, i], USER_CLIP_PERCENTILE))
        for i, col in enumerate(GRU_FEATURES)
    }
    user_clip_map[str(user_id)] = clip_vals

    def clip_seq(arr):
        out = arr.copy()
        for i, col in enumerate(GRU_FEATURES):
            out[:, :, i] = np.clip(out[:, :, i], None, clip_vals[col])
        return out

    gru_X_tr.extend(clip_seq(gru_w[:t_end]))
    gru_X_va.extend(clip_seq(gru_w[t_end:v_end]))
    gru_X_te.extend(clip_seq(gru_w[v_end:]))

    hgbr_X_tr.extend(hgbr_f[:t_end])
    hgbr_X_va.extend(hgbr_f[t_end:v_end])
    hgbr_X_te.extend(hgbr_f[v_end:])

    y_tr.extend(y_arr[:t_end])
    y_va.extend(y_arr[t_end:v_end])
    y_te.extend(y_arr[v_end:])

    tr_uids.extend([user_id] * t_end)
    va_uids.extend([user_id] * (v_end - t_end))
    te_uids.extend([user_id] * (n - v_end))

gru_X_tr  = np.array(gru_X_tr,  dtype=np.float32)
gru_X_va  = np.array(gru_X_va,  dtype=np.float32)
gru_X_te  = np.array(gru_X_te,  dtype=np.float32)
hgbr_X_tr = np.array(hgbr_X_tr, dtype=np.float32)
hgbr_X_va = np.array(hgbr_X_va, dtype=np.float32)
hgbr_X_te = np.array(hgbr_X_te, dtype=np.float32)
y_tr      = np.array(y_tr, dtype=np.float32).reshape(-1, 1)
y_va      = np.array(y_va, dtype=np.float32).reshape(-1, 1)
y_te      = np.array(y_te, dtype=np.float32).reshape(-1, 1)

print(f"  GRU   Train: {gru_X_tr.shape}  Val: {gru_X_va.shape}  Test: {gru_X_te.shape}")
print(f"  HGBR  Train: {hgbr_X_tr.shape}  Val: {hgbr_X_va.shape}  Test: {hgbr_X_te.shape}")

# ─────────────────────────────────────────
# 3. 標準化
# ─────────────────────────────────────────
print("\n📐 標準化（fit on train only）...")

# GRU 特徵 & 目標
gru_feat_scaler = StandardScaler()
gru_feat_scaler.fit(gru_X_tr.reshape(-1, len(GRU_FEATURES)))
gru_X_tr = gru_feat_scaler.transform(gru_X_tr.reshape(-1, len(GRU_FEATURES))).reshape(gru_X_tr.shape).astype(np.float32)
gru_X_va = gru_feat_scaler.transform(gru_X_va.reshape(-1, len(GRU_FEATURES))).reshape(gru_X_va.shape).astype(np.float32)
gru_X_te = gru_feat_scaler.transform(gru_X_te.reshape(-1, len(GRU_FEATURES))).reshape(gru_X_te.shape).astype(np.float32)

target_scaler = StandardScaler()
target_scaler.fit(y_tr)
y_tr_sc = target_scaler.transform(y_tr).astype(np.float32)
y_va_sc = target_scaler.transform(y_va).astype(np.float32)
y_te_sc = target_scaler.transform(y_te).astype(np.float32)

# HGBR 特徵（不需縮放目標，HGBR 直接預測原始值）
hgbr_feat_scaler = StandardScaler()
hgbr_feat_scaler.fit(hgbr_X_tr)
hgbr_X_tr = hgbr_feat_scaler.transform(hgbr_X_tr).astype(np.float32)
hgbr_X_va = hgbr_feat_scaler.transform(hgbr_X_va).astype(np.float32)
hgbr_X_te = hgbr_feat_scaler.transform(hgbr_X_te).astype(np.float32)

# ─────────────────────────────────────────
# 4. 儲存
# ─────────────────────────────────────────
print("\n💾 儲存至 artifacts/...")
np.save("artifacts/gru_X_train.npy",  gru_X_tr)
np.save("artifacts/gru_X_val.npy",    gru_X_va)
np.save("artifacts/gru_X_test.npy",   gru_X_te)
np.save("artifacts/hgbr_X_train.npy", hgbr_X_tr)
np.save("artifacts/hgbr_X_val.npy",   hgbr_X_va)
np.save("artifacts/hgbr_X_test.npy",  hgbr_X_te)
np.save("artifacts/y_train.npy",      y_tr)
np.save("artifacts/y_val.npy",        y_va)
np.save("artifacts/y_test.npy",       y_te)
np.save("artifacts/y_train_sc.npy",   y_tr_sc)
np.save("artifacts/y_val_sc.npy",     y_va_sc)
np.save("artifacts/y_test_sc.npy",    y_te_sc)
np.save("artifacts/train_uids.npy",   np.array(tr_uids))
np.save("artifacts/val_uids.npy",     np.array(va_uids))
np.save("artifacts/test_uids.npy",    np.array(te_uids))

with open("artifacts/gru_feature_scaler.pkl", "wb") as f:
    pickle.dump(gru_feat_scaler, f)
with open("artifacts/target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)
with open("artifacts/hgbr_feature_scaler.pkl", "wb") as f:
    pickle.dump(hgbr_feat_scaler, f)
with open("artifacts/user_clip_values.pkl", "wb") as f:
    pickle.dump(user_clip_map, f)

print("✅ 完成！下一步：python train_hgbr.py  或  python train_gru_scratch.py")
