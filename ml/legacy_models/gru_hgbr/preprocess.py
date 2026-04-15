"""
gru_hgbr/preprocess.py
========================
準備兩種特徵：
  1. GRU 序列輸入（30天視窗，7個特徵）→ 用於 extract_embeddings.py 提取 embedding
  2. HGBR 扁平特徵（22個特徵）         → 與 embedding 拼接後輸入 HGBR

注意：GRU 序列使用 ml_gru 原版的 StandardScaler（personal_feature_scaler.pkl），
以確保輸入分佈與 finetune_gru_v4 預訓練時一致，embedding 才有意義。

執行順序：
  python preprocess.py → python extract_embeddings.py → python train_hgbr.py → python predict.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

os.makedirs("artifacts", exist_ok=True)

DATA_PATH              = Path(__file__).resolve().parents[2] / "processed_data" / "artifacts" / "features_all.csv"
ML_GRU_ARTIFACTS       = "../ml_gru/artificats"   # 使用 ml_gru 的 scaler 保持一致性
INPUT_DAYS             = 30
USER_CLIP_PERCENTILE   = 99

GRU_FEATURES = [
    "daily_expense",
    "expense_7d_mean",
    "expense_30d_sum",
    "has_expense",
    "has_income",
    "net_30d_sum",
    "txn_30d_sum",
]

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
# 載入 ml_gru 的 scaler（GRU 輸入必須與訓練時一致）
# ─────────────────────────────────────────
print("📦 載入 ml_gru 的 feature_scaler 和 clip_values...")
with open(f"{ML_GRU_ARTIFACTS}/personal_feature_scaler.pkl", "rb") as f:
    gru_feat_scaler = pickle.load(f)
with open(f"{ML_GRU_ARTIFACTS}/personal_user_clip_values.pkl", "rb") as f:
    ml_gru_clip_map = pickle.load(f)
with open(f"{ML_GRU_ARTIFACTS}/personal_target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)
print("  ✅ 使用 ml_gru 原版 scaler（確保 GRU embedding 分佈正確）")

# ─────────────────────────────────────────
# 1. 讀取資料
# ─────────────────────────────────────────
print("\n📂 讀取 features_all.csv...")
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

gru_Xtr, gru_Xva, gru_Xte     = [], [], []
hgbr_Xtr, hgbr_Xva, hgbr_Xte = [], [], []
y_tr, y_va, y_te               = [], [], []
tr_uids, va_uids, te_uids      = [], [], []
new_clip_map = {}   # 儲存本次計算的 clip 值（作備份）

for user_id in df["user_id"].unique():
    u          = df[df["user_id"] == user_id].reset_index(drop=True)
    gru_feats  = u[GRU_FEATURES].values.astype(np.float32)
    hgbr_feats = u[HGBR_FEATURES].values.astype(np.float32)
    target     = u[TARGET_COL].values.astype(np.float32)

    gru_wins, hgbr_rows, y_vals = [], [], []
    for t in range(INPUT_DAYS, len(u)):
        gru_wins.append(gru_feats[t - INPUT_DAYS:t])
        hgbr_rows.append(hgbr_feats[t])
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

    # 優先使用 ml_gru 的 clip 值；若不存在則重新計算
    uid_key = str(user_id)
    if uid_key in ml_gru_clip_map:
        clip_vals = ml_gru_clip_map[uid_key]
    else:
        clip_vals = {
            col: float(np.percentile(gru_w[:t_end, :, i], USER_CLIP_PERCENTILE))
            for i, col in enumerate(GRU_FEATURES)
        }
    new_clip_map[uid_key] = clip_vals

    def clip_seq(arr):
        out = arr.copy()
        for i, col in enumerate(GRU_FEATURES):
            out[:, :, i] = np.clip(out[:, :, i], None, clip_vals[col])
        return out

    gru_Xtr.extend(clip_seq(gru_w[:t_end]))
    gru_Xva.extend(clip_seq(gru_w[t_end:v_end]))
    gru_Xte.extend(clip_seq(gru_w[v_end:]))

    hgbr_Xtr.extend(hgbr_f[:t_end])
    hgbr_Xva.extend(hgbr_f[t_end:v_end])
    hgbr_Xte.extend(hgbr_f[v_end:])

    y_tr.extend(y_arr[:t_end])
    y_va.extend(y_arr[t_end:v_end])
    y_te.extend(y_arr[v_end:])

    tr_uids.extend([user_id] * t_end)
    va_uids.extend([user_id] * (v_end - t_end))
    te_uids.extend([user_id] * (n - v_end))

gru_Xtr  = np.array(gru_Xtr,  dtype=np.float32)
gru_Xva  = np.array(gru_Xva,  dtype=np.float32)
gru_Xte  = np.array(gru_Xte,  dtype=np.float32)
hgbr_Xtr = np.array(hgbr_Xtr, dtype=np.float32)
hgbr_Xva = np.array(hgbr_Xva, dtype=np.float32)
hgbr_Xte = np.array(hgbr_Xte, dtype=np.float32)
y_tr = np.array(y_tr, dtype=np.float32).reshape(-1, 1)
y_va = np.array(y_va, dtype=np.float32).reshape(-1, 1)
y_te = np.array(y_te, dtype=np.float32).reshape(-1, 1)

print(f"  GRU  Train: {gru_Xtr.shape}  Val: {gru_Xva.shape}  Test: {gru_Xte.shape}")
print(f"  HGBR Train: {hgbr_Xtr.shape}  Val: {hgbr_Xva.shape}  Test: {hgbr_Xte.shape}")

# ─────────────────────────────────────────
# 3. 套用 ml_gru 的 GRU 特徵 scaler（不重新 fit）
# ─────────────────────────────────────────
print("\n📐 套用 ml_gru GRU 特徵 scaler...")
gru_Xtr = gru_feat_scaler.transform(gru_Xtr.reshape(-1, len(GRU_FEATURES))).reshape(gru_Xtr.shape).astype(np.float32)
gru_Xva = gru_feat_scaler.transform(gru_Xva.reshape(-1, len(GRU_FEATURES))).reshape(gru_Xva.shape).astype(np.float32)
gru_Xte = gru_feat_scaler.transform(gru_Xte.reshape(-1, len(GRU_FEATURES))).reshape(gru_Xte.shape).astype(np.float32)

# ─────────────────────────────────────────
# 4. HGBR 特徵 scaler（重新 fit，因為 22 個特徵）
# ─────────────────────────────────────────
print("📐 fit HGBR 特徵 scaler...")
hgbr_feat_scaler = StandardScaler()
hgbr_feat_scaler.fit(hgbr_Xtr)
hgbr_Xtr = hgbr_feat_scaler.transform(hgbr_Xtr).astype(np.float32)
hgbr_Xva = hgbr_feat_scaler.transform(hgbr_Xva).astype(np.float32)
hgbr_Xte = hgbr_feat_scaler.transform(hgbr_Xte).astype(np.float32)

# ─────────────────────────────────────────
# 5. 儲存
# ─────────────────────────────────────────
print("\n💾 儲存至 artifacts/...")
np.save("artifacts/gru_X_train.npy",  gru_Xtr)
np.save("artifacts/gru_X_val.npy",    gru_Xva)
np.save("artifacts/gru_X_test.npy",   gru_Xte)
np.save("artifacts/hgbr_X_train.npy", hgbr_Xtr)
np.save("artifacts/hgbr_X_val.npy",   hgbr_Xva)
np.save("artifacts/hgbr_X_test.npy",  hgbr_Xte)
np.save("artifacts/y_train.npy",      y_tr)
np.save("artifacts/y_val.npy",        y_va)
np.save("artifacts/y_test.npy",       y_te)
np.save("artifacts/train_uids.npy",   np.array(tr_uids))
np.save("artifacts/val_uids.npy",     np.array(va_uids))
np.save("artifacts/test_uids.npy",    np.array(te_uids))

with open("artifacts/hgbr_feat_scaler.pkl", "wb") as f:
    pickle.dump(hgbr_feat_scaler, f)

print("✅ 完成！下一步：python extract_embeddings.py")
