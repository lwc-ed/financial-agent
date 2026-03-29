"""
experiment_exclude_users.py
=============================
實驗：排除高消費不規律用戶（user4/5/6），看模型是否改善。

假設：user4/5/6 的極端消費模式污染了 StandardScaler，
      導致其他 13 個用戶的特徵在縮放後失去精度。
      排除後重新訓練，觀察 13 人的 MAE 是否下降。

同時對比：
  A. 全 16 人訓練，只評估 13 人的子集 MAE
  B. 只用 13 人訓練，評估 13 人的子集 MAE
"""

import os
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

os.makedirs("artifacts", exist_ok=True)

DATA_PATH            = "../ml_gru/features_all.csv"
EXCLUDE_USERS        = ["user4", "user5", "user6"]
INPUT_DAYS           = 30
USER_CLIP_PERCENTILE = 99
TARGET_COL           = "future_expense_7d_sum"

HGBR_FEATURES = [
    "daily_expense", "daily_income", "daily_net",
    "has_expense", "has_income",
    "dow", "is_weekend", "day", "month",
    "is_summer_vacation", "is_winter_vacation", "days_to_end_of_month",
    "expense_7d_sum", "expense_7d_mean", "net_7d_sum", "txn_7d_sum",
    "expense_30d_sum", "expense_30d_mean", "net_30d_sum", "txn_30d_sum",
    "expense_7d_30d_ratio", "expense_trend",
]

def smape(yt, yp):
    yt, yp = yt.flatten(), yp.flatten()
    d = (np.abs(yt) + np.abs(yp)) / 2
    m = d > 0
    return float(np.mean(np.abs(yp[m] - yt[m]) / d[m]) * 100)

def per_user_nmae(yt, yp, uids):
    r = []
    for u in np.unique(uids):
        mask = np.array(uids) == u
        mu = yt[mask].mean()
        if mu > 0:
            r.append(np.mean(np.abs(yp[mask] - yt[mask])) / mu * 100)
    return float(np.mean(r))

def build_dataset(df_src, user_clip_percentile=99):
    """從 dataframe 建立 HGBR 扁平特徵資料集"""
    Xtr, Xva, Xte = [], [], []
    ytr, yva, yte = [], [], []
    tr_uids, va_uids, te_uids = [], [], []
    clip_map = {}

    for uid in df_src["user_id"].unique():
        u      = df_src[df_src["user_id"] == uid].reset_index(drop=True)
        feats  = u[HGBR_FEATURES].values.astype(np.float32)
        target = u[TARGET_COL].values.astype(np.float32)

        rows, y_vals = [], []
        for t in range(INPUT_DAYS, len(u)):
            rows.append(feats[t])
            y_vals.append(target[t])

        if len(rows) == 0:
            continue

        n     = len(rows)
        t_end = int(n * 0.70)
        v_end = int(n * 0.85)
        if t_end == 0:
            continue

        f_arr = np.array(rows,   dtype=np.float32)
        y_arr = np.array(y_vals, dtype=np.float32)

        Xtr.extend(f_arr[:t_end]);  ytr.extend(y_arr[:t_end])
        Xva.extend(f_arr[t_end:v_end]); yva.extend(y_arr[t_end:v_end])
        Xte.extend(f_arr[v_end:]);  yte.extend(y_arr[v_end:])
        tr_uids.extend([uid] * t_end)
        va_uids.extend([uid] * (v_end - t_end))
        te_uids.extend([uid] * (n - v_end))

    Xtr = np.array(Xtr, dtype=np.float32)
    Xva = np.array(Xva, dtype=np.float32)
    Xte = np.array(Xte, dtype=np.float32)
    ytr = np.array(ytr, dtype=np.float32)
    yva = np.array(yva, dtype=np.float32)
    yte = np.array(yte, dtype=np.float32)

    scaler = StandardScaler()
    scaler.fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xva = scaler.transform(Xva).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    return (Xtr, ytr, np.array(tr_uids),
            Xva, yva, np.array(va_uids),
            Xte, yte, np.array(te_uids),
            scaler)


def train_hgbr(Xtr, ytr, Xva, yva):
    Xtv = np.concatenate([Xtr, Xva])
    ytv = np.concatenate([ytr, yva])
    model = HistGradientBoostingRegressor(
        max_iter=1000, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=0.1,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=42, verbose=0,
    )
    model.fit(Xtv, ytv)
    return model


# ─────────────────────────────────────────
# 讀取資料
# ─────────────────────────────────────────
print("📂 讀取 features_all.csv...")
df_all = pd.read_csv(DATA_PATH)
df_all["date"] = pd.to_datetime(df_all["date"])
df_all = df_all.sort_values(["user_id", "date"]).reset_index(drop=True)

all_users      = list(df_all["user_id"].unique())
included_users = [u for u in all_users if u not in EXCLUDE_USERS]
print(f"  全部用戶: {len(all_users)} 人")
print(f"  排除：{EXCLUDE_USERS}")
print(f"  保留：{len(included_users)} 人")

# ─────────────────────────────────────────
# 情境 A：全 16 人訓練，只看 13 人子集的結果
# ─────────────────────────────────────────
print("\n" + "="*60)
print("情境 A：全 16 人訓練，評估 13 人子集")
print("="*60)
Xtr_A, ytr_A, tr_A, Xva_A, yva_A, va_A, Xte_A, yte_A, te_A, sc_A = build_dataset(df_all)
model_A = train_hgbr(Xtr_A, ytr_A, Xva_A, yva_A)

# 只看 13 人的 test 結果
mask_13_A = np.array([u not in EXCLUDE_USERS for u in te_A])
yp_A  = model_A.predict(Xte_A[mask_13_A])
yt_A  = yte_A[mask_13_A]
uid_A = te_A[mask_13_A]

mae_A   = float(np.mean(np.abs(yp_A - yt_A)))
rmse_A  = float(np.sqrt(np.mean((yp_A - yt_A)**2)))
smape_A = smape(yt_A, yp_A)
nmae_A  = per_user_nmae(yt_A, yp_A, uid_A)
print(f"  13人子集 Test MAE : {mae_A:,.2f}")
print(f"  13人子集 Test RMSE: {rmse_A:,.2f}")
print(f"  13人子集 SMAPE    : {smape_A:.2f}%")
print(f"  13人子集 NMAE     : {nmae_A:.2f}%")

# ─────────────────────────────────────────
# 情境 B：只用 13 人訓練，只看 13 人子集的結果
# ─────────────────────────────────────────
print("\n" + "="*60)
print("情境 B：只用 13 人訓練，評估 13 人子集")
print("="*60)
df_13 = df_all[df_all["user_id"].isin(included_users)].copy()
Xtr_B, ytr_B, tr_B, Xva_B, yva_B, va_B, Xte_B, yte_B, te_B, sc_B = build_dataset(df_13)
model_B = train_hgbr(Xtr_B, ytr_B, Xva_B, yva_B)

yp_B  = model_B.predict(Xte_B)
mae_B   = float(np.mean(np.abs(yp_B - yte_B)))
rmse_B  = float(np.sqrt(np.mean((yp_B - yte_B)**2)))
smape_B = smape(yte_B, yp_B)
nmae_B  = per_user_nmae(yte_B, yp_B, te_B)
print(f"  13人子集 Test MAE : {mae_B:,.2f}")
print(f"  13人子集 Test RMSE: {rmse_B:,.2f}")
print(f"  13人子集 SMAPE    : {smape_B:.2f}%")
print(f"  13人子集 NMAE     : {nmae_B:.2f}%")

# ─────────────────────────────────────────
# Per-user 明細對比
# ─────────────────────────────────────────
print("\n" + "─"*60)
print(f"  Per-user 對比（情境A vs 情境B）")
print(f"  {'User':<10} {'n':>5} {'A: 16人訓練':>14} {'B: 13人訓練':>14} {'改善':>10}")
print(f"  {'─'*55}")

per_user_results = {}
for uid in sorted(set(te_B), key=str):
    m_A = uid_A == uid
    m_B = te_B  == uid
    if m_A.sum() == 0 or m_B.sum() == 0:
        continue
    mae_a = float(np.mean(np.abs(yp_A[m_A] - yt_A[m_A])))
    mae_b = float(np.mean(np.abs(yp_B[m_B] - yte_B[m_B])))
    imp   = mae_a - mae_b
    flag  = "✅" if imp > 0 else "❌"
    print(f"  {str(uid):<10} {int(m_B.sum()):>5} {mae_a:>14,.0f} {mae_b:>14,.0f} {imp:>+10,.0f} {flag}")
    per_user_results[str(uid)] = {"n_test": int(m_B.sum()),
                                   "A_16users_mae": round(mae_a, 2),
                                   "B_13users_mae": round(mae_b, 2),
                                   "improvement": round(imp, 2)}

print(f"  {'─'*55}")
print(f"  {'整體 (13人)':<10} {'':>5} {mae_A:>14,.0f} {mae_B:>14,.0f} {mae_A-mae_B:>+10,.0f} {'✅' if mae_A > mae_B else '❌'}")

# ─────────────────────────────────────────
# 結論
# ─────────────────────────────────────────
print("\n" + "="*60)
print("結論")
print("="*60)
delta = mae_A - mae_B
print(f"  排除 user4/5/6 後，13人 Test MAE：{mae_A:,.0f} → {mae_B:,.0f}")
print(f"  改善幅度：{delta:+,.0f} ({delta/mae_A*100:+.1f}%)")
if delta > 100:
    print("  ✅ 排除高消費用戶後有明顯改善，他們的資料確實在污染模型")
elif delta > 0:
    print("  ✅ 略有改善，但效果有限")
else:
    print("  ❌ 排除後沒有改善，問題不在於資料污染")
print(f"\n  ml_hgbr 原版參考（13人，不同切分）：MAE ~1,246")
print(f"  情境B（13人，per-user切分）：MAE {mae_B:,.0f}")

# ─────────────────────────────────────────
# 儲存
# ─────────────────────────────────────────
result = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "excluded_users": EXCLUDE_USERS,
    "scenario_A_16users_train_13users_eval": {
        "test_mae": round(mae_A, 2), "test_rmse": round(rmse_A, 2),
        "test_smape": round(smape_A, 4), "test_nmae": round(nmae_A, 4),
    },
    "scenario_B_13users_train_13users_eval": {
        "test_mae": round(mae_B, 2), "test_rmse": round(rmse_B, 2),
        "test_smape": round(smape_B, 4), "test_nmae": round(nmae_B, 4),
    },
    "improvement_mae": round(delta, 2),
    "improvement_pct": round(delta / mae_A * 100, 2),
    "per_user_detail": per_user_results,
}
with open("artifacts/experiment_exclude_users.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

with open("artifacts/hgbr_13users_model.pkl", "wb") as f:
    pickle.dump(model_B, f)
with open("artifacts/hgbr_13users_scaler.pkl", "wb") as f:
    pickle.dump(sc_B, f)

print("\n✅ 結果已儲存至 artifacts/experiment_exclude_users.json")
