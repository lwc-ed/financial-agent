"""
ml_gru_nopretrain/train_hgbr.py
================================
使用 22 個豐富特徵訓練 HistGradientBoostingRegressor。

與原版 HGBR (test_mae ~1200) 相比，新增：
  - 日曆特徵：dow, is_weekend, day, month, is_summer/winter_vacation, days_to_end_of_month
  - 更多滾動統計：net_7d_sum, txn_7d_sum, expense_7d_30d_ratio, expense_trend
  - 收入相關：daily_income, daily_net

HistGradientBoostingRegressor 在小資料集上通常優於 GRU，
因為它不需要 pretrain 且對 feature interaction 學習能力強。
"""

import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingRegressor

# ─────────────────────────────────────────
# 載入資料
# ─────────────────────────────────────────
print("📂 載入資料...")
X_train = np.load("artifacts/hgbr_X_train.npy")
X_val   = np.load("artifacts/hgbr_X_val.npy")
X_test  = np.load("artifacts/hgbr_X_test.npy")
y_train = np.load("artifacts/y_train.npy").flatten()
y_val   = np.load("artifacts/y_val.npy").flatten()
y_test  = np.load("artifacts/y_test.npy").flatten()
val_uids  = np.load("artifacts/val_uids.npy",  allow_pickle=True)
test_uids = np.load("artifacts/test_uids.npy", allow_pickle=True)
print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

# ─────────────────────────────────────────
# 合併 train + val，用 HGBR 內建 early stopping
# ─────────────────────────────────────────
X_trainval = np.concatenate([X_train, X_val], axis=0)
y_trainval = np.concatenate([y_train, y_val], axis=0)

# ─────────────────────────────────────────
# 模型（使用 HGBR 內建 early stopping）
# ─────────────────────────────────────────
model = HistGradientBoostingRegressor(
    max_iter=1000,
    learning_rate=0.05,
    max_leaf_nodes=31,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.15,   # 從 trainval 中取 15% 做 early stopping
    n_iter_no_change=30,
    random_state=42,
    verbose=1,
)

print("\n🚀 訓練 HGBR（22 個特徵，無 pretrain）...")
model.fit(X_trainval, y_trainval)
print(f"  訓練完成，最終迭代數：{model.n_iter_}")


# ─────────────────────────────────────────
# 評估
# ─────────────────────────────────────────
def smape(yt, yp):
    yt, yp = yt.flatten(), yp.flatten()
    d = (np.abs(yt) + np.abs(yp)) / 2
    m = d > 0
    return float(np.mean(np.abs(yp[m] - yt[m]) / d[m]) * 100)

def per_user_nmae(yt, yp, uids):
    yt, yp = yt.flatten(), yp.flatten()
    r = []
    for u in np.unique(uids):
        mask = np.array(uids) == u
        mu = yt[mask].mean()
        if mu > 0:
            r.append(np.mean(np.abs(yp[mask] - yt[mask])) / mu * 100)
    return float(np.mean(r))

y_val_pred  = model.predict(X_val)
y_test_pred = model.predict(X_test)

val_mae   = float(np.mean(np.abs(y_val_pred  - y_val)))
val_rmse  = float(np.sqrt(np.mean((y_val_pred  - y_val)**2)))
test_mae  = float(np.mean(np.abs(y_test_pred - y_test)))
test_rmse = float(np.sqrt(np.mean((y_test_pred - y_test)**2)))
val_smape  = smape(y_val,  y_val_pred)
test_smape = smape(y_test, y_test_pred)
val_nmae   = per_user_nmae(y_val,  y_val_pred,  val_uids)
test_nmae  = per_user_nmae(y_test, y_test_pred, test_uids)

print(f"\n{'='*60}")
print(f"  [HGBR 22 features]")
print(f"  Val  MAE: {val_mae:,.2f}  RMSE: {val_rmse:,.2f}  SMAPE: {val_smape:.2f}%  NMAE: {val_nmae:.2f}%")
print(f"  Test MAE: {test_mae:,.2f}  RMSE: {test_rmse:,.2f}  SMAPE: {test_smape:.2f}%  NMAE: {test_nmae:.2f}%")
print(f"  原版 HGBR (7 features) Test MAE ~1,200 | 原版最佳 GRU (V4) Test MAE ~5,773")
print(f"  {'✅ 超過原版 HGBR' if test_mae < 1200 else '⚠️  未超過原版 HGBR，但比 GRU 更好' if test_mae < 5773 else '❌ 需要再調整'}")
print(f"{'='*60}")

# ─────────────────────────────────────────
# 儲存
# ─────────────────────────────────────────
with open("artifacts/hgbr_model.pkl", "wb") as f:
    pickle.dump(model, f)

metrics = {
    "model": "HGBR_22features_nopretrain",
    "val_mae":   round(val_mae,   2), "val_rmse":  round(val_rmse,  2),
    "val_smape": round(val_smape, 4), "val_nmae":  round(val_nmae,  4),
    "test_mae":  round(test_mae,  2), "test_rmse": round(test_rmse, 2),
    "test_smape":round(test_smape,4), "test_nmae": round(test_nmae, 4),
    "n_iter":    model.n_iter_,
    "n_features": X_train.shape[1],
    "reference_original_hgbr_test_mae": 1200,
    "reference_gru_v4_test_mae": 5773,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
with open("artifacts/metrics_hgbr.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("\n✅ 模型已儲存至 artifacts/hgbr_model.pkl")
print("   下一步：python train_gru_scratch.py  或  python predict.py")
