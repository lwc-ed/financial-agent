"""
gru_hgbr/train_hgbr.py
========================
用 GRU embedding + HGBR 扁平特徵（共 86 維）訓練 HGBR。

輸入特徵：
  - GRU context embedding (64 維)：捕捉序列時序模式
  - HGBR 扁平特徵 (22 維)：日曆特徵、多種滾動統計

核心想法：
  「讓 GRU 做它擅長的（時序特徵提取），讓 HGBR 做它擅長的（最終迴歸）」
  兩者互補，避免各自的弱點（GRU 在小資料 overfit，HGBR 不擅長捕捉序列依賴）
"""

import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingRegressor

# ─────────────────────────────────────────
# 載入資料
# ─────────────────────────────────────────
print("📂 載入 combined 特徵（GRU embedding + 扁平特徵）...")
X_train_combined = np.load("artifacts/combined_X_train.npy")
X_val_combined   = np.load("artifacts/combined_X_val.npy")
X_test_combined  = np.load("artifacts/combined_X_test.npy")
y_train = np.load("artifacts/y_train.npy").flatten()
y_val   = np.load("artifacts/y_val.npy").flatten()
y_test  = np.load("artifacts/y_test.npy").flatten()
val_uids  = np.load("artifacts/val_uids.npy",  allow_pickle=True)
test_uids = np.load("artifacts/test_uids.npy", allow_pickle=True)

print(f"  X_train: {X_train_combined.shape}  (embedding + flat features)")
print(f"  X_test:  {X_test_combined.shape}")

# ─────────────────────────────────────────
# 對照組：只用扁平特徵（22 維）
# ─────────────────────────────────────────
X_train_flat = np.load("artifacts/hgbr_X_train.npy")
X_val_flat   = np.load("artifacts/hgbr_X_val.npy")
X_test_flat  = np.load("artifacts/hgbr_X_test.npy")

# ─────────────────────────────────────────
# 訓練兩個 HGBR 做對比
# ─────────────────────────────────────────
def make_hgbr():
    return HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.05,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42,
        verbose=1,
    )

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

results = {}

for label, X_tr, X_va, X_te in [
    ("HGBR_flat_only",     X_train_flat,     X_val_flat,     X_test_flat),
    ("HGBR_emb_and_flat",  X_train_combined, X_val_combined, X_test_combined),
]:
    print(f"\n🚀 訓練 {label}（{X_tr.shape[1]} 個特徵）...")
    X_tv = np.concatenate([X_tr, X_va], axis=0)
    y_tv = np.concatenate([y_train, y_val], axis=0)

    model = make_hgbr()
    model.fit(X_tv, y_tv)
    print(f"  最終迭代數：{model.n_iter_}")

    vp = model.predict(X_va)
    tp = model.predict(X_te)

    vm  = float(np.mean(np.abs(vp - y_val)))
    tm  = float(np.mean(np.abs(tp - y_test)))
    vr  = float(np.sqrt(np.mean((vp - y_val)**2)))
    tr  = float(np.sqrt(np.mean((tp - y_test)**2)))
    vs  = smape(y_val,  vp)
    ts  = smape(y_test, tp)
    vn  = per_user_nmae(y_val,  vp, val_uids)
    tn  = per_user_nmae(y_test, tp, test_uids)

    print(f"  Val  MAE: {vm:,.2f}  RMSE: {vr:,.2f}  SMAPE: {vs:.2f}%  NMAE: {vn:.2f}%")
    print(f"  Test MAE: {tm:,.2f}  RMSE: {tr:,.2f}  SMAPE: {ts:.2f}%  NMAE: {tn:.2f}%")

    results[label] = {
        "val_mae":   round(vm, 2),  "val_rmse":  round(vr, 2),
        "val_smape": round(vs, 4),  "val_nmae":  round(vn, 4),
        "test_mae":  round(tm, 2),  "test_rmse": round(tr, 2),
        "test_smape":round(ts, 4),  "test_nmae": round(tn, 4),
        "n_features": int(X_tr.shape[1]),
        "n_iter": model.n_iter_,
    }

    with open(f"artifacts/hgbr_{label}.pkl", "wb") as f:
        pickle.dump(model, f)

# ─────────────────────────────────────────
# 比較
# ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  GRU embedding 有沒有幫助？")
flat_mae  = results["HGBR_flat_only"]["test_mae"]
combo_mae = results["HGBR_emb_and_flat"]["test_mae"]
delta     = flat_mae - combo_mae
print(f"  HGBR (flat only)  Test MAE: {flat_mae:,.2f}")
print(f"  HGBR (emb+flat)   Test MAE: {combo_mae:,.2f}")
print(f"  改善幅度：{delta:+,.2f}  {'✅ embedding 有幫助！' if delta > 0 else '❌ embedding 沒幫助，建議只用 flat'}")
print(f"  原版 GRU V4 (參考): 5,773")
print(f"{'='*60}")

# ─────────────────────────────────────────
# 儲存
# ─────────────────────────────────────────
metrics = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "embedding_dim": 64,
    "flat_feature_dim": 22,
    "combined_dim": 86,
    "gru_source": "finetune_gru_v4.pth",
    "models": results,
    "embedding_helps": delta > 0,
}
with open("artifacts/metrics_hgbr.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("\n✅ 模型已儲存至 artifacts/hgbr_*.pkl")
print("   下一步：python predict.py")
