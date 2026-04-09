"""
Step 1：Walmart 資料前處理（分類任務）
=======================================
使用與個人資料相同的 12 個 aligned 特徵
Label：future_7d_sum > roll_30d_mean × 7 × 1.5 → 1（預警），否則 → 0

輸出：
  - artifacts_clf/walmart_clf_X_train.npy
  - artifacts_clf/walmart_clf_y_train.npy   （binary float32）
  - artifacts_clf/walmart_clf_X_val.npy
  - artifacts_clf/walmart_clf_y_val.npy
  - artifacts_clf/walmart_clf_pos_weight.npy（用於 BCEWithLogitsLoss）
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import (
    compute_aligned_features, compute_alert_label,
    load_walmart_daily, ALIGNED_FEATURE_COLS, INPUT_DAYS, ALERT_RATIO
)

ARTIFACTS_DIR = "artifacts_clf"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

VAL_RATIO = 0.15

print("📂 載入 Walmart 資料...")
df = load_walmart_daily()
print(f"  共 {len(df):,} 筆 | {df['store_id'].nunique()} 家門市")

print(f"\n📊 計算特徵與分類標籤（alert_ratio={ALERT_RATIO}）...")
X_train_list, y_train_list = [], []
X_val_list,   y_val_list   = [], []
pos_count, neg_count        = 0, 0

for store_id in sorted(df["store_id"].unique()):
    u = df[df["store_id"] == store_id].sort_values("date").reset_index(drop=True)

    feats = compute_aligned_features(u["daily_expense"], u["date"])
    alert_labels, _, _ = compute_alert_label(u["daily_expense"])

    feats["alert_label"] = alert_labels.values

    # 丟掉 label 為 NaN 的最後幾行（shift(-7) 導致）
    feats = feats.dropna(subset=["alert_label"]).reset_index(drop=True)
    if len(feats) <= INPUT_DAYS + 10:
        continue

    feat_arr  = feats[ALIGNED_FEATURE_COLS].values.astype(np.float32)
    label_arr = feats["alert_label"].values.astype(np.float32)

    windows_X, windows_y = [], []
    for t in range(INPUT_DAYS, len(feats)):
        windows_X.append(feat_arr[t - INPUT_DAYS : t])
        windows_y.append(label_arr[t])

    n     = len(windows_X)
    v_end = int(n * (1 - VAL_RATIO))

    X_train_list.extend(windows_X[:v_end])
    y_train_list.extend(windows_y[:v_end])
    X_val_list.extend(windows_X[v_end:])
    y_val_list.extend(windows_y[v_end:])

X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)
X_val   = np.array(X_val_list,   dtype=np.float32)
y_val   = np.array(y_val_list,   dtype=np.float32)

pos_count = int(y_train.sum())
neg_count = int((y_train == 0).sum())
pos_weight = neg_count / (pos_count + 1e-8)

print(f"  Train: {X_train.shape}  Val: {X_val.shape}")
print(f"  Train 標籤分佈：正例={pos_count} ({pos_count/len(y_train)*100:.1f}%)，"
      f"負例={neg_count} ({neg_count/len(y_train)*100:.1f}%)")
print(f"  pos_weight（用於 BCEWithLogitsLoss）= {pos_weight:.3f}")

np.save(f"{ARTIFACTS_DIR}/walmart_clf_X_train.npy", X_train)
np.save(f"{ARTIFACTS_DIR}/walmart_clf_y_train.npy", y_train)
np.save(f"{ARTIFACTS_DIR}/walmart_clf_X_val.npy",   X_val)
np.save(f"{ARTIFACTS_DIR}/walmart_clf_y_val.npy",   y_val)
np.save(f"{ARTIFACTS_DIR}/walmart_clf_pos_weight.npy", np.array([pos_weight]))

print(f"\n✅ 儲存至 {ARTIFACTS_DIR}/")
print(f"🎉 完成！下一步：2_preprocess_personal_clf.py")
