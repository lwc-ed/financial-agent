"""
[NO-ALIGN pipeline] Step 1：IBM 前處理（raw 特徵，不做 Domain Alignment）
=========================================================================
與 ml_ibm/bigru_TL_alignment/1_preprocess_ibm.py 相同流程，唯一差別：
  特徵改用 no_alignment_utils.compute_raw_features（原始絕對金額）。

輸出 → ml/ml_temp/artifacts_noalign/
  ibm_X_train.npy / ibm_y_train.npy / ibm_X_val.npy / ibm_y_val.npy
  ibm_X_test.npy  / ibm_y_test.npy  / ibm_target_scaler.pkl
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ORIG = HERE.parent / "ml_ibm" / "bigru_TL_alignment"
for p in (HERE, ORIG):
    sys.path.insert(0, str(p))

from alignment_utils import INPUT_DAYS, TARGET_COL  # noqa: E402  （只讀重用）
from no_alignment_utils import RAW_FEATURE_COLS, compute_raw_features, load_ibm_daily_sample  # noqa: E402

ART = HERE / "artifacts_noalign"
ART.mkdir(parents=True, exist_ok=True)

N_USERS = 2000
MIN_DAYS_PER_USER = INPUT_DAYS + 7 + 10


def build_windows(df):
    X_list, y_list, skipped = [], [], 0
    for _, grp in df.groupby("user_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < MIN_DAYS_PER_USER:
            skipped += 1
            continue
        feats = compute_raw_features(grp["daily_expense"], grp["date"])
        target = grp["daily_expense"].rolling(7).sum().shift(-7)
        feats[TARGET_COL] = target.values
        feats = feats[feats[TARGET_COL].notna()].reset_index(drop=True)
        feat_arr = feats[RAW_FEATURE_COLS].values.astype(np.float32)
        target_arr = feats[TARGET_COL].values.astype(np.float32)
        for t in range(INPUT_DAYS, len(feats)):
            X_list.append(feat_arr[t - INPUT_DAYS : t])
            y_list.append([target_arr[t]])
    print(f"[INFO] 跳過資料不足的用戶：{skipped} 位")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def main():
    print(f"📂 輸出資料夾：{ART}")
    df = load_ibm_daily_sample(N_USERS)

    print(f"\n📊 計算 {len(RAW_FEATURE_COLS)} 個 raw 特徵並建立滑動視窗...")
    X, y = build_windows(df)
    print(f"[INFO] 全部視窗數：{len(X):,}，shape: {X.shape}")

    train_end = int(len(X) * 0.70)
    val_end = int(len(X) * 0.85)
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    print(f"[INFO] Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    print("\n📐 標準化 IBM Target...")
    scaler = StandardScaler().fit(y_train)
    np.save(ART / "ibm_X_train.npy", X_train)
    np.save(ART / "ibm_y_train.npy", scaler.transform(y_train).astype(np.float32))
    np.save(ART / "ibm_X_val.npy", X_val)
    np.save(ART / "ibm_y_val.npy", scaler.transform(y_val).astype(np.float32))
    np.save(ART / "ibm_X_test.npy", X_test)
    np.save(ART / "ibm_y_test.npy", scaler.transform(y_test).astype(np.float32))
    with open(ART / "ibm_target_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n✅ IBM（raw）前處理完成！y_train mean={y_train.mean():.2f} std={y_train.std():.2f}")


if __name__ == "__main__":
    main()
