"""
[NO-ALIGN pipeline] Step 2：個人資料前處理（raw 特徵）
====================================================
與 ml_ibm/bigru_TL_alignment/2_preprocess_personal.py 相同流程（含 metadata、
per-user 70/15/15），唯一差別：特徵改用 compute_raw_features。

輸出 → ml/ml_temp/artifacts_noalign/
  personal_X_*.npy / personal_y_*.npy / personal_y_test_raw.npy
  personal_*_user_ids.npy / metadata.csv / personal_target_scaler.pkl
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ORIG = HERE.parent / "ml_ibm" / "bigru_TL_alignment"
for p in (HERE, ORIG):
    sys.path.insert(0, str(p))

from alignment_utils import INPUT_DAYS, TARGET_COL, load_personal_daily  # noqa: E402
from no_alignment_utils import RAW_FEATURE_COLS, compute_raw_features  # noqa: E402

ART = HERE / "artifacts_noalign"
ART.mkdir(parents=True, exist_ok=True)


def main():
    print(f"📂 輸出資料夾：{ART}")
    df = load_personal_daily()
    print(f"  共 {len(df):,} 筆 | {df['user_id'].nunique()} 位用戶")

    print(f"\n📊 計算 {len(RAW_FEATURE_COLS)} 個 raw 特徵...")
    result_list = []
    for user_id in sorted(df["user_id"].unique()):
        u = df[df["user_id"] == user_id].sort_values("date").reset_index(drop=True)
        feats = compute_raw_features(u["daily_expense"], u["date"])
        u[TARGET_COL] = u["daily_expense"].rolling(7).sum().shift(-7)
        feats[TARGET_COL] = u[TARGET_COL].values
        feats["user_id"] = user_id
        feats["date"] = u["date"].values
        result_list.append(feats)

    daily = pd.concat(result_list, ignore_index=True).dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"  有效筆數：{len(daily):,}")

    print("\n🪟 滑動視窗 + per-user 70/15/15（含 metadata）...")
    Xtr, ytr, tr_uid = [], [], []
    Xva, yva, va_uid = [], [], []
    Xte, yte, te_uid = [], [], []
    meta = []

    for user_id in sorted(daily["user_id"].unique()):
        u = daily[daily["user_id"] == user_id].reset_index(drop=True)
        fa = u[RAW_FEATURE_COLS].values.astype(np.float32)
        ta = u[TARGET_COL].values.astype(np.float32)
        da = u["date"].values

        wX, wy, wm = [], [], []
        for t in range(INPUT_DAYS, len(u)):
            wX.append(fa[t - INPUT_DAYS : t])
            wy.append([ta[t]])
            wm.append({"user_id": user_id, "date": da[t]})

        n = len(wX)
        if n < 5:
            print(f"  ⚠️  {user_id} 資料不足（{n}），跳過")
            continue
        te_end = int(n * 0.70)
        ve_end = int(n * 0.85)
        if te_end == 0:
            continue

        Xtr.extend(wX[:te_end]); ytr.extend(wy[:te_end])
        for m in wm[:te_end]: m["split"] = "train"
        Xva.extend(wX[te_end:ve_end]); yva.extend(wy[te_end:ve_end])
        for m in wm[te_end:ve_end]: m["split"] = "val"
        Xte.extend(wX[ve_end:]); yte.extend(wy[ve_end:])
        for m in wm[ve_end:]: m["split"] = "test"
        meta.extend(wm)
        tr_uid.extend([user_id] * te_end)
        va_uid.extend([user_id] * (ve_end - te_end))
        te_uid.extend([user_id] * (n - ve_end))

    X_train = np.array(Xtr, dtype=np.float32); y_train = np.array(ytr, dtype=np.float32)
    X_val = np.array(Xva, dtype=np.float32); y_val = np.array(yva, dtype=np.float32)
    X_test = np.array(Xte, dtype=np.float32); y_test = np.array(yte, dtype=np.float32)
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    print("\n📐 標準化 Target（fit on personal train）...")
    scaler = StandardScaler().fit(y_train)

    pd.DataFrame(meta).to_csv(ART / "metadata.csv", index=False)
    np.save(ART / "personal_X_train.npy", X_train)
    np.save(ART / "personal_y_train.npy", scaler.transform(y_train).astype(np.float32))
    np.save(ART / "personal_X_val.npy", X_val)
    np.save(ART / "personal_y_val.npy", scaler.transform(y_val).astype(np.float32))
    np.save(ART / "personal_X_test.npy", X_test)
    np.save(ART / "personal_y_test.npy", scaler.transform(y_test).astype(np.float32))
    np.save(ART / "personal_y_test_raw.npy", y_test)
    np.save(ART / "personal_train_user_ids.npy", np.array(tr_uid))
    np.save(ART / "personal_val_user_ids.npy", np.array(va_uid))
    np.save(ART / "personal_test_user_ids.npy", np.array(te_uid))
    with open(ART / "personal_target_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n🎉 [Step 2 raw] 完成！檔案在：{ART}")


if __name__ == "__main__":
    main()
