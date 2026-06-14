"""
Step 2：個人資料前處理 (含 Metadata 追蹤)
=======================
1. 使用與 IBM 完全相同的 10 個 aligned 特徵
2. per-user 70/15/15 切分
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts_transformer_tl"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MY_DIR))
from alignment_utils import ALIGNED_FEATURE_COLS, INPUT_DAYS, TARGET_COL, compute_aligned_features, load_personal_daily


def main():
    print(f"📂 工作目錄鎖定為: {MY_DIR}")
    print("📂 載入個人資料...")
    df = load_personal_daily()
    print(f"  共 {len(df):,} 筆 | {df['user_id'].nunique()} 位用戶")

    print(f"\n📊 計算 {len(ALIGNED_FEATURE_COLS)} 個 aligned 特徵...")
    result_list = []
    for user_id in sorted(df["user_id"].unique()):
        u = df[df["user_id"] == user_id].sort_values("date").reset_index(drop=True)
        feats = compute_aligned_features(u["daily_expense"], u["date"])
        u[TARGET_COL] = u["daily_expense"].rolling(7).sum().shift(-7)
        feats[TARGET_COL] = u[TARGET_COL].values
        feats["user_id"] = user_id
        feats["date"] = u["date"].values
        result_list.append(feats)

    daily = pd.concat(result_list, ignore_index=True)
    daily = daily.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"  有效筆數：{len(daily):,}")

    print(f"\n🪟 滑動視窗 + per-user 70/15/15 切分 (含 Metadata 追蹤)...")
    X_train_list, y_train_list, train_uid = [], [], []
    X_val_list, y_val_list, val_uid = [], [], []
    X_test_list, y_test_list, test_uid = [], [], []
    all_meta_list = []

    for user_id in sorted(daily["user_id"].unique()):
        u = daily[daily["user_id"] == user_id].reset_index(drop=True)
        feat_arr = u[ALIGNED_FEATURE_COLS].values.astype(np.float32)
        target_arr = u[TARGET_COL].values.astype(np.float32)
        date_arr = u["date"].values

        windows_X, windows_y, windows_meta = [], [], []
        for t in range(INPUT_DAYS, len(u)):
            windows_X.append(feat_arr[t - INPUT_DAYS : t])
            windows_y.append([target_arr[t]])
            windows_meta.append({"user_id": user_id, "date": date_arr[t]})

        n = len(windows_X)
        if n < 5:
            print(f"  ⚠️  {user_id} 資料不足（{n}），跳過")
            continue

        t_end = int(n * 0.70)
        v_end = int(n * 0.85)
        if t_end == 0:
            continue

        X_train_list.extend(windows_X[:t_end])
        y_train_list.extend(windows_y[:t_end])
        for m in windows_meta[:t_end]:
            m['split'] = 'train'

        X_val_list.extend(windows_X[t_end:v_end])
        y_val_list.extend(windows_y[t_end:v_end])
        for m in windows_meta[t_end:v_end]:
            m['split'] = 'val'

        X_test_list.extend(windows_X[v_end:])
        y_test_list.extend(windows_y[v_end:])
        for m in windows_meta[v_end:]:
            m['split'] = 'test'

        all_meta_list.extend(windows_meta)
        train_uid.extend([user_id] * t_end)
        val_uid.extend([user_id] * (v_end - t_end))
        test_uid.extend([user_id] * (n - v_end))

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.float32)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    print("\n📐 標準化 Target（fit on personal train）...")
    target_scaler = StandardScaler()
    target_scaler.fit(y_train)
    y_train_s = target_scaler.transform(y_train).astype(np.float32)
    y_val_s = target_scaler.transform(y_val).astype(np.float32)
    y_test_s = target_scaler.transform(y_test).astype(np.float32)

    print(f"\n💾 儲存檔案至 {ARTIFACTS_DIR}...")
    meta_df = pd.DataFrame(all_meta_list)
    meta_df.to_csv(ARTIFACTS_DIR / "metadata.csv", index=False)
    print(f"  ✅ 已存出 metadata.csv (共 {len(meta_df)} 筆)")

    np.save(ARTIFACTS_DIR / "personal_X_train.npy", X_train)
    np.save(ARTIFACTS_DIR / "personal_y_train.npy", y_train_s)
    np.save(ARTIFACTS_DIR / "personal_X_val.npy", X_val)
    np.save(ARTIFACTS_DIR / "personal_y_val.npy", y_val_s)
    np.save(ARTIFACTS_DIR / "personal_X_test.npy", X_test)
    np.save(ARTIFACTS_DIR / "personal_y_test.npy", y_test_s)
    np.save(ARTIFACTS_DIR / "personal_y_test_raw.npy", y_test)
    np.save(ARTIFACTS_DIR / "personal_train_user_ids.npy", np.array(train_uid))
    np.save(ARTIFACTS_DIR / "personal_val_user_ids.npy", np.array(val_uid))
    np.save(ARTIFACTS_DIR / "personal_test_user_ids.npy", np.array(test_uid))
    with open(ARTIFACTS_DIR / "personal_target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    print(f"\n🎉 [Step 2] 完成！檔案已存放在: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
