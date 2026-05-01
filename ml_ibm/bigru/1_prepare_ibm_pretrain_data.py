import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── 路徑設定 ─────────────────────────────────────────────
MY_DIR = Path(__file__).resolve().parent
IBM_FILE = MY_DIR.parent / "processed_data" / "artifacts" / "ibm_daily.csv"
ARTIFACTS_DIR = MY_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 超參數 ─────────────────────────────────────────────
SEQ_LEN = 30

# ✅ 改成與 own data 一致（5個特徵）
FEATURE_COLS = [
    "daily_expense",
    "roll_7d_mean",
    "roll_30d_mean",
    "dow_sin",
    "dow_cos",
]

TARGET_COL = "target"


def main():
    print("🚀 [IBM Pretrain Step 1] 準備 IBM BiGRU 3D 資料...")

    if not IBM_FILE.exists():
        raise FileNotFoundError(f"找不到 IBM daily 檔案：{IBM_FILE}")

    df = pd.read_csv(IBM_FILE)
    df["date"] = pd.to_datetime(df["date"])

    # 🔥 測試用（你可以保留）
    sample_users = df["user_id"].unique()[:5]
    df = df[df["user_id"].isin(sample_users)].copy()

    # ── 建立與 own data 一致的特徵 ─────────────────────────
    df["roll_7d_mean"] = df["expense_7d_mean"]
    df["roll_30d_mean"] = df["expense_30d_mean"]

    dow = df["dow"]
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    print(f"✅ IBM daily 載入完成：{len(df)} 筆，{df['user_id'].nunique()} 位 user")
    print(f"🪟 開始切割 sequence，SEQ_LEN={SEQ_LEN}")

    X_train_list, X_val_list, X_test_list = [], [], []
    y_train_list, y_val_list, y_test_list = [], [], []
    metadata_rows = []

    for user_id, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("date").reset_index(drop=True)

        feats_arr = user_df[FEATURE_COLS].values.astype(np.float32)
        target_arr = user_df[TARGET_COL].values.astype(np.float32)

        user_samples = []

        for i in range(SEQ_LEN, len(user_df)):
            x_window = feats_arr[i - SEQ_LEN:i]
            y_target = [target_arr[i]]
            sample_date = pd.to_datetime(user_df.loc[i, "date"]).strftime("%Y-%m-%d")

            user_samples.append({
                "X": x_window,
                "y": y_target,
                "user_id": user_id,
                "date": sample_date,
            })

        n_samples = len(user_samples)
        if n_samples < 10:
            continue

        t_end = int(n_samples * 0.70)
        v_end = int(n_samples * 0.85)

        split_sets = [
            ("train", user_samples[:t_end], X_train_list, y_train_list),
            ("valid", user_samples[t_end:v_end], X_val_list, y_val_list),
            ("test", user_samples[v_end:], X_test_list, y_test_list),
        ]

        for split_name, samples, X_list, y_list in split_sets:
            for s in samples:
                X_list.append(s["X"])
                y_list.append(s["y"])
                metadata_rows.append({
                    "user_id": s["user_id"],
                    "date": s["date"],
                    "split": split_name,
                })

    if not X_train_list or not X_test_list:
        raise RuntimeError("❌ 沒有足夠 IBM 樣本可切出 train/test sequence")

    X_train = np.array(X_train_list, dtype=np.float32)
    X_val = np.array(X_val_list, dtype=np.float32)
    X_test = np.array(X_test_list, dtype=np.float32)

    y_train = np.array(y_train_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)

    print("📊 IBM sequence 切割完成")
    print(f"   ibm_X_train: {X_train.shape}")
    print(f"   ibm_X_val  : {X_val.shape}")
    print(f"   ibm_X_test : {X_test.shape}")

    # ── 標準化 target ─────────────────────────────────────
    print("📏 標準化 target...")
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train).astype(np.float32)
    y_val_scaled = target_scaler.transform(y_val).astype(np.float32)
    y_test_scaled = target_scaler.transform(y_test).astype(np.float32)

    # ── 儲存 ─────────────────────────────────────────────
    np.save(ARTIFACTS_DIR / "ibm_X_train.npy", X_train)
    np.save(ARTIFACTS_DIR / "ibm_X_val.npy", X_val)
    np.save(ARTIFACTS_DIR / "ibm_X_test.npy", X_test)

    np.save(ARTIFACTS_DIR / "ibm_y_train_scaled.npy", y_train_scaled)
    np.save(ARTIFACTS_DIR / "ibm_y_val_scaled.npy", y_val_scaled)
    np.save(ARTIFACTS_DIR / "ibm_y_test_scaled.npy", y_test_scaled)

    np.save(ARTIFACTS_DIR / "ibm_y_train_raw.npy", y_train)
    np.save(ARTIFACTS_DIR / "ibm_y_val_raw.npy", y_val)
    np.save(ARTIFACTS_DIR / "ibm_y_test_raw.npy", y_test)

    pd.DataFrame(metadata_rows).to_csv(
        ARTIFACTS_DIR / "ibm_sample_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(ARTIFACTS_DIR / "ibm_target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    print("🎉 [IBM Pretrain Step 1] 完成！")


if __name__ == "__main__":
    main()