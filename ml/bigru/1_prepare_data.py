import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── 1. 路徑設定 ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"🔍 尋找 Excel 檔案的目標資料夾: {DATA_DIR}")

# 排除不要的使用者，跟現有 baseline 一致
EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]

# ── 2. 超參數設定 ────────────────────────────────────────────────────────
SEQ_LEN = 30
PREDICT_DAYS = 7

FEATURE_COLS = [
    "daily_expense",
    "roll_7d_mean",
    "roll_30d_mean",
    "dow_sin",
    "dow_cos",
]
TARGET_COL = "future_7d_sum"


def main():
    print("🚀 [Step 1] 開始準備 Bi-GRU baseline 專用 3D 資料...")

    all_excel_files = sorted(DATA_DIR.glob("raw_transactions_*.xlsx"))
    print(f"🔍 搜尋完畢！共找到 {len(all_excel_files)} 個 Excel 檔案。")

    if len(all_excel_files) == 0:
        print("❌ 找不到 raw_transactions_*.xlsx，請確認 ml/data/ 是否已放入原始交易資料。")
        return

    all_users_data = []

    for file_path in all_excel_files:
        user_id = file_path.stem.replace("raw_transactions_", "")
        if user_id in EXCLUDE_USERS:
            continue

        df = pd.read_excel(file_path)
        df["time_stamp"] = pd.to_datetime(df["time_stamp"]).dt.normalize()
        df = df[df["transaction_type"] == "Expense"].copy()

        daily = df.groupby("time_stamp")["amount"].sum().reset_index()
        daily.rename(columns={"time_stamp": "date", "amount": "daily_expense"}, inplace=True)

        date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = daily.set_index("date").reindex(date_range, fill_value=0).reset_index()
        daily.columns = ["date", "daily_expense"]

        daily["roll_7d_mean"] = daily["daily_expense"].rolling(7, min_periods=1).mean()
        daily["roll_30d_mean"] = daily["daily_expense"].rolling(30, min_periods=1).mean()

        dow = daily["date"].dt.dayofweek
        daily["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        daily["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        daily[TARGET_COL] = daily["daily_expense"].rolling(PREDICT_DAYS).sum().shift(-PREDICT_DAYS)
        daily = daily.dropna().reset_index(drop=True)
        daily["user_id"] = user_id
        all_users_data.append(daily)

    if not all_users_data:
        print("❌ 可用使用者資料為 0，請確認原始交易資料與排除名單。")
        return

    full_df = pd.concat(all_users_data, ignore_index=True)
    print(f"✅ 資料載入完畢！共 {len(full_df)} 筆日資料。")

    print(f"🪟 正在切割 3D 視窗 (看過去 {SEQ_LEN} 天)...")

    X_list, y_list = [], []
    train_uid, val_uid, test_uid = [], [], []

    for user_id in full_df["user_id"].unique():
        user_df = full_df[full_df["user_id"] == user_id].reset_index(drop=True)

        feats_arr = user_df[FEATURE_COLS].values.astype(np.float32)
        target_arr = user_df[TARGET_COL].values.astype(np.float32)

        u_X, u_y = [], []
        for i in range(SEQ_LEN, len(user_df)):
            u_X.append(feats_arr[i - SEQ_LEN : i])
            u_y.append([target_arr[i]])

        n_samples = len(u_X)
        if n_samples < 10:
            continue

        t_end = int(n_samples * 0.70)
        v_end = int(n_samples * 0.85)

        X_list.append((u_X[:t_end], u_X[t_end:v_end], u_X[v_end:]))
        y_list.append((u_y[:t_end], u_y[t_end:v_end], u_y[v_end:]))

        train_uid.extend([user_id] * len(u_X[:t_end]))
        val_uid.extend([user_id] * len(u_X[t_end:v_end]))
        test_uid.extend([user_id] * len(u_X[v_end:]))

    if not X_list:
        print("❌ 沒有足夠的樣本可切出 3D 視窗。")
        return

    X_train = np.concatenate([x[0] for x in X_list], axis=0)
    X_val = np.concatenate([x[1] for x in X_list], axis=0)
    X_test = np.concatenate([x[2] for x in X_list], axis=0)

    y_train = np.concatenate([y[0] for y in y_list], axis=0)
    y_val = np.concatenate([y[1] for y in y_list], axis=0)
    y_test = np.concatenate([y[2] for y in y_list], axis=0)

    print("📊 3D 切割完成！")
    print(f"   X_train 形狀: {X_train.shape}")
    print(f"   X_val   形狀: {X_val.shape}")
    print(f"   X_test  形狀: {X_test.shape}")

    print("📏 正在對 Target 進行標準化 (StandardScaler)...")
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train).astype(np.float32)
    y_val_scaled = target_scaler.transform(y_val).astype(np.float32)
    y_test_scaled = target_scaler.transform(y_test).astype(np.float32)

    print(f"💾 正在儲存檔案至 {ARTIFACTS_DIR} ...")
    np.save(ARTIFACTS_DIR / "my_X_train.npy", X_train)
    np.save(ARTIFACTS_DIR / "my_X_val.npy", X_val)
    np.save(ARTIFACTS_DIR / "my_X_test.npy", X_test)

    np.save(ARTIFACTS_DIR / "my_y_train_scaled.npy", y_train_scaled)
    np.save(ARTIFACTS_DIR / "my_y_val_scaled.npy", y_val_scaled)
    np.save(ARTIFACTS_DIR / "my_y_test_scaled.npy", y_test_scaled)
    np.save(ARTIFACTS_DIR / "my_y_test_raw.npy", y_test)

    with open(ARTIFACTS_DIR / "my_target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    print("🎉 [Step 1] 完成！接著可執行 2_train_bigru.py。")


if __name__ == "__main__":
    main()
