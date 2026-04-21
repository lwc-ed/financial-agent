"""
Step 1：Bi-LSTM 資料準備 (含 Metadata 追蹤版本)
=======================
1. 讀取原始 Excel 交易資料
2. 建立 3D 滑動視窗 (SEQ_LEN=30)
3. 新增：存出 metadata.csv 以供後續統一評估使用
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ── 1. 路徑設定 ──────────────────────────────────────────────────────────
# 指向專案共用 data/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 排除不要的使用者 (跟團隊約定一致)
EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]

# ── 2. 超參數設定 ────────────────────────────────────────────────────────
SEQ_LEN = 30       # 觀測過去 30 天
PREDICT_DAYS = 7   # 預測未來 7 天總花費

# 特徵欄位
FEATURE_COLS = [
    "daily_expense", 
    "roll_7d_mean", 
    "roll_30d_mean", 
    "dow_sin", 
    "dow_cos"
]
TARGET_COL = "future_7d_sum"

def main():
    print("🚀 [Step 1] 開始準備 Bi-LSTM (No Pretrain) 專用 3D 資料...")

    all_excel_files = sorted(DATA_DIR.glob("raw_transactions_*.xlsx"))
    print(f"🔍 搜尋完畢！共找到 {len(all_excel_files)} 個 Excel 檔案。")
    
    if len(all_excel_files) == 0:
        print("❌ 找不到原始交易資料，請確認 ml/data/ 路徑。")
        return

    all_users_data = []
    
    for file_path in all_excel_files:
        user_id = file_path.stem.replace("raw_transactions_", "")
        if user_id in EXCLUDE_USERS:
            continue
            
        df = pd.read_excel(file_path)
        df["time_stamp"] = pd.to_datetime(df["time_stamp"]).dt.normalize()
        df = df[df["transaction_type"] == "Expense"].copy()
        
        # 彙整日資料
        daily = df.groupby("time_stamp")["amount"].sum().reset_index()
        daily.rename(columns={"time_stamp": "date", "amount": "daily_expense"}, inplace=True)
        
        # 補齊日期
        date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = daily.set_index("date").reindex(date_range, fill_value=0).reset_index()
        daily.columns = ["date", "daily_expense"]
        
        # 建立特徵
        daily["roll_7d_mean"] = daily["daily_expense"].rolling(7, min_periods=1).mean()
        daily["roll_30d_mean"] = daily["daily_expense"].rolling(30, min_periods=1).mean()
        
        dow = daily["date"].dt.dayofweek
        daily["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        daily["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        
        # 建立 Target
        daily[TARGET_COL] = daily["daily_expense"].rolling(PREDICT_DAYS).sum().shift(-PREDICT_DAYS)
        
        daily = daily.dropna().reset_index(drop=True)
        daily["user_id"] = user_id
        all_users_data.append(daily)

    full_df = pd.concat(all_users_data, ignore_index=True)
    print(f"✅ 資料載入完畢！共 {len(full_df)} 筆日資料。")

    # ── 切割成 3D 滑動視窗 ──
    print(f"🪟 正在切割 3D 視窗...")
    
    X_list, y_list = [], []
    meta_list = [] # 存放 metadata
    
    for user_id in full_df["user_id"].unique():
        user_df = full_df[full_df["user_id"] == user_id].reset_index(drop=True)
        
        feats_arr = user_df[FEATURE_COLS].values.astype(np.float32)
        target_arr = user_df[TARGET_COL].values.astype(np.float32)
        dates_arr = user_df["date"].values 
        
        u_X, u_y, u_meta = [], [], []
        for i in range(SEQ_LEN, len(user_df)):
            u_X.append(feats_arr[i - SEQ_LEN : i])
            u_y.append([target_arr[i]])
            u_meta.append({
                "user_id": user_id,
                "date": dates_arr[i]
            })
            
        n_samples = len(u_X)
        if n_samples < 10: continue
            
        # 70/15/15 切分
        t_end = int(n_samples * 0.70)
        v_end = int(n_samples * 0.85)
        
        # 標記 Split
        for i, m in enumerate(u_meta):
            if i < t_end: m['split'] = 'train'
            elif i < v_end: m['split'] = 'val'
            else: m['split'] = 'test'
        
        meta_list.extend(u_meta)
        X_list.append((u_X[:t_end], u_X[t_end:v_end], u_X[v_end:]))
        y_list.append((u_y[:t_end], u_y[t_end:v_end], u_y[v_end:]))

    # 合併資料
    X_train = np.concatenate([x[0] for x in X_list], axis=0)
    X_val   = np.concatenate([x[1] for x in X_list], axis=0)
    X_test  = np.concatenate([x[2] for x in X_list], axis=0)
    
    y_train = np.concatenate([y[0] for y in y_list], axis=0)
    y_val   = np.concatenate([y[1] for y in y_list], axis=0)
    y_test  = np.concatenate([y[2] for y in y_list], axis=0)
    
    # 標準化
    print("📏 正在對 Target 進行標準化...")
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train).astype(np.float32)
    y_val_scaled   = target_scaler.transform(y_val).astype(np.float32)
    y_test_scaled  = target_scaler.transform(y_test).astype(np.float32)

    # ── 儲存檔案 ──
    print(f"💾 正在儲存檔案至 {ARTIFACTS_DIR} ...")
    
    # 儲存 Metadata CSV
    metadata_df = pd.DataFrame(meta_list)
    metadata_df.to_csv(ARTIFACTS_DIR / "metadata.csv", index=False)
    print(f"  ✅ 已存出 metadata.csv (共 {len(metadata_df)} 筆)")

    # 儲存 Numpy 檔案
    np.save(ARTIFACTS_DIR / "my_X_train.npy", X_train)
    np.save(ARTIFACTS_DIR / "my_X_val.npy", X_val)
    np.save(ARTIFACTS_DIR / "my_X_test.npy", X_test)
    np.save(ARTIFACTS_DIR / "my_y_train_scaled.npy", y_train_scaled)
    np.save(ARTIFACTS_DIR / "my_y_val_scaled.npy", y_val_scaled)
    np.save(ARTIFACTS_DIR / "my_y_test_scaled.npy", y_test_scaled)
    np.save(ARTIFACTS_DIR / "my_y_test_raw.npy", y_test)
    
    with open(ARTIFACTS_DIR / "my_target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    print("🎉 [Step 1] Bi-LSTM 資料處理完成！")

if __name__ == "__main__":
    main()