import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ── 1. 路徑設定 ──────────────────────────────────────────────────────────
# 假設這支程式放在 ml_alignment_ckh/ 底下
# 直接指向 ml/data/，避免 rglob 找到 .claude/worktrees 裡的重複備份
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

print(f"🔍 尋找 Excel 檔案的目標資料夾: {DATA_DIR}")

# 📍 這裡修改了：用 .parent 直接抓到這支程式所在的資料夾 (ml_alignment_ckh)
MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 排除不要的使用者 (跟團隊約定好的一致)
EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]

# ── 2. 超參數設定 ────────────────────────────────────────────────────────
SEQ_LEN = 30       # 讓神經網路看過去 30 天的資料
PREDICT_DAYS = 7   # 預測未來 7 天的總花費

# 定義要給神經網路看的特徵欄位
FEATURE_COLS = [
    "daily_expense", 
    "roll_7d_mean", 
    "roll_30d_mean", 
    "dow_sin", 
    "dow_cos"
]
TARGET_COL = "future_7d_sum"

def main():
    print("🚀 [Step 1] 開始準備神經網路專用 3D 資料...")

    # ── 讀取與彙整原始資料 ───────────────────────────────────────────────
    # 直接在 ml/data/ 搜尋，避免找到 worktrees 裡的重複備份
    all_excel_files = sorted(DATA_DIR.glob("raw_transactions_*.xlsx"))
    print(f"🔍 搜尋完畢！共找到 {len(all_excel_files)} 個 Excel 檔案。")
    
    # 加上安全鎖：如果真的沒半個檔案，就印出警告並停止，不要往下跑導致當機
    if len(all_excel_files) == 0:
        print("❌ 慘了，整個專案底下完全找不到 raw_transactions_*.xlsx！")
        print("   請確認一下檔案是不是還沒放進來，或是檔名不對？")
        return

    all_users_data = []
    
    # 把尋找目標換成我們剛剛搜出來的檔案清單
    for file_path in sorted(all_excel_files):
        user_id = file_path.stem.replace("raw_transactions_", "")
        # ... 後面 (if user_id in EXCLUDE_USERS: 開始) 都不用動！
        if user_id in EXCLUDE_USERS:
            continue
            
        df = pd.read_excel(file_path)
        df["time_stamp"] = pd.to_datetime(df["time_stamp"]).dt.normalize()
        df = df[df["transaction_type"] == "Expense"].copy()
        
        # 彙整成每天的總花費
        daily = df.groupby("time_stamp")["amount"].sum().reset_index()
        daily.rename(columns={"time_stamp": "date", "amount": "daily_expense"}, inplace=True)
        
        # 補齊沒有消費的日期 (補 0)
        date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = daily.set_index("date").reindex(date_range, fill_value=0).reset_index()
        daily.columns = ["date", "daily_expense"]
        
        # ── 建立簡單特徵 ──
        daily["roll_7d_mean"] = daily["daily_expense"].rolling(7, min_periods=1).mean()
        daily["roll_30d_mean"] = daily["daily_expense"].rolling(30, min_periods=1).mean()
        
        # 時間特徵 (星期幾的三角函數編碼，讓網路知道週日跟週一連在一起)
        dow = daily["date"].dt.dayofweek
        daily["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        daily["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        
        # 目標值 (Y): 未來 7 天的總和
        daily[TARGET_COL] = daily["daily_expense"].rolling(PREDICT_DAYS).sum().shift(-PREDICT_DAYS)
        
        # 砍掉最後幾天沒有 Target 的資料
        daily = daily.dropna().reset_index(drop=True)
        daily["user_id"] = user_id
        all_users_data.append(daily)

    full_df = pd.concat(all_users_data, ignore_index=True)
    print(f"✅ 資料載入完畢！共 {len(full_df)} 筆日資料。")

    # ── 切割成 3D 滑動視窗 (Sliding Windows) ───────────────────────────
    print(f"🪟 正在切割 3D 視窗 (看過去 {SEQ_LEN} 天)...")
    
    X_list, y_list = [], []
    train_uid, val_uid, test_uid = [], [], []
    
    # 按照每個使用者獨立切割，避免資料混到別人
    for user_id in full_df["user_id"].unique():
        user_df = full_df[full_df["user_id"] == user_id].reset_index(drop=True)
        
        feats_arr = user_df[FEATURE_COLS].values.astype(np.float32)
        target_arr = user_df[TARGET_COL].values.astype(np.float32)
        
        u_X, u_y = [], []
        # 從第 30 天開始切，每天往後滑動一格
        for i in range(SEQ_LEN, len(user_df)):
            u_X.append(feats_arr[i - SEQ_LEN : i])  # 過去 30 天的特徵矩陣
            u_y.append([target_arr[i]])             # 第 30 天對應的未來花費
            
        n_samples = len(u_X)
        if n_samples < 10: 
            continue # 資料太少直接放棄
            
        # 按照時間切分 70% 訓練, 15% 驗證, 15% 測試
        t_end = int(n_samples * 0.70)
        v_end = int(n_samples * 0.85)
        
        X_list.append((u_X[:t_end], u_X[t_end:v_end], u_X[v_end:]))
        y_list.append((u_y[:t_end], u_y[t_end:v_end], u_y[v_end:]))
        
        train_uid.extend([user_id] * len(u_X[:t_end]))
        val_uid.extend([user_id] * len(u_X[t_end:v_end]))
        test_uid.extend([user_id] * len(u_X[v_end:]))

    # 把大家的心血全部合併起來
    X_train = np.concatenate([x[0] for x in X_list], axis=0)
    X_val   = np.concatenate([x[1] for x in X_list], axis=0)
    X_test  = np.concatenate([x[2] for x in X_list], axis=0)
    
    y_train = np.concatenate([y[0] for y in y_list], axis=0)
    y_val   = np.concatenate([y[1] for y in y_list], axis=0)
    y_test  = np.concatenate([y[2] for y in y_list], axis=0)
    
    print(f"📊 3D 切割完成！神經網路準備就緒：")
    print(f"   X_train 形狀: {X_train.shape} (樣本數, {SEQ_LEN}天, {len(FEATURE_COLS)}個特徵)")
    print(f"   y_train 形狀: {y_train.shape}")

    # ── 特徵標準化 (Scale) ────────────────────────────────────────────────
    # 神經網路不喜歡數字太大，我們把 Target 壓平
    print("📏 正在對 Target 進行標準化 (StandardScaler)...")
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train).astype(np.float32)
    y_val_scaled   = target_scaler.transform(y_val).astype(np.float32)
    y_test_scaled  = target_scaler.transform(y_test).astype(np.float32)

    # ── 儲存所有的彈藥 ────────────────────────────────────────────────────
    print(f"💾 正在儲存檔案至 {ARTIFACTS_DIR} ...")
    
    np.save(ARTIFACTS_DIR / "my_X_train.npy", X_train)
    np.save(ARTIFACTS_DIR / "my_X_val.npy", X_val)
    np.save(ARTIFACTS_DIR / "my_X_test.npy", X_test)
    
    np.save(ARTIFACTS_DIR / "my_y_train_scaled.npy", y_train_scaled)
    np.save(ARTIFACTS_DIR / "my_y_val_scaled.npy", y_val_scaled)
    np.save(ARTIFACTS_DIR / "my_y_test_scaled.npy", y_test_scaled)
    
    # 把沒有 scaled 的真實金額留下來，測試算 MAE 時會用到！
    np.save(ARTIFACTS_DIR / "my_y_test_raw.npy", y_test)
    
    with open(ARTIFACTS_DIR / "my_target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    print("🎉 [Step 1] 完美結束！可以直接進入模型訓練環節了。")

if __name__ == "__main__":
    main()