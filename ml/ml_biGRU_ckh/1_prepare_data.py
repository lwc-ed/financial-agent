import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

print("🚀 [Step 1] 開始手工打造 Alignment 對齊特徵...")

# ── 1. 路徑設定 ────────────────────────────────────────────────────────
# 設定輸出資料夾 (就在目前這支程式旁邊建立 artifacts_aligned)
BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "artifacts_aligned"
SAVE_DIR.mkdir(exist_ok=True)

# 💡 假設你最初始的特徵檔放在上一層的 ml_gru 資料夾裡
# 如果你的 features_all.csv 放在別的地方，請幫我修改這行路徑！
FEATURES_PATH = BASE_DIR.parent / "ml_gru" / "features_all.csv"

try:
    print(f"📂 正在讀取原始資料: {FEATURES_PATH} ...")
    df = pd.read_csv(FEATURES_PATH)
except FileNotFoundError:
    print(f"❌ 找不到 features_all.csv！請確認路徑是否正確：{FEATURES_PATH}")
    exit()

df["date"] = pd.to_datetime(df["date"])
EXCLUDE_USERS = ["user4", "user5", "user6", "user14"]
df = df[~df["user_id"].isin(EXCLUDE_USERS)].reset_index(drop=True)
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

# ── 2. 手工計算 7 大 Alignment 特徵 ──────────────────────────────────────
print("⚙️ 正在計算 Rolling Z-score 與各項對齊特徵...")

# 計算 7天與 30天的滾動平均與標準差
df["roll_7d_mean"] = df.groupby("user_id")["daily_expense"].transform(lambda x: x.rolling(7, min_periods=1).mean())
df["roll_7d_std"]  = df.groupby("user_id")["daily_expense"].transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))
df["roll_30d_mean"] = df.groupby("user_id")["daily_expense"].transform(lambda x: x.rolling(30, min_periods=1).mean())
df["roll_30d_std"]  = df.groupby("user_id")["daily_expense"].transform(lambda x: x.rolling(30, min_periods=1).std().fillna(0))

# [特徵 1 & 2] Z-score (避免除以零，加上 1e-8)
df["zscore_7d"] = (df["daily_expense"] - df["roll_7d_mean"]) / (df["roll_7d_std"] + 1e-8)
df["zscore_30d"] = (df["daily_expense"] - df["roll_30d_mean"]) / (df["roll_30d_std"] + 1e-8)

# [特徵 3] 變動率 (Percent Change)
df["prev_day"] = df.groupby("user_id")["daily_expense"].shift(1).fillna(df["daily_expense"])
df["pct_change"] = (df["daily_expense"] - df["prev_day"]) / (df["prev_day"] + 1e-8)
# 把極端值稍微夾住，避免爆炸
df["pct_change_norm"] = df["pct_change"].clip(-5, 5)

# [特徵 4 & 5] 波動度與是否大於均線
df["volatility_7d"] = df["roll_7d_std"]
df["is_above_mean_30d"] = (df["daily_expense"] > df["roll_30d_mean"]).astype(float)

# [特徵 6 & 7] 星期幾的週期編碼
df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

# 把我們要的 7 個特徵列出來
ALIGNED_FEATURES = [
    "zscore_7d", "zscore_30d", "pct_change_norm", 
    "volatility_7d", "is_above_mean_30d", "dow_sin", "dow_cos"
]
TARGET_COL = "future_expense_7d_sum"

# ── 3. 切割 3D 視窗與 Train/Val/Test ───────────────────────────────────
print("🪟 正在將資料切割成 3D 彈藥包 (30天視窗)...")
INPUT_DAYS = 30

X_train_l, y_train_l = [], []
X_val_l,   y_val_l   = [], []
X_test_l,  y_test_l  = [], []

for uid in df["user_id"].unique():
    u = df[df["user_id"] == uid].reset_index(drop=True)
    feat = u[ALIGNED_FEATURES].values.astype(np.float32)
    target = u[TARGET_COL].values.astype(np.float32)
    
    windows_X, windows_y = [], []
    for i in range(INPUT_DAYS, len(u)):
        windows_X.append(feat[i - INPUT_DAYS: i])
        windows_y.append([target[i]])
        
    n = len(windows_X)
    if n < 10: continue
    
    # 70% 訓練, 15% 驗證, 15% 測試
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)
    
    X_train_l.extend(windows_X[:t_end]);    y_train_l.extend(windows_y[:t_end])
    X_val_l.extend(windows_X[t_end:v_end]); y_val_l.extend(windows_y[t_end:v_end])
    X_test_l.extend(windows_X[v_end:]);     y_test_l.extend(windows_y[v_end:])

X_train = np.array(X_train_l, dtype=np.float32)
y_train = np.array(y_train_l, dtype=np.float32)
X_val   = np.array(X_val_l, dtype=np.float32)
y_val   = np.array(y_val_l, dtype=np.float32)
X_test  = np.array(X_test_l, dtype=np.float32)
y_test  = np.array(y_test_l, dtype=np.float32)

print(f"✅ 切割完成！ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ── 4. 標準化真實金額 (Target Scaler) ──────────────────────────────────
print("📏 正在建立並儲存 Target Scaler...")
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train).astype(np.float32)
y_val_scaled   = target_scaler.transform(y_val).astype(np.float32)
y_test_scaled  = target_scaler.transform(y_test).astype(np.float32)

# ── 5. 存檔到 artifacts_aligned ───────────────────────────────────────
print(f"💾 正在將彈藥包存入 {SAVE_DIR} ...")
np.save(SAVE_DIR / "personal_aligned_X_train.npy", X_train)
np.save(SAVE_DIR / "personal_aligned_y_train.npy", y_train_scaled)
np.save(SAVE_DIR / "personal_aligned_X_val.npy", X_val)
np.save(SAVE_DIR / "personal_aligned_y_val.npy", y_val_scaled)
np.save(SAVE_DIR / "personal_aligned_X_test.npy", X_test)
np.save(SAVE_DIR / "personal_aligned_y_test.npy", y_test_scaled) # scaled target
np.save(SAVE_DIR / "personal_aligned_y_test_raw.npy", y_test)    # 未縮放的原始金額 (評估算 MAE 用的)

with open(SAVE_DIR / "personal_aligned_target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)

print("🎉 [Step 1] 大功告成！專屬 Alignment 彈藥包已準備完畢！")
print("👉 下一步：您可以直接執行 4_train_aligned_bigru.py 了！")