import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ── 1. 路徑鎖定 ──────────────────────────────────────────────────────────
MY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MY_DIR / "artifacts_bigru_tl"
sys.path.insert(0, str(MY_DIR))
from alignment_utils import load_personal_daily

def get_label(ratio):
    if ratio < 0.8: return 0
    if 0.8 <= ratio < 1.0: return 1
    if 1.0 <= ratio < 1.2: return 2
    return 3

def main():
    print("🎯 正在利用 Metadata 進行『絕對對齊』標籤計算...")
    
    # 1. 載入原始日資料 (算公式用)
    df_daily = load_personal_daily()
    
    # 2. 載入 Metadata (對齊 X 矩陣用)
    if not (ARTIFACTS_DIR / "metadata.csv").exists():
        print("❌ 錯誤：找不到 metadata.csv，請先執行 python3 2_preprocess_personal.py")
        return
    meta = pd.read_csv(ARTIFACTS_DIR / "metadata.csv", parse_dates=["date"])
    
    # 檢查是否具備收入欄位
    has_income = 'daily_income' in df_daily.columns
    if not has_income:
        print("⚠️  警告：找不到 'daily_income'，啟動預算估算模式。")

    # 3. 逐筆計算 Risk Ratio
    official_labels = []
    
    # 先預算每位用戶的月收入/預算，存成字典加速
    user_budget_map = {}
    for uid, grp in df_daily.groupby("user_id"):
        if has_income:
            monthly_cash = grp["daily_income"].sum() / (max(len(grp)/30, 1))
        else:
            monthly_cash = (grp["daily_expense"].mean() * 30) * 1.2
        user_budget_map[uid] = monthly_cash if monthly_cash > 0 else 30000

    # 根據 metadata 的順序，確保每一筆 X 都有對應的 y
    for _, row in meta.iterrows():
        uid = row['user_id']
        t_date = row['date']
        
        # 找到當天往後 7 天的真實支出
        user_data = df_daily[df_daily['user_id'] == uid].sort_values('date')
        # 抓取該日期之後的 7 筆資料
        future_expense = user_data[user_data['date'] > t_date].head(7)['daily_expense'].sum()
        
        # 算 Risk Ratio
        budget_7d = user_budget_map[uid] / 4
        ratio = future_expense / budget_7d
        official_labels.append(get_label(ratio))

    meta['target_label'] = official_labels

    # 4. 根據 metadata 裡的 split 標記存檔
    for s in ['train', 'val', 'test']:
        subset = meta[meta['split'] == s]
        y_arr = subset['target_label'].values.astype(np.int64)
        np.save(ARTIFACTS_DIR / f"personal_y_{s}_label.npy", y_arr)
        print(f"   ✅ {s} 標籤已存出：{len(y_arr)} 筆")

    print("\n🚀 [Step 2b] 絕對對齊完成！現在 X 和 y 的數量保證一模一樣了。")

if __name__ == "__main__":
    main()