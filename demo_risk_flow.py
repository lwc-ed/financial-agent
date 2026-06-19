"""
demo_risk_flow.py — 模擬「使用者記帳後，系統做的一系列事」
用真實 BiGRU ensemble + 報告中的風險公式，不需連資料庫。
執行： .venv/bin/python demo_risk_flow.py
"""
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
BIGRU_DIR = HERE / "ml" / "ml_ibm" / "bigru_TL_alignment"
ART = BIGRU_DIR / "artifacts_bigru_tl"
sys.path.insert(0, str(BIGRU_DIR))
from alignment_utils import ALIGNED_FEATURE_COLS, compute_aligned_features
from model_bigru import BiGRUWithAttention

# ── 1. 模擬輸入：一位使用者最近 30 天的每日支出 ──────────────────
np.random.seed(7)
base = np.random.normal(600, 200, 30).clip(0)        # 平日 ~600/天
base[[5, 12, 19, 26]] += np.array([2500, 1800, 3000, 2200])  # 幾筆大額消費
daily_expense = np.round(base, 0)
today = pd.Timestamp("2026-06-19")
dates = pd.date_range(today - pd.Timedelta(days=29), today, freq="D")

# 使用者剛剛記的這筆（demo 用，加進最後一天）
new_expense = 500
daily_expense[-1] += new_expense

print("="*60)
print("【步驟1】使用者記帳：晚餐 500 元 → 寫入 records 表")
print(f"  最近30天總支出 = {daily_expense.sum():,.0f} 元")
print(f"  日均支出 = {daily_expense.mean():,.0f} 元/天")

# ── 2. 計算 aligned features + BiGRU ensemble 推論 ───────────────
feat_df = compute_aligned_features(pd.Series(daily_expense), pd.Series(dates))
X = feat_df[ALIGNED_FEATURE_COLS].values
X_t = torch.tensor(X[np.newaxis, :, :], dtype=torch.float32)

device = torch.device("cpu")
models = []
for pth in sorted(ART.glob("finetune_bigru_seed*.pth")):
    m = BiGRUWithAttention(len(ALIGNED_FEATURE_COLS), 48, 2, 1, 0.4).to(device)
    ckpt = torch.load(pth, map_location=device, weights_only=True)
    m.load_state_dict(ckpt["model_state"]); m.eval(); models.append(m)
with open(ART / "personal_target_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with torch.no_grad():
    preds = [m(X_t).cpu().numpy().flatten()[0] for m in models]
scaled = float(np.mean(preds))
predicted_7d = max(0.0, float(scaler.inverse_transform([[scaled]])[0][0]))

print("\n【步驟2】30 個 seed 模型 ensemble 推論")
print(f"  預測未來 7 天總支出 = {predicted_7d:,.0f} 元")

# ── 3. 風險指標計算（可自行調整收入情境）────────────────────────
monthly_income_avg = 30000.0     # 最近6個月平均月收入
income_30d         = 30000.0     # 最近30天收入
expense_30d        = float(daily_expense.sum())

future_available_7d = (monthly_income_avg / 30.0) * 7.0
spending_pressure   = min(predicted_7d / future_available_7d, 99.0)

net_cash_flow = income_30d - expense_30d
saving_rate   = float(np.clip(net_cash_flow / monthly_income_avg, -2.0, 2.0))
cf_risk       = 1.0 - saving_rate

risk_score = 0.6 * spending_pressure + 0.4 * cf_risk

def level(r):
    return 1 if r <= 0.85 else 2 if r <= 1.0 else 3 if r <= 1.2 else 4
lv = level(risk_score)
name = {1:"安全",2:"注意",3:"警示",4:"危險"}[lv]

print("\n【步驟3】風險指標計算")
print(f"  月可動用收入        = {monthly_income_avg:,.0f}")
print(f"  未來7天可動用收入   = {future_available_7d:,.0f}  (= {monthly_income_avg:,.0f}/30*7)")
print(f"  消費壓力            = {spending_pressure:.3f}  (= {predicted_7d:,.0f}/{future_available_7d:,.0f})")
print(f"  淨現金流(30天)      = {net_cash_flow:,.0f}  (= {income_30d:,.0f}-{expense_30d:,.0f})")
print(f"  儲蓄率              = {saving_rate:.3f}")
print(f"  現金流風險          = {cf_risk:.3f}  (= 1 - {saving_rate:.3f})")
print(f"  風險分數            = {risk_score:.3f}  (= 0.6*{spending_pressure:.3f} + 0.4*{cf_risk:.3f})")
print(f"\n【步驟4】風險等級 = 等級{lv}（{name}）")
print(f"【步驟5】notification_service 判斷冷卻/升降級 → 決定是否推播")
print("="*60)
