"""
Step 1：診斷 Domain Gap
=======================
量化並視覺化 Walmart 與個人資料在「原始特徵空間」vs「Aligned 特徵空間」的分佈差異
輸出：
  - MMD 分數（before / after alignment）
  - 特徵分佈圖（before / after）
  - 結論：alignment 是否有效縮小 gap
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from alignment_utils import (
    compute_aligned_features,
    load_walmart_daily,
    load_personal_daily,
    ALIGNED_FEATURE_COLS,
)

ARTIFACTS_DIR = "artifacts_aligned"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 載入原始資料
# ─────────────────────────────────────────────────────────────────────────────
print("📂 載入 Walmart 與個人原始資料...")
walmart_daily  = load_walmart_daily()
personal_daily = load_personal_daily()
print(f"  Walmart  : {len(walmart_daily):,} 筆 | {walmart_daily['store_id'].nunique()} 家店")
print(f"  Personal : {len(personal_daily):,} 筆 | {personal_daily['user_id'].nunique()} 位用戶")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 原始特徵（模擬現有 pipeline：直接用絕對值 + StandardScaler）
# ─────────────────────────────────────────────────────────────────────────────
print("\n📊 計算原始特徵（現有 Naive TL pipeline）...")
from sklearn.preprocessing import StandardScaler

ORIG_FEATURES = ["daily_expense", "expense_7d_mean", "expense_30d_sum",
                 "has_expense",   "has_income",       "net_30d_sum", "txn_30d_sum"]

def build_original_features(df, id_col):
    rows = []
    for uid in df[id_col].unique():
        u = df[df[id_col] == uid].sort_values("date").reset_index(drop=True)
        u["expense_7d_mean"] = u["daily_expense"].rolling(7,  min_periods=1).mean()
        u["expense_30d_sum"] = u["daily_expense"].rolling(30, min_periods=1).sum()
        u["has_expense"]     = (u["daily_expense"] > 0).astype(float)
        u["has_income"]      = 0.0
        u["net_30d_sum"]     = -u["expense_30d_sum"]
        u["txn_30d_sum"]     = u["daily_expense"].rolling(30, min_periods=1).count()
        rows.append(u)
    result = pd.concat(rows).reset_index(drop=True)
    return result[ORIG_FEATURES].values

X_wm_orig_raw = build_original_features(walmart_daily,  "store_id")
X_ps_orig_raw = build_original_features(personal_daily, "user_id")

# 各自 fit StandardScaler（模擬現有 pipeline）
scaler_wm = StandardScaler().fit(X_wm_orig_raw)
scaler_ps = StandardScaler().fit(X_ps_orig_raw)
X_wm_orig = scaler_wm.transform(X_wm_orig_raw)
X_ps_orig = scaler_ps.transform(X_ps_orig_raw)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Aligned 特徵（Rolling Z-score，兩邊用同一函式）
# ─────────────────────────────────────────────────────────────────────────────
print("📊 計算 Aligned 特徵（Rolling Z-score）...")

def build_aligned_features(df, id_col):
    rows = []
    for uid in df[id_col].unique():
        u = df[df[id_col] == uid].sort_values("date").reset_index(drop=True)
        feats = compute_aligned_features(u["daily_expense"], u["date"])
        rows.append(feats)
    return pd.concat(rows).reset_index(drop=True)[ALIGNED_FEATURE_COLS].values

X_wm_aligned = build_aligned_features(walmart_daily,  "store_id")
X_ps_aligned = build_aligned_features(personal_daily, "user_id")

print(f"  原始空間   : Walmart {X_wm_orig.shape} | Personal {X_ps_orig.shape}")
print(f"  Aligned空間 : Walmart {X_wm_aligned.shape} | Personal {X_ps_aligned.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 計算 MMD（Maximum Mean Discrepancy）
# ─────────────────────────────────────────────────────────────────────────────
def compute_mmd(X, Y, sigma=1.0, n_sample=3000, seed=42):
    """RBF Kernel MMD，量化兩個分佈的距離（越大 = gap 越大）"""
    rng = np.random.default_rng(seed)
    if len(X) > n_sample:
        X = X[rng.choice(len(X), n_sample, replace=False)]
    if len(Y) > n_sample:
        Y = Y[rng.choice(len(Y), n_sample, replace=False)]

    def rbf(A, B):
        dist_sq = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-dist_sq / (2 * sigma ** 2))

    kXX = rbf(X, X).mean()
    kYY = rbf(Y, Y).mean()
    kXY = rbf(X, Y).mean()
    return float(kXX + kYY - 2 * kXY)

print("\n📐 計算 MMD（Maximum Mean Discrepancy）...")
mmd_orig    = compute_mmd(X_wm_orig,    X_ps_orig)
mmd_aligned = compute_mmd(X_wm_aligned, X_ps_aligned)
reduction   = (mmd_orig - mmd_aligned) / (mmd_orig + 1e-10) * 100

print(f"\n{'='*52}")
print(f"  MMD（原始特徵，Naive TL pipeline）  : {mmd_orig:.6f}")
print(f"  MMD（Aligned 特徵，Rolling Z-score）: {mmd_aligned:.6f}")
print(f"  Domain Gap 縮減幅度                : {reduction:.1f}%")
print(f"{'='*52}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 視覺化
# ─────────────────────────────────────────────────────────────────────────────
print("\n🎨 繪製特徵分佈對比圖...")

rng    = np.random.default_rng(42)
n_draw = 3000

def sample(X, n):
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    return X[idx]

s_wm_o  = sample(X_wm_orig,    n_draw)
s_ps_o  = sample(X_ps_orig,    n_draw)
s_wm_a  = sample(X_wm_aligned, n_draw)
s_ps_a  = sample(X_ps_aligned, n_draw)

n_cols = max(len(ORIG_FEATURES), len(ALIGNED_FEATURE_COLS))
fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3.2, 7))

fig.suptitle(
    f"Domain Gap 診斷（Rolling Z-score Alignment）\n"
    f"MMD 原始空間: {mmd_orig:.4f}  →  Aligned 空間: {mmd_aligned:.4f}"
    f"  （Gap 縮減 {reduction:.1f}%）",
    fontsize=12, fontweight="bold"
)

C_WM = "#E74C3C"   # 紅：Walmart
C_PS = "#3498DB"   # 藍：Personal（原始）
C_AL = "#2ECC71"   # 綠：Personal（aligned）

# 上排：原始特徵
for i, fname in enumerate(ORIG_FEATURES):
    ax = axes[0, i]
    ax.hist(s_wm_o[:, i], bins=40, alpha=0.6, color=C_WM, label="Walmart",  density=True)
    ax.hist(s_ps_o[:, i], bins=40, alpha=0.6, color=C_PS, label="Personal", density=True)
    ax.set_title(fname, fontsize=8)
    ax.tick_params(labelsize=6)
    if i == 0:
        ax.set_ylabel("❌ 原始特徵（Naive TL）", fontsize=8, color=C_WM, fontweight="bold")
    ax.legend(fontsize=6)

# 下排：Aligned 特徵
for i, fname in enumerate(ALIGNED_FEATURE_COLS):
    ax = axes[1, i]
    ax.hist(s_wm_a[:, i], bins=40, alpha=0.6, color=C_WM, label="Walmart",  density=True)
    ax.hist(s_ps_a[:, i], bins=40, alpha=0.6, color=C_AL, label="Personal", density=True)
    ax.set_title(fname, fontsize=8)
    ax.tick_params(labelsize=6)
    if i == 0:
        ax.set_ylabel("✅ Aligned 特徵（Rolling Z-score）", fontsize=8, color=C_AL, fontweight="bold")
    ax.legend(fontsize=6)

plt.tight_layout()
plot_path = f"{ARTIFACTS_DIR}/domain_gap_diagnosis.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  ✅ 分佈圖儲存至 {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 儲存 MMD 結果
# ─────────────────────────────────────────────────────────────────────────────
result = {
    "mmd_original_features"   : round(mmd_orig,    6),
    "mmd_aligned_features"    : round(mmd_aligned, 6),
    "gap_reduction_pct"       : round(reduction,   2),
    "original_feature_names"  : ORIG_FEATURES,
    "aligned_feature_names"   : ALIGNED_FEATURE_COLS,
    "conclusion": (
        f"Rolling Z-score alignment 將 domain gap（MMD）縮減了 {reduction:.1f}%，"
        f"說明 aligned 特徵空間中 Walmart 與個人資料的分佈更接近，"
        f"pretrain 的知識更容易被 finetune 保留。"
    )
}
with open(f"{ARTIFACTS_DIR}/domain_gap_mmd.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"  ✅ MMD 結果儲存至 {ARTIFACTS_DIR}/domain_gap_mmd.json")
print(f"\n🎉 診斷完成！Rolling Z-score 將 Domain Gap 縮減了 {reduction:.1f}%")
print(f"   → 下一步：執行 2_preprocess_walmart_aligned.py")
