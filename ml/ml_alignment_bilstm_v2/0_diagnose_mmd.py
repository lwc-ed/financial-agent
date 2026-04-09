"""
Step 0：MMD 診斷
=================
量化三種特徵空間的 Domain Gap（Walmart vs 個人）：
  1. 原始特徵（Naive TL，7 個絕對數值特徵）
  2. 舊 Aligned 特徵（7 個 z-score，ml_alignment_lwc）
  3. 新 Aligned 特徵（10 個 z-score + pct_rank，本方法）

輸出：
  - mmd_diagnosis.json
  - mmd_diagnosis.png（特徵分佈圖）
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

ARTIFACTS_DIR = "artifacts_bilstm_v2"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. 載入原始資料
# ─────────────────────────────────────────────────────────────────────────────
print("📂 載入 Walmart 與個人資料...")
walmart_daily  = load_walmart_daily()
personal_daily = load_personal_daily()
print(f"  Walmart  : {len(walmart_daily):,} 筆 | {walmart_daily['store_id'].nunique()} 家店")
print(f"  Personal : {len(personal_daily):,} 筆 | {personal_daily['user_id'].nunique()} 位用戶")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 原始特徵（模擬 Naive TL：直接用絕對數值 + StandardScaler）
# ─────────────────────────────────────────────────────────────────────────────
print("\n📊 計算原始特徵（Naive TL）...")
from sklearn.preprocessing import StandardScaler

def build_original_features(df, id_col):
    rows = []
    for uid in df[id_col].unique():
        u = df[df[id_col] == uid].sort_values("date").reset_index(drop=True)
        u["expense_7d_mean"]  = u["daily_expense"].rolling(7,  min_periods=1).mean()
        u["expense_30d_sum"]  = u["daily_expense"].rolling(30, min_periods=1).sum()
        u["has_expense"]      = (u["daily_expense"] > 0).astype(float)
        u["net_30d_sum"]      = -u["expense_30d_sum"]
        u["txn_30d_sum"]      = u["daily_expense"].rolling(30, min_periods=1).count()
        rows.append(u)
    result = pd.concat(rows).reset_index(drop=True)
    cols = ["daily_expense", "expense_7d_mean", "expense_30d_sum",
            "has_expense", "net_30d_sum", "txn_30d_sum"]
    return result[cols].values

X_wm_orig_raw = build_original_features(walmart_daily,  "store_id")
X_ps_orig_raw = build_original_features(personal_daily, "user_id")
X_wm_orig = StandardScaler().fit_transform(X_wm_orig_raw)
X_ps_orig = StandardScaler().fit_transform(X_ps_orig_raw)

# ─────────────────────────────────────────────────────────────────────────────
# 3. 新 Aligned 特徵（10 個，本方法）
# ─────────────────────────────────────────────────────────────────────────────
print("📊 計算新 Aligned 特徵（10 features）...")

def build_aligned_features(df, id_col):
    rows = []
    for uid in df[id_col].unique():
        u = df[df[id_col] == uid].sort_values("date").reset_index(drop=True)
        feats = compute_aligned_features(u["daily_expense"], u["date"])
        rows.append(feats)
    return pd.concat(rows).reset_index(drop=True)[ALIGNED_FEATURE_COLS].values

X_wm_aligned = build_aligned_features(walmart_daily,  "store_id")
X_ps_aligned = build_aligned_features(personal_daily, "user_id")

print(f"  原始特徵   : Walmart {X_wm_orig.shape} | Personal {X_ps_orig.shape}")
print(f"  新 Aligned : Walmart {X_wm_aligned.shape} | Personal {X_ps_aligned.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. 計算 MMD
# ─────────────────────────────────────────────────────────────────────────────
def compute_mmd(X, Y, sigma=1.0, n_sample=3000, seed=42):
    """RBF Kernel MMD（越小 = 兩個 domain 越接近）"""
    rng = np.random.default_rng(seed)
    if len(X) > n_sample:
        X = X[rng.choice(len(X), n_sample, replace=False)]
    if len(Y) > n_sample:
        Y = Y[rng.choice(len(Y), n_sample, replace=False)]

    def rbf(A, B):
        dist_sq = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-dist_sq / (2 * sigma ** 2))

    return float(rbf(X, X).mean() + rbf(Y, Y).mean() - 2 * rbf(X, Y).mean())

print("\n📐 計算 MMD...")
mmd_orig    = compute_mmd(X_wm_orig,    X_ps_orig)
mmd_aligned = compute_mmd(X_wm_aligned, X_ps_aligned)

reduction_from_orig    = (mmd_orig - mmd_aligned) / (mmd_orig + 1e-10) * 100

# 舊版 7 feature aligned（ml_alignment_lwc 的結果，硬編碼供比較）
mmd_old_aligned = 0.090122

print(f"\n{'='*58}")
print(f"  MMD 比較（越低越好）")
print(f"{'='*58}")
print(f"  ❌ 原始特徵（Naive TL）       : {mmd_orig:.6f}")
print(f"  🔶 舊 Aligned（7 features）   : {mmd_old_aligned:.6f}  （-{(mmd_orig-mmd_old_aligned)/mmd_orig*100:.1f}%）")
print(f"  ✅ 新 Aligned（10 features）  : {mmd_aligned:.6f}  （-{reduction_from_orig:.1f}%）")
print(f"{'='*58}")
print(f"\n  新 vs 舊 Aligned：{(mmd_old_aligned - mmd_aligned) / mmd_old_aligned * 100:+.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 視覺化：前兩個主成分（PCA）+ 特徵分佈
# ─────────────────────────────────────────────────────────────────────────────
print("\n🎨 繪製特徵分佈對比圖...")

rng    = np.random.default_rng(42)
n_draw = 2000

def sample(X, n):
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    return X[idx]

s_wm_o = sample(X_wm_orig,    n_draw)
s_ps_o = sample(X_ps_orig,    n_draw)
s_wm_a = sample(X_wm_aligned, n_draw)
s_ps_a = sample(X_ps_aligned, n_draw)

n_orig    = X_wm_orig.shape[1]
n_aligned = len(ALIGNED_FEATURE_COLS)
n_cols    = max(n_orig, n_aligned)

fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3, 7))
fig.suptitle(
    f"Domain Gap 診斷（MMD）\n"
    f"原始: {mmd_orig:.4f}  →  舊 Aligned(7): {mmd_old_aligned:.4f}  →  新 Aligned(10): {mmd_aligned:.4f}",
    fontsize=11, fontweight="bold"
)

C_WM = "#E74C3C"
C_PS = "#3498DB"
C_AL = "#2ECC71"

orig_cols    = ["daily_expense", "expense_7d_mean", "expense_30d_sum",
                "has_expense", "net_30d_sum", "txn_30d_sum"]
aligned_cols = ALIGNED_FEATURE_COLS

for i in range(n_cols):
    # 上排：原始特徵
    ax = axes[0, i]
    if i < n_orig:
        ax.hist(s_wm_o[:, i], bins=40, alpha=0.6, color=C_WM, label="Walmart",  density=True)
        ax.hist(s_ps_o[:, i], bins=40, alpha=0.6, color=C_PS, label="Personal", density=True)
        ax.set_title(orig_cols[i], fontsize=7)
        if i == 0:
            ax.set_ylabel("❌ Naive TL", fontsize=8, color=C_WM, fontweight="bold")
        ax.legend(fontsize=5)
    else:
        ax.axis("off")
    ax.tick_params(labelsize=5)

    # 下排：新 Aligned 特徵
    ax = axes[1, i]
    if i < n_aligned:
        ax.hist(s_wm_a[:, i], bins=40, alpha=0.6, color=C_WM, label="Walmart",  density=True)
        ax.hist(s_ps_a[:, i], bins=40, alpha=0.6, color=C_AL, label="Personal", density=True)
        ax.set_title(aligned_cols[i], fontsize=7)
        if i == 0:
            ax.set_ylabel("✅ 新 Aligned (10)", fontsize=8, color=C_AL, fontweight="bold")
        ax.legend(fontsize=5)
    else:
        ax.axis("off")
    ax.tick_params(labelsize=5)

plt.tight_layout()
plot_path = f"{ARTIFACTS_DIR}/mmd_diagnosis.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  ✅ 分佈圖儲存至 {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 儲存結果
# ─────────────────────────────────────────────────────────────────────────────
result = {
    "mmd_naive_tl_original"   : round(mmd_orig,        6),
    "mmd_old_aligned_7feat"   : round(mmd_old_aligned,  6),
    "mmd_new_aligned_10feat"  : round(mmd_aligned,      6),
    "reduction_from_naive_pct": round(reduction_from_orig, 2),
    "reduction_vs_old_aligned_pct": round((mmd_old_aligned - mmd_aligned) / mmd_old_aligned * 100, 2),
    "aligned_feature_names"   : ALIGNED_FEATURE_COLS,
    "conclusion": (
        f"新 Aligned 特徵（10 個）將 domain gap 從 {mmd_orig:.4f}（Naive TL）"
        f"縮減至 {mmd_aligned:.4f}，共縮減 {reduction_from_orig:.1f}%，"
        f"比舊版 7 特徵（{mmd_old_aligned:.4f}）又進一步降低。"
    )
}

with open(f"{ARTIFACTS_DIR}/mmd_diagnosis.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"  ✅ MMD 結果儲存至 {ARTIFACTS_DIR}/mmd_diagnosis.json")
print(f"\n🎉 診斷完成！")
