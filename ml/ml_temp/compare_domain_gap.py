"""
compare_domain_gap.py
=====================
產生「Domain Alignment 前 vs 後」的對照資料：
  對齊前（raw，絕對金額）：source(IBM) 與 target(個人) 分布差很大
  對齊後（aligned，現用特徵）：兩者分布趨於一致

輸出：
  ml/ml_temp/artifacts_gap/
    gap_summary.csv          每個特徵在兩 domain 的 mean/std + 標準化差距
    gap_features_raw.npz     raw 特徵（source / target）供自行畫圖
    gap_features_aligned.npz aligned 特徵（source / target）
    domain_gap.png           代表性特徵的分布對照圖（對齊前 vs 後）

執行： .venv/bin/python ml/ml_temp/compare_domain_gap.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts_gap"
ART.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ml_ibm" / "bigru_TL_alignment"))

from no_alignment_utils import RAW_FEATURE_COLS, compute_raw_features, load_ibm_daily_sample
from alignment_utils import ALIGNED_FEATURE_COLS, compute_aligned_features, load_personal_daily

N_IBM_USERS = 200


def stack_features(df: pd.DataFrame, compute_fn, cols) -> np.ndarray:
    """對每位 user 各自算特徵後堆疊（避免跨 user 滾動污染）。"""
    parts = []
    for _, grp in df.groupby("user_id"):
        grp = grp.sort_values("date")
        feats = compute_fn(grp["daily_expense"], grp["date"])
        parts.append(feats[cols].values.astype(np.float64))
    return np.vstack(parts)


def standardized_gap(src: np.ndarray, tgt: np.ndarray, cols) -> pd.DataFrame:
    """每個特徵的 source/target mean、std，及標準化差距（越大代表 domain gap 越大）。"""
    rows = []
    for j, c in enumerate(cols):
        ms, ss = src[:, j].mean(), src[:, j].std()
        mt, st = tgt[:, j].mean(), tgt[:, j].std()
        pooled = np.sqrt((ss**2 + st**2) / 2) + 1e-9
        rows.append(
            {
                "feature": c,
                "source_mean": ms,
                "target_mean": mt,
                "source_std": ss,
                "target_std": st,
                "std_gap": abs(ms - mt) / pooled,
            }
        )
    return pd.DataFrame(rows)


def _balanced_subsample(src, tgt, n, seed=0):
    rng = np.random.default_rng(seed)
    m = min(len(src), len(tgt), n)
    s = src[rng.choice(len(src), m, replace=False)]
    t = tgt[rng.choice(len(tgt), m, replace=False)]
    return s, t


def proxy_a_distance(src, tgt, n=4000, seed=0):
    """訓練域分類器分 source/target。分得越準 → 域差距越大。
    A-distance = 2(1-2ε)，ε 為域分類器誤差；對齊後應趨近 0（分類器≈亂猜）。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    s, t = _balanced_subsample(src, tgt, n, seed)
    X = np.vstack([s, t])
    y = np.r_[np.zeros(len(s)), np.ones(len(t))]
    clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=seed, n_jobs=-1)
    acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=-1).mean()
    err = min(1.0 - acc, 0.5)
    return 2.0 * (1.0 - 2.0 * err), float(acc)


def mmd_rbf(src, tgt, n=1500, seed=0):
    """RBF kernel 的 MMD²（median-heuristic 帶寬，scale 自適應）。對齊後應顯著下降。"""
    from sklearn.metrics.pairwise import rbf_kernel
    from scipy.spatial.distance import pdist

    s, t = _balanced_subsample(src, tgt, n, seed)
    pooled = np.vstack([s, t])
    med = np.median(pdist(pooled[: min(len(pooled), 1000)], "euclidean"))
    gamma = 1.0 / (2.0 * (med**2) + 1e-12)
    Kxx, Kyy, Kxy = rbf_kernel(s, s, gamma), rbf_kernel(t, t, gamma), rbf_kernel(s, t, gamma)
    m, k = len(s), len(t)
    return float(Kxx.sum() / (m * m) + Kyy.sum() / (k * k) - 2 * Kxy.sum() / (m * k))


def per_feature_wasserstein(src, tgt, cols):
    from scipy.stats import wasserstein_distance

    return pd.DataFrame(
        {"feature": cols,
         "wasserstein": [wasserstein_distance(src[:, j], tgt[:, j]) for j in range(len(cols))]}
    )


def main():
    print("📂 載入資料...")
    ibm = load_ibm_daily_sample(N_IBM_USERS)
    personal = load_personal_daily()
    print(f"[INFO] 個人(target)：{personal['user_id'].nunique()} 位、{len(personal):,} 筆")

    print("\n🔧 計算特徵（raw 未對齊 / aligned 已對齊）...")
    src_raw = stack_features(ibm, compute_raw_features, RAW_FEATURE_COLS)
    tgt_raw = stack_features(personal, compute_raw_features, RAW_FEATURE_COLS)
    src_al = stack_features(ibm, compute_aligned_features, ALIGNED_FEATURE_COLS)
    tgt_al = stack_features(personal, compute_aligned_features, ALIGNED_FEATURE_COLS)

    gap_raw = standardized_gap(src_raw, tgt_raw, RAW_FEATURE_COLS)
    gap_al = standardized_gap(src_al, tgt_al, ALIGNED_FEATURE_COLS)
    gap_raw["alignment"] = "raw (對齊前)"
    gap_al["alignment"] = "aligned (對齊後)"

    summary = pd.concat([gap_raw, gap_al], ignore_index=True)
    summary.to_csv(ART / "gap_summary.csv", index=False, encoding="utf-8-sig")
    np.savez(ART / "gap_features_raw.npz", source=src_raw, target=tgt_raw, cols=RAW_FEATURE_COLS)
    np.savez(ART / "gap_features_aligned.npz", source=src_al, target=tgt_al, cols=ALIGNED_FEATURE_COLS)

    raw_mean_gap = gap_raw["std_gap"].mean()
    al_mean_gap = gap_al["std_gap"].mean()

    print("\n================= Domain Gap 對照 =================")
    print(f"對齊前 (raw)     平均標準化差距：{raw_mean_gap:.3f}")
    print(f"對齊後 (aligned) 平均標準化差距：{al_mean_gap:.3f}")
    print(f"縮小幅度：{(1 - al_mean_gap / raw_mean_gap) * 100:.1f}%")
    print("\n[對齊前] 各特徵 source vs target 平均值：")
    print(gap_raw[["feature", "source_mean", "target_mean", "std_gap"]].to_string(index=False))
    print("\n[對齊後] 各特徵 source vs target 平均值：")
    print(gap_al[["feature", "source_mean", "target_mean", "std_gap"]].to_string(index=False))

    # ── 進階 domain gap 指標（Proxy A-distance / MMD / Wasserstein）──────────
    print("\n================= 進階 Domain Gap 指標 =================")
    metric_rows = []
    try:
        pad_raw, acc_raw = proxy_a_distance(src_raw, tgt_raw)
        pad_al, acc_al = proxy_a_distance(src_al, tgt_al)
        print(f"Proxy A-distance   對齊前={pad_raw:.3f} (域分類器 acc={acc_raw:.1%}) | "
              f"對齊後={pad_al:.3f} (acc={acc_al:.1%})  〔越小越好，2=完全可分,0=不可分〕")
        metric_rows += [
            {"metric": "proxy_a_distance", "raw": pad_raw, "aligned": pad_al},
            {"metric": "domain_clf_acc", "raw": acc_raw, "aligned": acc_al},
        ]
    except Exception as e:
        print(f"[WARN] Proxy A-distance 略過：{e}")

    try:
        mmd_raw = mmd_rbf(src_raw, tgt_raw)
        mmd_al = mmd_rbf(src_al, tgt_al)
        print(f"MMD² (RBF)         對齊前={mmd_raw:.4f} | 對齊後={mmd_al:.4f}  〔越小越好〕")
        metric_rows.append({"metric": "mmd2_rbf", "raw": mmd_raw, "aligned": mmd_al})
    except Exception as e:
        print(f"[WARN] MMD 略過：{e}")

    try:
        w_raw = per_feature_wasserstein(src_raw, tgt_raw, RAW_FEATURE_COLS)
        w_al = per_feature_wasserstein(src_al, tgt_al, ALIGNED_FEATURE_COLS)
        w_raw.to_csv(ART / "wasserstein_raw.csv", index=False, encoding="utf-8-sig")
        w_al.to_csv(ART / "wasserstein_aligned.csv", index=False, encoding="utf-8-sig")
        print(f"Wasserstein 平均    對齊前={w_raw['wasserstein'].mean():.3f} | "
              f"對齊後={w_al['wasserstein'].mean():.3f}  〔逐特徵明細存 wasserstein_*.csv〕")
        metric_rows.append({"metric": "wasserstein_mean",
                            "raw": w_raw["wasserstein"].mean(), "aligned": w_al["wasserstein"].mean()})
    except Exception as e:
        print(f"[WARN] Wasserstein 略過：{e}")

    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(ART / "domain_gap_metrics.csv", index=False, encoding="utf-8-sig")

    # 代表性特徵分布圖：對齊前(daily_expense) vs 對齊後(zscore_7d)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang TC", "Heiti TC"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        # 對齊前：絕對金額用 log1p 尺度，讓 source(~4) 與 target(~300) 都看得見且分離明顯
        sr = np.log1p(src_raw[:, 0].clip(min=0))
        tr = np.log1p(tgt_raw[:, 0].clip(min=0))
        bins = np.linspace(0, max(sr.max(), tr.max()), 50)
        axes[0].hist(sr, bins=bins, alpha=0.5, label="Source (IBM)", density=True)
        axes[0].hist(tr, bins=bins, alpha=0.5, label="Target (個人)", density=True)
        axes[0].set_title(f"對齊前 raw: log(每日支出) (gap={gap_raw.iloc[0]['std_gap']:.2f})")
        axes[0].set_xlabel("log(1 + 每日支出)")
        axes[0].legend()
        # 對齊後：zscore_7d
        axes[1].hist(src_al[:, 0], bins=50, alpha=0.5, label="Source (IBM)", density=True)
        axes[1].hist(tgt_al[:, 0], bins=50, alpha=0.5, label="Target (個人)", density=True)
        axes[1].set_title(f"對齊後 aligned: zscore_7d (gap={gap_al.iloc[0]['std_gap']:.2f})")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(ART / "domain_gap.png", dpi=130)
        print(f"\n🖼  分布對照圖已存：{ART / 'domain_gap.png'}")
    except Exception as e:
        print(f"[WARN] 畫圖略過：{e}")

    print(f"\n✅ 完成，結果在 {ART}")


if __name__ == "__main__":
    main()
