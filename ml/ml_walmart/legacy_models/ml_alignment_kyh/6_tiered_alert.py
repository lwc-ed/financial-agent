"""
Step 6：三級警報評估（kyh aligned XGBoost）
===========================================
將回歸預測轉為三個等級：
  🟢 正常    ：預測值 < 個人基線 × 1.2
  🟡 低度警告：個人基線 × 1.2 ≤ 預測值 < 個人基線 × 1.8
  🔴 高度警告：預測值 ≥ 個人基線 × 1.8

輸出：results/tiered_alert_evaluation.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from alignment_utils import DATA_DIR, MODEL_DIR, RESULT_DIR, load_feature_schema

LOW_RATIO = 1.2
HIGH_RATIO = 1.8
TIER_NAMES = ["🟢 正常", "🟡 低度", "🔴 高度"]
TIER_LABELS = [0, 1, 2]

COST_MATRIX = np.array([
    [0.0, 0.5, 2.0],
    [2.0, 0.0, 1.0],
    [5.0, 3.0, 0.0],
])


def get_test_df() -> pd.DataFrame:
    own_data_path = DATA_DIR / "own_processed_aligned.csv"
    if not own_data_path.exists():
        raise FileNotFoundError(f"找不到資料檔案: {own_data_path}")

    df = pd.read_csv(own_data_path)
    if "target" not in df.columns:
        if "label" not in df.columns:
            raise ValueError("own_processed_aligned.csv 缺少 target/label 欄位，無法做三級警報評估")
        df["target"] = df["label"]

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    if len(test_df) == 0:
        raise ValueError("測試集為空，請檢查 own_processed_aligned.csv")
    return test_df


def load_or_generate_predictions(test_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    pred_path = RESULT_DIR / "predictions_finetuned_aligned.csv"
    required_cols = {"y_true", "y_pred"}

    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        if required_cols.issubset(pred_df.columns) and len(pred_df) == len(test_df):
            return pred_df
        print(f"[WARN] 既有 prediction 檔格式不符或筆數不一致，重新推論: {pred_path}")

    model_path = MODEL_DIR / "xgb_finetuned_own_aligned.json"
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型檔案: {model_path}")

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    preds = model.predict(test_df[feature_cols].copy())

    pred_df = test_df[["user_id", "date", "target"]].copy()
    pred_df["y_true"] = test_df["target"].values
    pred_df["y_pred"] = preds
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 已重新產生 prediction 檔 -> {pred_path}")
    return pred_df


def to_tier(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    ratio = values / (baseline + 1e-8)
    tiers = np.zeros(len(values), dtype=int)
    tiers[ratio >= LOW_RATIO] = 1
    tiers[ratio >= HIGH_RATIO] = 2
    return tiers


def tiered_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = len(y_true)
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    per_level = {}
    for lv in TIER_LABELS:
        tp = cm[lv, lv]
        fp = cm[:, lv].sum() - tp
        fn = cm[lv, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        per_level[TIER_NAMES[lv]] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "support": int(cm[lv].sum()),
        }

    exact_acc = float(np.mean(y_true == y_pred))
    ordinal_acc = float(np.mean(np.abs(y_true - y_pred) <= 1))
    severe_err = float(np.mean(np.abs(y_true - y_pred) == 2))
    expected_cost = float(sum(COST_MATRIX[t, p] for t, p in zip(y_true, y_pred)) / n)

    high_mask = y_true == 2
    high_fnr = float(np.sum((y_true == 2) & (y_pred != 2)) / high_mask.sum()) if high_mask.sum() > 0 else None

    return {
        "confusion_matrix": cm.tolist(),
        "per_level": per_level,
        "exact_accuracy": round(exact_acc, 4),
        "ordinal_accuracy": round(ordinal_acc, 4),
        "severe_error_rate": round(severe_err, 4),
        "expected_cost": round(expected_cost, 4),
        "high_alert_fnr": round(high_fnr, 4) if high_fnr is not None else None,
    }


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    test_df = get_test_df()
    feature_cols = load_feature_schema(DATA_DIR / "common_features.json")
    pred_df = load_or_generate_predictions(test_df, feature_cols)

    y_true = pred_df["y_true"].to_numpy(dtype=np.float32)
    y_pred = pred_df["y_pred"].to_numpy(dtype=np.float32)
    baseline_7d = (test_df["expense_30d_mean"].to_numpy(dtype=np.float32) * 7.0)

    y_true_tier = to_tier(y_true, baseline_7d)
    y_pred_tier = to_tier(y_pred, baseline_7d)
    user_ids = test_df["user_id"].astype(str).to_numpy()

    print(f"\n真實分佈：" + "  ".join(
        f"{TIER_NAMES[t]}={np.sum(y_true_tier == t)}({np.mean(y_true_tier == t) * 100:.1f}%)"
        for t in TIER_LABELS
    ))
    print(f"預測分佈：" + "  ".join(
        f"{TIER_NAMES[t]}={np.sum(y_pred_tier == t)}({np.mean(y_pred_tier == t) * 100:.1f}%)"
        for t in TIER_LABELS
    ))

    global_tiered = tiered_metrics(y_true_tier, y_pred_tier)
    per_user_tiered = {}
    for uid in np.unique(user_ids):
        mask = user_ids == uid
        m = tiered_metrics(y_true_tier[mask], y_pred_tier[mask])
        m["true_dist"] = {TIER_NAMES[t]: int(np.sum(y_true_tier[mask] == t)) for t in TIER_LABELS}
        m["pred_dist"] = {TIER_NAMES[t]: int(np.sum(y_pred_tier[mask] == t)) for t in TIER_LABELS}
        m["n_samples"] = int(mask.sum())
        per_user_tiered[uid] = m

    print(f"\n{'=' * 65}")
    print(f"  三級警報評估（kyh aligned XGBoost，thresholds: {LOW_RATIO}× / {HIGH_RATIO}×）")
    print(f"{'=' * 65}")
    print("\n  混淆矩陣（行=真實，列=預測）：")
    print(f"  {'':12s}  {'🟢預測正常':>10}  {'🟡預測低度':>10}  {'🔴預測高度':>10}")
    for i, row in enumerate(global_tiered["confusion_matrix"]):
        print(f"  {TIER_NAMES[i]:12s}  {row[0]:>10}  {row[1]:>10}  {row[2]:>10}")

    print("\n  Per-level 指標（one-vs-rest）：")
    for name, m in global_tiered["per_level"].items():
        print(f"  {name}  Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}  (support={m['support']})")

    print("\n  整體指標：")
    print(f"  Exact Accuracy   ：{global_tiered['exact_accuracy']:.4f}")
    print(f"  Ordinal Accuracy ：{global_tiered['ordinal_accuracy']:.4f}  （差 ≤ 1 級都算可接受）")
    print(f"  Severe Error Rate：{global_tiered['severe_error_rate']:.4f}  （跨兩級的嚴重誤判）")
    print(f"  Expected Cost    ：{global_tiered['expected_cost']:.4f}")
    if global_tiered["high_alert_fnr"] is not None:
        print(f"  高度警告 FNR     ：{global_tiered['high_alert_fnr']:.4f}")

    output = {
        "model": "kyh aligned XGBoost (finetuned)",
        "low_ratio": LOW_RATIO,
        "high_ratio": HIGH_RATIO,
        "tier_names": TIER_NAMES,
        "cost_matrix": COST_MATRIX.tolist(),
        "global_metrics": global_tiered,
        "true_distribution": {TIER_NAMES[t]: int(np.sum(y_true_tier == t)) for t in TIER_LABELS},
        "pred_distribution": {TIER_NAMES[t]: int(np.sum(y_pred_tier == t)) for t in TIER_LABELS},
        "per_user": per_user_tiered,
    }

    output_path = RESULT_DIR / "tiered_alert_evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=NpEncoder)

    print(f"\n✅ 儲存至 {output_path}")
    print("🎉 完成！")


if __name__ == "__main__":
    main()
