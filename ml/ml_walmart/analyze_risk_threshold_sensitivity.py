"""
Risk threshold sensitivity analysis.

Uses an existing predictions.csv to simulate how changing the three risk
boundaries affects classification metrics, predicted level distribution, and
the percentage of predictions that change level.

The prediction output contains risk ratios rather than the backend composite
risk score, so this is a threshold-policy proxy analysis. The same boundaries
are applied to true_risk_ratio and pred_risk_ratio for each scenario.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


LABELS = ["safe", "caution", "alert", "critical"]
DEFAULT_THRESHOLDS = (0.85, 1.0, 1.2)
PRIMARY_SCENARIOS = {
    "baseline": (0.85, 1.0, 1.2),
    "conservative": (0.75, 0.9, 1.05),
    "aggressive": (0.95, 1.1, 1.35),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("ml_ibm/model_outputs/bigru_TL_alignment/predictions.csv"),
        help="Prediction CSV containing true_risk_ratio and pred_risk_ratio.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/model_outputs/risk_threshold_sensitivity"),
        help="Directory for sensitivity-analysis outputs.",
    )
    parser.add_argument(
        "--thresholds",
        nargs=3,
        type=float,
        metavar=("SAFE_MAX", "CAUTION_MAX", "ALERT_MAX"),
        default=DEFAULT_THRESHOLDS,
        help="Baseline risk thresholds. Default: 0.85 1.00 1.20.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="One-at-a-time threshold sweep step. Default: 0.05.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.15,
        help="One-at-a-time sweep distance around each baseline. Default: 0.15.",
    )
    return parser.parse_args()


def validate_thresholds(thresholds: tuple[float, float, float]) -> None:
    if not thresholds[0] < thresholds[1] < thresholds[2]:
        raise ValueError(f"Thresholds must be strictly increasing: {thresholds}")


def levels_from_ratio(
    ratios: pd.Series, thresholds: tuple[float, float, float]
) -> pd.Series:
    bins = [-np.inf, *thresholds, np.inf]
    return pd.cut(ratios, bins=bins, labels=LABELS, right=True).astype(str)


def evaluate_scenario(
    df: pd.DataFrame,
    scenario: str,
    thresholds: tuple[float, float, float],
    baseline_pred_levels: pd.Series,
    varied_threshold: str,
) -> dict[str, float | str]:
    true_levels = levels_from_ratio(df["true_risk_ratio"], thresholds)
    pred_levels = levels_from_ratio(df["pred_risk_ratio"], thresholds)
    distribution = pred_levels.value_counts(normalize=True)

    result: dict[str, float | str] = {
        "scenario": scenario,
        "varied_threshold": varied_threshold,
        "safe_max": thresholds[0],
        "caution_max": thresholds[1],
        "alert_max": thresholds[2],
        "accuracy": accuracy_score(true_levels, pred_levels),
        "macro_f1": f1_score(true_levels, pred_levels, labels=LABELS, average="macro"),
        "weighted_f1": f1_score(
            true_levels, pred_levels, labels=LABELS, average="weighted"
        ),
        "changed_from_baseline_rate": (pred_levels != baseline_pred_levels).mean(),
    }
    for label in LABELS:
        result[f"pred_{label}_rate"] = distribution.get(label, 0.0)
    return result


def build_scenarios(
    baseline: tuple[float, float, float], step: float, radius: float
) -> list[tuple[str, tuple[float, float, float], str]]:
    scenarios = [
        (name, thresholds, "primary")
        for name, thresholds in PRIMARY_SCENARIOS.items()
    ]
    if baseline != DEFAULT_THRESHOLDS:
        scenarios[0] = ("baseline", baseline, "primary")

    names = ("safe_max", "caution_max", "alert_max")
    offsets = np.arange(-radius, radius + step / 2, step)
    for idx, name in enumerate(names):
        for offset in offsets:
            if np.isclose(offset, 0):
                continue
            candidate = list(baseline)
            candidate[idx] = round(candidate[idx] + float(offset), 10)
            thresholds = tuple(candidate)
            if thresholds[0] > 0 and thresholds[0] < thresholds[1] < thresholds[2]:
                scenarios.append((f"{name}_{offset:+.2f}", thresholds, name))

    return scenarios


def main() -> None:
    args = parse_args()
    baseline = tuple(args.thresholds)
    validate_thresholds(baseline)

    df = pd.read_csv(args.predictions)
    required = {"true_risk_ratio", "pred_risk_ratio"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.dropna(subset=list(required))
    if df.empty:
        raise ValueError("No valid risk-ratio rows remain after removing missing values.")

    baseline_pred_levels = levels_from_ratio(df["pred_risk_ratio"], baseline)
    rows = [
        evaluate_scenario(df, scenario, thresholds, baseline_pred_levels, varied)
        for scenario, thresholds, varied in build_scenarios(
            baseline, args.step, args.radius
        )
    ]
    results = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "threshold_sensitivity.csv", index=False)
    primary_results = results.loc[results["varied_threshold"] == "primary"].copy()
    primary_results.to_csv(args.output_dir / "scenario_comparison.csv", index=False)

    baseline_row = results.loc[results["scenario"] == "baseline"].iloc[0]
    best_macro = results.loc[results["macro_f1"].idxmax()]
    most_stable = results.loc[
        results.loc[results["scenario"] != "baseline", "changed_from_baseline_rate"].idxmin()
    ]
    summary = {
        "source": str(args.predictions),
        "sample_count": len(df),
        "baseline_thresholds": baseline,
        "primary_scenarios": primary_results.to_dict(orient="records"),
        "baseline": baseline_row.to_dict(),
        "best_macro_f1_scenario": best_macro.to_dict(),
        "most_stable_non_baseline_scenario": most_stable.to_dict(),
        "limitation": (
            "predictions.csv contains risk_ratio, not the backend composite risk score; "
            "results are a threshold-policy proxy analysis."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    display_columns = [
        "scenario",
        "safe_max",
        "caution_max",
        "alert_max",
        "macro_f1",
        "weighted_f1",
        "changed_from_baseline_rate",
    ]
    print(primary_results[display_columns].to_string(index=False))
    print("\nBest Macro-F1 scenario:")
    print(best_macro[display_columns].to_string())
    print(f"\nOutputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
