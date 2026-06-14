"""
Diagnose RMSE tail errors for one model output folder.

This script is intentionally read-only. It inspects:
- ml_ibm/model_outputs/<model>/predictions.csv
- latest ml_ibm/model_outputs/<model>/per_seed_metrics_*.csv
- optional latest ml/model_outputs/<baseline_model>/per_seed_metrics_*.csv

The goal is to explain why RMSE may fail significance tests even when MAE and
F1 metrics improve.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


METRICS = ("MAE", "RMSE", "Binary_F1", "Weighted_F1")
LOWER_IS_BETTER = {"MAE", "RMSE"}


def latest_csv(model_dir: Path) -> Path | None:
    files = sorted(model_dir.glob("per_seed_metrics_*.csv"))
    return files[-1] if files else None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def rmse(errors: list[float]) -> float:
    if not errors:
        return math.nan
    return math.sqrt(mean([e * e for e in errors]))


def summarize_predictions(pred_path: Path, top_n: int) -> dict:
    rows = read_csv_rows(pred_path)
    enriched = []
    for idx, row in enumerate(rows, start=1):
        y_true = to_float(row, "y_true")
        y_pred = to_float(row, "y_pred")
        error = y_pred - y_true
        abs_error = abs(error)
        sq_error = error * error
        future_available_7d = to_float(row, "future_available_7d")
        enriched.append(
            {
                "row_number": idx,
                "user_id": row.get("user_id", ""),
                "date": row.get("date", ""),
                "y_true": y_true,
                "y_pred": y_pred,
                "error": error,
                "abs_error": abs_error,
                "sq_error": sq_error,
                "future_available_7d": future_available_7d,
                "true_risk_ratio": to_float(row, "true_risk_ratio"),
                "pred_risk_ratio": to_float(row, "pred_risk_ratio"),
                "true_risk_level": row.get("true_risk_level", ""),
                "pred_risk_level": row.get("pred_risk_level", ""),
                "true_alarm": row.get("true_alarm", ""),
                "pred_alarm": row.get("pred_alarm", ""),
            }
        )

    errors = [r["error"] for r in enriched]
    abs_errors = [r["abs_error"] for r in enriched]
    total_sq = sum(r["sq_error"] for r in enriched)
    top_rows = sorted(enriched, key=lambda r: r["sq_error"], reverse=True)[:top_n]
    top_sq = sum(r["sq_error"] for r in top_rows)

    user_groups: dict[str, list[dict]] = defaultdict(list)
    for row in enriched:
        user_groups[row["user_id"]].append(row)

    user_summary = []
    for user_id, group in user_groups.items():
        group_errors = [r["error"] for r in group]
        group_abs = [r["abs_error"] for r in group]
        group_sq = sum(r["sq_error"] for r in group)
        user_summary.append(
            {
                "user_id": user_id,
                "n": len(group),
                "mae": mean(group_abs),
                "rmse": rmse(group_errors),
                "bias": mean(group_errors),
                "sq_error_share": group_sq / total_sq if total_sq else math.nan,
                "top_error_rows": sum(1 for r in top_rows if r["user_id"] == user_id),
            }
        )
    user_summary.sort(key=lambda r: r["sq_error_share"], reverse=True)

    risk_pair_counts = Counter((r["true_risk_level"], r["pred_risk_level"]) for r in enriched)
    alarm_pair_counts = Counter((r["true_alarm"], r["pred_alarm"]) for r in enriched)
    high_tail_level_counts = Counter((r["true_risk_level"], r["pred_risk_level"]) for r in top_rows)

    return {
        "prediction_file": str(pred_path),
        "n_rows": len(enriched),
        "mae": mean(abs_errors) if abs_errors else math.nan,
        "rmse": rmse(errors),
        "bias": mean(errors) if errors else math.nan,
        "median_abs_error": median(abs_errors) if abs_errors else math.nan,
        "top_n": top_n,
        "top_sq_error_share": top_sq / total_sq if total_sq else math.nan,
        "top_rows": top_rows,
        "user_summary": user_summary,
        "risk_pair_counts": {f"{k[0]} -> {k[1]}": v for k, v in risk_pair_counts.most_common()},
        "alarm_pair_counts": {f"{k[0]} -> {k[1]}": v for k, v in alarm_pair_counts.most_common()},
        "high_tail_level_counts": {f"{k[0]} -> {k[1]}": v for k, v in high_tail_level_counts.most_common()},
    }


def read_metrics(path: Path) -> list[dict]:
    rows = []
    for row in read_csv_rows(path):
        parsed = {"seed": row.get("seed", "")}
        for metric in METRICS:
            parsed[metric] = to_float(row, metric, math.nan)
        rows.append(parsed)
    return rows


def summarize_metrics(current_path: Path, baseline_path: Path | None) -> dict:
    current = read_metrics(current_path)
    result = {
        "current_file": str(current_path),
        "current_mean": {
            metric: mean([r[metric] for r in current if not math.isnan(r[metric])])
            for metric in METRICS
        },
    }
    if not baseline_path:
        return result

    baseline = read_metrics(baseline_path)
    result["baseline_file"] = str(baseline_path)
    result["baseline_mean"] = {
        metric: mean([r[metric] for r in baseline if not math.isnan(r[metric])])
        for metric in METRICS
    }

    paired = []
    for b, c in zip(baseline, current):
        row = {"baseline_seed": b["seed"], "current_seed": c["seed"]}
        for metric in METRICS:
            delta = c[metric] - b[metric]
            improved = delta < 0 if metric in LOWER_IS_BETTER else delta > 0
            row[f"{metric}_delta"] = delta
            row[f"{metric}_improved"] = improved
        paired.append(row)

    result["paired_n"] = min(len(baseline), len(current))
    result["paired_improved_counts"] = {
        metric: sum(1 for row in paired if row[f"{metric}_improved"])
        for metric in METRICS
    }
    result["paired_mean_delta"] = {
        metric: mean([row[f"{metric}_delta"] for row in paired]) if paired else math.nan
        for metric in METRICS
    }
    return result


def fmt_float(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_text_report(path: Path, model: str, pred_summary: dict, metric_summary: dict) -> None:
    lines = []
    lines.append(f"RMSE Tail Diagnostics: {model}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("[Prediction Error Summary]")
    lines.append(f"rows: {pred_summary['n_rows']}")
    lines.append(f"MAE: {fmt_float(pred_summary['mae'])}")
    lines.append(f"RMSE: {fmt_float(pred_summary['rmse'])}")
    lines.append(f"Bias (pred - true): {fmt_float(pred_summary['bias'])}")
    lines.append(f"Median absolute error: {fmt_float(pred_summary['median_abs_error'])}")
    lines.append(
        f"Top {pred_summary['top_n']} squared-error share: "
        f"{fmt_float(pred_summary['top_sq_error_share'] * 100, 2)}%"
    )
    lines.append("")

    lines.append("[Per-Seed Metric Direction]")
    lines.append(f"current: {metric_summary['current_file']}")
    if "baseline_file" in metric_summary:
        lines.append(f"baseline: {metric_summary['baseline_file']}")
    for metric in METRICS:
        cur = metric_summary["current_mean"][metric]
        base = metric_summary.get("baseline_mean", {}).get(metric)
        if base is None:
            lines.append(f"{metric}: current_mean={fmt_float(cur)}")
            continue
        delta = cur - base
        direction = "better" if (delta < 0 if metric in LOWER_IS_BETTER else delta > 0) else "worse_or_flat"
        count = metric_summary["paired_improved_counts"][metric]
        paired_n = metric_summary["paired_n"]
        lines.append(
            f"{metric}: baseline={fmt_float(base)} current={fmt_float(cur)} "
            f"delta={fmt_float(delta)} direction={direction} "
            f"paired_improved={count}/{paired_n}"
        )
    lines.append("")

    lines.append("[Top Users By Squared Error Share]")
    lines.append("user_id,n,mae,rmse,bias,sq_error_share,top_error_rows")
    for row in pred_summary["user_summary"][:10]:
        lines.append(
            ",".join(
                [
                    row["user_id"],
                    str(row["n"]),
                    fmt_float(row["mae"]),
                    fmt_float(row["rmse"]),
                    fmt_float(row["bias"]),
                    fmt_float(row["sq_error_share"] * 100, 2) + "%",
                    str(row["top_error_rows"]),
                ]
            )
        )
    lines.append("")

    lines.append("[Top Squared-Error Rows]")
    header = [
        "row",
        "user_id",
        "date",
        "y_true",
        "y_pred",
        "error",
        "abs_error",
        "future_available_7d",
        "true_level",
        "pred_level",
        "true_alarm",
        "pred_alarm",
    ]
    lines.append(",".join(header))
    for row in pred_summary["top_rows"]:
        lines.append(
            ",".join(
                [
                    str(row["row_number"]),
                    row["user_id"],
                    row["date"],
                    fmt_float(row["y_true"], 2),
                    fmt_float(row["y_pred"], 2),
                    fmt_float(row["error"], 2),
                    fmt_float(row["abs_error"], 2),
                    fmt_float(row["future_available_7d"], 2),
                    row["true_risk_level"],
                    row["pred_risk_level"],
                    row["true_alarm"],
                    row["pred_alarm"],
                ]
            )
        )
    lines.append("")

    lines.append("[Risk Level Confusion Counts]")
    for key, value in pred_summary["risk_pair_counts"].items():
        lines.append(f"{key}: {value}")
    lines.append("")

    lines.append("[Top Tail Risk Level Counts]")
    for key, value in pred_summary["high_tail_level_counts"].items():
        lines.append(f"{key}: {value}")
    lines.append("")

    lines.append("[Alarm Confusion Counts]")
    for key, value in pred_summary["alarm_pair_counts"].items():
        lines.append(f"{key}: {value}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bigru_TL_alignment")
    parser.add_argument("--baseline-model", default="bigru")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    current_dir = root / "ml_ibm" / "model_outputs" / args.model
    baseline_dir = root / "ml" / "model_outputs" / args.baseline_model
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = current_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    current_metrics = latest_csv(current_dir)
    if current_metrics is None:
        raise FileNotFoundError(f"Missing per_seed_metrics_*.csv under {current_dir}")

    baseline_metrics = latest_csv(baseline_dir)
    pred_summary = summarize_predictions(pred_path, args.top_n)
    metric_summary = summarize_metrics(current_metrics, baseline_metrics)

    txt_path = out_dir / f"rmse_tail_diagnostics_{args.model}.txt"
    json_path = out_dir / f"rmse_tail_diagnostics_{args.model}.json"
    write_text_report(txt_path, args.model, pred_summary, metric_summary)
    json_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "prediction_summary": pred_summary,
                "metric_summary": metric_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Diagnostics written:\n  {txt_path}\n  {json_path}")


if __name__ == "__main__":
    main()
