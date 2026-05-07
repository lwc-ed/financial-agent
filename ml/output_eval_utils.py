"""
Shared output/evaluation utilities for `ml/OUTPUT_EVALUATION_SPEC.md`.

This module is intentionally model-agnostic. Each model only needs to provide:

1. prediction_input_df
   Required columns:
   - user_id
   - date
   - y_true
   - y_pred

2. split_metadata_df
   Required columns:
   - user_id
   - date
   - split

`split_metadata_df` must include at least each user's training rows so that
`monthly_available_cash` can be computed from training-period income only.
"""

from __future__ import annotations

import json
import math
from calendar import monthrange
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SPEC_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SPEC_ROOT / "data"
DEFAULT_OUTPUT_ROOT = SPEC_ROOT / "model_outputs"

PREDICTION_REQUIRED_COLUMNS = {"user_id", "date", "y_true", "y_pred"}
SPLIT_METADATA_REQUIRED_COLUMNS = {"user_id", "date", "split"}
PREDICTIONS_OUTPUT_COLUMNS = [
    "user_id",
    "date",
    "y_true",
    "y_pred",
    "monthly_available_cash",
    "spent_mtd",
    "future_available_7d",
    "true_risk_ratio",
    "pred_risk_ratio",
    "true_alarm",
    "pred_alarm",
    "true_risk_level",
    "pred_risk_level",
]
RISK_LABEL_ORDER = ["no_alarm", "low_risk", "mid_risk", "high_risk"]


@dataclass(frozen=True)
class EvaluationArtifacts:
    output_dir: Path
    metrics_regression_path: Path
    metrics_alarm_binary_path: Path
    metrics_risk_4class_path: Path
    predictions_path: Path
    summary_path: Path


def run_output_evaluation(
    model_name: str,
    prediction_input_df: pd.DataFrame,
    split_metadata_df: pd.DataFrame,
    transactions_df: pd.DataFrame | None = None,
    data_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict:
    """
    Build all spec outputs for one model.

    Parameters
    ----------
    model_name:
        Folder name under `ml/model_outputs/`.
    prediction_input_df:
        Sample-level predictions to evaluate.
        Required columns: `user_id`, `date`, `y_true`, `y_pred`.
    split_metadata_df:
        Sample-level metadata containing at least all training rows for each user.
        Required columns: `user_id`, `date`, `split`.
    transactions_df:
        Optional raw transaction dataframe. If omitted, transactions are loaded
        from `data_dir`.
    """
    pred_df = _prepare_prediction_input(prediction_input_df)
    split_df = _prepare_split_metadata(split_metadata_df)
    raw_txn_df = _prepare_transactions(transactions_df, data_dir=data_dir)

    monthly_cash_df = compute_monthly_available_cash(raw_txn_df, split_df)
    spent_lookup = build_spent_mtd_lookup(raw_txn_df)

    enriched_df = pred_df.merge(monthly_cash_df, on="user_id", how="left", validate="many_to_one")
    if enriched_df["monthly_available_cash"].isna().any():
        missing_users = sorted(enriched_df.loc[enriched_df["monthly_available_cash"].isna(), "user_id"].unique())
        raise ValueError(
            "Missing `monthly_available_cash` for users with no train-period income metadata: "
            f"{missing_users}"
        )

    enriched_df["spent_mtd"] = enriched_df.apply(
        lambda row: lookup_spent_mtd(spent_lookup, row["user_id"], row["date"]),
        axis=1,
    )
    enriched_df["future_available_7d"] = enriched_df.apply(
        lambda row: compute_future_available_7d(
            current_date=row["date"],
            monthly_available_cash=float(row["monthly_available_cash"]),
            spent_mtd=float(row["spent_mtd"]),
        ),
        axis=1,
    )
    enriched_df["true_risk_ratio"] = enriched_df.apply(
        lambda row: compute_risk_ratio(float(row["y_true"]), float(row["future_available_7d"])),
        axis=1,
    )
    enriched_df["pred_risk_ratio"] = enriched_df.apply(
        lambda row: compute_risk_ratio(float(row["y_pred"]), float(row["future_available_7d"])),
        axis=1,
    )
    enriched_df["true_alarm"] = enriched_df["true_risk_ratio"].apply(risk_ratio_to_alarm).astype(int)
    enriched_df["pred_alarm"] = enriched_df["pred_risk_ratio"].apply(risk_ratio_to_alarm).astype(int)
    enriched_df["true_risk_level"] = enriched_df["true_risk_ratio"].apply(risk_ratio_to_level)
    enriched_df["pred_risk_level"] = enriched_df["pred_risk_ratio"].apply(risk_ratio_to_level)

    predictions_df = enriched_df[PREDICTIONS_OUTPUT_COLUMNS].copy()
    predictions_df["date"] = predictions_df["date"].dt.strftime("%Y-%m-%d")

    regression_metrics = compute_regression_metrics(
        y_true=enriched_df["y_true"].to_numpy(dtype=float),
        y_pred=enriched_df["y_pred"].to_numpy(dtype=float),
    )
    binary_metrics = compute_binary_alarm_metrics(
        y_true=enriched_df["true_alarm"].to_numpy(dtype=int),
        y_pred=enriched_df["pred_alarm"].to_numpy(dtype=int),
    )
    four_class_metrics = compute_4class_risk_metrics(
        y_true=enriched_df["true_risk_level"].tolist(),
        y_pred=enriched_df["pred_risk_level"].tolist(),
    )

    artifacts = write_spec_outputs(
        model_name=model_name,
        predictions_df=predictions_df,
        regression_metrics=regression_metrics,
        binary_metrics=binary_metrics,
        four_class_metrics=four_class_metrics,
        output_root=output_root,
    )

    return {
        "artifacts": artifacts,
        "metrics_regression": regression_metrics,
        "metrics_alarm_binary": binary_metrics,
        "metrics_risk_4class": four_class_metrics,
        "predictions_df": predictions_df,
    }


def _prepare_prediction_input(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(PREDICTION_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"`prediction_input_df` missing required columns: {missing}")

    out = df.copy()
    out["user_id"] = out["user_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["y_true"] = pd.to_numeric(out["y_true"], errors="raise").astype(float)
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="raise").astype(float)
    out = out.sort_values(["user_id", "date"]).reset_index(drop=True)
    return out


def _prepare_split_metadata(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(SPLIT_METADATA_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"`split_metadata_df` missing required columns: {missing}")

    out = df.copy()
    out["user_id"] = out["user_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["split"] = out["split"].astype(str).str.lower()
    out = out.sort_values(["user_id", "date"]).reset_index(drop=True)

    if not (out["split"] == "train").any():
        raise ValueError("`split_metadata_df` must contain at least one `train` row.")

    return out


def _prepare_transactions(
    transactions_df: pd.DataFrame | None,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    if transactions_df is None:
        transactions_df = load_personal_transactions(data_dir=data_dir)

    required = {"user_id", "date", "transaction_type", "amount"}
    missing = sorted(required - set(transactions_df.columns))
    if missing:
        raise ValueError(f"`transactions_df` missing required columns: {missing}")

    out = transactions_df.copy()
    out["user_id"] = out["user_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["transaction_type"] = out["transaction_type"].astype(str)
    out["amount"] = pd.to_numeric(out["amount"], errors="raise").astype(float)
    out = out.sort_values(["user_id", "date"]).reset_index(drop=True)
    return out


def load_personal_transactions(
    data_dir: str | Path | None = None,
    exclude_users: Iterable[str] | None = None,
) -> pd.DataFrame:
    data_path = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    excluded = set(exclude_users or [])

    rows: list[pd.DataFrame] = []
    for file_path in sorted(data_path.glob("raw_transactions_*.xlsx")):
        user_id = file_path.stem.replace("raw_transactions_", "")
        if user_id in excluded:
            continue

        df = pd.read_excel(file_path)
        missing_cols = {"time_stamp", "transaction_type", "amount"} - set(df.columns)
        if missing_cols:
            raise ValueError(f"{file_path} missing columns: {sorted(missing_cols)}")

        rows.append(
            pd.DataFrame(
                {
                    "user_id": user_id,
                    "date": pd.to_datetime(df["time_stamp"]).dt.normalize(),
                    "transaction_type": df["transaction_type"],
                    "amount": pd.to_numeric(df["amount"], errors="raise"),
                }
            )
        )

    if not rows:
        raise ValueError(f"No transaction files found in {data_path}")

    return pd.concat(rows, ignore_index=True)


def compute_monthly_available_cash(
    transactions_df: pd.DataFrame,
    split_metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    train_rows = split_metadata_df.loc[split_metadata_df["split"] == "train", ["user_id", "date"]].copy()
    if train_rows.empty:
        raise ValueError("`split_metadata_df` contains no train rows.")

    train_periods = (
        train_rows.groupby("user_id")["date"]
        .agg(train_start_date="min", train_end_date="max")
        .reset_index()
    )
    train_periods["train_months_covered"] = train_periods.apply(
        lambda row: month_span_inclusive(row["train_start_date"], row["train_end_date"]),
        axis=1,
    )

    income_txn = transactions_df.loc[transactions_df["transaction_type"].str.lower() == "income"].copy()
    income_joined = income_txn.merge(train_periods, on="user_id", how="inner")
    in_train_mask = (
        (income_joined["date"] >= income_joined["train_start_date"])
        & (income_joined["date"] <= income_joined["train_end_date"])
    )
    income_joined = income_joined.loc[in_train_mask]

    income_sum = (
        income_joined.groupby("user_id")["amount"]
        .sum()
        .rename("train_income_sum")
        .reset_index()
    )

    result = train_periods.merge(income_sum, on="user_id", how="left")
    result["train_income_sum"] = result["train_income_sum"].fillna(0.0)
    result["monthly_available_cash"] = result["train_income_sum"] / result["train_months_covered"]
    return result[["user_id", "monthly_available_cash", "train_income_sum", "train_months_covered"]]


def build_spent_mtd_lookup(transactions_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    expense_df = transactions_df.loc[transactions_df["transaction_type"].str.lower() == "expense"].copy()
    if expense_df.empty:
        return {}

    daily_expense = (
        expense_df.groupby(["user_id", "date"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "daily_expense"})
    )
    daily_expense["month_period"] = daily_expense["date"].dt.to_period("M")
    daily_expense = daily_expense.sort_values(["user_id", "date"]).reset_index(drop=True)
    daily_expense["spent_mtd"] = daily_expense.groupby(["user_id", "month_period"])["daily_expense"].cumsum()

    lookup: dict[str, pd.DataFrame] = {}
    for user_id, group in daily_expense.groupby("user_id"):
        lookup[user_id] = group[["date", "month_period", "spent_mtd"]].copy().reset_index(drop=True)
    return lookup


def lookup_spent_mtd(spent_lookup: dict[str, pd.DataFrame], user_id: str, date: pd.Timestamp) -> float:
    user_lookup = spent_lookup.get(str(user_id))
    if user_lookup is None or user_lookup.empty:
        return 0.0

    target_date = pd.Timestamp(date).normalize()
    target_month = target_date.to_period("M")
    month_rows = user_lookup.loc[
        (user_lookup["month_period"] == target_month) & (user_lookup["date"] <= target_date)
    ]
    if month_rows.empty:
        return 0.0

    return float(month_rows.iloc[-1]["spent_mtd"])


def compute_future_available_7d(
    current_date: pd.Timestamp,
    monthly_available_cash: float,
    spent_mtd: float,
) -> float:
    current_date = pd.Timestamp(current_date).normalize()
    remaining_budget_month = monthly_available_cash - spent_mtd

    current_days_in_month = monthrange(current_date.year, current_date.month)[1]
    days_left_in_month = current_days_in_month - current_date.day + 1

    end_date = current_date + pd.Timedelta(days=6)
    if end_date.month == current_date.month and end_date.year == current_date.year:
        return remaining_budget_month * 7.0 / days_left_in_month

    d1 = days_left_in_month
    d2 = 7 - d1

    next_month_start = (current_date.replace(day=1) + pd.offsets.MonthBegin(1)).normalize()
    next_month_days = monthrange(next_month_start.year, next_month_start.month)[1]

    budget_part1 = remaining_budget_month * d1 / days_left_in_month
    budget_part2 = monthly_available_cash * d2 / next_month_days
    return float(budget_part1 + budget_part2)


def compute_risk_ratio(expense_7d: float, future_available_7d: float) -> float:
    if future_available_7d <= 0:
        return math.inf
    return float(expense_7d / future_available_7d)


def risk_ratio_to_alarm(risk_ratio: float) -> int:
    return 1 if risk_ratio >= 0.8 else 0


def risk_ratio_to_level(risk_ratio: float) -> str:
    if risk_ratio < 0.8:
        return "no_alarm"
    if risk_ratio < 1.0:
        return "low_risk"
    if risk_ratio < 1.2:
        return "mid_risk"
    return "high_risk"


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true - y_pred)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    positive_mask = y_true > 0
    if positive_mask.any():
        mape = float(np.mean(np.abs((y_true[positive_mask] - y_pred[positive_mask]) / y_true[positive_mask])) * 100)
    else:
        mape = None

    denom = np.abs(y_true) + np.abs(y_pred)
    nonzero_denom = denom > 0
    if nonzero_denom.any():
        smape = float(np.mean(2 * abs_err[nonzero_denom] / denom[nonzero_denom]) * 100)
    else:
        smape = None

    return {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "MAPE": round(mape, 6) if mape is not None else None,
        "SMAPE": round(smape, 6) if smape is not None else None,
    }


def compute_binary_alarm_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = len(y_true)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "Accuracy": round(float(accuracy), 6),
        "Precision": round(float(precision), 6),
        "Recall": round(float(recall), 6),
        "F1-score": round(float(f1), 6),
        "Confusion Matrix": [[tn, fp], [fn, tp]],
    }


def compute_4class_risk_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    label_to_idx = {label: idx for idx, label in enumerate(RISK_LABEL_ORDER)}
    cm = np.zeros((len(RISK_LABEL_ORDER), len(RISK_LABEL_ORDER)), dtype=int)

    for truth, pred in zip(y_true, y_pred):
        cm[label_to_idx[truth], label_to_idx[pred]] += 1

    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total else 0.0

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    supports = cm.sum(axis=1)

    for idx in range(len(RISK_LABEL_ORDER)):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    macro_precision = float(np.mean(per_class_precision)) if per_class_precision else 0.0
    macro_recall = float(np.mean(per_class_recall)) if per_class_recall else 0.0
    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0
    weighted_f1 = (
        float(np.average(per_class_f1, weights=supports)) if supports.sum() else 0.0
    )

    return {
        "labels": RISK_LABEL_ORDER,
        "Accuracy": round(accuracy, 6),
        "Macro Precision": round(macro_precision, 6),
        "Macro Recall": round(macro_recall, 6),
        "Macro F1": round(macro_f1, 6),
        "Weighted F1": round(weighted_f1, 6),
        "Confusion Matrix": cm.tolist(),
    }


def write_spec_outputs(
    model_name: str,
    predictions_df: pd.DataFrame,
    regression_metrics: dict,
    binary_metrics: dict,
    four_class_metrics: dict,
    output_root: str | Path | None = None,
) -> EvaluationArtifacts:
    root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    output_dir = root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_regression_path = output_dir / "metrics_regression.json"
    metrics_alarm_binary_path = output_dir / "metrics_alarm_binary.json"
    metrics_risk_4class_path = output_dir / "metrics_risk_4class.json"
    predictions_path = output_dir / "predictions.csv"
    summary_path = output_dir / "summary.txt"

    metrics_regression_path.write_text(
        json.dumps(regression_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics_alarm_binary_path.write_text(
        json.dumps(binary_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics_risk_4class_path.write_text(
        json.dumps(four_class_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    predictions_df.to_csv(predictions_path, index=False)
    summary_text = build_summary_text(
        model_name=model_name,
        output_dir=output_dir,
        regression_metrics=regression_metrics,
        binary_metrics=binary_metrics,
        four_class_metrics=four_class_metrics,
    )
    with open(summary_path, "a", encoding="utf-8") as f:
        if summary_path.stat().st_size > 0 if summary_path.exists() else False:
            f.write("\n" + "=" * 60 + "\n\n")
        f.write(summary_text)

    return EvaluationArtifacts(
        output_dir=output_dir,
        metrics_regression_path=metrics_regression_path,
        metrics_alarm_binary_path=metrics_alarm_binary_path,
        metrics_risk_4class_path=metrics_risk_4class_path,
        predictions_path=predictions_path,
        summary_path=summary_path,
    )


def build_summary_text(
    model_name: str,
    output_dir: Path,
    regression_metrics: dict,
    binary_metrics: dict,
    four_class_metrics: dict,
) -> str:
    from datetime import datetime
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    smape_text = "N/A" if regression_metrics.get("SMAPE") is None else str(regression_metrics["SMAPE"])
    mape_text = "N/A" if regression_metrics.get("MAPE") is None else str(regression_metrics["MAPE"])

    matrix_lines = "\n".join(str(row) for row in four_class_metrics["Confusion Matrix"])
    return (
        f"Generated at: {generated_at}\n"
        f"Model: {model_name}\n"
        f"Output Dir: {output_dir}\n\n"
        f"[Test 1] Binary Alarm\n"
        f"Accuracy: {binary_metrics['Accuracy']}\n"
        f"Precision: {binary_metrics['Precision']}\n"
        f"Recall: {binary_metrics['Recall']}\n"
        f"F1-score: {binary_metrics['F1-score']}\n"
        f"Confusion Matrix: {binary_metrics['Confusion Matrix']}\n\n"
        f"[Test 2] Risk 4-Class\n"
        f"Accuracy: {four_class_metrics['Accuracy']}\n"
        f"Macro Precision: {four_class_metrics['Macro Precision']}\n"
        f"Macro Recall: {four_class_metrics['Macro Recall']}\n"
        f"Macro F1: {four_class_metrics['Macro F1']}\n"
        f"Weighted F1: {four_class_metrics['Weighted F1']}\n"
        f"Confusion Matrix:\n"
        f"{matrix_lines}\n\n"
        f"[Test 3] Regression\n"
        f"MAE: {regression_metrics['MAE']}\n"
        f"RMSE: {regression_metrics['RMSE']}\n"
        f"MAPE: {mape_text}\n"
        f"SMAPE: {smape_text}\n\n"
        f"Notes:\n"
        f"- monthly_available_cash computed from training data only\n"
        f"- spent_mtd computed using only observed spending up to date t\n"
        f"- future_available_7d uses the shared team formula\n"
    )


def compute_per_seed_metrics(
    seed_preds_dict: dict,
    target_scaler,
    prediction_input_df: pd.DataFrame,
    split_metadata_df: pd.DataFrame,
    output_dir: "str | Path",
    transactions_df: "pd.DataFrame | None" = None,
    data_dir: "str | Path | None" = None,
) -> pd.DataFrame:
    """
    每個 seed 各自計算三個主要指標，儲存為 per_seed_metrics.csv，
    並 append 到 summary.txt。

    Parameters
    ----------
    seed_preds_dict : {seed: np.ndarray (scaled predictions on test set)}
    target_scaler   : 用來 inverse_transform 的 scaler
    prediction_input_df : 包含 user_id, date, y_true（raw）的 test DataFrame
    split_metadata_df   : 包含 user_id, date, split 的 DataFrame
    output_dir      : 要存檔的資料夾（model_outputs/<model_name>/）
    """
    import numpy as np

    output_dir = Path(output_dir)
    pred_df  = _prepare_prediction_input(prediction_input_df)
    split_df = _prepare_split_metadata(split_metadata_df)
    raw_txn_df = _prepare_transactions(transactions_df, data_dir=data_dir)

    # ── 預先計算靜態部分（所有 seed 共用）──────────────────────────────
    monthly_cash_df = compute_monthly_available_cash(raw_txn_df, split_df)
    spent_lookup    = build_spent_mtd_lookup(raw_txn_df)

    base_df = pred_df.merge(monthly_cash_df, on="user_id", how="left", validate="many_to_one")
    base_df["spent_mtd"] = base_df.apply(
        lambda row: lookup_spent_mtd(spent_lookup, row["user_id"], row["date"]), axis=1
    )
    base_df["future_available_7d"] = base_df.apply(
        lambda row: compute_future_available_7d(
            row["date"], float(row["monthly_available_cash"]), float(row["spent_mtd"])
        ), axis=1
    )
    base_df["true_risk_ratio"]  = base_df.apply(
        lambda row: compute_risk_ratio(float(row["y_true"]), float(row["future_available_7d"])), axis=1
    )
    base_df["true_alarm"]      = base_df["true_risk_ratio"].apply(risk_ratio_to_alarm).astype(int)
    base_df["true_risk_level"] = base_df["true_risk_ratio"].apply(risk_ratio_to_level)

    y_true   = base_df["y_true"].to_numpy(dtype=float)
    fav_7d   = base_df["future_available_7d"].to_numpy(dtype=float)
    t_alarm  = base_df["true_alarm"].to_numpy(dtype=int)
    t_level  = base_df["true_risk_level"].tolist()

    # ── 每個 seed 計算指標 ──────────────────────────────────────────────
    rows = []
    for seed in sorted(seed_preds_dict.keys()):
        scaled = seed_preds_dict[seed]
        y_pred = target_scaler.inverse_transform(scaled).ravel()

        pred_risk_ratio = np.array([compute_risk_ratio(float(p), float(f)) for p, f in zip(y_pred, fav_7d)])
        pred_alarm      = np.array([risk_ratio_to_alarm(r) for r in pred_risk_ratio], dtype=int)
        pred_level      = [risk_ratio_to_level(r) for r in pred_risk_ratio]

        reg  = compute_regression_metrics(y_true, y_pred)
        bin_m = compute_binary_alarm_metrics(t_alarm, pred_alarm)
        cls_m = compute_4class_risk_metrics(t_level, pred_level)

        rows.append({
            "seed":        seed,
            "MAE":         reg["MAE"],
            "RMSE":        reg["RMSE"],
            "Binary_F1":   bin_m["F1-score"],
            "Weighted_F1": cls_m["Weighted F1"],
        })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "per_seed_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ per_seed_metrics.csv 已儲存（{len(df)} 個 seed）→ {csv_path}")

    # ── Append 到 summary.txt ──────────────────────────────────────────
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "a", encoding="utf-8") as f:
        from datetime import datetime
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write("\n" + "=" * 60 + "\n\n")
        f.write(f"Generated at: {generated_at}\n")
        f.write("[Per-Seed Metrics（假設檢定用）]\n")
        f.write(f"{'Seed':>8}  {'MAE':>10}  {'RMSE':>10}  {'Binary_F1':>10}  {'Weighted_F1':>12}\n")
        f.write("-" * 58 + "\n")
        for _, row in df.iterrows():
            f.write(
                f"{int(row['seed']):>8}  {row['MAE']:>10.4f}  {row['RMSE']:>10.4f}"
                f"  {row['Binary_F1']:>10.4f}  {row['Weighted_F1']:>12.4f}\n"
            )
        f.write("\n")

    return df


def month_span_inclusive(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return (end.year - start.year) * 12 + (end.month - start.month) + 1
