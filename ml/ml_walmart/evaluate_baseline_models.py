"""
Evaluate deterministic baseline models on the same target test split.

Baselines:
- Naive 7d: sum of the 7 days before the prediction date.
- Moving Average 30d: mean daily expense over the 30 days before the
  prediction date, multiplied by 7.

These baselines do not use random initialization, so they produce one
deterministic result instead of 30-seed mean/std metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from output_eval_utils import (
    load_personal_transactions,
    run_output_evaluation,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "bigru" / "artifacts"
OUTPUT_ROOT = ROOT / "model_outputs"
METADATA_PATH = ARTIFACTS_DIR / "sample_metadata.csv"
Y_TEST_PATH = ARTIFACTS_DIR / "my_y_test_raw.npy"

EXCLUDE_USERS = {"user4", "user5", "user6"}


def build_daily_expense(transactions_df: pd.DataFrame) -> pd.DataFrame:
    expense_df = transactions_df.loc[
        transactions_df["transaction_type"].str.lower() == "expense"
    ].copy()

    daily_frames = []
    for user_id, group in expense_df.groupby("user_id"):
        grouped = (
            group.groupby("date")["amount"]
            .sum()
            .reset_index()
            .rename(columns={"amount": "daily_expense"})
            .sort_values("date")
        )

        date_range = pd.date_range(grouped["date"].min(), grouped["date"].max(), freq="D")
        daily = (
            grouped.set_index("date")
            .reindex(date_range, fill_value=0.0)
            .rename_axis("date")
            .reset_index()
        )
        daily["user_id"] = user_id
        daily_frames.append(daily[["user_id", "date", "daily_expense"]])

    if not daily_frames:
        raise ValueError("No expense transactions found.")

    return pd.concat(daily_frames, ignore_index=True).sort_values(["user_id", "date"])


def add_past_window_baselines(daily_df: pd.DataFrame) -> pd.DataFrame:
    daily_df = daily_df.sort_values(["user_id", "date"]).copy()
    grouped_expense = daily_df.groupby("user_id")["daily_expense"]

    daily_df["naive_7d_pred"] = grouped_expense.transform(
        lambda s: s.shift(1).rolling(7, min_periods=7).sum()
    )
    daily_df["moving_avg_30d_pred"] = grouped_expense.transform(
        lambda s: s.shift(1).rolling(30, min_periods=30).mean() * 7.0
    )
    return daily_df


def build_prediction_inputs() -> dict[str, pd.DataFrame]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")
    if not Y_TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test target file: {Y_TEST_PATH}")

    split_metadata_df = pd.read_csv(METADATA_PATH)
    split_metadata_df["date"] = pd.to_datetime(split_metadata_df["date"]).dt.normalize()

    test_metadata_df = (
        split_metadata_df.loc[split_metadata_df["split"].str.lower().eq("test")]
        .copy()
        .sort_values(["user_id", "date"])
        .reset_index(drop=True)
    )

    y_test = np.load(Y_TEST_PATH).reshape(-1).astype(float)
    if len(test_metadata_df) != len(y_test):
        raise ValueError(
            f"Test metadata rows ({len(test_metadata_df)}) do not match y_test rows ({len(y_test)})."
        )

    transactions_df = load_personal_transactions(DATA_DIR, exclude_users=EXCLUDE_USERS)
    daily_df = add_past_window_baselines(build_daily_expense(transactions_df))

    merged = test_metadata_df.merge(
        daily_df[["user_id", "date", "naive_7d_pred", "moving_avg_30d_pred"]],
        on=["user_id", "date"],
        how="left",
        validate="one_to_one",
    )

    missing = merged[["naive_7d_pred", "moving_avg_30d_pred"]].isna().sum()
    if int(missing.sum()) > 0:
        raise ValueError(f"Missing baseline predictions for test rows: {missing.to_dict()}")

    base_cols = pd.DataFrame(
        {
            "user_id": merged["user_id"].astype(str),
            "date": merged["date"],
            "y_true": y_test,
        }
    )

    return {
        "naive_7d_baseline": base_cols.assign(y_pred=merged["naive_7d_pred"].astype(float)),
        "moving_avg_30d_baseline": base_cols.assign(
            y_pred=merged["moving_avg_30d_pred"].astype(float)
        ),
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def format_row(values: list[str]) -> str:
        cells = [value.ljust(widths[index]) for index, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([format_row(headers), separator, *(format_row(row) for row in rows)])


def clear_existing_model_outputs(model_name: str) -> None:
    output_dir = OUTPUT_ROOT / model_name
    for file_name in [
        "metrics_regression.json",
        "metrics_alarm_binary.json",
        "metrics_risk_4class.json",
        "predictions.csv",
        "summary.txt",
    ]:
        path = output_dir / file_name
        if path.exists():
            path.unlink()


def main() -> None:
    split_metadata_df = pd.read_csv(METADATA_PATH)
    transactions_df = load_personal_transactions(DATA_DIR, exclude_users=EXCLUDE_USERS)
    prediction_inputs = build_prediction_inputs()

    summary_rows = []
    for model_name, prediction_input_df in prediction_inputs.items():
        clear_existing_model_outputs(model_name)
        result = run_output_evaluation(
            model_name=model_name,
            prediction_input_df=prediction_input_df,
            split_metadata_df=split_metadata_df,
            transactions_df=transactions_df,
            output_root=OUTPUT_ROOT,
        )

        regression = result["metrics_regression"]
        binary = result["metrics_alarm_binary"]
        risk_4class = result["metrics_risk_4class"]
        summary_rows.append(
            {
                "model": model_name,
                "MAE": regression["MAE"],
                "RMSE": regression["RMSE"],
                "MAPE": regression["MAPE"],
                "SMAPE": regression["SMAPE"],
                "Binary F1": binary["F1-score"],
                "Weighted F1": risk_4class["Weighted F1"],
                "note": "deterministic baseline; no random seed",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUTPUT_ROOT / "baseline_results.csv"
    summary_md = OUTPUT_ROOT / "baseline_results.md"
    summary_json = OUTPUT_ROOT / "baseline_results.json"

    summary_df.to_csv(summary_csv, index=False)
    summary_md.write_text(dataframe_to_markdown(summary_df), encoding="utf-8")
    summary_json.write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDeterministic baseline results on target test split:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_csv}")
    print(f"Saved: {summary_md}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
