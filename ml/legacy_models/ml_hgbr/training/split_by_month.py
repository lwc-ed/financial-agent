import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parents[1]
ML_ROOT = MODEL_DIR.parent.parent if MODEL_DIR.parent.name == "legacy_models" else MODEL_DIR.parent
ARTIFACT_DIR = MODEL_DIR / "artifacts"
PROCESSED_DIR = ML_ROOT / "processed_data" / "artifacts"
DEFAULT_LEDGER_PATH = PROCESSED_DIR / "daily_ledger_all.parquet"
DEFAULT_FEATURES_PATH = PROCESSED_DIR / "features_all.parquet"
DEFAULT_MIN_DAYS_PER_MONTH = 20 #調整活躍天數


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter inactive user-months and split features into train/val/test by month."
    )
    parser.add_argument(
        "--daily-ledger",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
        help="Path to daily_ledger_all.parquet",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Path to features_all.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help="Directory for split parquet/csv outputs",
    )
    parser.add_argument(
        "--min-days-per-month",
        type=int,
        default=DEFAULT_MIN_DAYS_PER_MONTH,
        help="Minimum active days required to keep a user-month",
    )
    return parser.parse_args()


def add_month_start(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["month_start"] = result["date"].dt.to_period("M").dt.to_timestamp()
    return result


def compute_kept_user_months(daily_ledger: pd.DataFrame, min_days_per_month: int) -> pd.DataFrame:
    ledger = add_month_start(daily_ledger)
    ledger["is_active_day"] = (ledger["txn_count"] > 0).astype(int)

    monthly_activity = (
        ledger.groupby(["user_id", "month_start"], as_index=False)
        .agg(active_days=("is_active_day", "sum"))
        .sort_values(["user_id", "month_start"])
        .reset_index(drop=True)
    )

    kept = monthly_activity.loc[monthly_activity["active_days"] >= min_days_per_month].copy()
    return kept


def compute_monthly_activity(daily_ledger: pd.DataFrame) -> pd.DataFrame:
    ledger = add_month_start(daily_ledger)
    ledger["is_active_day"] = (ledger["txn_count"] > 0).astype(int)
    monthly_activity = (
        ledger.groupby(["user_id", "month_start"], as_index=False)
        .agg(active_days=("is_active_day", "sum"))
        .sort_values(["user_id", "month_start"])
        .reset_index(drop=True)
    )
    return monthly_activity


def write_invalid_months_debug(
    monthly_activity: pd.DataFrame,
    min_days_per_month: int,
    output_dir: Path,
) -> Path:
    invalid_months = monthly_activity.loc[monthly_activity["active_days"] < min_days_per_month].copy()
    invalid_months["month_str"] = invalid_months["month_start"].dt.strftime("%Y-%m")

    debug_lines: list[str] = []
    all_users = sorted(monthly_activity["user_id"].unique().tolist())
    for user_id in all_users:
        debug_lines.append(f"{user_id}無效月份：")
        user_invalid = invalid_months.loc[invalid_months["user_id"] == user_id]
        if user_invalid.empty:
            debug_lines.append("無")
        else:
            for row in user_invalid.itertuples(index=False):
                debug_lines.append(f"{row.month_str} (active_days={row.active_days})")
        debug_lines.append("")

    debug_path = output_dir / "invalid_months_debug.txt"
    debug_path.write_text("\n".join(debug_lines).rstrip() + "\n", encoding="utf-8")
    return debug_path


def assign_split_by_month(months: list[pd.Timestamp]) -> dict[pd.Timestamp, str]:
    month_count = len(months)
    if month_count < 3:
        raise ValueError("Need at least 3 eligible months to build non-empty train/val/test splits.")

    train_end = int(np.floor(month_count * 0.6))
    val_end = int(np.floor(month_count * 0.8))

    train_end = max(1, min(train_end, month_count - 2))
    val_end = max(train_end + 1, min(val_end, month_count - 1))

    month_to_split: dict[pd.Timestamp, str] = {}
    for month in months[:train_end]:
        month_to_split[month] = "train"
    for month in months[train_end:val_end]:
        month_to_split[month] = "val"
    for month in months[val_end:]:
        month_to_split[month] = "test"
    return month_to_split


def build_split_assignments(eligible_months: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    assignments = []
    skipped_users = []

    for user_id, group in eligible_months.groupby("user_id"):
        months = sorted(group["month_start"].tolist())
        if len(months) < 3:
            skipped_users.append(user_id)
            continue

        month_to_split = assign_split_by_month(months)
        user_assignments = group.copy()
        user_assignments["split"] = user_assignments["month_start"].map(month_to_split)
        assignments.append(user_assignments)

    if not assignments:
        raise RuntimeError("No users have at least 3 eligible months after filtering.")

    assignment_df = pd.concat(assignments, ignore_index=True).sort_values(["user_id", "month_start"])
    return assignment_df, skipped_users


def save_split(df: pd.DataFrame, split_name: str, output_dir: Path) -> None:
    parquet_path = output_dir / f"features_{split_name}.parquet"
    csv_path = output_dir / f"features_{split_name}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(
        f"[{split_name.upper()}] saved {parquet_path.name} and {csv_path.name} "
        f"rows={len(df)} users={df['user_id'].nunique() if not df.empty else 0}"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    daily_ledger = pd.read_parquet(args.daily_ledger)
    features = pd.read_parquet(args.features)

    monthly_activity = compute_monthly_activity(daily_ledger=daily_ledger)
    kept_user_months = monthly_activity.loc[
        monthly_activity["active_days"] >= args.min_days_per_month
    ].copy()
    assignments, skipped_users = build_split_assignments(kept_user_months)
    debug_path = write_invalid_months_debug(
        monthly_activity=monthly_activity,
        min_days_per_month=args.min_days_per_month,
        output_dir=args.output_dir,
    )

    features = add_month_start(features)
    filtered = features.merge(
        assignments[["user_id", "month_start", "split"]],
        on=["user_id", "month_start"],
        how="inner",
        validate="many_to_one",
    )
    filtered = filtered.sort_values(["user_id", "date"]).reset_index(drop=True)

    split_datasets = {
        split_name: filtered.loc[filtered["split"] == split_name].drop(columns=["split"])
        for split_name in ["train", "val", "test"]
    }

    for split_name, split_df in split_datasets.items():
        if split_df.empty:
            raise RuntimeError(f"{split_name} split is empty after filtering. Check min-days or source data.")
        save_split(split_df, split_name, args.output_dir)

    assignment_path = args.output_dir / "month_split_assignments.csv"
    assignments.to_csv(assignment_path, index=False)

    summary = {
        "daily_ledger_path": str(args.daily_ledger),
        "features_path": str(args.features),
        "min_days_per_month": args.min_days_per_month,
        "eligible_user_months": int(len(kept_user_months)),
        "skipped_users_with_lt_3_months": skipped_users,
        "split_row_counts": {name: int(len(df)) for name, df in split_datasets.items()},
        "split_user_counts": {name: int(df["user_id"].nunique()) for name, df in split_datasets.items()},
        "split_month_counts": {
            name: int(df["month_start"].nunique()) for name, df in split_datasets.items()
        },
    }
    summary_path = args.output_dir / "month_split_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, default=str)

    print(f"[INFO] kept user-months={len(kept_user_months)} skipped_users={len(skipped_users)}")
    if skipped_users:
        print(f"[WARN] users skipped because eligible months < 3: {', '.join(skipped_users)}")
    print(f"[DEBUG] saved invalid months to {debug_path}")
    print(f"[DONE] saved split assignments to {assignment_path}")
    print(f"[DONE] saved summary to {summary_path}")


if __name__ == "__main__":
    main()
