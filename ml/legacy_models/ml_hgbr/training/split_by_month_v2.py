import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
ML_DIR = Path(__file__).resolve().parents[1]
ML_ROOT = ML_DIR.parent.parent if ML_DIR.parent.name == "legacy_models" else ML_DIR.parent
ARTIFACT_DIR = ML_DIR / "artifacts"
PROCESSED_DIR = ML_ROOT / "processed_data" / "artifacts"
DEFAULT_LEDGER_PATH = PROCESSED_DIR / "daily_ledger_all.parquet"
DEFAULT_FEATURES_PATH = PROCESSED_DIR / "features_all.parquet"
DEFAULT_MIN_DAYS_PER_MONTH = 15
DEFAULT_N_FOLDS = 3
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_TEST_FRACTION = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build month-based expanding-window folds per user and save fold datasets."
    )
    parser.add_argument("--daily-ledger", type=Path, default=DEFAULT_LEDGER_PATH, help="Path to daily_ledger_all.parquet")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH, help="Path to features_all.parquet")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR, help="Output directory")
    parser.add_argument(
        "--min-days-per-month",
        type=int,
        default=DEFAULT_MIN_DAYS_PER_MONTH,
        help="Drop user-months with active_days lower than this threshold",
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS, help="Number of expanding folds")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION, help="Validation fraction per fold")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Holdout test fraction from latest eligible months per user",
    )
    return parser.parse_args()


def add_month_start(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["month_start"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return out


def compute_monthly_activity(daily_ledger: pd.DataFrame) -> pd.DataFrame:
    ledger = add_month_start(daily_ledger)
    ledger["is_active_day"] = (ledger["txn_count"] > 0).astype(int)
    return (
        ledger.groupby(["user_id", "month_start"], as_index=False)
        .agg(active_days=("is_active_day", "sum"))
        .sort_values(["user_id", "month_start"])
        .reset_index(drop=True)
    )


def write_invalid_months_debug(
    monthly_activity: pd.DataFrame,
    min_days_per_month: int,
    output_dir: Path,
) -> Path:
    invalid = monthly_activity.loc[monthly_activity["active_days"] < min_days_per_month].copy()
    invalid["month_str"] = invalid["month_start"].dt.strftime("%Y-%m")

    lines = []
    for user_id in sorted(monthly_activity["user_id"].unique().tolist()):
        lines.append(f"{user_id}無效月份：")
        user_invalid = invalid.loc[invalid["user_id"] == user_id]
        if user_invalid.empty:
            lines.append("無")
        else:
            for row in user_invalid.itertuples(index=False):
                lines.append(f"{row.month_str} (active_days={row.active_days})")
        lines.append("")

    debug_path = output_dir / "invalid_months_debug_v2.txt"
    debug_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return debug_path


def build_expanding_assignments(
    eligible_months: pd.DataFrame,
    n_folds: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be between 0 and 1")
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be between 0 and 1")

    base_train_fraction = 0.5
    assignments = []
    holdout_assignments = []
    skipped_users = []

    for user_id, user_df in eligible_months.groupby("user_id"):
        months = sorted(user_df["month_start"].tolist())
        n_months = len(months)

        test_count = int(np.floor(n_months * test_fraction))
        test_count = max(1, min(test_count, n_months - 1))
        cv_months = months[:-test_count]
        holdout_months = months[-test_count:]

        if len(cv_months) < 2:
            skipped_users.append(user_id)
            continue

        for month in holdout_months:
            holdout_assignments.append(
                {
                    "user_id": user_id,
                    "month_start": month,
                    "split": "test",
                }
            )

        n_cv_months = len(cv_months)
        for fold_idx in range(n_folds):
            train_fraction = base_train_fraction + fold_idx * val_fraction
            train_end = int(np.floor(n_cv_months * train_fraction))
            train_end = max(1, min(train_end, n_cv_months))

            val_start = train_end
            val_end = int(np.floor(n_cv_months * (train_fraction + val_fraction)))
            val_end = max(val_start, min(val_end, n_cv_months))

            train_months = cv_months[:train_end]
            val_months = cv_months[val_start:val_end]
            if not val_months:
                continue

            for month in train_months:
                assignments.append(
                    {
                        "user_id": user_id,
                        "month_start": month,
                        "fold": fold_idx + 1,
                        "split": "train",
                    }
                )
            for month in val_months:
                assignments.append(
                    {
                        "user_id": user_id,
                        "month_start": month,
                        "fold": fold_idx + 1,
                        "split": "val",
                    }
                )

    if not assignments:
        raise RuntimeError("No fold assignments were generated. Check data size and split params.")
    if not holdout_assignments:
        raise RuntimeError("No holdout test assignments were generated. Check data size and split params.")

    assignment_df = (
        pd.DataFrame(assignments)
        .drop_duplicates(subset=["user_id", "month_start", "fold", "split"])
        .sort_values(["fold", "user_id", "month_start", "split"])
        .reset_index(drop=True)
    )
    holdout_df = (
        pd.DataFrame(holdout_assignments)
        .drop_duplicates(subset=["user_id", "month_start", "split"])
        .sort_values(["user_id", "month_start", "split"])
        .reset_index(drop=True)
    )
    return assignment_df, holdout_df, skipped_users


def save_fold_split(df: pd.DataFrame, fold: int, split: str, output_dir: Path) -> None:
    parquet_path = output_dir / f"features_fold{fold}_{split}.parquet"
    csv_path = output_dir / f"features_fold{fold}_{split}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(
        f"[FOLD {fold} {split.upper()}] saved {parquet_path.name} and {csv_path.name} "
        f"rows={len(df)} users={df['user_id'].nunique() if not df.empty else 0}"
    )


def save_named_split(df: pd.DataFrame, split_name: str, output_dir: Path) -> None:
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

    monthly_activity = compute_monthly_activity(daily_ledger)
    kept_user_months = monthly_activity.loc[
        monthly_activity["active_days"] >= args.min_days_per_month
    ].copy()
    debug_path = write_invalid_months_debug(
        monthly_activity=monthly_activity,
        min_days_per_month=args.min_days_per_month,
        output_dir=args.output_dir,
    )

    assignments, holdout_assignments, skipped_users = build_expanding_assignments(
        eligible_months=kept_user_months,
        n_folds=args.n_folds,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    features = add_month_start(features)
    features_with_folds = features.merge(
        assignments,
        on=["user_id", "month_start"],
        how="inner",
        validate="many_to_many",
    ).sort_values(["fold", "user_id", "date"]).reset_index(drop=True)

    fold_stats = {}
    for fold in sorted(features_with_folds["fold"].unique().tolist()):
        fold_df = features_with_folds.loc[features_with_folds["fold"] == fold]
        train_df = fold_df.loc[fold_df["split"] == "train"].drop(columns=["split"]).copy()
        val_df = fold_df.loc[fold_df["split"] == "val"].drop(columns=["split"]).copy()

        if train_df.empty or val_df.empty:
            print(f"[WARN] fold={fold} has empty split (train={len(train_df)}, val={len(val_df)}), skipped saving.")
            continue

        save_fold_split(train_df, fold, "train", args.output_dir)
        save_fold_split(val_df, fold, "val", args.output_dir)
        fold_stats[f"fold_{fold}"] = {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "train_users": int(train_df["user_id"].nunique()),
            "val_users": int(val_df["user_id"].nunique()),
            "train_months": int(train_df["month_start"].nunique()),
            "val_months": int(val_df["month_start"].nunique()),
        }

    pretest_features = features.merge(
        assignments[["user_id", "month_start"]].drop_duplicates(),
        on=["user_id", "month_start"],
        how="inner",
        validate="many_to_one",
    ).sort_values(["user_id", "date"]).reset_index(drop=True)
    holdout_test = features.merge(
        holdout_assignments[["user_id", "month_start"]],
        on=["user_id", "month_start"],
        how="inner",
        validate="many_to_one",
    ).sort_values(["user_id", "date"]).reset_index(drop=True)

    save_named_split(pretest_features, "pretest", args.output_dir)
    save_named_split(holdout_test, "holdout_test", args.output_dir)

    assignment_path = args.output_dir / "month_split_v2_assignments.csv"
    assignments.to_csv(assignment_path, index=False)
    holdout_path = args.output_dir / "month_split_v2_holdout_test.csv"
    holdout_assignments.to_csv(holdout_path, index=False)

    summary = {
        "daily_ledger_path": str(args.daily_ledger),
        "features_path": str(args.features),
        "min_days_per_month": args.min_days_per_month,
        "n_folds": args.n_folds,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "eligible_user_months": int(len(kept_user_months)),
        "skipped_users": skipped_users,
        "fold_stats": fold_stats,
        "pretest_rows": int(len(pretest_features)),
        "holdout_test_rows": int(len(holdout_test)),
        "holdout_test_users": int(holdout_test["user_id"].nunique()),
        "holdout_test_months": int(holdout_test["month_start"].nunique()),
    }
    summary_path = args.output_dir / "month_split_v2_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, default=str)

    print(f"[INFO] kept user-months={len(kept_user_months)} skipped_users={len(skipped_users)}")
    if skipped_users:
        print(f"[WARN] users skipped due to insufficient eligible months: {', '.join(skipped_users)}")
    print(f"[DEBUG] saved invalid months to {debug_path}")
    print(f"[DONE] saved fold assignments to {assignment_path}")
    print(f"[DONE] saved holdout assignments to {holdout_path}")
    print(f"[DONE] saved summary to {summary_path}")


if __name__ == "__main__":
    main()
