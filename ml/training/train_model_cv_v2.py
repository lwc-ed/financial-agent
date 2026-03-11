import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT_DIR = Path(__file__).resolve().parents[2]
ML_DIR = ROOT_DIR / "ml"
ARTIFACT_DIR = ML_DIR / "artifacts"
TARGET_CANDIDATES = ["future_7d_expense", "future_expense_7d_sum"]
DROP_COLUMNS = {"user_id", "date", "month_start", "fold"}
DEFAULT_FEATURE_COLUMNS = [
    "daily_expense",
    "expense_30d_sum",
    "expense_7d_mean",
    "has_expense",
    "has_income",
    "net_30d_sum",
    "txn_30d_sum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HGBR on split_by_month_v2 folds and summarize CV metrics."
    )
    parser.add_argument("--input-dir", type=Path, default=ARTIFACT_DIR, help="Directory containing fold parquet files")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR, help="Directory for CV outputs")
    parser.add_argument("--n-folds", type=int, default=3, help="Number of folds")
    parser.add_argument(
        "--target-column",
        default=None,
        help="Target column. Default auto-detect from future_7d_expense/future_expense_7d_sum.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--feature-columns",
        default="",
        help="Comma-separated features. If empty, use default 7-feature set.",
    )
    return parser.parse_args()


def resolve_target_column(dataframes: list[pd.DataFrame], requested_target: str | None) -> str:
    if requested_target is not None:
        missing = [i for i, df in enumerate(dataframes) if requested_target not in df.columns]
        if missing:
            raise ValueError(f"Target column {requested_target!r} missing from dataframes index={missing}")
        return requested_target

    for candidate in TARGET_CANDIDATES:
        if all(candidate in df.columns for df in dataframes):
            return candidate

    raise ValueError(
        "Unable to detect target column. Please pass --target-column explicitly."
    )


def resolve_feature_columns(train_df: pd.DataFrame, target_column: str, feature_columns_arg: str) -> list[str]:
    if feature_columns_arg.strip():
        feature_cols = [c.strip() for c in feature_columns_arg.split(",") if c.strip()]
    else:
        feature_cols = DEFAULT_FEATURE_COLUMNS.copy()

    missing = [col for col in feature_cols if col not in train_df.columns]
    if missing:
        raise ValueError(f"Selected feature columns missing in training data: {missing}")
    if target_column in feature_cols:
        raise ValueError(f"Target column {target_column} cannot be used as feature.")
    return feature_cols


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(np.sqrt(mse))}


def load_fold_data(input_dir: Path, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = input_dir / f"features_fold{fold}_train.parquet"
    val_path = input_dir / f"features_fold{fold}_val.parquet"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Missing fold files for fold={fold}: {train_path.name}, {val_path.name}"
        )
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    return train_df, val_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    fold_predictions = []
    feature_columns = None
    target_column = None

    all_train_val_dfs = []
    for fold in range(1, args.n_folds + 1):
        train_df, val_df = load_fold_data(args.input_dir, fold)
        all_train_val_dfs.extend([train_df, val_df])

    target_column = resolve_target_column(all_train_val_dfs, args.target_column)

    for fold in range(1, args.n_folds + 1):
        train_df, val_df = load_fold_data(args.input_dir, fold)
        if feature_columns is None:
            feature_columns = resolve_feature_columns(train_df, target_column, args.feature_columns)

        x_train = train_df[feature_columns].to_numpy(dtype=np.float32)
        y_train = train_df[target_column].to_numpy(dtype=np.float32)
        x_val = val_df[feature_columns].to_numpy(dtype=np.float32)
        y_val = val_df[target_column].to_numpy(dtype=np.float32)

        model = HistGradientBoostingRegressor(
            random_state=args.random_state,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
        )
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)

        metrics = compute_metrics(y_val, val_pred)
        fold_metrics.append(
            {
                "fold": fold,
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "iterations": int(model.n_iter_),
            }
        )
        print(
            f"[FOLD {fold}] train_rows={len(train_df)} val_rows={len(val_df)} "
            f"mae={metrics['mae']:.6f} rmse={metrics['rmse']:.6f} iters={model.n_iter_}"
        )

        pred_df = val_df[["user_id", "date", target_column]].copy()
        pred_df = pred_df.rename(columns={target_column: "y_true"})
        pred_df["y_pred"] = val_pred
        pred_df["fold"] = fold
        pred_df["model_name"] = "hgbr_cv_v2"
        pred_df["abs_error"] = np.abs(pred_df["y_true"] - pred_df["y_pred"])
        pred_df["error"] = pred_df["y_pred"] - pred_df["y_true"]
        fold_predictions.append(pred_df)

    metrics_df = pd.DataFrame(fold_metrics)
    mean_mae = float(metrics_df["mae"].mean())
    std_mae = float(metrics_df["mae"].std(ddof=0))
    mean_rmse = float(metrics_df["rmse"].mean())
    std_rmse = float(metrics_df["rmse"].std(ddof=0))

    summary = {
        "model_name": "hgbr_cv_v2",
        "target_column": target_column,
        "n_folds": args.n_folds,
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "cv_mean_mae": mean_mae,
        "cv_std_mae": std_mae,
        "cv_mean_rmse": mean_rmse,
        "cv_std_rmse": std_rmse,
        "fold_metrics": fold_metrics,
    }

    metrics_csv_path = args.output_dir / "cv_metrics_v2.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    metrics_json_path = args.output_dir / "cv_metrics_v2.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    predictions_path = args.output_dir / "cv_predictions_v2.csv"
    pd.concat(fold_predictions, ignore_index=True).to_csv(predictions_path, index=False)

    report_lines = [
        "Training CV Report v2",
        "model_name: hgbr_cv_v2",
        f"target_column: {target_column}",
        f"n_folds: {args.n_folds}",
        f"feature_count: {len(feature_columns)}",
        "",
        "Fold Metrics",
    ]
    for row in fold_metrics:
        report_lines.append(
            f"fold={row['fold']} train_rows={row['train_rows']} val_rows={row['val_rows']} "
            f"mae={row['mae']:.6f} rmse={row['rmse']:.6f} iters={row['iterations']}"
        )
    report_lines.extend(
        [
            "",
            "CV Summary",
            f"cv_mean_mae: {mean_mae:.6f}",
            f"cv_std_mae: {std_mae:.6f}",
            f"cv_mean_rmse: {mean_rmse:.6f}",
            f"cv_std_rmse: {std_rmse:.6f}",
            "",
            "Artifacts",
            f"metrics_csv: {metrics_csv_path}",
            f"metrics_json: {metrics_json_path}",
            f"predictions_csv: {predictions_path}",
        ]
    )

    report_path = args.output_dir / "training_report_cv_v2.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[CV SUMMARY] mean_mae={mean_mae:.6f} std_mae={std_mae:.6f} mean_rmse={mean_rmse:.6f} std_rmse={std_rmse:.6f}")
    print(f"saved report to {report_path}")
    print(f"saved metrics csv to {metrics_csv_path}")
    print(f"saved metrics json to {metrics_json_path}")
    print(f"saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()
