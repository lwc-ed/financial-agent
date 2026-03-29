# 迴圈找best combinition Regression

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings


ROOT_DIR = Path(__file__).resolve().parents[2]
ML_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ML_DIR / "artifacts"
DEFAULT_TRAIN_PATH = ARTIFACT_DIR / "features_train.parquet"
DEFAULT_VAL_PATH = ARTIFACT_DIR / "features_val.parquet"
DEFAULT_TEST_PATH = ARTIFACT_DIR / "features_test.parquet"
DEFAULT_OUTPUT_DIR = ARTIFACT_DIR
TARGET_CANDIDATES = ["future_7d_expense", "future_expense_7d_sum"]
DROP_COLUMNS = {"user_id", "date", "month_start"}
NAIVE_BASELINE_FEATURE = "expense_7d_sum"
MOVING_AVG_BASELINE_FEATURE = "expense_30d_mean"
DEFAULT_TRIALS = 500
DEFAULT_MIN_FEATURES = 5
DEFAULT_MAX_FEATURES = 15

# 固定模型設定，讓 v4_regression 的重點放在 feature subset search。
MLP_CONFIG = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "batch_size": 128,
    "learning_rate_init": 1e-3,
    "max_iter": 300,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train model version 6 with MLP random feature subset search."
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH, help="Train parquet path")
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH, help="Validation parquet path")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH, help="Test parquet path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Artifact output directory")
    parser.add_argument(
        "--target-column",
        default=None,
        help="Regression target column. Defaults to auto-detect from future_7d_expense/future_expense_7d_sum.",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of random subsets to evaluate")
    parser.add_argument("--min-features", type=int, default=DEFAULT_MIN_FEATURES, help="Minimum subset size")
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES, help="Maximum subset size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def resolve_target_column(dataframes: list[pd.DataFrame], requested_target: str | None) -> str:
    if requested_target is not None:
        missing = [index for index, df in enumerate(dataframes) if requested_target not in df.columns]
        if missing:
            raise ValueError(f"Target column {requested_target!r} is missing from dataset indexes {missing}.")
        return requested_target

    for candidate in TARGET_CANDIDATES:
        if all(candidate in df.columns for df in dataframes):
            return candidate

    raise ValueError(
        "Unable to detect target column. Provide --target-column explicitly. "
        f"Tried {TARGET_CANDIDATES}."
    )


def list_all_numeric_features(df: pd.DataFrame, target_column: str) -> list[str]:
    excluded_columns = DROP_COLUMNS | {target_column}
    numeric_columns = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    feature_columns = [column for column in numeric_columns if column not in excluded_columns]
    if not feature_columns:
        raise ValueError("No numeric feature columns available after excluding metadata and target.")
    return feature_columns


def prepare_xy(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> tuple[np.ndarray, np.ndarray]:
    x = df[feature_columns].astype(np.float32).to_numpy()
    y = df[target_column].astype(np.float32).to_numpy()
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("NaN detected in features or target. Clean the source features before training.")
    return x, y


def compute_regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(actual, prediction)
    mse = mean_squared_error(actual, prediction)
    return {
        "mae": float(mae),
        "rmse": float(np.sqrt(mse)),
    }


def fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> tuple[MLPRegressor, StandardScaler]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = MLPRegressor(
        **MLP_CONFIG,
        random_state=random_state,
    )
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    model.fit(x_train_scaled, y_train)
    return model, scaler


def sample_feature_subsets(
    feature_pool: list[str],
    trials: int,
    min_features: int,
    max_features: int,
    random_state: int,
) -> list[list[str]]:
    if min_features < 1:
        raise ValueError("--min-features must be >= 1.")
    if max_features > len(feature_pool):
        raise ValueError("--max-features cannot exceed available feature count.")
    if min_features > max_features:
        raise ValueError("--min-features cannot be greater than --max-features.")

    rng = np.random.default_rng(random_state)
    subsets: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    max_attempts = trials * 20
    attempts = 0

    while len(subsets) < trials and attempts < max_attempts:
        attempts += 1
        subset_size = int(rng.integers(min_features, max_features + 1))
        subset = tuple(sorted(rng.choice(feature_pool, size=subset_size, replace=False).tolist()))
        if subset in seen:
            continue
        seen.add(subset)
        subsets.append(list(subset))

    if len(subsets) < trials:
        print(
            f"[WARN] requested trials={trials}, but only generated {len(subsets)} unique subsets "
            f"within attempt budget."
        )
    return subsets


def build_predictions_long_df(
    test_df: pd.DataFrame,
    target_column: str,
    prediction_map: dict[str, np.ndarray],
) -> pd.DataFrame:
    frames = []
    for model_name, predictions in prediction_map.items():
        frame = test_df[["user_id", "date", target_column]].copy()
        frame = frame.rename(columns={target_column: "y_true"})
        frame["y_pred"] = predictions
        frame["model_name"] = model_name
        frame["abs_error"] = np.abs(frame["y_true"] - frame["y_pred"])
        frame["error"] = frame["y_pred"] - frame["y_true"]
        frames.append(frame[["user_id", "date", "y_true", "y_pred", "model_name", "abs_error", "error"]])
    return pd.concat(frames, ignore_index=True)


def save_pickle(payload: dict, path: Path) -> None:
    with path.open("wb") as file:
        pickle.dump(payload, file)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(args.train_path)
    val_df = pd.read_parquet(args.val_path)
    test_df = pd.read_parquet(args.test_path)

    target_column = resolve_target_column([train_df, val_df, test_df], args.target_column)
    feature_pool = list_all_numeric_features(train_df, target_column)

    print("== Model v4_regression Random Feature Search ==")
    print(f"target_column={target_column}")
    print(f"feature_pool_count={len(feature_pool)}")
    print(f"subset_size_range={args.min_features}-{args.max_features}")
    print(f"requested_trials={args.trials}")

    subsets = sample_feature_subsets(
        feature_pool=feature_pool,
        trials=args.trials,
        min_features=args.min_features,
        max_features=args.max_features,
        random_state=args.random_state,
    )

    search_records = []
    best_result = None
    best_model_path = args.output_dir / "best_mlp_model_v4_regression.pkl"

    for trial_id, feature_columns in enumerate(subsets, start=1):
        x_train, y_train = prepare_xy(train_df, feature_columns, target_column)
        x_val, y_val = prepare_xy(val_df, feature_columns, target_column)

        model, scaler = fit_mlp(
            x_train=x_train,
            y_train=y_train,
            random_state=args.random_state,
        )
        val_predictions = model.predict(scaler.transform(x_val))
        val_metrics = compute_regression_metrics(y_val, val_predictions)

        record = {
            "trial_id": trial_id,
            "feature_count": len(feature_columns),
            "feature_columns": "|".join(feature_columns),
            "val_mae": float(val_metrics["mae"]),
            "val_rmse": float(val_metrics["rmse"]),
            "iterations": int(getattr(model, "n_iter_", 0)),
            "model_name": "mlp_v4_regression",
        }
        search_records.append(record)

        if trial_id <= 5 or trial_id % 50 == 0 or trial_id == len(subsets):
            print(
                f"[v4_regression] trial={trial_id:03d}/{len(subsets):03d} "
                f"feature_count={len(feature_columns):02d} "
                f"val_mae={val_metrics['mae']:.6f} val_rmse={val_metrics['rmse']:.6f}"
            )

        is_better = best_result is None or (
            val_metrics["mae"] < best_result["val_metrics"]["mae"]
            or (
                np.isclose(val_metrics["mae"], best_result["val_metrics"]["mae"])
                and val_metrics["rmse"] < best_result["val_metrics"]["rmse"]
            )
        )
        if is_better:
            best_result = {
                "trial_id": trial_id,
                "feature_columns": feature_columns,
                "val_metrics": val_metrics,
                "iterations": int(getattr(model, "n_iter_", 0)),
                "model": model,
                "scaler": scaler,
            }
            save_pickle(
                {
                    "model_name": "mlp_v4_regression",
                    "target_column": target_column,
                    "feature_columns": feature_columns,
                    "best_val_metric": float(val_metrics["mae"]),
                    "best_val_rmse": float(val_metrics["rmse"]),
                    "trial_id": trial_id,
                    "iterations": int(getattr(model, "n_iter_", 0)),
                    "model": model,
                    "scaler": scaler,
                    "config": MLP_CONFIG,
                },
                best_model_path,
            )

    search_results_df = pd.DataFrame(search_records).sort_values(["val_mae", "val_rmse", "feature_count", "trial_id"])
    search_results_path = args.output_dir / "feature_search_results_v4_regression.csv"
    search_results_df.to_csv(search_results_path, index=False)

    with best_model_path.open("rb") as file:
        best_bundle = pickle.load(file)
    best_features = best_result["feature_columns"]
    x_test, y_test = prepare_xy(test_df, best_features, target_column)
    test_predictions = best_bundle["model"].predict(best_bundle["scaler"].transform(x_test))
    model_metrics = compute_regression_metrics(y_test, test_predictions)

    naive_predictions = test_df[NAIVE_BASELINE_FEATURE].astype(np.float32).to_numpy()
    moving_avg_predictions = test_df[MOVING_AVG_BASELINE_FEATURE].astype(np.float32).to_numpy() * 7.0
    naive_metrics = compute_regression_metrics(y_test, naive_predictions)
    moving_avg_metrics = compute_regression_metrics(y_test, moving_avg_predictions)

    print("== Model v4_regression Final Evaluation ==")
    print(f"best_trial_id={best_result['trial_id']}")
    print(f"best_val_mae={best_result['val_metrics']['mae']:.6f}")
    print(f"best_val_rmse={best_result['val_metrics']['rmse']:.6f}")
    print(f"test_mae={model_metrics['mae']:.6f} test_rmse={model_metrics['rmse']:.6f}")
    print(f"baseline_naive_7d mae={naive_metrics['mae']:.6f} rmse={naive_metrics['rmse']:.6f}")
    print(f"baseline_moving_avg_30d_x7 mae={moving_avg_metrics['mae']:.6f} rmse={moving_avg_metrics['rmse']:.6f}")

    beat_moving_avg = model_metrics["mae"] < moving_avg_metrics["mae"] and model_metrics["rmse"] < moving_avg_metrics["rmse"]
    print(f"beat_moving_avg_30d_x7={'YES' if beat_moving_avg else 'NO'}")

    best_features_payload = {
        "model_name": "mlp_v4_regression",
        "trial_id": int(best_result["trial_id"]),
        "target_column": target_column,
        "feature_count": int(len(best_features)),
        "feature_columns": best_features,
        "best_val_metric": float(best_result["val_metrics"]["mae"]),
        "best_val_rmse": float(best_result["val_metrics"]["rmse"]),
        "iterations": int(best_result["iterations"]),
        "config": MLP_CONFIG,
    }
    best_features_path = args.output_dir / "best_features_v4_regression.json"
    with best_features_path.open("w", encoding="utf-8") as file:
        json.dump(best_features_payload, file, ensure_ascii=False, indent=2)

    metrics_payload = {
        "model_name": "mlp_v4_regression",
        "target_column": target_column,
        "trial_id": int(best_result["trial_id"]),
        "feature_columns": best_features,
        "best_val_metric": float(best_result["val_metrics"]["mae"]),
        "best_val_rmse": float(best_result["val_metrics"]["rmse"]),
        "test_mae": float(model_metrics["mae"]),
        "test_rmse": float(model_metrics["rmse"]),
        "baseline_mae": {
            "naive_7d_sum": float(naive_metrics["mae"]),
            "moving_avg_30d_x7": float(moving_avg_metrics["mae"]),
        },
        "baseline_rmse": {
            "naive_7d_sum": float(naive_metrics["rmse"]),
            "moving_avg_30d_x7": float(moving_avg_metrics["rmse"]),
        },
        "beat_moving_avg_30d_x7": bool(beat_moving_avg),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_pool_count": int(len(feature_pool)),
        "searched_subsets": int(len(subsets)),
    }
    metrics_path = args.output_dir / "test_metrics_v4_regression.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, ensure_ascii=False, indent=2)

    predictions_path = args.output_dir / "predictions_test_v4_regression.csv"
    predictions_df = build_predictions_long_df(
        test_df=test_df,
        target_column=target_column,
        prediction_map={
            "mlp_v4_regression": test_predictions,
            "naive_7d_sum": naive_predictions,
            "moving_avg_30d_x7": moving_avg_predictions,
        },
    )
    predictions_df.to_csv(predictions_path, index=False)

    report_lines = [
        "Training Run Summary v4_regression",
        "model_name: mlp_v4_regression",
        f"target_column: {target_column}",
        f"searched_subsets: {len(subsets)}",
        f"subset_size_range: {args.min_features}-{args.max_features}",
        f"best_trial_id: {best_result['trial_id']}",
        f"best_val_metric: {best_result['val_metrics']['mae']:.6f}",
        f"best_val_rmse: {best_result['val_metrics']['rmse']:.6f}",
        f"test_mae: {model_metrics['mae']:.6f}",
        f"test_rmse: {model_metrics['rmse']:.6f}",
        "",
        "Dataset Sizes",
        f"train_rows: {len(train_df)}",
        f"val_rows: {len(val_df)}",
        f"test_rows: {len(test_df)}",
        f"feature_pool_count: {len(feature_pool)}",
        f"selected_feature_count: {len(best_features)}",
        "",
        "Selected Features",
    ]
    report_lines.extend([f"- {feature_name}" for feature_name in best_features])
    report_lines.extend(
        [
            "",
            "Baselines",
            f"naive_7d_sum mae: {naive_metrics['mae']:.6f}",
            f"naive_7d_sum rmse: {naive_metrics['rmse']:.6f}",
            f"moving_avg_30d_x7 mae: {moving_avg_metrics['mae']:.6f}",
            f"moving_avg_30d_x7 rmse: {moving_avg_metrics['rmse']:.6f}",
            "",
            f"beat_moving_avg_30d_x7: {beat_moving_avg}",
            "",
            "Artifacts",
            f"best_model: {best_model_path}",
            f"feature_search_results_csv: {search_results_path}",
            f"best_features_json: {best_features_path}",
            f"test_metrics_json: {metrics_path}",
            f"predictions_csv: {predictions_path}",
        ]
    )
    report_path = args.output_dir / "training_report_v4_regression.txt"
    report_text = "\n".join(report_lines) + "\n"
    report_path.write_text(report_text, encoding="utf-8")

    loop_report_path = args.output_dir / "training_report_loop.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with loop_report_path.open("a", encoding="utf-8") as file:
        if loop_report_path.exists() and loop_report_path.stat().st_size > 0:
            file.write("\n" + "=" * 80 + "\n\n")
        file.write(f"[run_timestamp] {timestamp}\n\n")
        file.write(report_text)

    print(f"saved best model to {best_model_path}")
    print(f"saved search results to {search_results_path}")
    print(f"saved best features to {best_features_path}")
    print(f"saved test metrics to {metrics_path}")
    print(f"saved predictions to {predictions_path}")
    print(f"saved report to {report_path}")
    print(f"appended report to {loop_report_path}")


if __name__ == "__main__":
    main()
