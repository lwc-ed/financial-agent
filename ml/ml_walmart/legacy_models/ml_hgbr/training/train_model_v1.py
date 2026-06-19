import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


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
FUTURE_LIKE_TOKENS = ("future", "lead", "t+")
PAST_EQUIVALENT_FEATURES = {
    "past_7d_sum": ["past_7d_sum", "expense_7d_sum"],
    "past_30d_sum": ["past_30d_sum", "expense_30d_sum"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train time-series regression models, generate diagnostics, and compare against baselines."
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
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs for the MLP candidate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on MLP validation loss")
    parser.add_argument("--hidden-dim-1", type=int, default=128, help="First MLP hidden layer size")
    parser.add_argument("--hidden-dim-2", type=int, default=64, help="Second MLP hidden layer size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="MLP learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="MLP L2 regularization")
    parser.add_argument("--batch-size", type=int, default=256, help="MLP batch size")
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


def build_feature_columns(train_df: pd.DataFrame, target_column: str) -> list[str]:
    excluded_columns = DROP_COLUMNS | {target_column}
    numeric_columns = train_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    feature_columns = [column for column in numeric_columns if column not in excluded_columns]
    if not feature_columns:
        raise ValueError("No numeric feature columns available after excluding metadata and target.")
    return feature_columns


def prepare_xy(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> tuple[np.ndarray, np.ndarray]:
    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in dataset: {missing_features}")

    x = df[feature_columns].astype(np.float32).to_numpy()
    y = df[target_column].astype(np.float32).to_numpy()
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("NaN detected in features or target. Clean the source features before training.")
    return x, y


def compute_regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(actual, prediction)
    mse = mean_squared_error(actual, prediction)
    rmse = float(np.sqrt(mse))
    return {
        "mae": float(mae),
        "rmse": float(rmse),
    }


def per_user_nmae(y_true: np.ndarray, y_pred: np.ndarray, user_ids: np.ndarray) -> float:
    """每個 user 的 MAE ÷ 該 user y_true 均值，再對所有 user 取平均（%）"""
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    nmae_list = []
    for u in np.unique(user_ids):
        mask = user_ids == u
        mean_u = y_true[mask].mean()
        if mean_u > 0:
            nmae_list.append(np.mean(np.abs(y_pred[mask] - y_true[mask])) / mean_u * 100)
    return float(np.mean(nmae_list))


def compute_target_distribution(y: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(y)),
        "median": float(np.median(y)),
        "p25": float(np.percentile(y, 25)),
        "p75": float(np.percentile(y, 75)),
        "p90": float(np.percentile(y, 90)),
        "max": float(np.max(y)),
    }


def format_metric_table(rows: list[dict[str, float | str]]) -> str:
    header = f"{'model':<24} {'mae':>12} {'rmse':>12}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(f"{str(row['model_name']):<24} {row['mae']:>12.6f} {row['rmse']:>12.6f}")
    return "\n".join(lines)


def detect_feature_patterns(feature_columns: list[str]) -> dict[str, object]:
    lowered = {column.lower(): column for column in feature_columns}
    equivalent_presence = {}
    for label, candidates in PAST_EQUIVALENT_FEATURES.items():
        matched = [candidate for candidate in candidates if candidate in lowered]
        equivalent_presence[label] = {
            "present": bool(matched),
            "matched_columns": matched,
        }

    future_like_columns = [
        column for column in feature_columns if any(token in column.lower() for token in FUTURE_LIKE_TOKENS)
    ]
    return {
        "past_equivalent_features": equivalent_presence,
        "future_like_columns": future_like_columns,
    }


def save_pickle(payload: dict, path: Path) -> None:
    with path.open("wb") as file:
        pickle.dump(payload, file)


def load_pickle(path: Path) -> dict:
    with path.open("rb") as file:
        return pickle.load(file)


def train_mlp_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    feature_columns: list[str],
    target_column: str,
) -> dict[str, object]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    model = MLPRegressor(
        hidden_layer_sizes=(args.hidden_dim_1, args.hidden_dim_2),
        activation="relu",
        solver="adam",
        alpha=args.weight_decay,
        batch_size=min(args.batch_size, len(x_train_scaled)),
        learning_rate_init=args.learning_rate,
        max_iter=1,
        warm_start=True,
        shuffle=False,
        random_state=args.random_state,
    )

    history_records = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    model_path = output_dir / "best_mlp_model.pkl"

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for epoch in range(1, args.epochs + 1):
        model.fit(x_train_scaled, y_train)

        train_predictions = model.predict(x_train_scaled)
        val_predictions = model.predict(x_val_scaled)
        train_loss = mean_squared_error(y_train, train_predictions)
        val_loss = mean_squared_error(y_val, val_predictions)
        train_mae = mean_absolute_error(y_train, train_predictions)
        val_mae = mean_absolute_error(y_val, val_predictions)

        history_records.append(
            {
                "model_name": "mlp",
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_mae": float(train_mae),
                "val_mae": float(val_mae),
            }
        )
        print(
            f"[mlp] epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_mae={train_mae:.6f} val_mae={val_mae:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            epochs_without_improvement = 0
            save_pickle(
                {
                    "model_name": "mlp",
                    "model": model,
                    "scaler": scaler,
                    "feature_columns": feature_columns,
                    "target_column": target_column,
                    "best_epoch": best_epoch,
                    "best_val_metric": float(val_mae),
                    "best_val_loss": best_val_loss,
                },
                model_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"[mlp] early_stopping epoch={epoch:03d} "
                    f"best_epoch={best_epoch:03d} best_val_loss={best_val_loss:.6f}"
                )
                break

    best_bundle = load_pickle(model_path)
    best_model = best_bundle["model"]
    best_scaler = best_bundle["scaler"]
    val_prediction = best_model.predict(best_scaler.transform(x_val))
    val_metrics = compute_regression_metrics(y_val, val_prediction)
    return {
        "model_name": "mlp",
        "artifact_path": model_path,
        "history_records": history_records,
        "val_metrics": val_metrics,
        "best_val_metric": float(val_metrics["mae"]),
        "best_val_loss": float(best_bundle["best_val_loss"]),
        "bundle": best_bundle,
    }


def train_hgbr_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
    feature_columns: list[str],
    target_column: str,
) -> dict[str, object]:
    candidate_configs = [
        {"learning_rate": 0.03, "max_depth": 3, "max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_depth": 3, "max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 0.1},
        {"learning_rate": 0.05, "max_depth": 4, "max_leaf_nodes": 31, "min_samples_leaf": 8, "l2_regularization": 0.0},
        {"learning_rate": 0.08, "max_depth": 3, "max_leaf_nodes": 31, "min_samples_leaf": 5, "l2_regularization": 0.1},
        {"learning_rate": 0.1, "max_depth": None, "max_leaf_nodes": 15, "min_samples_leaf": 5, "l2_regularization": 0.0},
    ]

    trial_records = []
    best_result = None
    model_path = output_dir / "best_hgbr_model.pkl"

    for trial_index, config in enumerate(candidate_configs, start=1):
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            max_leaf_nodes=config["max_leaf_nodes"],
            min_samples_leaf=config["min_samples_leaf"],
            l2_regularization=config["l2_regularization"],
            max_iter=500,
            early_stopping=True,
            validation_fraction=None,
            n_iter_no_change=25,
            random_state=args.random_state,
        )
        model.fit(x_train, y_train, X_val=x_val, y_val=y_val)
        val_prediction = model.predict(x_val)
        val_metrics = compute_regression_metrics(y_val, val_prediction)
        record = {
            "model_name": "hgbr",
            "trial": trial_index,
            "learning_rate": config["learning_rate"],
            "max_depth": config["max_depth"],
            "max_leaf_nodes": config["max_leaf_nodes"],
            "min_samples_leaf": config["min_samples_leaf"],
            "l2_regularization": config["l2_regularization"],
            "iterations": int(model.n_iter_),
            "val_mae": float(val_metrics["mae"]),
            "val_rmse": float(val_metrics["rmse"]),
        }
        trial_records.append(record)
        print(
            f"[hgbr] trial={trial_index:02d} "
            f"val_mae={val_metrics['mae']:.6f} val_rmse={val_metrics['rmse']:.6f} "
            f"iters={model.n_iter_} config={config}"
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
                "model": model,
                "config": config,
                "val_metrics": val_metrics,
                "iterations": int(model.n_iter_),
            }
            save_pickle(
                {
                    "model_name": "hgbr",
                    "model": model,
                    "scaler": None,
                    "feature_columns": feature_columns,
                    "target_column": target_column,
                    "best_val_metric": float(val_metrics["mae"]),
                    "best_val_rmse": float(val_metrics["rmse"]),
                    "best_config": config,
                    "iterations": int(model.n_iter_),
                },
                model_path,
            )

    best_bundle = load_pickle(model_path)
    return {
        "model_name": "hgbr",
        "artifact_path": model_path,
        "history_records": trial_records,
        "val_metrics": best_result["val_metrics"],
        "best_val_metric": float(best_result["val_metrics"]["mae"]),
        "best_val_loss": None,
        "bundle": best_bundle,
    }


def predict_with_bundle(bundle: dict, x_matrix: np.ndarray) -> np.ndarray:
    scaler = bundle.get("scaler")
    transformed = scaler.transform(x_matrix) if scaler is not None else x_matrix
    return bundle["model"].predict(transformed)


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


def build_diagnostics_markdown(
    target_stats: dict[str, float],
    metric_rows: list[dict[str, float | str]],
    feature_diagnostics: dict[str, object],
    target_column: str,
    feature_columns: list[str],
) -> str:
    lines = [
        "# Diagnostics",
        "",
        f"- target_column: `{target_column}`",
        f"- feature_count: {len(feature_columns)}",
        "",
        "## Test Target Distribution",
        "",
        f"- mean: {target_stats['mean']:.6f}",
        f"- median: {target_stats['median']:.6f}",
        f"- p25: {target_stats['p25']:.6f}",
        f"- p75: {target_stats['p75']:.6f}",
        f"- p90: {target_stats['p90']:.6f}",
        f"- max: {target_stats['max']:.6f}",
        "",
        "## Baseline Vs Model Metrics",
        "",
        "```text",
        format_metric_table(metric_rows),
        "```",
        "",
        "## Feature Checks",
        "",
    ]
    for label, status in feature_diagnostics["past_equivalent_features"].items():
        lines.append(
            f"- {label}: present={status['present']} matched_columns={status['matched_columns']}"
        )
    lines.append(
        f"- future-like columns in features: {feature_diagnostics['future_like_columns']}"
    )
    return "\n".join(lines) + "\n"


def build_training_report(metrics: dict, output_paths: dict[str, Path], beat_message: str) -> str:
    baseline_naive = metrics["baseline_metrics"]["naive_7d_sum"]
    baseline_moving_avg = metrics["baseline_metrics"]["moving_avg_30d_x7"]
    lines = [
        "Training Run Summary",
        f"model_name: {metrics['model_name']}",
        f"target_column: {metrics['target_column']}",
        f"best_val_metric: {metrics['best_val_metric']:.6f}",
        f"test_mae: {metrics['test_mae']:.6f}",
        f"test_rmse: {metrics['test_rmse']:.6f}",
        f"test_per_user_nmae: {metrics['test_per_user_nmae']:.4f} %",
        "",
        "Baselines",
        f"naive_7d_sum mae: {baseline_naive['mae']:.6f}",
        f"naive_7d_sum rmse: {baseline_naive['rmse']:.6f}",
        f"moving_avg_30d_x7 mae: {baseline_moving_avg['mae']:.6f}",
        f"moving_avg_30d_x7 rmse: {baseline_moving_avg['rmse']:.6f}",
        "",
        "Dataset Sizes",
        f"train_rows: {metrics['train_rows']}",
        f"val_rows: {metrics['val_rows']}",
        f"test_rows: {metrics['test_rows']}",
        f"feature_count: {metrics['feature_count']}",
        "",
        f"comparison: {beat_message}",
        "",
        "Artifacts",
        f"best_model: {output_paths['best_model']}",
        f"history_csv: {output_paths['history_csv']}",
        f"predictions_csv: {output_paths['predictions_csv']}",
        f"metrics_json: {output_paths['metrics_json']}",
        f"diagnostics_md: {output_paths['diagnostics_md']}",
    ]
    return "\n".join(lines) + "\n"


def summarize_failure_reasons(feature_columns: list[str], target_stats: dict[str, float]) -> list[str]:
    reasons = []
    if len(feature_columns) <= 25:
        reasons.append("Feature count is low relative to the problem, so the model may add little beyond rolling-expense baselines.")
    if target_stats["p90"] > 2 * max(target_stats["median"], 1.0):
        reasons.append("Target distribution is right-skewed, so a squared-error objective may be dominated by a few large-expense windows.")
    reasons.append("The strongest features may be near-equivalent to the baselines, so tree/MLP models have limited extra signal to exploit.")
    reasons.append("Train/validation sample sizes are small for a flexible regressor, which raises variance and weakens generalization.")
    return reasons


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(args.train_path)
    val_df = pd.read_parquet(args.val_path)
    test_df = pd.read_parquet(args.test_path)

    target_column = resolve_target_column(
        dataframes=[train_df, val_df, test_df],
        requested_target=args.target_column,
    )
    feature_columns = build_feature_columns(train_df=train_df, target_column=target_column)

    x_train, y_train = prepare_xy(train_df, feature_columns, target_column)
    x_val, y_val = prepare_xy(val_df, feature_columns, target_column)
    x_test, y_test = prepare_xy(test_df, feature_columns, target_column)

    if NAIVE_BASELINE_FEATURE not in test_df.columns:
        raise ValueError(f"Missing baseline feature column: {NAIVE_BASELINE_FEATURE}")
    if MOVING_AVG_BASELINE_FEATURE not in test_df.columns:
        raise ValueError(f"Missing baseline feature column: {MOVING_AVG_BASELINE_FEATURE}")

    print("== Diagnostics ==")
    target_stats = compute_target_distribution(y_test)
    for key in ["mean", "median", "p25", "p75", "p90", "max"]:
        print(f"target_{key}={target_stats[key]:.6f}")

    feature_diagnostics = detect_feature_patterns(feature_columns)
    for label, status in feature_diagnostics["past_equivalent_features"].items():
        print(
            f"feature_check {label} present={status['present']} matched_columns={status['matched_columns']}"
        )
    print(f"feature_check future_like_columns={feature_diagnostics['future_like_columns']}")

    mlp_result = train_mlp_candidate(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        args=args,
        output_dir=args.output_dir,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    hgbr_result = train_hgbr_candidate(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        args=args,
        output_dir=args.output_dir,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    candidate_results = [mlp_result, hgbr_result]
    best_candidate = min(candidate_results, key=lambda item: (item["best_val_metric"], item["val_metrics"]["rmse"]))
    best_bundle = best_candidate["bundle"]
    best_model_name = str(best_candidate["model_name"])
    best_model_path = Path(best_candidate["artifact_path"])

    mlp_test_predictions = predict_with_bundle(mlp_result["bundle"], x_test)
    hgbr_test_predictions = predict_with_bundle(hgbr_result["bundle"], x_test)
    candidate_test_metrics = {
        "mlp": compute_regression_metrics(y_test, mlp_test_predictions),
        "hgbr": compute_regression_metrics(y_test, hgbr_test_predictions),
    }
    candidate_test_predictions = {
        "mlp": mlp_test_predictions,
        "hgbr": hgbr_test_predictions,
    }
    model_predictions = candidate_test_predictions[best_model_name]
    model_metrics = candidate_test_metrics[best_model_name]
    naive_predictions = test_df[NAIVE_BASELINE_FEATURE].astype(np.float32).to_numpy()
    moving_avg_predictions = test_df[MOVING_AVG_BASELINE_FEATURE].astype(np.float32).to_numpy() * 7.0
    naive_metrics = compute_regression_metrics(y_test, naive_predictions)
    moving_avg_metrics = compute_regression_metrics(y_test, moving_avg_predictions)

    test_user_ids = test_df["user_id"].to_numpy()
    test_nmae = per_user_nmae(y_test, model_predictions, test_user_ids)

    metric_rows = [
        {"model_name": "mlp", "mae": candidate_test_metrics["mlp"]["mae"], "rmse": candidate_test_metrics["mlp"]["rmse"]},
        {"model_name": "hgbr", "mae": candidate_test_metrics["hgbr"]["mae"], "rmse": candidate_test_metrics["hgbr"]["rmse"]},
        {"model_name": "naive_7d_sum", "mae": naive_metrics["mae"], "rmse": naive_metrics["rmse"]},
        {"model_name": "moving_avg_30d_x7", "mae": moving_avg_metrics["mae"], "rmse": moving_avg_metrics["rmse"]},
    ]

    print("== Test Metric Comparison ==")
    print(format_metric_table(metric_rows))

    diagnostics_text = build_diagnostics_markdown(
        target_stats=target_stats,
        metric_rows=metric_rows,
        feature_diagnostics=feature_diagnostics,
        target_column=target_column,
        feature_columns=feature_columns,
    )
    diagnostics_path = args.output_dir / "diagnostics.md"
    diagnostics_path.write_text(diagnostics_text, encoding="utf-8")

    moving_avg_beaten = model_metrics["mae"] < moving_avg_metrics["mae"] and model_metrics["rmse"] < moving_avg_metrics["rmse"]
    if moving_avg_beaten:
        beat_message = (
            f"{best_model_name} beat moving_avg_30d_x7 on both MAE and RMSE "
            f"({model_metrics['mae']:.6f}/{model_metrics['rmse']:.6f} vs "
            f"{moving_avg_metrics['mae']:.6f}/{moving_avg_metrics['rmse']:.6f})."
        )
        print(f"beat_baseline=YES {beat_message}")
    else:
        beat_message = (
            f"{best_model_name} did not beat moving_avg_30d_x7 on both MAE and RMSE "
            f"({model_metrics['mae']:.6f}/{model_metrics['rmse']:.6f} vs "
            f"{moving_avg_metrics['mae']:.6f}/{moving_avg_metrics['rmse']:.6f})."
        )
        print(f"beat_baseline=NO {beat_message}")
        for reason in summarize_failure_reasons(feature_columns, target_stats):
            print(f"possible_reason: {reason}")
        print("next_steps: consider a log-target transform, a classification framing, richer calendar features, or additional user-history features.")

    predictions_df = build_predictions_long_df(
        test_df=test_df,
        target_column=target_column,
        prediction_map={
            best_model_name: model_predictions,
            "naive_7d_sum": naive_predictions,
            "moving_avg_30d_x7": moving_avg_predictions,
        },
    )
    predictions_path = args.output_dir / "predictions_test.csv"
    predictions_df.to_csv(predictions_path, index=False)

    history_path = args.output_dir / "training_history.csv"
    history_df = pd.concat(
        [
            pd.DataFrame(mlp_result["history_records"]),
            pd.DataFrame(hgbr_result["history_records"]),
        ],
        ignore_index=True,
        sort=False,
    )
    history_df.to_csv(history_path, index=False)

    metrics = {
        "model_name": best_model_name,
        "target_column": target_column,
        "best_val_metric": float(best_candidate["best_val_metric"]),
        "test_mae": float(model_metrics["mae"]),
        "test_rmse": float(model_metrics["rmse"]),
        "test_per_user_nmae": round(test_nmae, 4),
        "baseline_mae": {
            "naive_7d_sum": float(naive_metrics["mae"]),
            "moving_avg_30d_x7": float(moving_avg_metrics["mae"]),
        },
        "baseline_rmse": {
            "naive_7d_sum": float(naive_metrics["rmse"]),
            "moving_avg_30d_x7": float(moving_avg_metrics["rmse"]),
        },
        "baseline_metrics": {
            "naive_7d_sum": naive_metrics,
            "moving_avg_30d_x7": moving_avg_metrics,
        },
        "beat_moving_avg_30d_x7": bool(moving_avg_beaten),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
        "candidate_validation_metrics": {
            "mlp": mlp_result["val_metrics"],
            "hgbr": hgbr_result["val_metrics"],
        },
        "note": "Model selection uses validation only; test is evaluated once at the end on the selected model.",
    }
    metrics_path = args.output_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    report_path = args.output_dir / "training_report_v1.txt"
    report_text = build_training_report(
        metrics=metrics,
        output_paths={
            "best_model": best_model_path,
            "history_csv": history_path,
            "predictions_csv": predictions_path,
            "metrics_json": metrics_path,
            "diagnostics_md": diagnostics_path,
        },
        beat_message=beat_message,
    )
    report_path.write_text(report_text, encoding="utf-8")

    print(
        f"selected_model={best_model_name} best_val_metric={metrics['best_val_metric']:.6f} "
        f"test_mae={metrics['test_mae']:.6f} test_rmse={metrics['test_rmse']:.6f}"
    )
    print(f"saved diagnostics to {diagnostics_path}")
    print(f"saved best model to {best_model_path}")
    print(f"saved training history to {history_path}")
    print(f"saved test predictions to {predictions_path}")
    print(f"saved test metrics to {metrics_path}")
    print(f"saved training report to {report_path}")


if __name__ == "__main__":
    main()
