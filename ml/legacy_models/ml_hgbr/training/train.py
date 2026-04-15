import argparse
import json
import pickle
import shutil
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BACKEND_ARTIFACT_DIR = ROOT_DIR / "backend" / "ml_inference" / "artifacts"
DEFAULT_ML_ARTIFACT_DIR = ROOT_DIR / "ml" / "artifacts"


def train_model(
    dataset_path: Path,
    output_dir: Path,
    backend_artifact_dir: Path,
    target_column: str,
) -> None:
    df = pd.read_csv(dataset_path)
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    x = df.drop(columns=[target_column])
    y = df[target_column]

    model = LogisticRegression(max_iter=1000)
    model.fit(x, y)

    output_dir.mkdir(parents=True, exist_ok=True)
    backend_artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "risk_model.pkl"
    with model_path.open("wb") as file:
        pickle.dump(model, file)

    feature_config = {
        "model_version": "logistic-regression-v1",
        "threshold": 0.5,
        "feature_order": list(x.columns),
        "numeric_features": list(x.columns),
        "defaults": {column: 0.0 for column in x.columns},
    }
    config_path = output_dir / "feature_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(feature_config, file, ensure_ascii=False, indent=2)

    shutil.copy2(model_path, backend_artifact_dir / "risk_model.pkl")
    shutil.copy2(config_path, backend_artifact_dir / "feature_config.json")
    print(f"Saved model to {model_path}")
    print(f"Synced artifacts to {backend_artifact_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline risk model.")
    parser.add_argument("--dataset", required=True, type=Path, help="Training dataset CSV")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_ML_ARTIFACT_DIR,
        type=Path,
        help="Directory for training outputs",
    )
    parser.add_argument(
        "--backend-artifact-dir",
        default=DEFAULT_BACKEND_ARTIFACT_DIR,
        type=Path,
        help="Directory synced to backend inference artifacts",
    )
    parser.add_argument(
        "--target-column",
        default="label",
        help="Column name used as the training target",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        backend_artifact_dir=args.backend_artifact_dir,
        target_column=args.target_column,
    )

