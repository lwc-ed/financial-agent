import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def evaluate_model(model_path: Path, dataset_path: Path, target_column: str) -> None:
    df = pd.read_csv(dataset_path)
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    x = df.drop(columns=[target_column])
    y = df[target_column]

    with model_path.open("rb") as file:
        model = pickle.load(file)

    predictions = model.predict(x)
    print(f"accuracy={accuracy_score(y, predictions):.4f}")
    print(classification_report(y, predictions))

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)[:, 1]
        print(f"roc_auc={roc_auc_score(y, probabilities):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained risk model.")
    parser.add_argument("--model", required=True, type=Path, help="Path to the pickled model")
    parser.add_argument("--dataset", required=True, type=Path, help="Evaluation dataset CSV")
    parser.add_argument(
        "--target-column",
        default="label",
        help="Column name used as the evaluation target",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args.model, args.dataset, args.target_column)

