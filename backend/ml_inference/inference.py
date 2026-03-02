import pickle
from pathlib import Path
from typing import Any

from backend.ml_inference.feature_schema import build_feature_vector, load_feature_config


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "risk_model.pkl"


def load_model(model_path: Path | None = None) -> Any:
    path = model_path or MODEL_PATH
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Model artifact not found or empty: {path}. Run ml/training/train.py first."
        )

    with path.open("rb") as file:
        return pickle.load(file)


def predict_risk(payload: dict[str, Any]) -> dict[str, Any]:
    config = load_feature_config()
    features = build_feature_vector(payload, config)
    model = load_model()
    ordered_values = [features[name] for name in config["feature_order"]]

    score = None
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba([ordered_values])[0][1])
    elif hasattr(model, "predict"):
        score = float(model.predict([ordered_values])[0])
    else:
        raise TypeError("Loaded model does not implement predict or predict_proba")

    threshold = float(config.get("threshold", 0.5))
    return {
        "score": score,
        "label": "high_risk" if score >= threshold else "low_risk",
        "threshold": threshold,
        "feature_vector": features,
        "model_version": config.get("model_version", "unknown"),
    }

