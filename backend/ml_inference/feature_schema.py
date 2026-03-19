import json
from pathlib import Path
from typing import Any


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
FEATURE_CONFIG_PATH = ARTIFACT_DIR / "feature_config.json"


def load_feature_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or FEATURE_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_feature_vector(
    payload: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    cfg = config or load_feature_config()
    ordered_features = cfg.get("feature_order", [])
    defaults = cfg.get("defaults", {})
    numeric_only = set(cfg.get("numeric_features", ordered_features))

    features: dict[str, float] = {}
    for feature_name in ordered_features:
        raw_value = payload.get(feature_name, defaults.get(feature_name, 0.0))
        if feature_name in numeric_only:
            features[feature_name] = float(raw_value)
        else:
            raise ValueError(f"Unsupported feature type for {feature_name}")

    return features

