from pathlib import Path
import runpy


def main() -> None:
    model_dir = Path(__file__).resolve().parents[1]
    ml_root = model_dir.parent.parent if model_dir.parent.name == "legacy_models" else model_dir.parent
    target = ml_root / "processed_data" / "build_features.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
