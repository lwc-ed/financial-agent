from pathlib import Path
import shutil
import runpy


SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
CANONICAL_OUTPUT = ML_ROOT / "processed_data" / "artifacts" / "kaggle_processed_common.csv"
LEGACY_OUTPUT = SCRIPT_DIR / "data" / "processed" / "kaggle_processed_common.csv"


def main():
    build_script = ML_ROOT / "processed_data" / "build_kaggle_common.py"
    runpy.run_path(str(build_script), run_name="__main__")

    LEGACY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CANONICAL_OUTPUT, LEGACY_OUTPUT)

    print("同步完成")
    print("canonical 輸出:", CANONICAL_OUTPUT)
    print("legacy 輸出   :", LEGACY_OUTPUT)

if __name__ == "__main__":
    main()
