"""
ml vs ml_ibm 同名模型的統計檢定（Paired t-test，雙尾）
輸出：ml_ibm/statistical_test/output/
"""

import json
import unicodedata
from pathlib import Path
from scipy import stats
import numpy as np


def _visual_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)


def _rjust(s: str, width: int) -> str:
    padding = width - _visual_width(s)
    return ' ' * max(0, padding) + s

ROOT = Path(__file__).resolve().parents[2]
ML_OUTPUT = ROOT / "ml" / "model_outputs"
ML_IBM_OUTPUT = ROOT / "ml_ibm" / "model_outputs"
OUT_DIR = Path(__file__).resolve().parent / "output"

METRICS = ["MAE", "RMSE", "Binary_F1", "Weighted_F1"]

# MAE/RMSE: ml_ibm 較大 = 變差；F1: ml_ibm 較大 = 提升
DIRECTION = {
    "MAE":         "larger_is_worse",
    "RMSE":        "larger_is_worse",
    "Binary_F1":   "larger_is_better",
    "Weighted_F1": "larger_is_better",
}


def get_latest_per_seed_csv(model_dir: Path):
    files = sorted(model_dir.glob("per_seed_metrics_*.csv"))
    return files[-1] if files else None


def load_per_seed(csv_path: Path) -> dict[str, np.ndarray]:
    import csv
    data = {m: [] for m in METRICS}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for m in METRICS:
                if m in row:
                    data[m].append(float(row[m]))
    return {m: np.array(v) for m, v in data.items() if v}


def conclude(p_value: float, mean_ml: float, mean_ibm: float, direction: str, alpha: float) -> str:
    if p_value >= alpha:
        return "無顯著差異"
    ibm_larger = mean_ibm > mean_ml
    if direction == "larger_is_worse":
        return "顯著變差" if ibm_larger else "顯著提升"
    else:
        return "顯著提升" if ibm_larger else "顯著變差"


def run_test(model_name: str) -> dict | None:
    ml_csv = get_latest_per_seed_csv(ML_OUTPUT / model_name)
    ibm_csv = get_latest_per_seed_csv(ML_IBM_OUTPUT / model_name)

    if ml_csv is None or ibm_csv is None:
        return None

    ml_data = load_per_seed(ml_csv)
    ibm_data = load_per_seed(ibm_csv)

    results = {}
    for m in METRICS:
        if m not in ml_data or m not in ibm_data:
            continue
        a, b = ml_data[m], ibm_data[m]
        if len(a) != len(b):
            print(f"  [警告] {model_name} {m}: seed 數不一致 ({len(a)} vs {len(b)})，跳過")
            continue
        t_stat, p_value = stats.ttest_rel(a, b)
        results[m] = {
            "mean_ml":    float(np.mean(a)),
            "mean_ibm":   float(np.mean(b)),
            "t_stat":     float(t_stat),
            "p_value":    float(p_value),
            "conclude_05": conclude(p_value, np.mean(a), np.mean(b), DIRECTION[m], 0.05),
            "conclude_01": conclude(p_value, np.mean(a), np.mean(b), DIRECTION[m], 0.01),
        }
    return results if results else None


def format_table(model_name: str, results: dict) -> str:
    col_w = 16
    metrics_present = [m for m in METRICS if m in results]

    header = f"{'pair T-test':<20}" + "".join(f"{m:>{col_w}}" for m in metrics_present)
    rows = {
        "mean (ml)":       lambda m: f"{results[m]['mean_ml']:>{col_w}.4f}",
        "mean (ml_ibm)":   lambda m: f"{results[m]['mean_ibm']:>{col_w}.4f}",
        "p-value":         lambda m: f"{results[m]['p_value']:>{col_w}.4E}",
        "結論 (α=0.05)":   lambda m: _rjust(results[m]['conclude_05'], col_w),
        "結論 (α=0.01)":   lambda m: _rjust(results[m]['conclude_01'], col_w),
    }

    lines = [f"Model: {model_name}", "-" * (20 + col_w * len(metrics_present)), header]
    for label, fn in rows.items():
        lines.append(f"{label:<20}" + "".join(fn(m) for m in metrics_present))

    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ml/ 和 ml_ibm/ 共有的模型名稱
    ml_models = {p.name for p in ML_OUTPUT.iterdir() if p.is_dir()}
    ibm_models = {p.name for p in ML_IBM_OUTPUT.iterdir() if p.is_dir()}
    common = sorted(ml_models & ibm_models)

    if not common:
        print("找不到兩邊共同的模型資料夾。")
        return

    all_tables = []
    all_json = {}

    for model_name in common:
        print(f"處理：{model_name}")
        results = run_test(model_name)
        if results is None:
            print(f"  [跳過] 缺少 per_seed_metrics_*.csv")
            continue
        table = format_table(model_name, results)
        all_tables.append(table)
        all_json[model_name] = results

    if not all_tables:
        print("沒有可輸出的結果。")
        return

    # 輸出 txt
    txt_path = OUT_DIR / "pair_t_test_results.txt"
    with open(txt_path, "w") as f:
        f.write("\n\n".join(all_tables) + "\n")

    # 輸出 json
    json_path = OUT_DIR / "pair_t_test_results.json"
    with open(json_path, "w") as f:
        json.dump(all_json, f, ensure_ascii=False, indent=2)

    # 輸出 markdown
    md_path = OUT_DIR / "pair_t_test_results.md"
    with open(md_path, "w") as f:
        f.write("# Paired T-Test Results (ml vs ml_ibm)\n\n")
        for model_name, results in all_json.items():
            metrics_present = [m for m in METRICS if m in results]
            f.write(f"## {model_name}\n\n")
            header = "| " + " | ".join([""] + metrics_present) + " |"
            sep    = "| " + " | ".join(["---"] * (len(metrics_present) + 1)) + " |"
            f.write(header + "\n" + sep + "\n")
            rows_md = [
                ("mean (ml)",     lambda m: f"{results[m]['mean_ml']:.4f}"),
                ("mean (ml_ibm)", lambda m: f"{results[m]['mean_ibm']:.4f}"),
                ("p-value",       lambda m: f"{results[m]['p_value']:.4E}"),
                ("結論 (α=0.05)", lambda m: results[m]['conclude_05']),
                ("結論 (α=0.01)", lambda m: results[m]['conclude_01']),
            ]
            for label, fn in rows_md:
                f.write("| " + " | ".join([label] + [fn(m) for m in metrics_present]) + " |\n")
            f.write("\n")

    print(f"\n輸出完成：\n  {txt_path}\n  {json_path}\n  {md_path}")

    # 印出結果
    print("\n" + "=" * 70)
    print("\n\n".join(all_tables))


if __name__ == "__main__":
    main()
