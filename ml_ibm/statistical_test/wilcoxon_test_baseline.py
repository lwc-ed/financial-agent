"""
統計檢定（Wilcoxon Signed-Rank Test，雙尾）
比較：ml/ no-TL baseline vs ml_ibm/ TL 版本
  ml/bigru              vs ml_ibm/bigru_TL_alignment
  ml/bilstm             vs ml_ibm/bilstm_TL_alignment
  ml/gru                vs ml_ibm/gru_TL_alignment
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


ROOT          = Path(__file__).resolve().parents[2]
ML_OUTPUT     = ROOT / "ml" / "model_outputs"
ML_IBM_OUTPUT = ROOT / "ml_ibm" / "model_outputs"
OUT_DIR       = Path(__file__).resolve().parent / "output"

METRICS = ["MAE", "RMSE", "Binary_F1", "Weighted_F1"]

DIRECTION = {
    "MAE":         "larger_is_worse",
    "RMSE":        "larger_is_worse",
    "Binary_F1":   "larger_is_better",
    "Weighted_F1": "larger_is_better",
}

PAIRS = [
    ("bigru",  "bigru_TL_alignment"),
    ("bilstm", "bilstm_TL_alignment"),
    ("gru",    "gru_TL_alignment"),
]


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


def conclude(p_value: float, mean_baseline: float, mean_tl: float, direction: str, alpha: float) -> str:
    if p_value >= alpha:
        return "無顯著差異"
    tl_larger = mean_tl > mean_baseline
    if direction == "larger_is_worse":
        return "顯著變差" if tl_larger else "顯著提升"
    else:
        return "顯著提升" if tl_larger else "顯著變差"


def run_test(baseline_name: str, tl_name: str) -> dict | None:
    baseline_csv = get_latest_per_seed_csv(ML_OUTPUT / baseline_name)
    tl_csv       = get_latest_per_seed_csv(ML_IBM_OUTPUT / tl_name)

    if baseline_csv is None or tl_csv is None:
        return None

    baseline_data = load_per_seed(baseline_csv)
    tl_data       = load_per_seed(tl_csv)

    results = {}
    for m in METRICS:
        if m not in baseline_data or m not in tl_data:
            continue
        a, b = baseline_data[m], tl_data[m]
        if len(a) != len(b):
            print(f"  [警告] {baseline_name} vs {tl_name} {m}: seed 數不一致 ({len(a)} vs {len(b)})，跳過")
            continue
        stat, p_value = stats.wilcoxon(a, b, alternative="two-sided")
        results[m] = {
            "mean_baseline": float(np.mean(a)),
            "mean_tl":       float(np.mean(b)),
            "statistic":     float(stat),
            "p_value":       float(p_value),
            "conclude_05":   conclude(p_value, np.mean(a), np.mean(b), DIRECTION[m], 0.05),
            "conclude_01":   conclude(p_value, np.mean(a), np.mean(b), DIRECTION[m], 0.01),
        }
    return results if results else None


def format_table(baseline_name: str, tl_name: str, results: dict) -> str:
    col_w = 16
    metrics_present = [m for m in METRICS if m in results]

    header = f"{'Wilcoxon':<20}" + "".join(f"{m:>{col_w}}" for m in metrics_present)
    rows = {
        f"mean ({baseline_name})": lambda m: f"{results[m]['mean_baseline']:>{col_w}.4f}",
        f"mean ({tl_name})":       lambda m: f"{results[m]['mean_tl']:>{col_w}.4f}",
        "p-value":                 lambda m: f"{results[m]['p_value']:>{col_w}.4E}",
        "結論 (α=0.05)":           lambda m: _rjust(results[m]['conclude_05'], col_w),
        "結論 (α=0.01)":           lambda m: _rjust(results[m]['conclude_01'], col_w),
    }

    sep = "-" * (20 + col_w * len(metrics_present))
    lines = [f"ml/{baseline_name}  vs  ml_ibm/{tl_name}", sep, header]
    for label, fn in rows.items():
        lines.append(f"{label:<20}" + "".join(fn(m) for m in metrics_present))

    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tables = []
    all_json   = {}

    for baseline_name, tl_name in PAIRS:
        print(f"處理：ml/{baseline_name} vs ml_ibm/{tl_name}")
        results = run_test(baseline_name, tl_name)
        if results is None:
            print(f"  [跳過] 缺少 per_seed_metrics_*.csv")
            continue
        table = format_table(baseline_name, tl_name, results)
        all_tables.append(table)
        all_json[f"{baseline_name}_vs_{tl_name}"] = results

    if not all_tables:
        print("沒有可輸出的結果。")
        return

    txt_path = OUT_DIR / "wilcoxon_baseline_results.txt"
    with open(txt_path, "w") as f:
        f.write("\n\n".join(all_tables) + "\n")

    json_path = OUT_DIR / "wilcoxon_baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(all_json, f, ensure_ascii=False, indent=2)

    md_path = OUT_DIR / "wilcoxon_baseline_results.md"
    with open(md_path, "w") as f:
        f.write("# Wilcoxon Signed-Rank Test Results (no-TL baseline vs ml_ibm TL)\n\n")
        for key, results in all_json.items():
            baseline_name, tl_name = key.split("_vs_")
            metrics_present = [m for m in METRICS if m in results]
            f.write(f"## ml/{baseline_name} vs ml_ibm/{tl_name}\n\n")
            header = "| " + " | ".join([""] + metrics_present) + " |"
            sep    = "| " + " | ".join(["---"] * (len(metrics_present) + 1)) + " |"
            f.write(header + "\n" + sep + "\n")
            rows_md = [
                (f"mean ({baseline_name})", lambda m: f"{results[m]['mean_baseline']:.4f}"),
                (f"mean ({tl_name})",       lambda m: f"{results[m]['mean_tl']:.4f}"),
                ("p-value",                 lambda m: f"{results[m]['p_value']:.4E}"),
                ("結論 (α=0.05)",           lambda m: results[m]['conclude_05']),
                ("結論 (α=0.01)",           lambda m: results[m]['conclude_01']),
            ]
            for label, fn in rows_md:
                f.write("| " + " | ".join([label] + [fn(m) for m in metrics_present]) + " |\n")
            f.write("\n")

    print(f"\n輸出完成：\n  {txt_path}\n  {json_path}\n  {md_path}")
    print("\n" + "=" * 70)
    print("\n\n".join(all_tables))


if __name__ == "__main__":
    main()
