#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/venv/bin/python}"
RESULT_LOG="$SCRIPT_DIR/rf_output/pipeline_log.txt"

run_step() {
  local label="$1"
  local script_name="$2"

  echo
  echo "============================================================"
  echo "[$label] ${script_name}"
  echo "============================================================"
  "$PYTHON_BIN" "$SCRIPT_DIR/$script_name"
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "找不到可執行的 Python: $PYTHON_BIN" >&2
  echo "你可以改用自訂路徑執行，例如：" >&2
  echo "PYTHON_BIN=/path/to/python $0" >&2
  exit 1
fi

mkdir -p "$SCRIPT_DIR/rf_output"

echo "使用 Python: $PYTHON_BIN"
echo "工作目錄: $SCRIPT_DIR"

cd "$SCRIPT_DIR"

{
  run_step "1/2" "my_first_rf.py"
  run_step "2/2" "analyze_rf_features.py"
} | tee "$RESULT_LOG"

echo
echo "Pipeline 完成。結果請查看:"
echo "  $SCRIPT_DIR/rf_output/training_report_rf.txt"
echo "  $SCRIPT_DIR/rf_output/feature_importance_report.txt"
echo "  $RESULT_LOG"
