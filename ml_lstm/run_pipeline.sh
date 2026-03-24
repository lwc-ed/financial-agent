#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/liweichen/financial-agent/ml_gru/venv_ml_gru/bin/python}"

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

echo "使用 Python: $PYTHON_BIN"
echo "工作目錄: $SCRIPT_DIR"

cd "$SCRIPT_DIR"

run_step "1/5" "preprocess_wallmart.py"
run_step "2/5" "pretrain_lstm.py"
run_step "3/5" "preprocess_personal.py"
run_step "4/5" "finetune_lstm.py"
run_step "5/5" "predict.py"

echo
echo "Pipeline 完成。結果請查看:"
echo "  $SCRIPT_DIR/artificats/metrics.json"
echo "  $SCRIPT_DIR/artificats/training_summary.txt"
