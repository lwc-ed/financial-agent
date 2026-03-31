#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/venv/bin/python}"
RESULT_TXT="$SCRIPT_DIR/results/transfer_results.txt"

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

mkdir -p "$SCRIPT_DIR/results"

echo "使用 Python: $PYTHON_BIN"
echo "工作目錄: $ML_DIR"

# XGBoost 腳本內部使用 "ml_XGboost/..." 相對路徑，須從 ml/ 執行
cd "$ML_DIR"

{
  run_step "1/6" "preprocess_kaggle.py"
  run_step "2/6" "preprocess_kaggle_common.py"
  run_step "3/6" "preprocess_own.py"
  #run_step "4/7" "train_kaggle_base.py"
  run_step "4/6" "train_own_baseline.py"
  run_step "5/6" "test_transfer.py"
  run_step "6/6" "finetune_own.py"
} | tee "$RESULT_TXT"

echo
echo "Pipeline 完成。結果請查看:"
echo "  $RESULT_TXT"
echo "  $SCRIPT_DIR/models/xgb_kaggle_base.json"
echo "  $SCRIPT_DIR/models/xgb_kaggle_common.json"
echo "  $SCRIPT_DIR/models/xgb_finetuned_own.json"
