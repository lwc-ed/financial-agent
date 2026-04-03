#!/bin/bash
# GRU 無 Pretrain Pipeline
# 直接用個人資料從頭訓練，對比是否需要 Walmart pretrain

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${SCRIPT_DIR}/venv_ml_gru/bin/python" ]]; then
  PYTHON_BIN="${SCRIPT_DIR}/venv_ml_gru/bin/python"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

cd "${SCRIPT_DIR}"
echo "Using Python: ${PYTHON_BIN}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Step 1：個人資料前處理（若已有 .npy 可跳過）        ║"
echo "╚══════════════════════════════════════════════════════╝"
"${PYTHON_BIN}" preprocess_personal.py

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Step 2：GRU 無 Pretrain 訓練（3-seed Ensemble）    ║"
echo "╚══════════════════════════════════════════════════════╝"
"${PYTHON_BIN}" train_gru_nopretrain.py

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Step 3：預測 + 評估，輸出 result_nopretrain.txt    ║"
echo "╚══════════════════════════════════════════════════════╝"
"${PYTHON_BIN}" predict_nopretrain.py

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ALL DONE: GRU nopretrain pipeline 完成             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  結果：result_nopretrain.txt"
