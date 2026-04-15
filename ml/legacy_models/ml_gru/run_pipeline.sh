#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$(dirname "${SCRIPT_DIR}")")" == "legacy_models" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

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
echo "========== STEP 1: preprocess_wallmart =========="
"${PYTHON_BIN}" preprocess_wallmart.py

echo "========== STEP 2: pretrain_gru =========="
"${PYTHON_BIN}" pretrain_gru.py

echo "========== STEP 3: preprocess_personal =========="
"${PYTHON_BIN}" preprocess_personal.py

echo "========== STEP 4: finetune_gru =========="
"${PYTHON_BIN}" finetune_gru.py

echo "========== STEP 5: predict =========="
"${PYTHON_BIN}" predict.py

echo "========== ML_GRU PIPELINE FINISHED =========="
