#!/bin/bash
# V1 Pipeline：僅執行 pretrain_v1 → finetune_v1 → predict_v1
# 個人資料前處理沿用已有 artifacts（personal_X_train.npy 等）

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
echo "========== V1 STEP 1: pretrain_gru_v1 (hidden=128, Attention, Huber) =========="
"${PYTHON_BIN}" pretrain_gru_v1.py

echo "========== V1 STEP 2: finetune_gru_v1 (full fine-tune, AdamW, Cosine) =========="
"${PYTHON_BIN}" finetune_gru_v1.py

echo "========== V1 STEP 3: predict_v1 (eval + report) =========="
"${PYTHON_BIN}" predict_v1.py

echo "========== V1 PIPELINE FINISHED =========="
echo "Results saved to artificats/metrics_v1.json and artificats/training_summary_v1.txt"
