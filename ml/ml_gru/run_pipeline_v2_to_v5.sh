#!/bin/bash
# V2~V5 Pipeline
# V2 pretrain 已完成，從 predict_v2 開始（跳過已訓練的部分）

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
echo "║  V2 pretrain                       ║"
echo "╚══════════════════════════════════════════════════════╝"
"${PYTHON_BIN}" pretrain_gru_v2.py


echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  V2 predict（模型已訓練完畢）                        ║"
echo "╚══════════════════════════════════════════════════════╝"
"${PYTHON_BIN}" finetune_gru_v2.py
"${PYTHON_BIN}" predict_v2.py

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  V3：Log1p 目標轉換 + 輸入噪聲增強                  ║"
echo "╚══════════════════════════════════════════════════════╝"
"${PYTHON_BIN}" finetune_gru_v3.py
"${PYTHON_BIN}" predict_v3.py

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  V4：Layer-wise LR Decay + 混合損失 + WarmRestarts   ║"
echo "╚══════════════════════════════════════════════════════╝"
"${PYTHON_BIN}" finetune_gru_v4.py
"${PYTHON_BIN}" predict_v4.py

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  V5：三模型 Ensemble + Per-user 偏差修正             ║"
echo "╚══════════════════════════════════════════════════════╝"
"${PYTHON_BIN}" finetune_gru_v5.py
"${PYTHON_BIN}" predict_v5.py

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ALL DONE: V2~V5 Pipeline 完成                      ║"
echo "╚══════════════════════════════════════════════════════╝"
