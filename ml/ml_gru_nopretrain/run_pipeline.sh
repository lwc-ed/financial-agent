set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

cd "${SCRIPT_DIR}"
echo "Using Python: ${PYTHON_BIN}"

echo "========== STEP 1: 準備 HGBR 22 特徵 + GRU 序列 =========="
"${PYTHON_BIN}" preprocess.py

echo "========== STEP 2: 訓練 HGBR（22 features） =========="
"${PYTHON_BIN}" train_hgbr.py

echo "========== STEP 3: 訓練 GRU from scratch（對照組） =========="
"${PYTHON_BIN}" train_gru_scratch.py

echo "========== STEP 4: 比較兩者 vs baselines =========="
"${PYTHON_BIN}" predict.py
