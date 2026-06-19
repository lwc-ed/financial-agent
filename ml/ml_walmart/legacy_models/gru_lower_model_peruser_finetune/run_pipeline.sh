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

echo "========== STEP 1: 準備全域 + per-user 資料 =========="
"${PYTHON_BIN}" preprocess.py

echo "========== STEP 2: 訓練小 GRU（hidden=32, layers=1） =========="
"${PYTHON_BIN}" train_global.py

echo "========== STEP 3: 每個 user 各自 fine-tune（兩階段：凍結→解凍） =========="
"${PYTHON_BIN}" finetune_peruser.py

echo "========== STEP 4: 比較 global vs per-user =========="
"${PYTHON_BIN}" predict.py
