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

echo "========== STEP 1: 準備 GRU 序列 + HGBR 扁平特徵 =========="
"${PYTHON_BIN}" preprocess.py

echo "========== STEP 2: 從 V4 fine-tuned GRU 提取 64 維 embedding =========="
"${PYTHON_BIN}" extract_embeddings.py

echo "========== STEP 3: 訓練 HGBR（flat only vs embedding+flat 對比） =========="
"${PYTHON_BIN}" train_hgbr.py

echo "========== STEP 4: 完整比較 =========="
"${PYTHON_BIN}" predict.py

