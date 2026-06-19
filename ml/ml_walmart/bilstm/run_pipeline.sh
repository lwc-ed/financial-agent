#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "BiLSTM Baseline Pipeline (macOS)"
echo "Location: $SCRIPT_DIR"
echo "=============================================="

if [[ ! -d "venv" ]]; then
  echo "❌ 找不到 venv。"
  echo "請先在 $SCRIPT_DIR 建立虛擬環境，例如："
  echo "  python3 -m venv venv"
  exit 1
fi

echo ""
echo "📦 啟用虛擬環境..."
source "venv/bin/activate"

if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ 啟用 venv 後找不到 python3"
  exit 1
fi

echo "🐍 Python: $(python3 --version)"

echo ""
echo "📥 安裝/更新本子專案所需套件..."
pip install -r my_requirements_mac.txt

echo ""
echo "=============================================="
echo "STEP 1/3  準備資料"
echo "=============================================="
python3 1_prepare_data.py

echo ""
echo "=============================================="
echo "STEP 2/3  訓練 BiLSTM"
echo "=============================================="
python3 2_train_lstm.py

echo ""
echo "=============================================="
echo "STEP 3/3  評估模型（回歸 + 決策 + 成本 + Per-user）"
echo "=============================================="
python3 4_evaluate_decisions.py

echo ""
echo "=============================================="
echo "✅ Pipeline 完成"
echo "報告位置:"
echo "  $SCRIPT_DIR/artifacts/decision_evaluation.json"
echo "=============================================="
