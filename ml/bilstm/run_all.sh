#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run_step() {
    local step="$1"
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  ▶  $step"
    echo "════════════════════════════════════════════════════"
    python "$step"
}

run_step 1_prepare_data.py
run_step 2_train_lstm.py
run_step 4_evaluate_decisions.py

echo ""
echo "🎉  bilstm baseline 全部完成！"
