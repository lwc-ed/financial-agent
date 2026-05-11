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
run_step 2_train_bigru.py
run_step 3_evaluate_standard_bigru.py

echo ""
echo "🎉  bigru baseline 全部完成！"
