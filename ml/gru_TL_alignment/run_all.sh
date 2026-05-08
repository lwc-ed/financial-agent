#!/usr/bin/env bash
# gru_TL_alignment 完整流程（Walmart pretrain）
# 用法：cd ml/gru_TL_alignment && bash run_all.sh
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

run_step 2_preprocess_walmart_aligned.py
run_step 3_preprocess_personal_aligned.py
run_step 4_pretrain_aligned.py
run_step 5_finetune_aligned.py
run_step 6_predict_aligned.py

echo ""
echo "🎉  gru_TL_alignment (Walmart) 全部完成！"
