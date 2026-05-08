#!/usr/bin/env bash
# bilstm_TL_alignment 完整流程（Walmart pretrain）
# 用法：cd ml/bilstm_TL_alignment && bash run_all.sh
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

run_step 1_preprocess_walmart.py
run_step 2_preprocess_personal.py
run_step 3_pretrain_bilstm.py
run_step 4_finetune_bilstm.py
run_step 5_predict_bilstm.py

echo ""
echo "🎉  bilstm_TL_alignment (Walmart) 全部完成！"
