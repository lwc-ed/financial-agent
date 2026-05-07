#!/usr/bin/env bash
# bigru_TL_alignment 完整流程（IBM pretrain，無 SMOTE）
# 用法：cd ml_ibm/bigru_TL_alignment && bash run_all.sh
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

run_step 1_preprocess_ibm.py
run_step 2_preprocess_personal.py
run_step 3_pretrain_bigru.py
run_step 4_finetune_bigru.py
run_step 5_predict_bigru.py

echo ""
echo "🎉  bigru_TL_alignment (IBM) 全部完成！"
