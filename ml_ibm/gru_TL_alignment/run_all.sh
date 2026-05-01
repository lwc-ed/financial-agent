#!/usr/bin/env bash
# gru_TL_alignment 完整流程
# 用法：cd ml_ibm/gru_TL_alignment && bash run_all.sh
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
run_step 2_preprocess_personal_aligned.py
run_step 3_pretrain_aligned.py
run_step 4_finetune_aligned.py
run_step 5_predict_aligned.py
run_step 6_evaluate_decisions.py
run_step 7_threshold_sweep.py
run_step 8_tiered_alert.py

echo ""
echo "🎉  gru_TL_alignment 全部完成！"



#caffeinate -i bash -c "cd /Users/liweichen/financial-agent/ml_ibm/gru_TL_alignment && python 1_preprocess_ibm.py && rm -f artifacts_aligned/pretrain_aligned_gru.pth artifacts_aligned/pretrain_aligned_history.pkl artifacts_aligned/finetune_aligned_gru_seed*.pth && python 3_pretrain_aligned.py && python 4_finetune_aligned.py && python 5_predict_aligned.py && python 6_evaluate_decisions.py && python 7_threshold_sweep.py && python 8_tiered_alert.py"
