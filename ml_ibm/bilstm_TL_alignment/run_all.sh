#!/usr/bin/env bash
# bilstm_TL_alignment 完整流程
# 用法：cd ml_ibm/bilstm_TL_alignment && bash run_all.sh
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
run_step 3_pretrain_bilstm.py
run_step 4_finetune_bilstm.py
run_step 5_predict_bilstm.py
run_step 6_evaluate_decisions.py
run_step 7_tiered_alert.py

echo ""
echo "🎉  bilstm_TL_alignment 全部完成！"


#caffeinate -i bash -c "cd /Users/liweichen/financial-agent/ml_ibm/bilstm_TL_alignment && python 1_preprocess_ibm.py && rm -f artifacts_bilstm_v2/pretrain_bilstm.pth artifacts_bilstm_v2/pretrain_history.pkl artifacts_bilstm_v2/finetune_bilstm_seed*.pth && python 3_pretrain_bilstm.py && python 4_finetune_bilstm.py && python 5_predict_bilstm.py && python 6_evaluate_decisions.py && python 7_tiered_alert.py"
