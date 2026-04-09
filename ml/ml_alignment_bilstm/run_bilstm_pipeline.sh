#!/bin/bash
# ============================================================
# Bi-LSTM Aligned Pretrain Pipeline
# 執行方式：
#   cd ~/financial-agent/ml
#   bash ml_alignment_bilstm/run_bilstm_pipeline.sh
#   lwc + 卷毛的融合（失敗）
# ============================================================
set -e
cd "$(dirname "$0")"

VENV="../venv"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate" && echo "✅ venv 啟動"

echo ""
echo "======================================================"
echo "  Step 1：Bi-LSTM Pretrain（Walmart aligned data）"
echo "======================================================"
python3 1_pretrain_bilstm.py

echo ""
echo "======================================================"
echo "  Step 2：Bi-LSTM Global Finetune（個人資料，7 seeds）"
echo "======================================================"
python3 2_finetune_bilstm.py

echo ""
echo "======================================================"
echo "  Step 3：評估 + 四方比較"
echo "======================================================"
python3 3_predict_bilstm.py

echo ""
echo "🎉 Pipeline 完成！"
echo "   結果：ml_alignment_bilstm/artifacts_bilstm/bilstm_result.txt"
