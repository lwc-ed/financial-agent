#!/bin/bash
# ============================================================
# Aligned Pretrain Pipeline
# 執行順序：診斷 → 前處理 → Pretrain → Finetune → 預測評估
# 請在 ml/ 目錄下執行：
#   cd ~/financial-agent/ml && bash gru_TL_alignment/run_alignment_pipeline.sh
# ============================================================

set -e  # 任何步驟失敗即停止

cd "$(dirname "$0")"   # 切換到 gru_TL_alignment/

VENV="../venv"
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
    echo "✅ 已啟動虛擬環境"
fi

echo ""
echo "======================================================"
echo "  Step 1：診斷 Domain Gap（MMD）"
echo "======================================================"
python3 1_diagnose_gap.py

echo ""
echo "======================================================"
echo "  Step 2：Walmart Aligned 前處理"
echo "======================================================"
python3 2_preprocess_walmart_aligned.py

echo ""
echo "======================================================"
echo "  Step 3：個人資料 Aligned 前處理"
echo "======================================================"
python3 3_preprocess_personal_aligned.py

echo ""
echo "======================================================"
echo "  Step 4：Aligned Pretrain（Walmart）"
echo "======================================================"
python3 4_pretrain_aligned.py

echo ""
echo "======================================================"
echo "  Step 5：Aligned Finetune（個人資料，3 seeds）"
echo "======================================================"
python3 5_finetune_aligned.py

echo ""
echo "======================================================"
echo "  Step 6：預測與三方比較"
echo "======================================================"
python3 6_predict_aligned.py

echo ""
echo "🎉 Pipeline 完成！"
echo "   結果：gru_TL_alignment/artifacts_aligned/aligned_result.txt"
echo "   圖表：gru_TL_alignment/artifacts_aligned/domain_gap_diagnosis.png"
