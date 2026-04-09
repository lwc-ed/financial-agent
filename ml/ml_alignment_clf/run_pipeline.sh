#!/bin/bash
# ============================================================
# ml_alignment_clf：分類任務 Pipeline
# ============================================================
# 任務：直接預測「下週是否超標」（binary classification）
# 特徵：12 個 domain-invariant 特徵（含 spike_ratio, max_ratio_30d）
# 策略：Walmart 預訓練 → 個人 fine-tune（Transfer Learning）
#
# 執行方式：
#   cd ml/
#   bash ml_alignment_clf/run_pipeline.sh
# ============================================================

set -e
cd "$(dirname "$0")"    # 確保在 ml_alignment_clf/ 目錄下執行

PYTHON=${PYTHON:-python3}

echo ""
echo "======================================================="
echo "  ml_alignment_clf  分類任務 Pipeline"
echo "  12 features (10 aligned + 2 spike) | Walmart TL"
echo "======================================================="
echo ""

echo "▶ Step 1：Walmart 資料前處理（分類標籤）"
$PYTHON 1_preprocess_walmart_clf.py
echo ""

echo "▶ Step 2：個人資料前處理（分類標籤）"
$PYTHON 2_preprocess_personal_clf.py
echo ""

echo "▶ Step 3：Walmart 預訓練（GRU Classifier）"
$PYTHON 3_pretrain_clf.py
echo ""

echo "▶ Step 4：個人資料 Fine-tune（多 seeds）"
$PYTHON 4_finetune_clf.py
echo ""

echo "▶ Step 5：三層評估（分類版）"
$PYTHON 5_evaluate_clf.py
echo ""

echo "▶ Step 0：MMD 診斷（Raw + Encoder 表示）"
$PYTHON 0_diagnose_mmd.py
echo ""

echo "======================================================="
echo "  Pipeline 完成！"
echo "  結果：artifacts_clf/clf_evaluation.json"
echo "        artifacts_clf/mmd_diagnosis.json"
echo "======================================================="
