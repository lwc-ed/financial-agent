#!/bin/bash
set -e
echo "🚀 [開始執行] IBM 2000 人完整實驗流程"

# 這裡直接指定 Conda 環境裡的 Python 路徑
ENV_PYTHON="/home/server_605_4090/financial-agent/mlenv/bin/python3"

echo "------------------------------------------"
echo "📂 Step 1 & 2: Preprocessing (2000 users)..."
$ENV_PYTHON 1_preprocess_ibm.py
$ENV_PYTHON 2_preprocess_personal.py

echo "🧠 Step 3: Pre-training (大魔王階段)..."
$ENV_PYTHON 3_pretrain_bigru.py

echo "🎯 Step 4: Fine-tuning 30 Seeds..."
$ENV_PYTHON 4_finetune_bigru.py

echo "📊 Step 5: Final Evaluation..."
$ENV_PYTHON 5_predict_bigru.py

echo "✅ [完成] 實驗全部跑完囉！明天早上來看結果。"
