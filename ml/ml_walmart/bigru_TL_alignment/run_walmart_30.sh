#!/bin/bash
set -e
echo "🚀 [開始執行] Walmart 實驗組 - 30 Seeds 流程"

# 使用之前確認成功的 Python 環境路徑
ENV_PYTHON="/home/server_605_4090/financial-agent/mlenv/bin/python3"

echo "------------------------------------------"
echo "📂 Step 1 & 2: Preprocessing..."
$ENV_PYTHON 1_preprocess_walmart.py
$ENV_PYTHON 2_preprocess_personal.py

echo "🧠 Step 3: Pre-training (Walmart Source)..."
$ENV_PYTHON 3_pretrain_bigru.py

echo "🎯 Step 4: Fine-tuning 30 Seeds..."
$ENV_PYTHON 4_finetune_bigru.py

echo "📊 Step 5: Final Evaluation..."
$ENV_PYTHON 5_predict_bigru.py

echo "✅ [完成] Walmart 30 個實驗全部跑完囉！"
