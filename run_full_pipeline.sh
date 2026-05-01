#!/bin/bash
set -e  # 如果其中一個步驟報錯，就立刻停止，不會亂跑下去

echo "🚀 Step 1: Bigru Alignment (Preprocessing & Training)"
cd ~/financial-agent/ml_ibm/bigru_TL_alignment
python 1_preprocess_ibm.py
python 3_pretrain_bigru.py
python 4_finetune_bigru.py
python 5_predict_bigru.py

echo "🚀 Step 2: Bi-LSTM Alignment (Training & Evaluation)"
cd ../bilstm
python 3_pretrain_bilstm.py
python 4_finetune_bilstm.py
python 5_predict_bilstm.py

echo "🎉 全量數據流程跑完囉！今晚開會見！"
