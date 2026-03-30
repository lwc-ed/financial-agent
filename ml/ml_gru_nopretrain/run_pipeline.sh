set -e

echo "========== STEP 1: 準備 HGBR 22 特徵 + GRU 序列 =========="
python preprocess.py 

echo "========== STEP 2: 訓練 HGBR（22 features） =========="
python train_hgbr.py 

echo "========== STEP 3: 訓練 GRU from scratch（對照組） =========="
python train_gru_scratch.py 

echo "========== STEP 4: 比較兩者 vs baselines =========="
python predict.py

