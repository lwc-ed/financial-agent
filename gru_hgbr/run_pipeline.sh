set -e

echo "========== STEP 1: 準備 GRU 序列 + HGBR 扁平特徵 =========="
python preprocess.py  

echo "========== STEP 2: 從 V4 fine-tuned GRU 提取 64 維 embedding =========="
python extract_embeddings.py

echo "========== STEP 3: 訓練 HGBR（flat only vs embedding+flat 對比） =========="
python train_hgbr.py     

echo "========== STEP 4: 完整比較 =========="
python predict.py

