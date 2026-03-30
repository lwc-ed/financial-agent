set -e

echo "========== STEP 1: 準備全域 + per-user 資料 =========="
python preprocess.py

echo "========== STEP 2: 訓練小 GRU（hidden=32, layers=1） =========="
python train_global.py 

echo "========== STEP 3: 每個 user 各自 fine-tune（兩階段：凍結→解凍） =========="
python finetune_peruser.py

echo "========== STEP 4: 比較 global vs per-user =========="
python predict.py 


