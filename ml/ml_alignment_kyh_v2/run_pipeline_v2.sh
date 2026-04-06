#!/bin/bash
set -e

echo "========== STEP 1: preprocess kaggle aligned =========="
python3 preprocess_kaggle_common_aligned.py

echo "========== STEP 2: preprocess own aligned =========="
python3 preprocess_own_aligned.py

echo "========== STEP 3: train kaggle base aligned =========="
python3 train_kaggle_base_aligned.py

echo "========== STEP 4: finetune on own aligned =========="
python3 finetune_own_aligned.py

echo "========== STEP 5: test transfer aligned =========="
python3 test_transfer_aligned.py

echo "========== DONE =========="