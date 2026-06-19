#!/bin/bash
#先找出best feature
set -e

echo "========== STEP 1: make_daily_ledgers =========="
python3 training/make_daily_ledgers.py

echo "========== STEP 2: make_features =========="
python3 training/make_features.py

echo "========== STEP 3: split_by_month =========="
python3 training/split_by_month.py

echo "========== STEP 4: train_model_v1 =========="
python3 training/train_model_v1.py

echo "========== STEP 5: train_model_v2 =========="
python3 training/train_model_v2.py --feature-set user_selected_v1

echo "========== STEP 6: train_model_v3_loop_search_hgbr =========="
python3 training/train_model_v3_hgbr.py --trials 1000 --random-state 1999 --min-features 5 --max-features 21

echo "========== STEP 7: train_model_v4_loop_search_regression =========="
python3 training/train_model_v4_regression.py --trials 2000 --random-state 9192 --min-features 7 --max-features 21

echo "========== TRAINING PIPELINE FINISHED =========="
