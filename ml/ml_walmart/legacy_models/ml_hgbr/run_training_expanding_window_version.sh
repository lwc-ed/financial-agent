#!/bin/bash
# 再用找到的feature去做expanding window
set -e

echo "========== STEP 1: make_daily_ledgers =========="
python3 training/make_daily_ledgers.py

echo "========== STEP 2: make_features =========="
python3 training/make_features.py

echo "========== STEP 3: split_by_month =========="
python3 training/split_by_month_v2.py

echo "========== STEP 3: expanding_window_training =========="
python3 training/train_model_cv_v2.py

echo "========== TRAINING PIPELINE FINISHED =========="

