#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Step 1: run gru_hgbr"
bash "$SCRIPT_DIR/gru_hgbr/run_pipeline.sh"

echo "Step 2: run gru_lower_model_peruser_finetune"
bash "$SCRIPT_DIR/gru_lower_model_peruser_finetune/run_pipeline.sh"

echo "Step 3: run ml_gru"
bash "$SCRIPT_DIR/ml_gru/run_pipeline.sh"
bash "$SCRIPT_DIR/ml_gru/run_pipeline_v1.sh"
bash "$SCRIPT_DIR/ml_gru/run_pipeline_v2_to_v5.sh"

echo "Step 4: run ml_gru_nopretrain"
bash "$SCRIPT_DIR/ml_gru_nopretrain/run_pipeline.sh"

echo "Step 5: run ml_hgbr"
bash "$SCRIPT_DIR/ml_hgbr/run_training_pipeline.sh"
bash "$SCRIPT_DIR/ml_hgbr/run_training_expanding_window_version.sh"

echo "Step 6: run ml_lstm"
bash "$SCRIPT_DIR/ml_lstm/run_pipeline.sh"

echo "Pipeline done"