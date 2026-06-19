#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ML_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

pip install -r "$SCRIPT_DIR/gru_hgbr/requirements.txt"
pip install -r "$SCRIPT_DIR/gru_lower_model_peruser_finetune/requirements.txt"
pip install -r "$SCRIPT_DIR/ml_gru/requirements.txt"
pip install -r "$SCRIPT_DIR/ml_gru_nopretrain/requirements.txt"
pip install -r "$SCRIPT_DIR/ml_hgbr/requirements.txt"
pip install -r "$SCRIPT_DIR/ml_lstm/requirements.txt"
pip install -r "$SCRIPT_DIR/ml_rf/requirements.txt"
pip install -r "$SCRIPT_DIR/ml_XGboost/requirements.txt"
