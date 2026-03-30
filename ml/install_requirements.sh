#!/bin/bash
set -e

pip install -r ./gru_hgbr/requirements.txt
pip install -r ./gru_lower_model_peruser_finetune/requirements.txt
pip install -r ./ml_gru/requirements.txt
pip install -r ./ml_gru_nopretrain/requirements.txt
pip install -r ./ml_hgbr/requirements.txt
pip install -r ./ml_lstm/requirements.txt

