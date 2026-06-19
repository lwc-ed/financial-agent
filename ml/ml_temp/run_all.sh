#!/usr/bin/env bash
# 一鍵執行無對齊對照 pipeline（1→5）。需 GPU / 足夠記憶體。
set -e
cd "$(dirname "$0")"
PY="../../.venv/bin/python"

echo "===== [1/5] IBM 前處理（raw） ====="
$PY 1_preprocess_ibm.py
echo "===== [2/5] 個人前處理（raw） ====="
$PY 2_preprocess_personal.py
echo "===== [3/5] 預訓練 ====="
$PY 3_pretrain.py
echo "===== [4/5] 微調（30 seeds） ====="
$PY 4_finetune.py
echo "===== [5/5] 預測與評估 ====="
$PY 5_predict.py
echo "🎉 全部完成！報告：ml_temp/model_outputs/bigru_noalign/"
