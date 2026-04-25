# gru_TL_alignment

這個資料夾是 rolling z-score alignment 的主實驗。

## 目標

把 Walmart 與個人資料都投影到同一組相對動態特徵空間，再做：

1. source pretrain
2. target finetune
3. MMD-based alignment

任務仍然是：
- 用過去 30 天序列
- 預測未來 7 天總支出 `future_expense_7d_sum`

## 主要特徵

目前 aligned features 定義在 `alignment_utils.py`：

- `zscore_7d`
- `zscore_14d`
- `zscore_30d`
- `pct_change_norm`
- `volatility_7d`
- `is_above_mean_30d`
- `pct_rank_7d`
- `pct_rank_30d`
- `dow_sin`
- `dow_cos`

## 主要流程

1. `1_diagnose_gap.py`
診斷 Walmart 與個人資料的 domain gap。

2. `2_preprocess_walmart_aligned.py`
處理 Walmart aligned 特徵。

3. `3_preprocess_personal_aligned.py`
處理個人 aligned 特徵。

4. `4_pretrain_aligned.py`
在 Walmart aligned 資料上 pretrain。

5. `5_finetune_aligned.py`
在個人 aligned 資料上 finetune。

6. `6_predict_aligned.py`
做最終評估。

其他腳本：
- `7_evaluate_decisions.py`
- `8_threshold_sweep.py`
- `9_tiered_alert.py`
- `ablation_mmd_lambda.py`

## 資料來源

這條線目前沒有改成讀 `processed_data/features_all.csv` 當主輸入。  
它自己的個人資料來源仍在 `alignment_utils.py` 中，直接從 `ml/data/raw_transactions_user*.xlsx` 聚合成日級支出。

## 輸出

主要輸出在：
- `artifacts_aligned/`

例如：
- `personal_aligned_X_*.npy`
- `personal_aligned_y_*.npy`
- `pretrain_aligned_gru.pth`
- `finetune_aligned_gru_seed*.pth`
- `aligned_metrics.json`
- `aligned_result.txt`

## 如何重跑

```bash
cd /Users/liweichen/financial-agent/ml/gru_TL_alignment
./run_alignment_pipeline.sh
```

## 維護原則

1. 若要改 aligned feature 定義，先改 `alignment_utils.py`。
2. 若未來要和 `processed_data` 完全整合，要先重新定義這條線是不是仍保留自己從原始交易聚合的流程。
