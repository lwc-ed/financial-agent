# ML Overview

`ml/` 目前分成三層：

1. `processed_data/`
共用資料前處理入口。負責從 `ml/data/raw_transactions_user*.xlsx/csv/tsv` 產生 daily ledger 與 `features_all`。

2. 主要模型目錄
- `bilstm_TL_alignment/`
- `bilstm/`
- `bigru/`
- `xgboost_TL_alignment/`
- `gru_TL_alignment/`
- `bigru_TL_alignment/`
- 其他仍在使用中的資料夾

3. `legacy_models/`
舊實驗模型與對照組，保留可重跑，但不再作為共用資料來源。

## 目前的共用資料規則

共用資料統一放在：

- `ml/processed_data/artifacts/daily_ledger_all.parquet`
- `ml/processed_data/artifacts/daily_ledger_all.csv`
- `ml/processed_data/artifacts/features_all.parquet`
- `ml/processed_data/artifacts/features_all.csv`

所有需要共用特徵表的模型，應優先讀 `processed_data/artifacts/features_all.*`。  
不要再把 `ml_gru/features_all.csv` 當成共用來源。

## 建議執行順序

### 情境 1：原始交易資料有更新

```bash
cd /Users/liweichen/financial-agent/ml/processed_data
python3 build_daily_ledgers.py
python3 build_features.py
```

之後再到各模型資料夾重跑自己的 training / predict。

## 主要資料夾說明

### `processed_data/`
共用 preprocessing。  
如果要改 daily ledger 或 `features_all` 的欄位定義，先從這裡改。

### `gru_TL_alignment/`
Alignment 實驗主線。  
這條線目前仍用自己的 pipeline 處理個人原始資料與 Walmart 資料，不依賴 `features_all.csv` 當主要輸入。

### `bigru/`
Bi-GRU baseline。  
不走 transfer learning，直接從 `ml/data/raw_transactions_*.xlsx` 準備資料後訓練與評估。

### `bigru_TL_alignment/`
Bi-GRU 的 TL / alignment 實驗資料夾。  
目前先保留 `artifacts_aligned/` 與對齊特徵訓練流程，後續要再補成真正的 Walmart pretrain + personal finetune pipeline。

### `legacy_models/`
舊模型、對照組、已整理收納的實驗資料夾。  
這些模型若需要共用特徵表，現在也已改讀 `processed_data/artifacts/features_all.csv`。
其中 `ml_gru/`、`ml_hgbr/`、`ml_XGboost/` 已移入這裡；若要重跑舊流程，請從 `legacy_models/` 內對應資料夾執行。

## 維護原則

1. 共用特徵表只由 `processed_data/` 產生。
2. 模型專屬 scaler / clip values / npy 留在各模型自己的 `artifacts` 或 `artificats`。
3. 若要新增共用欄位，先改 `processed_data/build_features.py`。
4. 若是某模型特殊前處理，放在該模型資料夾內，不要覆蓋共用資料表。
