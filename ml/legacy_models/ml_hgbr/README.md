# legacy_models/ml_hgbr

這個資料夾已改列為 legacy 模型，但仍可重跑 HGBR / MLP 訓練流程。

## 目前角色

`ml_hgbr` 現在不再負責維護共用 `features_all` 主來源。  
共用資料已抽到：

- `../../processed_data/artifacts/daily_ledger_all.parquet`
- `../../processed_data/artifacts/features_all.parquet`

`ml_hgbr` 主要負責：
- 讀取共用資料
- 做 month-based split
- 訓練 HGBR / MLP
- 產出自己的模型與評估結果到 `legacy_models/ml_hgbr/artifacts/`

## 主要腳本

### `run_training_pipeline.sh`
完整入口。現在會先呼叫：

1. `training/make_daily_ledgers.py`（wrapper 到 `../../processed_data/build_daily_ledgers.py`）
2. `training/make_features.py`（wrapper 到 `../../processed_data/build_features.py`）

之後再執行：

3. `training/split_by_month.py`
4. `training/train_model_v1.py`
5. `training/train_model_v2.py`
6. `training/train_model_v3_hgbr.py`
7. `training/train_model_v4_regression.py`

### `training/split_by_month.py`
讀取 `processed_data/artifacts` 的共用資料，產生：
- `features_train.*`
- `features_val.*`
- `features_test.*`

這些 split 檔案輸出在 `legacy_models/ml_hgbr/artifacts/`。

### `training/split_by_month_v2.py`
expanding-window 版本切分。

## 共用資料與模型輸出分界

### 共用資料
放在：
- `ml/processed_data/artifacts/`

### `ml_hgbr` 自己的輸出
放在：
- `ml/legacy_models/ml_hgbr/artifacts/`

例如：
- `features_train.parquet`
- `features_val.parquet`
- `features_test.parquet`
- `best_hgbr_model*.pkl`
- `test_metrics*.json`
- `predictions_test*.csv`

## 如何重跑

```bash
cd /Users/liweichen/financial-agent/ml/legacy_models/ml_hgbr
./run_training_pipeline.sh
```

如果共用資料已經是最新的，也可以只跑：

```bash
python3 training/split_by_month.py
python3 training/train_model_v1.py
```

或其他你需要的 `train_model_*` 版本。

## 維護原則

1. 若是 daily ledger / `features_all` 欄位定義問題，改 `processed_data/`。
2. 若是 split 規則問題，改 `training/split_by_month*.py`。
3. 若是模型本身問題，改 `training/train_model*.py`。
