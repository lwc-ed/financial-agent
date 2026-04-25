# processed_data

這個資料夾是 `ml/` 的共用資料前處理入口。

## 目的

把所有模型都會共用的資料生成流程集中在同一個地方，避免：

- 各模型互相依賴彼此資料夾
- 同一份 `features_all.csv` 在不同地方各存一份
- 修改特徵欄位時不知道應該改哪裡

## 目前輸出

輸出都放在 `artifacts/`：

- `daily_ledger_all.parquet`
- `daily_ledger_all.csv`
- `features_all.parquet`
- `features_all.csv`
- `kaggle_processed_common.csv`

## 腳本

### `build_daily_ledgers.py`
讀取 `ml/data/raw_transactions_user*.xlsx/csv/tsv`，建立 daily ledger。

主要欄位：
- `user_id`
- `date`
- `daily_income`
- `daily_expense`
- `txn_count`
- `daily_net`
- `has_income`
- `has_expense`
- `dow`
- `is_weekend`
- `day`
- `month`

### `build_features.py`
讀取 `daily_ledger_all.parquet`，建立共用特徵表 `features_all`。

目前包含的重點欄位：
- `daily_income`
- `daily_expense`
- `daily_net`
- `txn_count`
- `expense_7d_sum`
- `expense_7d_mean`
- `expense_30d_sum`
- `expense_30d_mean`
- `net_7d_sum`
- `net_30d_sum`
- `txn_7d_sum`
- `txn_30d_sum`
- `expense_7d_30d_ratio`
- `expense_trend`
- `days_to_end_of_month`
- `future_expense_7d_sum`

### `build_kaggle_common.py`
讀取 `ml/wallmart/` 的 Kaggle Walmart 原始資料，建立：

- `artifacts/kaggle_processed_common.csv`

這份資料是給需要 Walmart / Kaggle 共用欄位格式的模型使用，例如 `xgboost_TL_alignment`。

## 執行方式

```bash
cd /Users/liweichen/financial-agent/ml/processed_data
python3 build_daily_ledgers.py
python3 build_features.py
python3 build_kaggle_common.py
```

## 使用規則

1. 共用特徵表只從這裡產生。
2. 其他模型應讀 `processed_data/artifacts/features_all.csv` 或 `.parquet`。
3. Kaggle / Walmart 的共用對齊資料應讀 `processed_data/artifacts/kaggle_processed_common.csv`。
4. 不要再把 `ml_gru/features_all.csv` 當成共用主來源。

## 最低相依套件

如果你在 `processed_data` 底下使用獨立 venv，至少要安裝：

```bash
pip install -r requirements.txt
```

其中：
- `openpyxl`：讀 `.xlsx`
- `pyarrow`：讀寫 parquet

如果沒有 `pyarrow`，目前腳本仍會輸出 CSV，並在讀取時自動優先 fallback 到 CSV。
