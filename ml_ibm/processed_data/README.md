# ml_ibm/processed_data

這個資料夾是 `ml_ibm/` 的共用資料前處理入口。

## 目的

將 IBM Credit Card Transactions 原始資料聚合成所有模型共用的日粒度格式，
讓 `ml_ibm/` 底下各模型的 pretrain 腳本只需讀取同一份 `ibm_daily.csv`。

## 輸出

輸出放在 `artifacts/`（不上傳）：

- `ibm_daily.csv` — 所有模型共用的日粒度資料

## 腳本

### `build_ibm_daily.py`

讀取 `ml_ibm/ibm_data/credit_card_transactions-ibm_v2.csv`，
聚合成 per-user 日粒度，輸出 `artifacts/ibm_daily.csv`。

主要欄位：

| 欄位 | 說明 |
|---|---|
| `user_id` | 用戶 ID（0~1999）|
| `date` | 日期 |
| `daily_expense` | 當日支出（log1p 壓縮）|
| `daily_income` | 固定為 0（IBM 無收入資料）|
| `txn_count` | 當日交易筆數 |
| `daily_net` | = `-daily_expense` |
| `dow` | 星期幾（0=週一）|
| `is_weekend` | 0 或 1 |
| `day` | 幾號（1~31）|
| `month` | 幾月（1~12）|
| `expense_7d_sum` | 7 日支出總和 |
| `expense_7d_mean` | 7 日支出平均 |
| `expense_30d_sum` | 30 日支出總和 |
| `expense_30d_mean` | 30 日支出平均 |
| `zscore_7d` | 7 日 z-score |
| `zscore_14d` | 14 日 z-score |
| `zscore_30d` | 30 日 z-score |
| `target` | 未來 7 天支出總和（log1p）|

## 執行方式

```bash
cd /Users/liweichen/financial-agent/ml_ibm/processed_data
pip install -r requirements.txt
python build_ibm_daily.py
```

原始資料需先下載並放置於：

```
ml_ibm/ibm_data/credit_card_transactions-ibm_v2.csv
```

## 最低相依套件

```bash
pip install -r requirements.txt
```
