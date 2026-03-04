# ML Pipeline Handoff

這份文件是 `financial-agent/ml` 目前離線訓練流程的交接說明，重點放在：

- 每次要先跑哪些程式
- 每支程式會產生哪些檔案
- 這些檔案的用途是什麼

## 環境建立

建議使用獨立的 `venv_ml`：

```bash
cd /Users/liweichen/financial-agent
python3 -m venv ml/venv_ml
source ml/venv_ml/bin/activate
pip install -r ml/requirements.txt
```

之後所有指令都可以從 repo root 執行：

```bash
cd /Users/liweichen/financial-agent
```

也可以先進到 `ml/` 目錄再執行 `python3 training/...`。

## Pipeline 概觀

目前資料流程分成四步：

1. 原始交易資料整理成 daily ledger
2. 從 daily ledger 產生特徵與 label
3. 依月份做 per-user 時序切分
4. 訓練模型並輸出 test 評估與 baseline 對照

## 執行順序

每次從原始資料重跑時，建議照下面順序：

```bash
python3 ml/training/make_daily_ledgers.py
python3 ml/training/make_features.py
python3 ml/training/split_by_month.py
python3 ml/training/train_model.py
```

如果 `daily_ledger_all.parquet` 和 `features_all.parquet` 已經是最新的，只需要跑：

```bash
python3 ml/training/split_by_month.py
python3 ml/training/train_model.py
```

## 各腳本用途

### 1. `ml/training/make_daily_ledgers.py`

作用：
- 讀取 `ml/data/raw_transactions_user*.xlsx/csv/tsv`
- 轉成每位 user 的 daily ledger
- 補齊日期，沒有交易的日期也會保留

主要輸出：
- `ml/artifacts/daily_ledger_user{n}.parquet`
- `ml/artifacts/daily_ledger_user{n}.csv`
- `ml/artifacts/daily_ledger_all.parquet`
- `ml/artifacts/daily_ledger_all.csv`

重要欄位：
- `user_id`
- `date`
- `txn_count`
- `daily_expense`
- `daily_income`
- `daily_net`
- `is_weekend`

用途：
- 作為後續特徵工程與月份活躍天數篩選的基礎資料

### 2. `ml/training/make_features.py`

作用：
- 讀取 `daily_ledger_all.parquet`
- 對每位 user 依時間排序產生 rolling features
- 建立未來 7 天支出總和作為回歸 label

主要輸出：
- `ml/artifacts/features_all.parquet`
- `ml/artifacts/features_all.csv`

目前重要特徵與 label：
- `expense_7d_sum`
- `expense_30d_mean`
- `expense_30d_sum`
- `txn_7d_sum`
- `txn_30d_sum`
- `future_expense_7d_sum` 或其他指定 target 欄位

用途：
- 作為切分與模型訓練的輸入資料

### 3. `ml/training/split_by_month.py`

作用：
- 用 `daily_ledger_all.parquet` 計算每個 user 每個月的 `active_days`
- `active_day` 定義是該日 `txn_count > 0`
- 若某 user 某月份 `active_days < 15`，整個 user-month 會被丟掉
- 用剩下月份按時間做 per-user 的 60/20/20 時序切分
- 同一個月份不會同時落在不同 split

預設命令：

```bash
python3 ml/training/split_by_month.py
```

若要改活躍天數門檻：

```bash
python3 ml/training/split_by_month.py --min-days-per-month 12
```

主要輸出：
- `ml/artifacts/features_train.parquet`
- `ml/artifacts/features_train.csv`
- `ml/artifacts/features_val.parquet`
- `ml/artifacts/features_val.csv`
- `ml/artifacts/features_test.parquet`
- `ml/artifacts/features_test.csv`
- `ml/artifacts/month_split_assignments.csv`
- `ml/artifacts/month_split_summary.json`
- `ml/artifacts/invalid_months_debug.txt`

各輸出用途：
- `features_train.*`: 訓練集
- `features_val.*`: 驗證集
- `features_test.*`: 最終測試集
- `month_split_assignments.csv`: 每個 user-month 被分到哪個 split
- `month_split_summary.json`: 切分後的統計摘要
- `invalid_months_debug.txt`: 每個 user 哪些月份因為 `active_days` 不足而無效

補充：
- 若某 user 過濾後剩下不到 3 個有效月份，該 user 會被整個跳過，不進 train/val/test

### 4. `ml/training/train_model.py`

作用：
- 讀取 `features_train/val/test.parquet`
- 先輸出 diagnostics，包括 test target 分布、baseline vs 模型比較表、feature 檢查
- 訓練兩個候選模型：
  - `MLPRegressor`
  - `HistGradientBoostingRegressor`
- `MLPRegressor` 會逐 epoch 輸出 `train_loss`、`val_loss`、`train_mae`、`val_mae`
- `HistGradientBoostingRegressor` 會在 validation set 上做多組超參數 trial 比較
- 用 validation 表現選最佳模型
- 訓練結束後只對選中的最佳模型在 test set 上做一次最終評估
- 同時計算兩個 baseline 在同一個 test set 上的 MAE/RMSE
- 印出是否 beat `moving_avg_30d_x7` baseline；若沒有，會輸出可能原因與下一步建議

預設命令：

```bash
python3 ml/training/train_model.py
```

若 target 欄位需要手動指定：

```bash
python3 ml/training/train_model.py --target-column future_expense_7d_sum
```

目前 baseline：

1. Naive baseline
   用 `expense_7d_sum` 預測未來 7 天支出

2. Moving average baseline
   用 `expense_30d_mean * 7` 預測未來 7 天支出

主要輸出：
- `ml/artifacts/diagnostics.md`
- `ml/artifacts/best_mlp_model.pkl`
- `ml/artifacts/best_hgbr_model.pkl`
- `ml/artifacts/training_history.csv`
- `ml/artifacts/test_metrics.json`
- `ml/artifacts/predictions_test.csv`
- `ml/artifacts/training_report.txt`

各輸出用途：
- `diagnostics.md`: test target 統計、baseline vs 模型表格、feature leakage / future-like 名稱檢查
- `best_mlp_model.pkl`: MLP 最佳 checkpoint
- `best_hgbr_model.pkl`: HistGradientBoosting 最佳 checkpoint
- `training_history.csv`: MLP 的 epoch 訓練紀錄，加上 HGBR 的 validation trial 紀錄
- `test_metrics.json`: 最終選中模型的 `model_name`、`best_val_metric`、`test_mae`、`test_rmse`、baseline metrics
- `predictions_test.csv`: 長表格式 test 預測結果，包含 `user_id`、`date`、`y_true`、`y_pred`、`model_name`
- `training_report.txt`: 給人閱讀的訓練摘要，包含 dataset sizes、baseline 對照與 artifact 路徑

## 重要輸入與輸出檔案整理

### 原始輸入

- `ml/data/raw_transactions_user*.xlsx`

### 中間產物

- `ml/artifacts/daily_ledger_all.parquet`
- `ml/artifacts/features_all.parquet`

### 切分產物

- `ml/artifacts/features_train.parquet`
- `ml/artifacts/features_val.parquet`
- `ml/artifacts/features_test.parquet`

### 訓練產物

- `ml/artifacts/diagnostics.md`
- `ml/artifacts/best_mlp_model.pkl`
- `ml/artifacts/best_hgbr_model.pkl`
- `ml/artifacts/training_history.csv`
- `ml/artifacts/test_metrics.json`
- `ml/artifacts/predictions_test.csv`
- `ml/artifacts/training_report.txt`

## 推薦的日常操作

### 情境 A：原始交易資料有更新

```bash
python3 ml/training/make_daily_ledgers.py
python3 ml/training/make_features.py
python3 ml/training/split_by_month.py
python3 ml/training/train_model.py
```

### 情境 B：只想重做切分與重訓

```bash
python3 ml/training/split_by_month.py
python3 ml/training/train_model.py
```

### 情境 C：只想重新訓練模型

前提是 `features_train.parquet`、`features_val.parquet`、`features_test.parquet` 已存在且是最新的。

```bash
python3 ml/training/train_model.py
```

## 目前流程的設計原則

- 先做 rolling feature，再按月份切分
- rolling feature 在 `make_features.py` 裡已用 `shift(1)` 避免偷看當天或未來資訊
- split 以月份為單位，避免同一個月份同時出現在 train/val/test
- 模型選擇只用 validation，test 不參與調參
- final metrics 只在 test set 上做一次
- baseline 與候選模型使用同一個 test set，方便公平比較
- 若 tree-based model 明顯優於 MLP，最終會選 tree-based model 作為最佳模型

## 目前最佳結果範例

某次已完成的訓練結果如下：

- `model_name: hgbr`
- `target_column: future_expense_7d_sum`
- `test_mae: 988.881727`
- `test_rmse: 1275.818003`
- `moving_avg_30d_x7 mae: 1217.206299`
- `moving_avg_30d_x7 rmse: 1662.021360`

代表 `HistGradientBoostingRegressor` 已在同一個 test set 上優於 `moving_avg_30d_x7` baseline。

## 交接時組員最常需要看的檔案

- [make_daily_ledgers.py](/Users/liweichen/financial-agent/ml/training/make_daily_ledgers.py)
- [make_features.py](/Users/liweichen/financial-agent/ml/training/make_features.py)
- [split_by_month.py](/Users/liweichen/financial-agent/ml/training/split_by_month.py)
- [train_model.py](/Users/liweichen/financial-agent/ml/training/train_model.py)

如果要檢查某次切分或訓練結果，優先看：

- [month_split_summary.json](/Users/liweichen/financial-agent/ml/artifacts/month_split_summary.json)
- [invalid_months_debug.txt](/Users/liweichen/financial-agent/ml/artifacts/invalid_months_debug.txt)
- [diagnostics.md](/Users/liweichen/financial-agent/ml/artifacts/diagnostics.md)
- [test_metrics.json](/Users/liweichen/financial-agent/ml/artifacts/test_metrics.json)
- [predictions_test.csv](/Users/liweichen/financial-agent/ml/artifacts/predictions_test.csv)
- [training_report.txt](/Users/liweichen/financial-agent/ml/artifacts/training_report.txt)
