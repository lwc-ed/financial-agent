# ml_gru

這個資料夾是用 GRU 做「未來 7 天總消費預測」的實驗版本，流程分成兩段：

1. 用 Walmart 公開資料做 `pretrain`
2. 用個人記帳特徵做 `finetune`

目標欄位是 `future_expense_7d_sum`，也就是未來 7 天總支出。

## 資料夾結構

- `preprocess_wallmart.py`: Walmart 資料前處理，建立 pretrain 用的序列資料
- `pretrain_gru.py`: 用 Walmart 資料預訓練 GRU
- `preprocess_personal.py`: 個人記帳特徵前處理，建立 finetune 用的序列資料
- `finetune_gru.py`: 載入 pretrain 權重後，用個人資料做微調
- `predict.py`: 用 finetune 完的模型跑 validation / test，輸出 metrics 與 predictions
- `run_pipeline.sh`: 重跑整條 pipeline 的 shell script
- `features_all.csv`: 個人資料特徵表
- `wallmart/`: Walmart 原始資料
- `artificats/`: 模型、scaler、npy、metrics、predictions 等輸出

注意：資料夾名稱目前是 `artificats/`，拼字沿用既有專案，不是 `artifacts/`。

## 模型設定

模型在 [`pretrain_gru.py`](/Users/liweichen/financial-agent/ml_gru/pretrain_gru.py) 與 [`finetune_gru.py`](/Users/liweichen/financial-agent/ml_gru/finetune_gru.py) 中定義，結構一致：

- 模型類型：`GRU`
- `input_size = 7`
- `hidden_size = 64`
- `num_layers = 2`
- `dropout = 0.2`
- `output_size = 1`

輸入張量形狀：

- `X`: `(樣本數, 30天, 7特徵)`
- `y`: `(樣本數, 1)`

也就是用過去 `30` 天的特徵序列，預測未來 `7` 天總支出。

## 目前使用的特徵

個人資料與 Walmart 資料目前都對齊成 7 個特徵：

- `daily_expense`
- `expense_7d_mean`
- `expense_30d_sum`
- `has_expense`
- `has_income`
- `net_30d_sum`
- `txn_30d_sum`

目標欄位：

- `future_expense_7d_sum`

## 訓練流程

### 1. Walmart pretrain

[`preprocess_wallmart.py`](/Users/liweichen/financial-agent/ml_gru/preprocess_wallmart.py) 會做以下事情：

- 讀取 `wallmart/train.csv`、`features.csv`、`stores.csv`
- 先把週資料彙整成 store-level weekly sales
- 將週資料展成日資料
- 建立與個人資料對齊的 7 個特徵
- 建立 `30` 天 sliding windows
- 切成 `train / val / test = 70 / 15 / 15`
- 用 `train` fit 全域 clipping 與 scaler
- 輸出 `walmart_X_train.npy`、`walmart_y_train.npy`、`walmart_X_val.npy`、`walmart_y_val.npy`、`walmart_X_test.npy`、`walmart_y_test.npy`

[`pretrain_gru.py`](/Users/liweichen/financial-agent/ml_gru/pretrain_gru.py) 會：

- 載入 Walmart 的 train / val npy
- 用 `MSELoss` 訓練 GRU
- optimizer：`Adam`
- learning rate：`0.001`
- batch size：`64`
- epochs：`50`
- scheduler：`ReduceLROnPlateau`
- gradient clipping：`1.0`
- early stopping patience：`10`

輸出：

- `artificats/pretrain_gru.pth`
- `artificats/pretrain_history.pkl`

### 2. Personal finetune

[`preprocess_personal.py`](/Users/liweichen/financial-agent/ml_gru/preprocess_personal.py) 會：

- 讀取 `features_all.csv`
- 依照 `user_id` 與 `date` 排序
- 每個 user 各自建立 `30` 天 sliding windows
- 每個 user 各自做時間切分：`70 / 15 / 15`
- 每個 user 只用自己的 `train` 視窗估 `P95` clipping threshold
- 將這個 user 的 threshold 套到自己的 `train / val / test`
- 再用所有 clipped 後的 train 資料 fit 全域 `StandardScaler`
- 將 scaler 套到 val / test

這樣做的目的：

- 避免 validation / test 洩漏到前處理統計
- 避免高消費 user 的極端值主導所有人的 clipping threshold

目前 personal preprocess 會輸出：

- `personal_X_train.npy`
- `personal_y_train.npy`
- `personal_X_val.npy`
- `personal_y_val.npy`
- `personal_X_test.npy`
- `personal_y_test.npy`
- `personal_train_user_ids.npy`
- `personal_val_user_ids.npy`
- `personal_test_user_ids.npy`
- `personal_feature_scaler.pkl`
- `personal_target_scaler.pkl`
- `personal_user_clip_values.pkl`

[`finetune_gru.py`](/Users/liweichen/financial-agent/ml_gru/finetune_gru.py) 會：

- 載入 `pretrain_gru.pth`
- 建立相同結構的 GRU
- 直接繼承全部 pretrain 權重
- 凍結第二層 GRU 的 `weight_hh_l1` 與 `bias_hh_l1`
- 其餘層繼續訓練，以適應個人消費模式

訓練設定：

- loss：`MSELoss`
- optimizer：`Adam`
- learning rate：`0.0005`
- batch size：`16`
- epochs：`150`
- scheduler：`ReduceLROnPlateau`
- gradient clipping：`1.0`
- early stopping patience：`20`

輸出：

- `artificats/finetune_gru.pth`
- `artificats/finetune_history.pkl`

## 評估方式

[`predict.py`](/Users/liweichen/financial-agent/ml_gru/predict.py) 會讀取 finetune 模型與 personal scaler，然後在 validation / test 上評估。

目前輸出指標包含：

- `val_mae`
- `val_rmse`
- `val_smape`
- `val_per_user_nmae`
- `test_mae`
- `test_rmse`
- `test_smape`
- `test_per_user_nmae`

另外也會和 baseline 比較：

- `naive_7d_sum`
- `moving_avg_30d_x7`

相關輸出檔案：

- `artificats/metrics.json`
- `artificats/training_summary.txt`
- `artificats/predictions_val.csv`
- `artificats/predictions_test.csv`

## 風險評估

`predict.py` 裡還有一段示範性的風險評估邏輯，會把「預測未來 7 天支出」除以「目前帳戶餘額」，分成：

- `安全`: `< 30%`
- `注意`: `< 50%`
- `警告`: `< 80%`
- `危險`: `>= 80%`

這一段目前是 demo 用，餘額在腳本內是手動設定，不是正式系統串接版本。

## 如何重跑

### 方式一：直接跑整條 pipeline

```bash
cd /Users/liweichen/financial-agent/ml_gru
./run_pipeline.sh
```

`run_pipeline.sh` 會依序執行：

1. `preprocess_wallmart.py`
2. `pretrain_gru.py`
3. `preprocess_personal.py`
4. `finetune_gru.py`
5. `predict.py`

### 方式二：手動逐步跑

```bash
cd /Users/liweichen/financial-agent/ml_gru
/Users/liweichen/financial-agent/.venv/bin/python preprocess_wallmart.py
/Users/liweichen/financial-agent/.venv/bin/python pretrain_gru.py
/Users/liweichen/financial-agent/.venv/bin/python preprocess_personal.py
/Users/liweichen/financial-agent/.venv/bin/python finetune_gru.py
/Users/liweichen/financial-agent/.venv/bin/python predict.py
```

## 目前設計上的重點與限制

- 這版是 transfer learning：先學 Walmart，再微調到個人資料
- 個人資料目前是 per-user 時間切分，不是全資料混在一起切
- 個人資料目前有 per-user clipping，但不是直接刪除 outlier row
- evaluation 現在有絕對誤差與相對誤差兩套指標
- feature set 是目前固定的 7 個特徵，未來仍可再做 feature engineering 或 feature search

## 建議交接時補充的口頭說明

- 如果要改 preprocessing，優先確認有沒有破壞 train / val / test 的切分一致性
- 如果要改 relative metric，先確認 `predict.py` 裡的公式與業務定義一致
- 如果要重新比較版本，請固定同一份 `features_all.csv`、同一組 split 邏輯，再看 metrics
