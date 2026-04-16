# Output And Evaluation Spec

本文件是 `ml/` 目錄下所有模型的唯一正式輸出與評估規格。

適用模型包括但不限於：
- `bigru`
- `bigru_TL_alignment`
- `gru_TL_alignment`
- `bilstm`
- `bilstm_TL_alignment`
- `xgboost_TL_alignment`

若任何模型自己的 README、腳本註解、歷史輸出格式與本文件衝突，一律以本文件為準。

## 1. 統一輸出位置

所有模型的評估輸出必須統一寫入：

`/Users/liweichen/financial-agent/ml/model_outputs`

每個模型必須在此資料夾下建立自己的子資料夾，名稱固定使用模型資料夾名稱，例如：

- `model_outputs/bigru`
- `model_outputs/bigru_TL_alignment`
- `model_outputs/gru_TL_alignment`
- `model_outputs/bilstm`
- `model_outputs/bilstm_TL_alignment`
- `model_outputs/xgboost_TL_alignment`

## 2. 每個模型必須輸出的檔案

每個模型都必須輸出以下檔案：

- `metrics_regression.json`
- `metrics_alarm_binary.json`
- `metrics_risk_4class.json`
- `predictions.csv`
- `summary.txt`

## 3. 共同前提

所有 test 一律先做以下步驟，不可自行更改定義。

### 3.1 模型輸出

`predicted_expense_7d = 模型預測的未來 7 天支出總額`

### 3.2 每個 user 的月可動用資金

`monthly_available_cash_u = 該 user 訓練資料總收入 / 該 user 訓練資料涵蓋月數`

### 3.3 本月已花費

`spent_mtd(t) = 該 user 在日期 t 所在月份，從當月 1 日到 t 為止的實際累積支出`

### 3.4 本月剩餘預算

`remaining_budget_month(t) = monthly_available_cash_u - spent_mtd(t)`

### 3.5 未來 7 天可動用資金

使用固定方法 2。

若未來 7 天都在同一個月：

`future_available_7d(t) = remaining_budget_month(t) * 7 / days_left_in_month(t)`

若未來 7 天跨月：

- `d1 = 本月從 t 到月底的天數`
- `d2 = 7 - d1`
- `budget_part1 = remaining_budget_month(t) * d1 / days_left_in_month(t)`
- `budget_part2 = monthly_available_cash_u * d2 / next_month_days`
- `future_available_7d(t) = budget_part1 + budget_part2`

### 3.6 風險比值

`risk_ratio(t) = predicted_expense_7d / future_available_7d(t)`

### 3.7 安全下限

若 `future_available_7d(t) <= 0`，一律視為極高風險，避免分母為 0 或負值。

## 4. 重要原則

- `monthly_available_cash_u` 只能用訓練資料計算，不能使用驗證或測試資料。
- `spent_mtd(t)` 只能用日期 `t` 當下以前的真實支出計算，不能使用未來資料。
- 所有模型都必須使用同一套 `future_available_7d(t)` 定義，不可各自改分母。
- 所有模型都必須先做回歸預測，再由 `risk_ratio` 產生分類結果。
- 所有模型輸出的檔案名稱、欄位名稱、summary 格式都必須一致。

## 5. Test 1: alarm or not

目標：二元分類

### 5.1 預測標籤規則

- `alarm = 1` 若 `risk_ratio >= 0.8`
- `alarm = 0` 若 `risk_ratio < 0.8`

### 5.2 真實標籤規則

- `true_alarm = 1` 若 `true_expense_7d / future_available_7d(t) >= 0.8`
- `true_alarm = 0` 否則

### 5.3 評估指標

- `Accuracy`
- `Precision`
- `Recall`
- `F1-score`
- `Confusion Matrix`

輸出檔案：
- `metrics_alarm_binary.json`

## 6. Test 2: no alarm / low risk / mid risk / high risk

目標：四元分類

### 6.1 預測標籤規則

- `no_alarm` 若 `risk_ratio < 0.8`
- `low_risk` 若 `0.8 <= risk_ratio < 1.0`
- `mid_risk` 若 `1.0 <= risk_ratio < 1.2`
- `high_risk` 若 `risk_ratio >= 1.2`

### 6.2 真實標籤規則

使用 `true_expense_7d / future_available_7d(t)` 套用完全相同切點：

- `no_alarm`
- `low_risk`
- `mid_risk`
- `high_risk`

### 6.3 評估指標

- `Accuracy`
- `Macro Precision`
- `Macro Recall`
- `Macro F1`
- `Weighted F1`
- `Confusion Matrix`

輸出檔案：
- `metrics_risk_4class.json`

## 7. Test 3: regression

目標：回歸評估

### 7.1 定義

- `y_pred = predicted_expense_7d`
- `y_true = true_expense_7d`

### 7.2 評估指標

- `MAE = mean(|y_pred - y_true|)`
- `RMSE = sqrt(mean((y_pred - y_true)^2))`
- `MAPE = mean(|(y_pred - y_true) / y_true|) * 100%`，只對 `y_true > 0` 的樣本計算
- 可另外補報 `SMAPE`

輸出檔案：
- `metrics_regression.json`

## 8. predictions.csv 欄位規格

`predictions.csv` 至少必須包含以下欄位：

- `user_id`
- `date`
- `y_true`
- `y_pred`
- `monthly_available_cash`
- `spent_mtd`
- `future_available_7d`
- `true_risk_ratio`
- `pred_risk_ratio`
- `true_alarm`
- `pred_alarm`
- `true_risk_level`
- `pred_risk_level`

若模型需要額外欄位，可以附加，但不可刪除上述欄位。

## 9. summary.txt 統一格式

每個模型的 `summary.txt` 必須使用以下固定格式：

```txt
Model: <model_name>
Output Dir: /Users/liweichen/financial-agent/ml/model_outputs/<model_name>

[Test 1] Binary Alarm
Accuracy: <value>
Precision: <value>
Recall: <value>
F1-score: <value>
Confusion Matrix: [[TN, FP], [FN, TP]]

[Test 2] Risk 4-Class
Accuracy: <value>
Macro Precision: <value>
Macro Recall: <value>
Macro F1: <value>
Weighted F1: <value>
Confusion Matrix:
<4x4 matrix>

[Test 3] Regression
MAE: <value>
RMSE: <value>
MAPE: <value>
SMAPE: <value or N/A>

Notes:
- monthly_available_cash computed from training data only
- spent_mtd computed using only observed spending up to date t
- future_available_7d uses the shared team formula
```

## 10. 全組固定流程

1. 先用回歸模型輸出 `predicted_expense_7d`
2. 用固定公式算 `future_available_7d(t)`
3. 用 `risk_ratio = predicted_expense_7d / future_available_7d(t)` 做 Test 1 和 Test 2
4. 用 `predicted_expense_7d` 對 `true_expense_7d` 做 Test 3

## 11. 一句話版

- Test 1 看警報有沒有響對
- Test 2 看風險等級分對沒
- Test 3 看支出金額預測準不準

## 12. 六個主模型接入共用 evaluator 的必要工作

以下六個模型最後都必須接到共用輸出/評估模組。

共用模組目前預期每個模型至少提供兩份資料：

### 12.1 prediction_input_df

必要欄位：

- `user_id`
- `date`
- `y_true`
- `y_pred`

說明：

- `y_true = true_expense_7d`
- `y_pred = predicted_expense_7d`
- `date = 該 sample 的預測時點 t`

### 12.2 split_metadata_df

必要欄位：

- `user_id`
- `date`
- `split`

說明：

- `split` 至少必須包含 `train`
- 共用 evaluator 會用 `train` 區間去計算 `monthly_available_cash`

### 12.3 六個模型共同最低要求

所有模型在正式接入前，都必須先確認自己能輸出：

- `user_id`
- `date`
- `y_true`
- `y_pred`
- `split`

若做不到，代表該模型還不能直接接共用 evaluator。

### 12.4 各模型目前需要補的事

#### A. `gru_TL_alignment`

目前狀況：

- 已有 `user_id`
- 已能產生 `y_pred`
- 尚未正式保存每個 sample 對應的 `date`

接入前必做：

- 在 preprocessing 階段補存 train/val/test 的 sample metadata
- metadata 至少包含 `user_id`、`date`、`split`
- predict script 要能組出 `prediction_input_df`
- 舊的 `decision_evaluation` 與 `tiered_alert` 不可再視為正式 spec 輸出

#### B. `bilstm_TL_alignment`

目前狀況：

- 已有 `user_id`
- 已能產生 `y_pred`
- 尚未正式保存每個 sample 對應的 `date`

接入前必做：

- 在 preprocessing 階段補存 train/val/test 的 sample metadata
- metadata 至少包含 `user_id`、`date`、`split`
- predict script 要能組出 `prediction_input_df`
- 舊的 `decision_evaluation` 與 `tiered_alert` 不可再視為正式 spec 輸出

#### C. `bigru_TL_alignment`

目前狀況：

- 已有 `user_id`
- 已能產生 `y_pred`
- 尚未正式保存每個 sample 對應的 `date`

接入前必做：

- 在 preprocessing 階段補存 train/val/test 的 sample metadata
- metadata 至少包含 `user_id`、`date`、`split`
- predict script 要能組出 `prediction_input_df`
- 正式輸出必須改寫到 `ml/model_outputs/bigru_TL_alignment`

#### D. `bigru`

目前狀況：

- preprocessing 內部有建立 train/val/test user 順序
- 但目前沒有正式存出 `user_id`
- 也沒有正式存出每個 sample 對應的 `date`

接入前必做：

- preprocessing 必須正式輸出 train/val/test metadata
- metadata 至少包含 `user_id`、`date`、`split`
- evaluate/predict script 必須能組出 `prediction_input_df`
- 不可只留下回歸報告 txt 而沒有標準化輸出

#### E. `bilstm`

目前狀況：

- preprocessing 內部有建立 train/val/test user 順序
- 但目前沒有正式存出 `user_id`
- 也沒有正式存出每個 sample 對應的 `date`
- 後續 decision 腳本仍在重建 user_id，表示 metadata 設計不完整

接入前必做：

- preprocessing 必須正式輸出 train/val/test metadata
- metadata 至少包含 `user_id`、`date`、`split`
- evaluate/predict script 必須能組出 `prediction_input_df`
- 舊的 decision / tiered alert 腳本不可再當正式 spec 流程

#### F. `xgboost_TL_alignment`

目前狀況：

- 已有 `user_id`
- 已有 `date`
- 已能產生 `y_true`
- 已能產生 `y_pred`
- 目前最接近可直接接入共用 evaluator

接入前必做：

- 補出 `split_metadata_df`
- 必須明確標示哪些 row 屬於 `train`，哪些屬於 `test`
- 正式輸出必須改寫到 `ml/model_outputs/xgboost_TL_alignment`

## 13. AI 協作規則

本節是寫給任何協助本專案的 AI。

### 13.0 本專案目前的共用評估程式

本專案目前的共用評估程式是：

- `ml/output_eval_utils.py`

其中正式入口函式是：

- `run_output_evaluation(...)`

此函式的用途是：

- 接收各模型已經算好的 `y_pred`
- 接收 sample-level metadata
- 依本規格自動產生正式輸出

也就是說：

- 模型自己負責「預測」
- `ml/output_eval_utils.py` 負責「統一評估與統一輸出」

### 13.0.1 各模型接入共用評估程式的最低介面

每個模型修改完成後，至少必須能組出以下兩份資料：

#### `prediction_input_df`

必要欄位：

- `user_id`
- `date`
- `y_true`
- `y_pred`

#### `split_metadata_df`

必要欄位：

- `user_id`
- `date`
- `split`

限制：

- `split` 必須至少包含 `train`

### 13.0.2 各模型接入共用評估程式的標準呼叫方式

各模型在完成自己的預測之後，正式流程必須呼叫：

```python
from ml.output_eval_utils import run_output_evaluation

run_output_evaluation(
    model_name="<model_name>",
    prediction_input_df=prediction_input_df,
    split_metadata_df=split_metadata_df,
)
```

其中 `model_name` 必須使用模型資料夾名稱：

- `bigru`
- `bigru_TL_alignment`
- `gru_TL_alignment`
- `bilstm`
- `bilstm_TL_alignment`
- `xgboost_TL_alignment`

### 13.0.3 接入完成後應該產生的正式輸出位置

呼叫 `run_output_evaluation(...)` 後，正式輸出應寫到：

- `ml/model_outputs/<model_name>/`

並且至少包含：

- `metrics_regression.json`
- `metrics_alarm_binary.json`
- `metrics_risk_4class.json`
- `predictions.csv`
- `summary.txt`

### 13.0.4 模型端修改完成的判定標準

若某模型聲稱自己「已經接好共用 evaluator」，AI 或組員必須至少檢查：

1. 該模型的 predict/evaluate script 是否真的有 import：
   `from ml.output_eval_utils import run_output_evaluation`
2. 是否真的有建立 `prediction_input_df`
3. 是否真的有建立 `split_metadata_df`
4. 是否真的有呼叫 `run_output_evaluation(...)`
5. `ml/model_outputs/<model_name>/` 是否真的出現 spec 要求的 5 個正式輸出檔

只要以上任一項不成立，就不能算完成接入。

### 13.0.5 六個模型建議接入點

以下是六個主模型目前最合理的共用 evaluator 接入位置。

#### `bigru`

建議接入 script：

- `ml/bigru/3_evaluate_standard_bigru.py`

原因：

- 這支已經在做最終 test 預測
- 已經有 `y_test_raw`
- 已經會得到 `predictions_real`

接入前還要先補：

- `ml/bigru/1_prepare_data.py` 必須正式存出 train/val/test metadata
- 至少要能提供 `user_id`、`date`、`split`

#### `bigru_TL_alignment`

建議接入 script：

- `ml/bigru_TL_alignment/5_predict_bigru.py`

原因：

- 這支已經會做 ensemble 推論
- 已經會得到 `test_preds`
- 已經是目前最接近正式輸出的入口

接入前還要先補：

- `ml/bigru_TL_alignment/2_preprocess_personal.py` 必須正式存出 sample metadata

#### `gru_TL_alignment`

建議接入 script：

- `ml/gru_TL_alignment/6_predict_aligned.py`

原因：

- 這支已經會做 best combo / ensemble 預測
- 已經會得到 `test_preds`
- 舊的 `7_evaluate_decisions.py` 與 `9_tiered_alert.py` 應退出正式流程

接入前還要先補：

- `ml/gru_TL_alignment/3_preprocess_personal_aligned.py` 必須正式存出 sample metadata

#### `bilstm`

建議接入 script：

- `ml/bilstm/3_evaluate_model.py`

原因：

- 這支已經在做最終 test 預測
- 已經會得到 `predictions_real`
- 舊的 `4_evaluate_decisions.py` 與 `5_tiered_alert.py` 不應繼續作為正式 spec 流程

接入前還要先補：

- `ml/bilstm/1_prepare_data.py` 必須正式存出 train/val/test metadata

#### `bilstm_TL_alignment`

建議接入 script：

- `ml/bilstm_TL_alignment/5_predict_bilstm.py`

原因：

- 這支已經會做 best combo / ensemble 預測
- 已經會得到 `test_preds`
- 舊的 `6_evaluate_decisions.py` 與 `7_tiered_alert.py` 應退出正式流程

接入前還要先補：

- `ml/bilstm_TL_alignment/2_preprocess_personal.py` 必須正式存出 sample metadata

#### `xgboost_TL_alignment`

建議接入 script：

- `ml/xgboost_TL_alignment/test_transfer_aligned.py`

原因：

- 這支已經會輸出 `user_id`、`date`、`y_true`、`y_pred`
- 這支目前最接近直接呼叫共用 evaluator

接入前還要先補：

- 依目前的 train/test 切法建立 `split_metadata_df`
- 讓 `test_transfer_aligned.py` 在算完 prediction dataframe 後直接呼叫 `run_output_evaluation(...)`

### 13.1 AI 在開始動作前必須先反問

若使用者只說「幫我接 spec」、「幫我改輸出」、「幫我接共用 evaluator」這類要求，
AI 不可直接開始修改。

AI 必須先明確反問：

- 「你目前負責的是哪一個模型？」

若使用者沒有明確回答模型名稱，AI 不可直接往下做。

### 13.2 AI 必須先核對模型責任範圍

在得到模型名稱後，AI 必須先依本文件第 12 節檢查該模型目前還缺什麼，
再告訴使用者：

- 這個模型目前已具備哪些條件
- 這個模型目前還缺哪些欄位或 metadata
- 接共用 evaluator 前，必須先補哪些步驟

AI 不可跳過這個核對步驟。

### 13.3 AI 不可假設六個模型狀態相同

AI 必須知道：

- `gru_TL_alignment`
- `bilstm_TL_alignment`
- `bigru_TL_alignment`
- `bigru`
- `bilstm`
- `xgboost_TL_alignment`

這六個模型目前完成度不同，缺口不同。

因此 AI 不可直接套用同一套修改到所有模型。

### 13.4 AI 必須檢查使用者是否真的完成前置工作

若使用者聲稱：

- 已經存了 metadata
- 已經有 `date`
- 已經可以組出 `prediction_input_df`
- 已經有 `split_metadata_df`

AI 不可直接相信，必須先檢查程式或輸出檔是否真的存在。

AI 至少要核對：

- 檔案是否真的有被存出
- 欄位名稱是否正確
- `split` 是否真的包含 `train`
- `prediction_input_df` 是否真的包含 `user_id/date/y_true/y_pred`

若上述條件沒被滿足，AI 必須明講缺口，不可直接繼續。

### 13.5 AI 的正確工作順序

AI 協助本專案時，正確順序必須是：

1. 先反問使用者負責哪個模型
2. 依第 12 節檢查該模型目前缺口
3. 先確認 metadata 是否齊全
4. 再決定要補 preprocessing、predict，還是接 evaluator
5. 最後才允許正式修改與輸出

若 AI 沒有先做第 1 步與第 2 步，代表流程不合格。
