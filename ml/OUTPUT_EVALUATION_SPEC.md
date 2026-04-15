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



