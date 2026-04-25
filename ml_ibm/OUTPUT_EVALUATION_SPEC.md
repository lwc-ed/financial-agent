# Output And Evaluation Spec — ml_ibm

本文件是 `ml_ibm/` 目錄下所有模型的唯一正式輸出與評估規格。

**與 `ml/` 的差異：**
- Source domain 由 **Walmart** 改為 **IBM Credit Card Transactions**
  （Kaggle: `ealtman2019/credit-card-transactions`，2,400 萬筆，~2,000 位美國消費者）
- 所有輸出路徑前綴改為 `ml_ibm/`
- 個人原始資料（`raw_transactions_*.xlsx`）仍共用 `ml/processed_data/` 的輸出

若本文件與任何腳本的 README、註解或歷史輸出格式衝突，一律以本文件為準。

適用模型：
- `bigru`
- `bigru_TL_alignment`
- `gru_TL_alignment`
- `bilstm`
- `bilstm_TL_alignment`
- `xgboost_TL_alignment`

---

## 1. 統一輸出位置

所有模型的評估輸出必須統一寫入：

`/Users/liweichen/financial-agent/ml_ibm/model_outputs`

每個模型必須在此資料夾下建立自己的子資料夾：

- `model_outputs/bigru`
- `model_outputs/bigru_TL_alignment`
- `model_outputs/gru_TL_alignment`
- `model_outputs/bilstm`
- `model_outputs/bilstm_TL_alignment`
- `model_outputs/xgboost_TL_alignment`

---

## 2. 每個模型必須輸出的檔案

- `metrics_regression.json`
- `metrics_alarm_binary.json`
- `metrics_risk_4class.json`
- `predictions.csv`
- `summary.txt`

---

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

---

## 4. 重要原則

- `monthly_available_cash_u` 只能用訓練資料計算，不能使用驗證或測試資料。
- `spent_mtd(t)` 只能用日期 `t` 當下以前的真實支出計算，不能使用未來資料。
- 所有模型都必須使用同一套 `future_available_7d(t)` 定義，不可各自改分母。
- 所有模型都必須先做回歸預測，再由 `risk_ratio` 產生分類結果。
- 所有模型輸出的檔案名稱、欄位名稱、summary 格式都必須一致。

---

## 5. Test 1：Alarm or Not（二元分類）

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

輸出檔案：`metrics_alarm_binary.json`

---

## 6. Test 2：四分類風險等級

### 6.1 預測標籤規則

- `no_alarm`  若 `risk_ratio < 0.8`
- `low_risk`  若 `0.8 <= risk_ratio < 1.0`
- `mid_risk`  若 `1.0 <= risk_ratio < 1.2`
- `high_risk` 若 `risk_ratio >= 1.2`

### 6.2 真實標籤規則

使用 `true_expense_7d / future_available_7d(t)` 套用完全相同切點。

### 6.3 評估指標

- `Accuracy`
- `Macro Precision`
- `Macro Recall`
- `Macro F1`
- `Weighted F1`
- `Confusion Matrix`

輸出檔案：`metrics_risk_4class.json`

---

## 7. Test 3：Regression

### 7.1 定義

- `y_pred = predicted_expense_7d`
- `y_true = true_expense_7d`

### 7.2 評估指標

- `MAE = mean(|y_pred - y_true|)`
- `RMSE = sqrt(mean((y_pred - y_true)^2))`
- `MAPE = mean(|(y_pred - y_true) / y_true|) * 100%`（只對 `y_true > 0` 的樣本計算）
- 可另外補報 `SMAPE`

輸出檔案：`metrics_regression.json`

---

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

---

## 9. summary.txt 統一格式

```txt
Model: <model_name>
Source Domain: IBM Credit Card Transactions (ealtman2019/credit-card-transactions)
Output Dir: /Users/liweichen/financial-agent/ml_ibm/model_outputs/<model_name>

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
- Source domain: IBM Credit Card Transactions (~24M txns, ~2000 users)
- monthly_available_cash computed from training data only
- spent_mtd computed using only observed spending up to date t
- future_available_7d uses the shared team formula
```

---

## 10. 全組固定流程

1. 先用回歸模型輸出 `predicted_expense_7d`
2. 用固定公式算 `future_available_7d(t)`
3. 用 `risk_ratio = predicted_expense_7d / future_available_7d(t)` 做 Test 1 和 Test 2
4. 用 `predicted_expense_7d` 對 `true_expense_7d` 做 Test 3

---

## 11. 共用 Evaluator

本專案的共用評估程式為：

```
ml/output_eval_utils.py
```

`ml_ibm/` 的模型直接引用同一支，不另行複製。引用時使用：

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.output_eval_utils import run_output_evaluation

run_output_evaluation(
    model_name="<model_name>",
    prediction_input_df=prediction_input_df,
    split_metadata_df=split_metadata_df,
    output_root=Path(__file__).resolve().parents[1] / "model_outputs",
)
```

其中 `model_name` 必須使用模型資料夾名稱：
- `bigru`
- `bigru_TL_alignment`
- `gru_TL_alignment`
- `bilstm`
- `bilstm_TL_alignment`
- `xgboost_TL_alignment`

---

## 12. 各模型接入共用 Evaluator 的最低介面

### 12.1 prediction_input_df

必要欄位：

| 欄位 | 說明 |
|---|---|
| `user_id` | 使用者 ID |
| `date` | 預測時點 t |
| `y_true` | 真實未來 7 天支出 |
| `y_pred` | 模型預測未來 7 天支出 |

### 12.2 split_metadata_df

必要欄位：

| 欄位 | 說明 |
|---|---|
| `user_id` | 使用者 ID |
| `date` | 日期 |
| `split` | `train` / `valid` / `test` |

限制：`split` 必須至少包含 `train`（evaluator 用 train 區間計算 `monthly_available_cash`）。

---

## 13. 資料切分規格

所有模型統一採用 **per-user 70/15/15** 時間序切分：

```
每位 user：
├── 前 70% → train
├── 中 15% → valid（用於 early stopping / 超參數選擇）
└── 後 15% → test（唯一正式評估集）
```

- 切分必須在每位 user 的時間序內各自進行，不可跨 user 混合切分。
- Test set 必須是每位 user 最晚的 15% 資料，確保無資料洩漏。

---

## 14. Source Domain 說明（IBM vs Walmart）

| 項目 | ml/（Walmart） | ml_ibm/（IBM） |
|---|---|---|
| Source 資料集 | Walmart Weekly Sales | IBM Credit Card Transactions |
| 資料量 | ~45 stores × 143 週 | ~2,000 users × 數十年 |
| 資料粒度 | 週 → 日（÷7） | Transaction → 日（聚合） |
| 特徵對齊方式 | zscore / pct_rank / volatility | 相同（需重新 fit scaler） |
| 預訓練腳本 | `1_preprocess_walmart.py` | `1_preprocess_ibm.py`（待建立） |

`ml_ibm/` 的 pretrain 資料前處理腳本尚待建立，建立時請參考 `ml/` 對應的 `1_preprocess_walmart.py` 修改欄位對應。

---

## 15. 接入完成的判定標準

若某模型聲稱「已經接好共用 evaluator」，必須至少確認：

1. predict/evaluate script 有 import `from ml.output_eval_utils import run_output_evaluation`
2. 有建立 `prediction_input_df`（含 `user_id / date / y_true / y_pred`）
3. 有建立 `split_metadata_df`（含 `user_id / date / split`，且 `split` 包含 `train`）
4. 有呼叫 `run_output_evaluation(...)`，且 `output_root` 指向 `ml_ibm/model_outputs/`
5. `ml_ibm/model_outputs/<model_name>/` 出現 5 個正式輸出檔

只要以上任一項不成立，就不能算完成接入。

---

## 16. AI 協作規則

### 16.1 開始動作前必須先反問

若使用者只說「幫我接 spec」、「幫我改輸出」這類要求，AI 不可直接開始修改。

AI 必須先明確反問：「你目前負責的是哪一個模型？」

### 16.2 必須先核對模型缺口

在得到模型名稱後，AI 必須依第 12 節檢查該模型目前還缺什麼，並告知：
- 目前已具備哪些條件
- 目前還缺哪些欄位或 metadata
- 接共用 evaluator 前，必須先補哪些步驟

### 16.3 不可假設六個模型狀態相同

六個模型完成度不同，AI 不可直接套用同一套修改到所有模型。

### 16.4 必須確認輸出路徑正確

`ml_ibm/` 的模型輸出必須寫到 `ml_ibm/model_outputs/`，
不可誤寫到 `ml/model_outputs/`。

### 16.5 正確工作順序

1. 先反問使用者負責哪個模型
2. 依第 12 節檢查該模型目前缺口
3. 先確認 metadata 是否齊全
4. 再決定要補 preprocessing、predict，還是接 evaluator
5. 最後才允許正式修改與輸出

若 AI 沒有先做第 1 步與第 2 步，代表流程不合格。
