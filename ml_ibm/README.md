# ml_ibm

本目錄是以 **IBM Credit Card Transactions** 作為 pretrain source domain 的遷移學習實驗。
架構與 `ml/`（Walmart source）完全相同，只替換了 pretrain 資料來源。

---

## Source Domain

| 項目 | 說明 |
|---|---|
| 資料集 | IBM Credit Card Transactions |
| Kaggle | `ealtman2019/credit-card-transactions` |
| 規模 | 約 2,400 萬筆交易，~2,000 位美國消費者 |
| 幣值 | USD |
| 粒度 | Transaction-level → 日粒度聚合 |

---

## 目錄結構

```
ml_ibm/
├── ibm_data/                    ← 原始資料（不上傳，自行下載）
│   └── credit_card_transactions-ibm_v2.csv
│
├── processed_data/
│   ├── build_ibm_daily.py       ← 【第一步跑這個】原始資料 → 日粒度
│   └── artifacts/               ← 產出結果（不上傳）
│       └── ibm_daily.csv        ← 所有模型共用的 daily 資料
│
├── bigru/
├── bigru_TL_alignment/
├── bilstm/
├── bilstm_TL_alignment/
├── gru_TL_alignment/
├── xgboost_TL_alignment/
│
├── model_outputs/               ← 最終評估輸出（不上傳）
└── OUTPUT_EVALUATION_SPEC.md    ← 統一輸出規格
```

---

## 上傳規則

**上傳（commit）：**
- 所有 `.py` 腳本
- `README.md`、`requirements.txt`
- `OUTPUT_EVALUATION_SPEC.md`

**不上傳（.gitignore 已設定）：**
- `ibm_data/` — 原始 CSV 太大
- `processed_data/artifacts/` — 中間產物
- `**/artifacts*/` — 所有模型的 pretrain/finetune 產出
- `**/models/` — 訓練好的模型權重
- `**/results/` — 評估結果
- `model_outputs/` — 最終 spec 輸出

---

## 快速開始

### 1. 下載原始資料

至 Kaggle 下載 `ealtman2019/credit-card-transactions`，
將 `transactions.csv` 重新命名為 `credit_card_transactions-ibm_v2.csv` 並放到：

```
ml_ibm/ibm_data/credit_card_transactions-ibm_v2.csv
```

### 2. 產生共用日粒度資料（所有模型都要先跑這步）

```bash
cd ml_ibm/processed_data
pip install -r requirements.txt
python build_ibm_daily.py
```

輸出：`ml_ibm/processed_data/artifacts/ibm_daily.csv`

### 3. 跑各模型的 pretrain

每個模型各自讀取 `ibm_daily.csv` 做 pretrain，詳見各模型資料夾的腳本。

```bash
# DL 模型範例（bigru_TL_alignment）
cd ml_ibm/bigru_TL_alignment
python 1_preprocess_ibm.py    # ibm_daily.csv → sliding windows
python 3_pretrain_bigru.py    # pretrain
python 4_finetune_bigru.py    # finetune on personal data
python 5_predict_bigru.py     # 產生預測

# XGBoost
cd ml_ibm/xgboost_TL_alignment
python preprocess_ibm_common_aligned.py
python train_ibm_base_aligned.py
python finetune_own_aligned.py
```

### 4. 產生最終評估輸出

各模型跑完後呼叫共用 evaluator：

```python
from ml.output_eval_utils import run_output_evaluation

run_output_evaluation(
    model_name="<model_name>",
    prediction_input_df=prediction_input_df,
    split_metadata_df=split_metadata_df,
    output_root=Path("ml_ibm/model_outputs"),
)
```

輸出位置：`ml_ibm/model_outputs/<model_name>/`

---

## ibm_daily.csv 欄位說明

所有模型共用同一份 `ibm_daily.csv`，欄位如下：

| 欄位 | 說明 |
|---|---|
| `user_id` | 用戶 ID（0~1999） |
| `date` | 日期 |
| `daily_expense` | 當日支出（log1p 壓縮） |
| `daily_income` | 固定為 0（IBM 無收入資料） |
| `txn_count` | 當日交易筆數 |
| `daily_net` | = `-daily_expense` |
| `dow` | 星期幾（0=週一） |
| `is_weekend` | 0 或 1 |
| `day` | 幾號（1~31） |
| `month` | 幾月（1~12） |
| `expense_7d_sum` | 7 日支出總和 |
| `expense_7d_mean` | 7 日支出平均 |
| `expense_30d_sum` | 30 日支出總和 |
| `expense_30d_mean` | 30 日支出平均 |
| `zscore_7d` | 7 日 z-score（幣值對齊用） |
| `zscore_14d` | 14 日 z-score |
| `zscore_30d` | 30 日 z-score |
| `target` | 未來 7 天支出總和（log1p） |

> `daily_expense` 已做 log1p 壓縮，目的是縮小 USD/TWD 幣值差距，幫助 pretrain 與 finetune 特徵分佈對齊。

---

## 注意事項

- `daily_income = 0` 不影響模型預測，income 只在最終評估階段計算 `monthly_available_cash` 時使用，該值從個人原始資料取得。
- 個人原始資料（`raw_transactions_*.xlsx`）共用 `ml/processed_data/` 的輸出，不需要另外處理。
- 評估規格請參考 `OUTPUT_EVALUATION_SPEC.md`。
