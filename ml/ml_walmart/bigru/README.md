# bigru

`bigru/` 是不走 transfer learning 的 Bi-GRU baseline。

流程：

1. `1_prepare_data.py`
從 `ml/data/raw_transactions_*.xlsx` 直接整理個人原始交易資料，輸出 `artifacts/my_X_*` 與 `my_y_*`。

2. `2_train_bigru.py`
直接用 baseline 特徵訓練 Bi-GRU，不做 pretrain / alignment / finetune。

3. `3_evaluate_standard_bigru.py`
讀取 `artifacts/` 內的測試資料與模型，輸出 baseline 評估結果。

這條線的用途是做「從原始個人資料直接訓練」的對照組。
