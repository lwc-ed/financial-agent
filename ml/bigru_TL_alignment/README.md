# bigru_TL_alignment

`bigru_TL_alignment/` 的目標是成為真正的 Bi-GRU transfer learning / alignment pipeline。

目前資料夾內現況：

1. `1_prepare_data.py`
從 `processed_data/artifacts/features_all.csv` 產生 personal alignment features，輸出到 `artifacts_aligned/`。

2. `4_train_aligned_bigru.py`
直接用 personal aligned data 訓練 Bi-GRU。

3. `5_evaluate_aligned_bigru.py`
評估 personal aligned Bi-GRU。

也就是說，現在還沒有真正的 TL，只有 alignment feature engineering。

## 要做的事情

1. 補 Walmart preprocessing
建立 source domain 的 sequence data，讓 Walmart 和 personal 兩邊使用相容的 Bi-GRU 輸入格式。

2. 補 pretrain 腳本
先用 Walmart/source domain 預訓練 Bi-GRU backbone。

3. 補 personal finetune 腳本
讀取 pretrain 權重，再用 personal aligned data 做微調。

4. 明確切開 source / target artifacts
例如拆成 `artifacts_walmart/`、`artifacts_personal/` 或其他更清楚的結構，避免現在全部混在 `artifacts_aligned/` 的命名邏輯。

5. 補完整 pipeline 文件
整理成一致的 `preprocess -> pretrain -> finetune -> evaluate` 執行順序。

6. 跟 baseline `ml/bigru/` 做對照
`ml/bigru/` 應保持 no-TL baseline。這裡應該是 TL / alignment 版本，之後比較結果才有意義。

如果要跑不含 TL 的 baseline，請改到 `ml/bigru/`。
