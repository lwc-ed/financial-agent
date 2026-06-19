# ml_temp — 無 Domain Alignment 對照 pipeline

目的：跑一條**完全沒有三層 Domain Alignment**的 BiGRU 遷移學習管線，
與 aligned 版（`ml_ibm/bigru_TL_alignment`）比較 MAE / RMSE / Binary F1 / Weighted F1，
量化「Domain Alignment」對遷移效果的貢獻。

## 設計原則

- **完全自包含在 `ml_temp/`**，所有產物寫到 `ml_temp/artifacts_noalign/` 與 `ml_temp/model_outputs/`。
- **不修改** `ml_ibm/bigru_TL_alignment` 任何程式碼；僅以 `sys.path` **import 重用**：
  - `model_bigru.BiGRUWithAttention`（模型架構）
  - `alignment_utils`（`load_personal_daily` / `INPUT_DAYS` / `TARGET_COL`）
  - `ml_walmart/output_eval_utils`（正式評估器，與 aligned 同一套指標）
- 與 aligned 版**唯一差異 = 特徵**：用 `no_alignment_utils.compute_raw_features`
  （原始絕對金額與滾動統計），不做 z-score / 百分位 / sin-cos 對齊。

## 檔案

| 檔案 | 作用 |
|---|---|
| `no_alignment_utils.py` | raw 特徵 `compute_raw_features` + `RAW_FEATURE_COLS`(10) + IBM 抽樣讀取 |
| `1_preprocess_ibm.py` | IBM(raw) → 滑動視窗 → `artifacts_noalign/ibm_*.npy` |
| `2_preprocess_personal.py` | 個人(raw) → per-user 70/15/15 + metadata |
| `3_pretrain.py` | 在 IBM(raw) 預訓練 BiGRU → `pretrain_bigru.pth` |
| `4_finetune.py` | 個人微調（兩階段、30 seeds）→ `finetune_bigru_seed*.pth` |
| `5_predict.py` | 30-seed ensemble + 正式評估 → `model_outputs/bigru_noalign/` |
| `compare_domain_gap.py` | （另用）只看特徵分布的 domain gap 對齊前後對照 + 圖 |
| `run_all.sh` | 一鍵依序執行 1→5 |

## 執行（需 GPU / 足夠記憶體；預訓練在 5.7M 視窗上很重）

```bash
cd ml/ml_temp
bash run_all.sh
# 或手動：
../../.venv/bin/python 1_preprocess_ibm.py
../../.venv/bin/python 2_preprocess_personal.py
../../.venv/bin/python 3_pretrain.py
../../.venv/bin/python 4_finetune.py
../../.venv/bin/python 5_predict.py
```

## 比較

跑完後對照兩份報告即得「有無 Domain Alignment」的效能差距：
- 無對齊：`ml_temp/model_outputs/bigru_noalign/`
- 有對齊：`ml_ibm/model_outputs/bigru_TL_alignment/`

> 注意：`5_predict.py` 採乾淨的全 seed ensemble，不含 aligned 版專屬的
> calibration / 最佳 seed 組合搜尋，以確保對照基準公平。
