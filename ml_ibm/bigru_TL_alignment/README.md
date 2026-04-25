# bigru_TL_alignment

`bigru_TL_alignment/` 現在已補成 Bi-GRU transfer learning / alignment pipeline。

## Canonical TL Pipeline

1. `1_preprocess_walmart.py`
用和 target 相同的 10 個 aligned features 建立 Walmart source domain 序列資料。

2. `2_preprocess_personal.py`
用同一套 aligned features 建立 personal target domain 序列資料。

3. `3_pretrain_bigru.py`
先在 Walmart 上做 Bi-GRU pretrain，並用 personal train features 做 MMD alignment。

4. `4_finetune_bigru.py`
載入 `pretrain_bigru.pth`，再用 personal data finetune，同時保持 MMD alignment。

5. `5_predict_bigru.py`
用 finetune 後的多 seed 模型做推論，暴力搜尋最佳 ensemble 組合並輸出 metrics。

主要輸出會放在：

- `artifacts_bigru_tl/`

其中包括：

- `walmart_X_*.npy`
- `personal_X_*.npy`
- `pretrain_bigru.pth`
- `finetune_bigru_seed*.pth`
- `result.txt`
- `metrics.json`

## 建議執行順序

```bash
cd /Users/liweichen/financial-agent/ml/bigru_TL_alignment
python3 1_preprocess_walmart.py
python3 2_preprocess_personal.py
python3 3_pretrain_bigru.py
python3 4_finetune_bigru.py
python3 5_predict_bigru.py
```

如果要跑不含 TL 的 baseline，請改到 `ml/bigru/`。
