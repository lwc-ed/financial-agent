# gru_lower_model_peruser_finetune

這是舊版實驗：先訓練一個較小的全域 GRU，再對每個 user 做 per-user fine-tune。

目前已移到 `legacy_models/`，代表它是保留的對照組，不是主要主線。

## 共用資料來源

- `../../processed_data/artifacts/features_all.csv`

## 主要流程

1. `preprocess.py`
從共用 `features_all.csv` 產生：
- 全域資料
- per-user 資料字典

2. `train_global.py`
先訓練一個全域小型 GRU。

3. `finetune_peruser.py`
對每個 user 各自微調。

4. `predict.py`
比較全域模型、per-user 模型與 baseline。

## 重跑方式

```bash
cd /Users/liweichen/financial-agent/ml/legacy_models/gru_lower_model_peruser_finetune
python preprocess.py
python train_global.py
python finetune_peruser.py
python predict.py
```

## 備註

這條線已經改成讀 `processed_data` 的共用特徵表，不再依賴 `ml_gru/features_all.csv`。
