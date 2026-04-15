# ml_gru_nopretrain

這是舊版對照組：不使用 Walmart pretrain，直接從個人資料訓練。

目前已移到 `legacy_models/`，主要用途是：
- 做 no-pretrain baseline
- 和 `ml_gru` 主線比較

## 共用資料來源

- `../../processed_data/artifacts/features_all.csv`

## 主要流程

1. `preprocess.py`
從共用 `features_all.csv` 產生：
- HGBR 用的扁平特徵
- GRU from scratch 用的序列資料

2. `train_hgbr.py`
訓練 HGBR baseline。

3. `train_gru_scratch.py`
從頭訓練 GRU。

4. `predict.py`
比較 HGBR、GRU scratch 與 baseline。

其他補充實驗：
- `experiment_exclude_users.py`
- `experiment_gru_exclude.py`

## 重跑方式

```bash
cd /Users/liweichen/financial-agent/ml/legacy_models/ml_gru_nopretrain
python preprocess.py
python train_hgbr.py
python train_gru_scratch.py
python predict.py
```

## 備註

這條線現在已改成讀 `processed_data` 的共用特徵表，不再依賴 `ml_gru/features_all.csv`。
