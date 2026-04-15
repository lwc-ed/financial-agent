# gru_hgbr

這是舊版 hybrid 實驗：GRU embedding + HGBR。

目前已移到 `legacy_models/`，代表：
- 保留可重跑
- 不再是主要維護主線

## 現在依賴什麼

### 共用資料
- `../../processed_data/artifacts/features_all.csv`

### GRU 既有 artifacts
- `../ml_gru/artificats/personal_feature_scaler.pkl`
- `../ml_gru/artificats/personal_target_scaler.pkl`
- `../ml_gru/artificats/personal_user_clip_values.pkl`
- `../ml_gru/artificats/finetune_gru.pth`

## 主要流程

1. `preprocess.py`
讀取共用 `features_all.csv`，建立：
- GRU 序列輸入
- HGBR 扁平特徵

2. `extract_embeddings.py`
用 frozen GRU 提取 embedding。

3. `train_hgbr.py`
訓練 HGBR。

4. `predict.py`
做完整評估。

## 重跑方式

```bash
cd /Users/liweichen/financial-agent/ml/legacy_models/gru_hgbr
python preprocess.py
python extract_embeddings.py
python train_hgbr.py
python predict.py
```

## 備註

這個資料夾最容易壞的地方是它同時依賴：
- `processed_data`
- `ml_gru/artificats`

如果你之後要再整理 `ml_gru` 的 artifacts 結構，這個資料夾要一起修。
