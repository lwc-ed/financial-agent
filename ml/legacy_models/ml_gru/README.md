# legacy_models/ml_gru

這個資料夾已改列為 legacy 模型，但仍可重跑。

任務：
- 用過去 30 天序列
- 預測未來 7 天總支出 `future_expense_7d_sum`

## 資料來源

`ml_gru` 不再維護自己的共用 `features_all.csv`。  
目前統一讀：

- `../../processed_data/artifacts/features_all.csv`

這份特徵表由 `ml/processed_data/` 生成。

## 主要流程

1. `preprocess_wallmart.py`
處理 Walmart 資料，準備 pretrain 用資料。

2. `pretrain_gru.py`
用 Walmart 資料做 pretrain。

3. `preprocess_personal.py`
讀取 `processed_data/artifacts/features_all.csv`，建立個人資料的視窗、split、scaler。

4. `finetune_gru.py`
把 pretrain 權重微調到個人資料。

5. `predict.py`
評估主版模型並輸出 metrics / predictions。

另外還有：
- `predict_v1.py` ~ `predict_v5.py`
- `predict_nopretrain.py`
- 對應不同版本實驗

## 重要輸入

共用輸入：
- `../../processed_data/artifacts/features_all.csv`

模型內部輸入 / 輸出：
- `artificats/personal_X_*.npy`
- `artificats/personal_y_*.npy`
- `artificats/personal_feature_scaler.pkl`
- `artificats/personal_target_scaler.pkl`
- `artificats/personal_user_clip_values.pkl`
- `artificats/finetune_gru*.pth`
- `artificats/metrics*.json`

注意：資料夾名稱仍沿用既有拼字 `artificats/`。

## 如何重跑

### 主流程

```bash
cd /Users/liweichen/financial-agent/ml/legacy_models/ml_gru
./run_pipeline.sh
```

### 多版本流程

```bash
./run_pipeline_v1.sh
./run_pipeline_v2_to_v5.sh
./run_nopretrain.sh
```

## 維護原則

1. 若只改共用特徵表，先改 `processed_data/build_features.py`。
2. 若改 GRU 視窗切法、clip、scaler，改 `preprocess_personal.py`。
3. 不要再把 `ml_gru` 當共用資料來源；它現在只是 legacy 模型消費者。
