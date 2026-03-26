# ml_gru

這個資料夾是用 GRU 做「未來 7 天總消費預測」的實驗版本，流程分成兩段：

1. 用 Walmart 公開資料做 `pretrain`
2. 用個人記帳特徵做 `finetune`

目標欄位是 `future_expense_7d_sum`，也就是未來 7 天總支出。

## 資料夾結構

- `preprocess_wallmart.py`: Walmart 資料前處理，建立 pretrain 用的序列資料
- `pretrain_gru.py`: 用 Walmart 資料預訓練 GRU
- `preprocess_personal.py`: 個人記帳特徵前處理，建立 finetune 用的序列資料
- `finetune_gru.py`: 載入 pretrain 權重後，用個人資料做微調
- `predict.py`: 用 finetune 完的模型跑 validation / test，輸出 metrics 與 predictions
- `run_pipeline.sh`: 重跑整條 pipeline 的 shell script
- `features_all.csv`: 個人資料特徵表
- `wallmart/`: Walmart 原始資料
- `artificats/`: 模型、scaler、npy、metrics、predictions 等輸出

注意：資料夾名稱目前是 `artificats/`，拼字沿用既有專案，不是 `artifacts/`。

## 模型設定

模型在 [`pretrain_gru.py`](/Users/liweichen/financial-agent/ml_gru/pretrain_gru.py) 與 [`finetune_gru.py`](/Users/liweichen/financial-agent/ml_gru/finetune_gru.py) 中定義，結構一致：

- 模型類型：`GRU`
- `input_size = 7`
- `hidden_size = 64`
- `num_layers = 2`
- `dropout = 0.2`
- `output_size = 1`

輸入張量形狀：

- `X`: `(樣本數, 30天, 7特徵)`
- `y`: `(樣本數, 1)`

也就是用過去 `30` 天的特徵序列，預測未來 `7` 天總支出。

## 目前使用的特徵

個人資料與 Walmart 資料目前都對齊成 7 個特徵：

- `daily_expense`
- `expense_7d_mean`
- `expense_30d_sum`
- `has_expense`
- `has_income`
- `net_30d_sum`
- `txn_30d_sum`

目標欄位：

- `future_expense_7d_sum`

## 訓練流程

### 1. Walmart pretrain

[`preprocess_wallmart.py`](/Users/liweichen/financial-agent/ml_gru/preprocess_wallmart.py) 會做以下事情：

- 讀取 `wallmart/train.csv`、`features.csv`、`stores.csv`
- 先把週資料彙整成 store-level weekly sales
- 將週資料展成日資料
- 建立與個人資料對齊的 7 個特徵
- 建立 `30` 天 sliding windows
- 切成 `train / val / test = 70 / 15 / 15`
- 用 `train` fit 全域 clipping 與 scaler
- 輸出 `walmart_X_train.npy`、`walmart_y_train.npy`、`walmart_X_val.npy`、`walmart_y_val.npy`、`walmart_X_test.npy`、`walmart_y_test.npy`

[`pretrain_gru.py`](/Users/liweichen/financial-agent/ml_gru/pretrain_gru.py) 會：

- 載入 Walmart 的 train / val npy
- 用 `MSELoss` 訓練 GRU
- optimizer：`Adam`
- learning rate：`0.001`
- batch size：`64`
- epochs：`50`
- scheduler：`ReduceLROnPlateau`
- gradient clipping：`1.0`
- early stopping patience：`10`

輸出：

- `artificats/pretrain_gru.pth`
- `artificats/pretrain_history.pkl`

### 2. Personal finetune

[`preprocess_personal.py`](/Users/liweichen/financial-agent/ml_gru/preprocess_personal.py) 會：

- 讀取 `features_all.csv`
- 依照 `user_id` 與 `date` 排序
- 每個 user 各自建立 `30` 天 sliding windows
- 每個 user 各自做時間切分：`70 / 15 / 15`
- 每個 user 只用自己的 `train` 視窗估 `P95` clipping threshold
- 將這個 user 的 threshold 套到自己的 `train / val / test`
- 再用所有 clipped 後的 train 資料 fit 全域 `StandardScaler`
- 將 scaler 套到 val / test

這樣做的目的：

- 避免 validation / test 洩漏到前處理統計
- 避免高消費 user 的極端值主導所有人的 clipping threshold

目前 personal preprocess 會輸出：

- `personal_X_train.npy`
- `personal_y_train.npy`
- `personal_X_val.npy`
- `personal_y_val.npy`
- `personal_X_test.npy`
- `personal_y_test.npy`
- `personal_train_user_ids.npy`
- `personal_val_user_ids.npy`
- `personal_test_user_ids.npy`
- `personal_feature_scaler.pkl`
- `personal_target_scaler.pkl`
- `personal_user_clip_values.pkl`

[`finetune_gru.py`](/Users/liweichen/financial-agent/ml_gru/finetune_gru.py) 會：

- 載入 `pretrain_gru.pth`
- 建立相同結構的 GRU
- 直接繼承全部 pretrain 權重
- 凍結第二層 GRU 的 `weight_hh_l1` 與 `bias_hh_l1`
- 其餘層繼續訓練，以適應個人消費模式

訓練設定：

- loss：`MSELoss`
- optimizer：`Adam`
- learning rate：`0.0005`
- batch size：`16`
- epochs：`150`
- scheduler：`ReduceLROnPlateau`
- gradient clipping：`1.0`
- early stopping patience：`20`

輸出：

- `artificats/finetune_gru.pth`
- `artificats/finetune_history.pkl`

## 評估方式

[`predict.py`](/Users/liweichen/financial-agent/ml_gru/predict.py) 會讀取 finetune 模型與 personal scaler，然後在 validation / test 上評估。

目前輸出指標包含：

- `val_mae`
- `val_rmse`
- `val_smape`
- `val_per_user_nmae`
- `test_mae`
- `test_rmse`
- `test_smape`
- `test_per_user_nmae`

另外也會和 baseline 比較：

- `naive_7d_sum`
- `moving_avg_30d_x7`

相關輸出檔案：

- `artificats/metrics.json`
- `artificats/training_summary.txt`
- `artificats/predictions_val.csv`
- `artificats/predictions_test.csv`

## 風險評估

`predict.py` 裡還有一段示範性的風險評估邏輯，會把「預測未來 7 天支出」除以「目前帳戶餘額」，分成：

- `安全`: `< 30%`
- `注意`: `< 50%`
- `警告`: `< 80%`
- `危險`: `>= 80%`

這一段目前是 demo 用，餘額在腳本內是手動設定，不是正式系統串接版本。

## 如何重跑

### 方式一：直接跑整條 pipeline

```bash
cd /Users/liweichen/financial-agent/ml_gru
./run_pipeline.sh
```

`run_pipeline.sh` 會依序執行：

1. `preprocess_wallmart.py`
2. `pretrain_gru.py`
3. `preprocess_personal.py`
4. `finetune_gru.py`
5. `predict.py`

### 方式二：手動逐步跑

```bash
cd /Users/liweichen/financial-agent/ml_gru
/Users/liweichen/financial-agent/.venv/bin/python preprocess_wallmart.py
/Users/liweichen/financial-agent/.venv/bin/python pretrain_gru.py
/Users/liweichen/financial-agent/.venv/bin/python preprocess_personal.py
/Users/liweichen/financial-agent/.venv/bin/python finetune_gru.py
/Users/liweichen/financial-agent/.venv/bin/python predict.py
```

## 目前設計上的重點與限制

- 這版是 transfer learning：先學 Walmart，再微調到個人資料
- 個人資料目前是 per-user 時間切分，不是全資料混在一起切
- 個人資料目前有 per-user clipping，但不是直接刪除 outlier row
- evaluation 現在有絕對誤差與相對誤差兩套指標
- feature set 是目前固定的 7 個特徵，未來仍可再做 feature engineering 或 feature search

## 建議交接時補充的口頭說明

- 如果要改 preprocessing，優先確認有沒有破壞 train / val / test 的切分一致性
- 如果要改 relative metric，先確認 `predict.py` 裡的公式與業務定義一致
- 如果要重新比較版本，請固定同一份 `features_all.csv`、同一組 split 邏輯，再看 metrics

---

## Claude 優化實驗紀錄（V0 → V5）

以下為 Claude 對 GRU 模型進行的迭代優化實驗，共 6 個版本，每版均保留對應的 `.pth` 權重與 `claude_verN.txt` 檢討報告。

### Baseline 比較基準

| 方法 | Test MAE |
|------|----------|
| Naive 7d Sum | 8,130 元 |
| Moving Avg 30d | 7,019 元 |

### 各版本實驗結果

| 版本 | 關鍵改動 | Val MAE | Test MAE | Test RMSE | Test SMAPE | Test NMAE |
|------|---------|--------:|--------:|----------:|-----------:|----------:|
| **V0**（原始基準） | 原始 GRU，hidden=64，MSELoss，Adam，部分凍結 | 4,624 | 5,953 | 14,906 | 90.18% | 139.68% |
| **V1** | GRUWithAttention，hidden=128，HuberLoss，AdamW，100 epochs | 4,584 | 6,050 | 15,862 | 78.44% | 138.14% |
| **V2** | hidden=64（減少過擬合），dropout=0.4，Gradual Unfreezing，CosineAnnealingLR | 4,303 | 6,259 | 15,855 | 83.92% | 410.02% |
| **V3** | Log1p 目標轉換，Gaussian 輸入噪聲增強（σ=0.02），複用 V2 pretrain | 4,476 | 6,347 | 17,265 | 91.63% | 61.40% |
| **V4** | LLRD（層別學習率衰減），混合損失（0.7×Huber+0.3×MSE），CosineAnnealingWarmRestarts | 4,541 | **5,773** | 14,513 | 82.23% | 396.20% |
| **V5** | 3 模型 Ensemble（seeds 42/123/777），Log1p+Noise+LLRD，Per-user 偏差修正 | 4,360 | 5,821 | 15,299 | **69.82%** | **60.84%** |

### 各指標最佳版本

| 指標 | 最佳版本 | 數值 | vs V0 改善幅度 |
|------|---------|------|--------------|
| Test MAE（絕對誤差最小） | **V4** | 5,773 元 | ▼ 3.0% |
| Test SMAPE（相對誤差最小） | **V5** | 69.82% | ▼ 22.6 pp |
| Test NMAE（跨用戶標準化誤差最小） | **V5** | 60.84% | ▼ 56.4 pp |
| Val MAE（驗證集最低） | **V2** | 4,303 元 | ▼ 7.0% |

### 各版本技術摘要

#### V0：原始基準
- 架構：標準 GRU（hidden=64, layers=2, dropout=0.2）
- 訓練：MSELoss、Adam（lr=5e-4）、部分凍結（weight_hh_l1）
- 問題：SMAPE 高達 90%，NMAE 不穩定

#### V1：Attention + 擴容
- **新增**：Temporal Attention（softmax over hidden states）、LayerNorm、fc1→fc2 兩層輸出頭
- **新增**：hidden 從 64 擴至 128，HuberLoss，AdamW
- **結果**：SMAPE 大幅改善（90% → 78%），但 hidden=128 參數量 160K 對 4067 筆訓練資料仍過大，導致 early stop 在第 5 epoch
- **檔案**：`pretrain_gru_v1.py`、`finetune_gru_v1.py`、`predict_v1.py`

#### V2：縮容 + Gradual Unfreezing
- **新增**：hidden 從 128 縮回 64，dropout 0.3→0.4，weight_decay 1e-4→5e-4
- **新增**：Gradual Unfreezing（ULMFiT 風格）：Phase 1 凍結 GRU 60 epochs → Phase 2 全量微調 140 epochs
- **結果**：Val MAE 最低（4,303），但 Test NMAE 異常飆升至 410%（Val/Test 分佈差距問題）
- **檔案**：`pretrain_gru_v2.py`、`finetune_gru_v2.py`、`predict_v2.py`

#### V3：Log1p 目標轉換 + 噪聲增強
- **新增**：對 `y` 做 log1p 轉換再 StandardScaler，預測時 inverse（expm1）
- **新增**：訓練時對 `X` 加入 Gaussian 噪聲（σ=0.02）作為資料增強
- **結果**：NMAE 大幅改善（410% → 61%，最接近 V5 水準），但 SMAPE 惡化至 91%（log 空間誤差放大效應）
- **檔案**：`finetune_gru_v3.py`、`predict_v3.py`、`artificats/log_target_scaler_v3.pkl`

#### V4：LLRD + 混合損失 + WarmRestarts
- **新增**：Layer-wise Learning Rate Decay：GRU L0=5e-6、GRU L1=1e-5、Attention=3e-5、FC=1e-4
- **新增**：混合損失 = 0.7×HuberLoss + 0.3×MSELoss
- **新增**：CosineAnnealingWarmRestarts（T_0=40, T_mult=2）幫助跳出局部最優
- **結果**：Test MAE 全版本最低（5,773），RMSE 也最低（14,513）
- **檔案**：`finetune_gru_v4.py`、`predict_v4.py`

#### V5：3-Seed Ensemble + Per-user 偏差修正
- **新增**：3 個模型獨立訓練（seeds 42/123/777），預測結果在 real space 平均
- **整合**：每個 seed 各自使用 Log1p + Noise + Gradual Unfreezing + LLRD（V3+V4 技術組合）
- **新增**：Per-user 偏差修正：用 val set 計算每位用戶的中位數誤差，預測時扣除
- **偏差修正效果**：Test MAE 6,776 → 5,821（改善 14%）
- **結果**：SMAPE（69.82%）與 NMAE（60.84%）全版本最佳
- **檔案**：`finetune_gru_v5.py`、`predict_v5.py`、`artificats/log_target_scaler_v5.pkl`、`artificats/finetune_gru_v5_seed{42,123,777}.pth`

### 如何使用特定版本

```bash
cd /Users/liweichen/financial-agent/ml_gru

# 跑 V4（MAE 最佳）
VIRTUAL_ENV="/Users/liweichen/financial-agent/.venv" bash -c "
  source $VIRTUAL_ENV/bin/activate
  python finetune_gru_v4.py && python predict_v4.py
"

# 跑 V5（SMAPE/NMAE 最佳，Ensemble）
VIRTUAL_ENV="/Users/liweichen/financial-agent/.venv" bash -c "
  source $VIRTUAL_ENV/bin/activate
  python finetune_gru_v5.py && python predict_v5.py
"
```

### 已保存的模型檔案

| 檔案 | 說明 |
|------|------|
| `artificats/pretrain_gru_v1.pth` | V1 Walmart pretrain（hidden=128） |
| `artificats/pretrain_gru_v2.pth` | V2 Walmart pretrain（hidden=64，V3/V4/V5 共用） |
| `artificats/finetune_gru_v1.pth` | V1 finetune 權重 |
| `artificats/finetune_gru_v2.pth` | V2 finetune 權重 |
| `artificats/finetune_gru_v3.pth` | V3 finetune 權重（log1p 空間） |
| `artificats/finetune_gru_v4.pth` | V4 finetune 權重（Test MAE 最佳） |
| `artificats/finetune_gru_v5_seed42.pth` | V5 Ensemble seed 42 |
| `artificats/finetune_gru_v5_seed123.pth` | V5 Ensemble seed 123 |
| `artificats/finetune_gru_v5_seed777.pth` | V5 Ensemble seed 777 |

### 各版本檢討報告

每版訓練完畢後，Claude 會生成對應的文字檢討報告：

- `claude_ver1.txt`：V1 實驗分析
- `claude_ver2.txt`：V2 實驗分析
- `claude_ver3.txt`：V3 實驗分析
- `claude_ver4.txt`：V4 實驗分析
- `claude_ver5.txt`：V5 實驗分析
