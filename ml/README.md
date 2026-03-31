# ML 資料夾使用說明

## 🎯 專案目標
每位成員需各自實作一個 **alignment GRU 模型**，並應用於 **Domain Adaptation**。

模型表現需達成以下要求：
- 優於 baseline
- 優於原本的 GRU + finetune 結果(如表一、表二)

## 📊 GRU Model Experiment Results

### 🔹 1. Evaluation Metrics（主要評估）(表一)

| Model | Test MAE ↓ | Test RMSE ↓ | Val MAE | Val RMSE | Notes |
|------|-----------|------------|--------------|----------------|------|
| v0 (baseline transfer) | 842.21 | 1291.91 | 813.14 | 1221.99 | Stable baseline |
| v1 (enhanced) | 874.96 | 1279.94 | 837.69 | 1212.27 | No improvement |
| v2 | **822.82 ** | **1183.05 ** | 861.55 | 1216.41 | Best overall accuracy |
| v3 (log1p) | 910.09  | 1365.71  | 910.02 | 1360.37 | Log transform failed |
| v4 | 846.74 | 1211.90 | 877.62 | 1229.30 | Unstable |
| v5 (ensemble + bias) | 851.48 | 1265.93 | **770.02 ** | 1203.94 | Best personalization |

---
### 🔹 2. Baseline Comparison (表二)

| Method | MAE | RMSE |
|-------|-----|------|
| naive_7d | 989.76 | 1649.28 |
| moving_avg_30d | 926.37 | 1425.73 |

---

## 📁 資料夾規範
每個人需建立自己的模型資料夾，並符合以下規範：

### 1️⃣ README.md
需說明：
- 如何執行整個 pipeline（step-by-step）
- 各個檔案的功能
- 模型的運作流程（pretrain / finetune / predict）

### 2️⃣ requirements.txt
- 列出所有必要套件
- 方便他人在不同電腦快速建立虛擬環境

---

## 🧠 原型模型（Baseline Pipeline）
目前提供的 GRU 基本流程如下：

- `/ml_gru/pretrain.py` → 預訓練模型
- `/ml_gru/finetune.py` → 個人化微調
- `/ml_gru/predict.py` → 進行預測

---


## 🧩 建議實作方向（Domain Adaptation）
你們的 alignment 可以考慮以下方向：

- Feature space alignment（例如 scaling / normalization / representation mapping）
- Loss-level alignment（例如加 domain loss）
- Pretrain source domain → Finetune target domain
- Multi-domain joint training



## ⚠️ 資料過濾設定(可以先略過)
以下檔案中已設定：
- 排除 user4、user5、user6
- 額外排除 user14

請確認自己的實驗是否有使用相同設定（避免資料 leakage 或不一致）

涉及檔案如下：

### GRU / Domain Adaptation 相關
- `/ml/gru_lower_model_peruser_finetune/preprocess.py`
- `/ml/ml_gru/preprocess_personal.py`

### GRU 預測版本（多版本實驗）
- `/ml/ml_gru/predict.py`
- `/ml/ml_gru/predict_v1.py`
- `/ml/ml_gru/predict_v2.py`
- `/ml/ml_gru/predict_v3.py`
- `/ml/ml_gru/predict_v4.py`
- `/ml/ml_gru/predict_v5.py`

### 無 pretrain 設定
- `/ml/ml_gru_nopretrain/preprocess.py`

### Feature Engineering（HGBR）
- `/ml/ml_hgbr/training/make_features.py`

### 混合模型（GRU + HGBR）
- `/ml/gru_hgbr/preprocess.py`

### LSTM 模型
- `/ml/ml_lstm/preprocess_personal.py`
- `/ml/ml_lstm/predict.py`

---
