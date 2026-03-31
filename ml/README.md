# ML 資料夾使用說明

## 🎯 專案目標
每位成員需各自實作一個 **alignment GRU 模型**，並應用於 **Domain Adaptation**。

模型表現需達成以下要求：
- 優於 baseline
- 優於原本的 GRU + finetune 結果

下面是整理成 README 可直接貼的「數據比較表 + 重點摘要」版本（乾淨、可讀性高、方便你們討論）：

⸻

📊 GRU 實驗結果總覽（v0 ~ v5）

🔹 1. Test Metrics 比較（主要評估）

Model	Test MAE ↓	Test RMSE ↓	Test SMAPE ↓	Per-user NMAE ↓	備註
v0 (baseline transfer)	842.21	1291.91	85.66%	112.00%	穩定 baseline
v1 (enhanced)	874.96	1279.94	85.16%	132.09%	未改善
v2	822.82 ⭐	1183.05 ⭐	79.21%	168.52% ❌	整體最佳
v3 (log1p)	910.09 ❌	1365.71 ❌	88.68% ❌	60.33%	log 失敗
v4	846.74	1211.90	79.42%	208.24% ❌	不穩定
v5 (ensemble + bias)	851.48	1265.93	69.04% ⭐	59.45% ⭐	個人化最佳


⸻

🔹 2. Validation Metrics（參考）

Model	Val MAE	Val RMSE	Val SMAPE	Val per-user NMAE
v0	813.14	1221.99	91.61%	92.01%
v1	837.69	1212.27	90.36%	88.38%
v2	861.55	1216.41	87.64%	105.92%
v3	910.02	1360.37	105.01%	86.41%
v4	877.62	1229.30	87.61%	121.30%
v5	770.02 ⭐	1203.94	81.76% ⭐	70.99% ⭐


🔹 3. Baseline 比較

Method	MAE	RMSE
naive_7d	989.76	1649.28
moving_avg_30d	926.37	1425.73



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



---

## ✅ 總結
請確保你的資料夾具備：
- 可重現（reproducible）
- 有清楚說明（README）
- 可直接執行（requirements + pipeline）

並且最終結果需：
👉 明確 outperform baseline
👉 明確優於原本 GRU + finetune
