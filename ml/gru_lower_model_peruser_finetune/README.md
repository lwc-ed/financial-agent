# gru_lower_model_peruser_finetune

## 這個資料夾在做什麼？

本資料夾的核心策略是：**「先讓模型學通用模式，再讓它記住每個人的習慣」**。

原版 `ml_gru` 的根本問題在於：
- 模型太大（~70K 參數）但資料太少（2,850 筆）→ 每個使用者平均只有 ~178 筆
- Walmart pretrain 的 domain 差太遠（商業銷售 ≠ 個人消費）
- 所有使用者共用一個模型，無法捕捉個人習慣差異

本實驗用兩個步驟解決：
1. **縮小模型**：hidden=32, layers=1（~15K 參數），讓參數/樣本比從 1:40 改善到 1:190
2. **Per-user fine-tune**：全域訓練後，對每個使用者各自微調，讓模型記住個人的消費節奏

---

## 主模型架構

### SmallGRU

```
Input (B, 30, 7)
  → GRU (hidden=32, layers=1)          ← 比原版 V2 縮小一半
  → Temporal Attention (softmax)
  → LayerNorm
  → Dropout(0.3)
  → FC1: 32 → 16 + ReLU
  → FC2: 16 → 1
```

**為什麼縮小？**

| 版本 | 參數量 | 訓練筆數 | 參數/樣本比 |
|---|---|---|---|
| ml_gru V2 (hidden=64) | ~70K | 2,850 | 1 : 40 |
| SmallGRU (hidden=32)  | ~15K | 2,850 | 1 : 190 |

參數/樣本比越低越不容易 overfit，在小資料集上泛化能力更好。

---

## 訓練流程

### Step 1：全域訓練（`train_global.py`）

所有使用者的資料合併，訓練一個「通用消費模式」的全域模型。

```
Optimizer : AdamW (lr=5e-4, weight_decay=5e-4)
Loss      : 0.7 × HuberLoss + 0.3 × MSELoss
Scheduler : CosineAnnealingLR
Epochs    : 150 (early stopping patience=25)
Batch     : 32
```

全域模型的目的是讓 GRU 學會「消費序列的一般規律」，例如月末消費通常較高、有收入當天後幾天支出會增加等通用模式。

---

### Step 2：Per-user Fine-tune（`finetune_peruser.py`）

對 16 個使用者各自微調，每個使用者都從相同的全域模型出發，互不干擾。

**兩階段訓練策略**：

```
Phase 1（30 epochs, LR=1e-3）：
  凍結 GRU + Attention 層
  只訓練 FC head
  → 目的：讓輸出快速對齊該使用者的消費量級（大量級用戶 vs 省吃儉用型）

Phase 2（50 epochs, LR=1e-4）：
  解凍所有層
  整體微調
  → 目的：讓 GRU 捕捉個人特有的時序模式（例如每週二固定買菜）
```

**為什麼要兩階段？**
- 如果一開始就解凍所有層用大 LR 微調，全域模型學到的通用特徵會被覆蓋（catastrophic forgetting）
- Phase 1 先讓輸出層穩定，Phase 2 再細調整個模型，更安全

每個使用者的模型獨立儲存：`artifacts/user_{user_id}.pth`

---

## 執行順序

```bash
# 在 gru_lower_model_peruser_finetune/ 目錄下執行
python preprocess.py       # 準備全域資料 + per-user 資料字典
python train_global.py     # 訓練全域小 GRU
python finetune_peruser.py # 對每個使用者個別 fine-tune（16 個模型）
python predict.py          # 比較 global vs per-user vs baselines
```

**輸出路徑**：
```
artifacts/
  global_model.pth             # 全域模型
  user_{user_id}.pth           # 每個使用者的 fine-tuned 模型（16 個）
  peruser_finetune_summary.json # 每個使用者的 fine-tune 結果
  predict_results.json         # 整體比較 + per-user 明細
  predictions_test.csv         # 測試集預測（含全域 vs fine-tuned 對比）
```

---

## 預期結果

| 設定 | 預期效果 | 說明 |
|---|---|---|
| Global SmallGRU | 略優於 GRU V4 | 較小模型 + 無錯誤 pretrain |
| Per-user Fine-tuned | 進一步改善 | 個人消費習慣被捕捉 |
| 消費規律明顯的使用者 | 改善顯著 | 如固定月薪族 |
| 消費不規律的使用者 | 改善有限 | 如臨時性消費多 |

`predict.py` 會輸出 per-user 明細，讓你看出哪些使用者受益最多。

---

## 資料說明

- 來源：`../ml_gru/features_all.csv`（16 個使用者）
- 特徵：與 `ml_gru` 相同的 7 個核心特徵
- 切分：Per-user 70% / 15% / 15%（時間序列，無洩漏）
- Scaler：全域 StandardScaler（fit on train only），per-user 微調在 scaled space 進行
