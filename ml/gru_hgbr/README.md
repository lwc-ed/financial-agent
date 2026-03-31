# gru_hgbr

## 這個資料夾在做什麼？

本資料夾實作「GRU 當特徵提取器 + HGBR 做最終預測」的 Hybrid 架構。

**核心想法**：讓 GRU 和 HGBR 各做自己最擅長的事。

| 模型 | 擅長的事 | 不擅長的事 |
|---|---|---|
| GRU | 捕捉序列時序依賴（趨勢、週期） | 小資料 overfit、需要大量訓練資料 |
| HGBR | 小資料高效學習、特徵交互 | 時序序列的長程依賴 |

**做法**：把已 fine-tuned 的 GRU（V4）當作「固定的特徵提取器」，取出 context vector（64 維 embedding），再把這個 embedding 和 22 個扁平特徵拼接（共 86 維），最後用 HGBR 做最終預測。

這樣既保留了 GRU 提取時序特徵的能力，又讓 HGBR 在最終預測時充分利用所有資訊，避免 GRU 在小資料上 overfit 的問題。

---

## 架構圖

```
輸入：30 天消費序列 (B, 30, 7)
        ↓
   ┌────────────────────────────┐
   │  GRU V4 Fine-tuned (frozen)│  ← 不更新權重，只提取特徵
   │  GRU → Attention → LN      │
   └────────────┬───────────────┘
                ↓
         context vector (64 維)    ← 捕捉序列的時序模式

輸入：當前時間點的 22 個特徵
        ↓
   日曆特徵 + 滾動統計 (22 維)

                ↓  concat
        ┌───────────────┐
        │  合併特徵       │  (64 + 22 = 86 維)
        └───────┬───────┘
                ↓
        HistGradientBoostingRegressor
                ↓
        預測：future_expense_7d_sum
```

---

## GRU Embedding 說明

使用 `ml_gru/artificats/finetune_gru_v4.pth`（V4 fine-tuned 模型）作為 embedding 來源。

**為什麼選 V4 fine-tuned 而不是 pretrain？**
- V4 模型已經在個人財務資料上 fine-tune 過，context vector 的語義更貼近個人消費模式
- 直接用 Walmart pretrain 的 embedding，等於用商業零售的時序表示來描述個人消費，domain 差距太大
- V4 是 ml_gru 系列中 MAE 最低的版本（Test MAE 5,773）

**Context vector 是什麼？**

GRU 處理 30 天序列後，用 Attention 對每個時間步加權平均，得到一個 64 維向量：

```python
gru_out, _ = self.gru(x)              # (B, 30, 64)
attn_w     = softmax(attention(gru_out))  # (B, 30, 1)
context    = (gru_out * attn_w).sum(dim=1)  # (B, 64) ← 這就是 embedding
context    = LayerNorm(context)
```

這個向量壓縮了 30 天序列的所有時序資訊（近期趨勢、波動幅度、消費節奏等）。

---

## 訓練流程

### Step 1：`preprocess.py`
- 讀取 `features_all.csv`，建立：
  - GRU 序列（30天視窗，7特徵）→ 供 embedding 提取用
  - HGBR 扁平特徵（22特徵）→ 供最終 HGBR 使用
- **重要**：GRU 序列使用 `ml_gru` 原版的 `personal_feature_scaler.pkl`（不重新 fit），確保輸入分佈與 V4 訓練時一致，embedding 才有意義

### Step 2：`extract_embeddings.py`
- 載入 frozen GRU V4，提取 train/val/test 的 context embedding（64 維）
- 拼接 embedding + HGBR 扁平特徵 → combined 特徵（86 維）

### Step 3：`train_hgbr.py`
同時訓練兩個 HGBR 做比較：
- `HGBR_flat_only`：只用 22 個扁平特徵
- `HGBR_emb_and_flat`：64 維 embedding + 22 個扁平特徵

直接對比讓你看出 GRU embedding 是否真的帶來了額外的預測能力。

```
max_iter=1000, learning_rate=0.05, max_leaf_nodes=31
min_samples_leaf=20, l2_regularization=0.1
early_stopping=True, validation_fraction=0.15
```

### Step 4：`predict.py`
完整評估三個模型：
- 純 GRU V4（參考）
- HGBR flat only
- HGBR embedding + flat（hybrid）

並輸出 per-user 明細，顯示各使用者的改善幅度。

---

## 執行順序

```bash
# 在 gru_hgbr/ 目錄下執行
python preprocess.py           # 準備序列資料 + 扁平特徵
python extract_embeddings.py   # 提取 GRU embedding 並拼接（86 維）
python train_hgbr.py           # 訓練兩種 HGBR（flat vs hybrid）
python predict.py              # 完整評估與比較
```

**輸出路徑**：
```
artifacts/
  gru_X_train/val/test.npy         # GRU 序列（scaled，用於 embedding）
  hgbr_X_train/val/test.npy        # HGBR 扁平特徵
  emb_train/val/test.npy           # GRU context embedding（64 維）
  combined_X_train/val/test.npy    # embedding + 扁平特徵（86 維）
  hgbr_HGBR_flat_only.pkl          # HGBR flat-only 模型
  hgbr_HGBR_emb_and_flat.pkl       # HGBR hybrid 模型
  metrics_hgbr.json                # 兩個模型的指標 + embedding 是否有幫助
  predict_results.json             # 完整比較結果
  predictions_test.csv             # 預測結果 CSV
```

---

## 預期結果

| 模型 | 預期 Test MAE | 說明 |
|---|---|---|
| 純 GRU V4 | ~5,773 | 參考（ml_gru 最佳） |
| HGBR flat only | < 5,773 | 豐富特徵 + 強大迴歸器 |
| **HGBR hybrid** | **< HGBR flat** | 若 embedding 有效 |

`train_hgbr.py` 最後會直接印出：
```
GRU embedding 有沒有幫助？
HGBR (flat only)  Test MAE: X,XXX
HGBR (emb+flat)   Test MAE: Y,YYY
改善幅度：+ZZZ  ✅ embedding 有幫助！
```

若 embedding 沒幫助（delta ≤ 0），代表 V4 fine-tuned GRU 雖然可以做預測，但它的 context vector 並未包含 HGBR 無法從扁平特徵中自行學到的額外資訊。

---

## 資料說明

- 來源：`../ml_gru/features_all.csv`（16 個使用者）
- GRU Scaler：使用 `../ml_gru/artificats/personal_feature_scaler.pkl`（與 V4 一致）
- HGBR Scaler：本資料夾內重新 fit（22 個特徵）
- GRU Model：使用 `../ml_gru/artificats/finetune_gru_v4.pth`（frozen，不更新）

