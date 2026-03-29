# ml_gru_nopretrain

## 這個資料夾在做什麼？

本資料夾測試兩個核心假設：

1. **HGBR 配豐富特徵是否優於 GRU transfer learning？**
   原版 HGBR 只使用 7 個特徵，但 `features_all.csv` 中其實有 22 個可用特徵（包含大量日曆特徵、多種滾動統計），本實驗將這些特徵全部交給 HGBR，看能否超越原版的 Test MAE ~1,200。

2. **Walmart pretrain 到底是幫助還是拖累？**
   `train_gru_scratch.py` 使用與 `ml_gru` V2 完全相同的架構，但不載入任何 pretrain 權重，直接從頭訓練。若 GRU scratch 的 MAE 接近甚至優於 pretrain GRU（V4 Test MAE ~5,773），就確認 Walmart pretrain 對個人財務預測沒有幫助，應放棄 transfer learning。

---

## 主模型說明

### 模型 1：HGBR（22 個特徵）

**核心概念**：HGBR（HistGradientBoostingRegressor）在小資料集上天然優於深度學習，原因是：
- 不需要 pretrain，直接對小資料有效
- 對特徵交互作用學習能力強
- 對離群值、右偏分佈較穩健
- `features_all.csv` 裡的日曆特徵（月初/月末、週末、寒暑假）對個人消費規律非常重要，HGBR 可以直接利用

**使用特徵（22 個）**：

| 類別 | 特徵 |
|---|---|
| 當日基礎 | `daily_expense`, `daily_income`, `daily_net`, `has_expense`, `has_income` |
| 日曆 | `dow`, `is_weekend`, `day`, `month`, `is_summer_vacation`, `is_winter_vacation`, `days_to_end_of_month` |
| 7 日統計 | `expense_7d_sum`, `expense_7d_mean`, `net_7d_sum`, `txn_7d_sum` |
| 30 日統計 | `expense_30d_sum`, `expense_30d_mean`, `net_30d_sum`, `txn_30d_sum` |
| 趨勢 | `expense_7d_30d_ratio`, `expense_trend` |

**超參數**：
```
max_iter=1000, learning_rate=0.05, max_leaf_nodes=31
min_samples_leaf=20, l2_regularization=0.1
early_stopping=True, validation_fraction=0.15, n_iter_no_change=30
```

---

### 模型 2：GRU from scratch（對照組）

**核心概念**：與 `ml_gru` V2 完全相同架構，但**完全不使用 Walmart pretrain**，直接用個人資料從隨機初始化開始訓練。

**架構**：
```
Input (B, 30, 7)
  → GRU (hidden=64, layers=2, dropout=0.4)
  → Temporal Attention (softmax weighted)
  → LayerNorm
  → Dropout(0.4)
  → FC1: 64 → 32 + ReLU
  → FC2: 32 → 1
```

**訓練設定**：
```
Optimizer : AdamW (lr=1e-3, weight_decay=5e-4)
Loss      : 0.7 × HuberLoss + 0.3 × MSELoss
Scheduler : CosineAnnealingWarmRestarts (T_0=40, T_mult=2)
Epochs    : 200 (early stopping patience=30)
Batch     : 16
```

---

## 執行順序

```bash
# 在 ml_gru_nopretrain/ 目錄下執行
python preprocess.py          # 準備 HGBR 22 特徵 + GRU 序列資料
python train_hgbr.py          # 訓練 HGBR
python train_gru_scratch.py   # 訓練 GRU from scratch（可與 train_hgbr.py 平行執行）
python predict.py             # 比較兩個模型 vs baselines
```

**輸出路徑**：
```
artifacts/
  hgbr_model.pkl              # 訓練完成的 HGBR 模型
  gru_scratch.pth             # 訓練完成的 GRU 模型
  metrics_hgbr.json           # HGBR 指標
  comparison_results.json     # 兩個模型的完整比較
  predictions_test.csv        # 測試集預測結果
```

---

## 預期結果

| 模型 | 預期 Test MAE | 備註 |
|---|---|---|
| Naive 7-day | ~8,130 | baseline |
| Moving Avg 30d | ~7,019 | baseline |
| 原版 HGBR (7 特徵) | ~1,200 | 歷史最佳 |
| **HGBR (22 特徵)** | **< 1,200** | 本實驗目標 |
| 原版 GRU V4 (pretrain) | ~5,773 | 參考 |
| **GRU from scratch** | 待測定 | 驗證 pretrain 效果 |

若 GRU scratch 和 pretrain GRU 差不多，代表 Walmart pretrain 對此任務無效，應完全放棄 transfer learning 策略。

---

## 資料說明

- 來源：`../ml_gru/features_all.csv`（16 個使用者，共 ~4,067 個訓練樣本）
- 切分：Per-user 70% train / 15% val / 15% test（時間序列切分）
- Per-user P99 clipping：防止各使用者的極端值影響全域 scaler
