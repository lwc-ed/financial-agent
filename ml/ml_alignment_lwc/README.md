# 整體架構
## Part 1: 診斷為什麼 Naive TL 失敗（量化domain gap）






## Part 2: 設計 Aligned Pretrain 讓 Walmart TL 贏過 no pretrain
### Experiement 1:
Method: 把兩個資料集的 feature 都轉換成「相對於自己歷史的偏離程度」，讓 GRU 學到的不是「Walmart的絕對銷售數字」，而是「消費序列的動態規律」，這個規律才能遷移到個人資料。
```
Walmart 原始資料                    個人原始資料
(週級銷售額，百萬尺度)               (日級支出，百元尺度)
        ↓                                   ↓
  Rolling Z-score                     Rolling Z-score
        ↓                                   ↓
  domain-invariant features          domain-invariant features
  (zscore_7d, pct_change, ...)       (zscore_7d, pct_change, ...)
        ↓                                   ↓
        └──────────── 同一個feature空間 ─────┘
                            ↓
                       Pretrain GRU
                      (用 Walmart 大量資料
                       學習「相對消費動態」)
                            ↓
                       Finetune GRU
                      (用個人資料適應
                       個人消費習慣)
                            ↓
                         Predict
                      (預測個人未來支出)

```
#### 今天實際做的實驗內容
1. 先用 `1_diagnose_gap.py` 檢查 Walmart 與個人資料的 domain gap，確認直接把原始數值拿來做 transfer learning 會受到尺度差異影響。
2. 用 `2_preprocess_walmart_aligned.py` 把 Walmart 週資料先展成日資料，再轉成 aligned feature，並建立 `30` 天輸入視窗去預測未來 `7` 天支出總和。
3. 用 `3_preprocess_personal_aligned.py` 對個人資料套用完全相同的 feature engineering pipeline；每位 user 先聚合成日支出、補齊沒有消費的日期為 `0`，再做 per-user `70/15/15` 切分。
4. 用 `4_pretrain_aligned.py` 在 Walmart aligned 資料上先訓練 GRU + Attention，模型架構維持和原本 `ml_gru` pipeline 相同，只改成新的 aligned feature 空間，方便直接比較是不是 feature alignment 帶來改善。
5. 用 `5_finetune_aligned.py` 載入 pretrained weights，在個人資料上 finetune，並跑 `7` 個 seeds（`42, 123, 456, 777, 789, 999, 2024`）做 ensemble，降低小資料量造成的波動。
6. 用 `6_predict_aligned.py` 做最終預測與三方比較：`No Pretrain`、`Naive TL (V5)`、`Aligned Pretrain`。

#### 這次新增的 aligned features
這次不是直接餵原始金額，而是把兩邊資料都映射到同一組「相對動態」特徵空間：

- `zscore_7d`：今天相對最近 7 天平均的偏離程度
- `zscore_30d`：今天相對最近 30 天平均的偏離程度
- `pct_change_norm`：今天與前一天的變化量，再用 30 日均值正規化
- `volatility_7d`：最近 7 天波動率，表示消費是否穩定
- `is_above_mean_30d`：今天是否高於 30 日平均
- `dow_sin`
- `dow_cos`

#### 今天這版實驗的重點設定
- Walmart 的週銷售先平均展成 daily level，再與個人資料統一成日級序列
- 兩邊都使用同一份 `compute_aligned_features()` 邏輯，避免 source / target feature definition 不一致
- input window 固定為 `30` 天，target 固定為未來 `7` 天支出加總 `future_expense_7d_sum`
- feature 不再做額外 scaler，因為 rolling z-score 特徵本身已經落在相對穩定的標準空間
- target 仍各自在 train split 上做 `StandardScaler`，讓 GRU 訓練更穩
- pretrain 與 finetune 都維持原本 GRU + Attention + LayerNorm 架構，讓比較聚焦在 alignment 方法本身
- bias correction 最後停用，因為這次實驗觀察到不做 correction 的 test MAE 更低

#### 目前結果
使用 MMD 衡量 source domain 與 target domain 的分布差距
- `Before Alignment` MMD: 0.193
- `After Alignment` MMD: 0.090
- `MMD reduction ` ≈ 53.34%
表示兩個資料集在 aligned feature space 中變得更接近

這次 aligned pretrain 的測試結果為：
- `No Pretrain` Test MAE: `836.29`
- `Naive TL (V5)` Test MAE: `851.48`
- `Aligned Pretrain` Test MAE: `795.24`

代表這次實驗至少驗證了一件事：只做 pretrain 不夠，必須先把 source domain 與 target domain 投影到可遷移的共同 feature space，transfer learning 才有機會真的贏過 no pretrain。
### 為什麼不選 XGBoost（即使它 MAE 更低）？
XGBoost 雖然在現有測試資料上表現最佳（Test MAE: 622），但考慮到實際上線情境，它有兩個根本限制：
1.冷啟動品質差：新用戶資料不足時，rolling features（30日均值、30日加總）嚴重失真，預測品質不穩定
2.無法個人化：XGBoost 是靜態模型，無法針對個別用戶的消費習慣做 finetune 調整，所有用戶共用同一套預測邏輯
因此我們選擇 GRU + Domain Adaptation：犧牲一部分當下的準確率，換取一個能隨用戶成長、持續進化的系統設計。

|階段	|XGBoost	|GRU（無pretrain）	|GRU（Aligned Pretrain）
|------|-----------|------------|--------------|
| < 30 天	| ⚠️ 能預測但不準	| ❌ 無法預測	| ❌ 無法預測
| 30–60 天	| ⚠️ 特徵還不穩	| ⚠️ weights 不穩	| ✅ pretrain 補足先驗
| 60 天以上	| ✅ 穩定	| ✅ 穩定	| ✅ + 可 per-user finetune
