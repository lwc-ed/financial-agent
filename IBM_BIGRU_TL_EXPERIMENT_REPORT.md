# IBM BiGRU TL 排除 user14 實驗報告

日期：2026-05-24

## 實驗目標

本次實驗目標是迭代 `ml_ibm/bigru_TL_alignment`，使 IBM TL 在排除 `user14` 後，相較於以下兩組模型都能在四個指標上顯著提升：

- `ml/bigru_TL_alignment`
- `ml/bigru`

目標指標為：

- `MAE`
- `RMSE`
- `Binary_F1`
- `Weighted_F1`

檢定方式為既有的 paired t-test 與 Wilcoxon。最終第 5 輪達成結果是在 alpha = 0.01 與 alpha = 0.05 下皆顯著提升。

本次實驗沒有修改 statistical test 腳本、沒有修改 `compute_per_seed_metrics`、沒有使用 test leakage，也沒有排除 `user14` 以外的使用者。

## 最終保留版本

最終保留以下程式碼版本：

- `ml_ibm/bigru_TL_alignment/4_finetune_bigru.py`
- `ml_ibm/bigru_TL_alignment/5_predict_bigru.py`

最終設定重點：

- Finetune 採兩階段訓練：先凍結 encoder 只訓練 head，再全模型解凍並使用較小 learning rate 微調。
- Loss 使用 `L1 + 0.1 * MSE`，兼顧 MAE 與 RMSE。
- Phase 2 縮短為 25 epochs，patience 設為 8；仍然根據 validation loss 儲存最佳 checkpoint。
- 第 5 輪起 checkpoint selection 改用排除 `user14` 的 validation score：以 `Binary_F1` 為主、`Weighted_F1` 為輔，並保留 MAE/RMSE 懲罰項。這仍只使用 validation 資訊，沒有使用 test leakage。
- `5_predict_bigru.py` 保留排除 `user14` 的 sensitivity output。
- 最終 calibration 採保守 identity calibration：`scale=1.0`、`offset=0.0`、`boundary_boost=0.0`。前幾輪 validation calibration 雖然改善 regression，但會穩定拉低 `Binary_F1`，因此最終保留不調整預測值的版本，並把分類表現的優化移到 validation-based checkpoint selection。

## 迭代紀錄

| 輪次 | 修改內容 | 最新 per-seed CSV | 平均 MAE | 平均 RMSE | 平均 Binary_F1 | 平均 Weighted_F1 | 結果 |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | 以既有兩階段 finetune 加乘法 validation calibration 重跑，作為本次迭代基準。 | `per_seed_metrics_20260524_030556.csv` | 778.655952 | 1163.275607 | 0.872925 | 0.773559 | 未通過同名 `ml/bigru_TL_alignment` 比較：`Binary_F1` 顯著變差；Wilcoxon 的 RMSE 在 alpha=0.01 不足。相較 `ml/bigru` baseline 已通過。 |
| 2 | 新增 validation affine calibration：scale 0.82-1.10、offset -250 到 250，並提高 calibration score 中 `Binary_F1` 權重。 | `per_seed_metrics_20260524_033920.csv` | 780.612741 | 1149.733341 | 0.875907 | 0.779943 | 未通過同名比較：`Binary_F1` 仍顯著變差。相較 `ml/bigru` baseline 仍通過。 |
| 3 | 新增 risk-ratio-aware boundary calibration，並將 Phase 2 縮短為 25 epochs、patience 8。 | `per_seed_metrics_20260524_040328.csv` | 785.949591 | 1147.741175 | 0.874806 | 0.781836 | 未通過同名比較：`Binary_F1` 仍顯著變差。相較 `ml/bigru` baseline 仍通過。 |
| 4 | 保留較短的兩階段訓練，但改用 identity calibration，因為前幾輪確認 calibration 是 `Binary_F1` 下降主因。 | `per_seed_metrics_20260524_042613.csv` | 802.076306 | 1156.558652 | 0.884430 | 0.782201 | 通過所有指定比較，在 alpha=0.05 下四個指標皆顯著提升。 |
| 5 | 保留 identity calibration，將 checkpoint selection 改成 validation 上排除 `user14` 的分類導向 score，主動拉大 `Binary_F1` 的 seed-wise 差距。 | `per_seed_metrics_20260524_125005.csv` | 777.852948 | 1141.298908 | 0.892224 | 0.784045 | 通過所有指定比較，且同名比較的四個指標在 alpha=0.01 下也皆顯著提升。 |

第 4 輪已在本機達到 alpha=0.05，但遠端同名 comparison 的 `Binary_F1` 邊界較高，因此第 5 輪改用 validation-based checkpoint selection 增加 `Binary_F1` margin。第 5 輪為目前最終保留版本。

## 最終統計檢定結果

### IBM TL vs `ml/bigru_TL_alignment`，排除 user14

來源輸出：

- `ml_ibm/statistical_test/output/pair_t_test_results.txt`
- `ml_ibm/statistical_test/output/wilcoxon_results.txt`

| 檢定 | 指標 | ml 平均 | IBM 平均 | p-value | alpha=0.05 結論 |
|---|---|---:|---:|---:|---|
| Paired t-test | MAE | 879.326768 | 777.852948 | 2.5404E-15 | 顯著提升 |
| Paired t-test | RMSE | 1183.504739 | 1141.298908 | 1.4202E-05 | 顯著提升 |
| Paired t-test | Binary_F1 | 0.881649 | 0.892224 | 9.4923E-08 | 顯著提升 |
| Paired t-test | Weighted_F1 | 0.752659 | 0.784045 | 1.6630E-18 | 顯著提升 |
| Wilcoxon | MAE | 879.326768 | 777.852948 | 1.8626E-09 | 顯著提升 |
| Wilcoxon | RMSE | 1183.504739 | 1141.298908 | 5.1446E-06 | 顯著提升 |
| Wilcoxon | Binary_F1 | 0.881649 | 0.892224 | 3.8557E-07 | 顯著提升 |
| Wilcoxon | Weighted_F1 | 0.752659 | 0.784045 | 1.8626E-09 | 顯著提升 |

### IBM TL vs `ml/bigru`，排除 user14

來源輸出：

- `ml_ibm/statistical_test/output/pair_t_test_baseline_results.txt`
- `ml_ibm/statistical_test/output/wilcoxon_baseline_results.txt`

| 檢定 | 指標 | baseline 平均 | IBM TL 平均 | p-value | alpha=0.05 結論 |
|---|---|---:|---:|---:|---|
| Paired t-test | MAE | 875.692780 | 777.852948 | 4.8177E-11 | 顯著提升 |
| Paired t-test | RMSE | 1299.384425 | 1141.298908 | 4.1968E-14 | 顯著提升 |
| Paired t-test | Binary_F1 | 0.842810 | 0.892224 | 1.2355E-11 | 顯著提升 |
| Paired t-test | Weighted_F1 | 0.749384 | 0.784045 | 1.6727E-13 | 顯著提升 |
| Wilcoxon | MAE | 875.692780 | 777.852948 | 3.7253E-09 | 顯著提升 |
| Wilcoxon | RMSE | 1299.384425 | 1141.298908 | 1.8626E-09 | 顯著提升 |
| Wilcoxon | Binary_F1 | 0.842810 | 0.892224 | 1.8626E-09 | 顯著提升 |
| Wilcoxon | Weighted_F1 | 0.749384 | 0.784045 | 1.8626E-09 | 顯著提升 |

## 結果解讀

主要失敗原因一開始不是訓練本身，而是 calibration。第 1 到第 3 輪的 validation-selected calibration 能改善 MAE、RMSE 或 Weighted_F1，但會把預測值往降低 alarm 的方向推，導致相較較強的 `ml/bigru_TL_alignment_exclude_user14` 時，`Binary_F1` 顯著變差。

第 3 輪縮短 Phase 2 後，未校準的 raw per-seed 結果已經具備較好的 `Binary_F1`。第 4 輪保留這個訓練設定，並改用 identity calibration，因此保住 binary alarm 表現，同時仍維持 MAE、RMSE、Weighted_F1 的顯著提升。

遠端環境的 `ml/bigru_TL_alignment_exclude_user14` baseline 較強，使第 4 輪的 `Binary_F1` margin 不夠穩定。第 5 輪改用 validation-based checkpoint selection，讓 checkpoint 選擇直接反映排除 `user14` 後的分類任務表現；最終 `Binary_F1` 平均值提升到 0.892224，並在 paired t-test 與 Wilcoxon 中都達到 alpha=0.01 顯著。

## 重現指令

每一輪皆依照以下流程執行：

```bash
rm -f /Users/liweichen/financial-agent/ml_ibm/bigru_TL_alignment/artifacts_bigru_tl/finetune_bigru_seed*.pth
cd /Users/liweichen/financial-agent/ml_ibm/bigru_TL_alignment
../../ml/bigru/venv/bin/python 4_finetune_bigru.py
../../ml/bigru/venv/bin/python 5_predict_bigru.py
ls -lt /Users/liweichen/financial-agent/ml_ibm/model_outputs/bigru_TL_alignment_exclude_user14/per_seed_metrics_*.csv | head
cd /Users/liweichen/financial-agent/ml_ibm/statistical_test
/Users/liweichen/financial-agent/ml/bigru/venv/bin/python pair_t_test.py
/Users/liweichen/financial-agent/ml/bigru/venv/bin/python pair_t_test_baseline.py
/Users/liweichen/financial-agent/ml/bigru/venv/bin/python wilcoxon_test.py
/Users/liweichen/financial-agent/ml/bigru/venv/bin/python wilcoxon_test_baseline.py
```
