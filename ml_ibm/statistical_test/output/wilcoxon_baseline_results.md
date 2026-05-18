# Wilcoxon Signed-Rank Test Results (no-TL baseline vs ml_ibm TL)

## ml/bigru vs ml_ibm/bigru_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (bigru) | 1098.8679 | 1684.3895 | 0.8043 | 0.7225 |
| mean (bigru_TL_alignment) | 1062.4328 | 1577.5892 | 0.8659 | 0.7463 |
| p-value | 4.9710E-02 | 3.8009E-04 | 1.8626E-09 | 3.7253E-09 |
| 結論 (α=0.05) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |
| 結論 (α=0.01) | 無顯著差異 | 顯著提升 | 顯著提升 | 顯著提升 |

## ml/bilstm vs ml_ibm/bilstm_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (bilstm) | 1067.7855 | 1673.4847 | 0.8360 | 0.7414 |
| mean (bilstm_TL_alignment) | 1067.1412 | 1596.8720 | 0.8706 | 0.7625 |
| p-value | 7.6107E-01 | 5.2263E-02 | 1.8626E-09 | 4.7125E-07 |
| 結論 (α=0.05) | 無顯著差異 | 無顯著差異 | 顯著提升 | 顯著提升 |
| 結論 (α=0.01) | 無顯著差異 | 無顯著差異 | 顯著提升 | 顯著提升 |

## ml/gru vs ml_ibm/gru_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (gru) | 1065.3673 | 1551.9080 | 0.8343 | 0.7374 |
| mean (gru_TL_alignment) | 1134.4145 | 1654.5529 | 0.8549 | 0.7384 |
| p-value | 1.4193E-06 | 6.1467E-08 | 2.0489E-07 | 7.3034E-01 |
| 結論 (α=0.05) | 顯著變差 | 顯著變差 | 顯著提升 | 無顯著差異 |
| 結論 (α=0.01) | 顯著變差 | 顯著變差 | 顯著提升 | 無顯著差異 |

