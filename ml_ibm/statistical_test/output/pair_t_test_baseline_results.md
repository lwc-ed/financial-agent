# Paired T-Test Results (no-TL baseline vs ml_ibm TL)

## ml/bigru vs ml_ibm/bigru_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (bigru) | 1098.8679 | 1684.3895 | 0.8043 | 0.7225 |
| mean (bigru_TL_alignment) | 1062.4328 | 1577.5892 | 0.8659 | 0.7463 |
| p-value | 2.5067E-02 | 5.4264E-04 | 1.6362E-14 | 9.6459E-09 |
| 結論 (α=0.05) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |
| 結論 (α=0.01) | 無顯著差異 | 顯著提升 | 顯著提升 | 顯著提升 |

## ml/bilstm vs ml_ibm/bilstm_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (bilstm) | 1067.7855 | 1673.4847 | 0.8360 | 0.7414 |
| mean (bilstm_TL_alignment) | 1067.1412 | 1596.8720 | 0.8706 | 0.7625 |
| p-value | 9.7334E-01 | 3.8832E-02 | 8.8695E-13 | 5.4534E-08 |
| 結論 (α=0.05) | 無顯著差異 | 顯著提升 | 顯著提升 | 顯著提升 |
| 結論 (α=0.01) | 無顯著差異 | 無顯著差異 | 顯著提升 | 顯著提升 |

## ml/gru vs ml_ibm/gru_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (gru) | 1065.3673 | 1551.9080 | 0.8343 | 0.7374 |
| mean (gru_TL_alignment) | 1134.4145 | 1654.5529 | 0.8549 | 0.7384 |
| p-value | 3.4326E-08 | 3.0912E-09 | 3.2205E-07 | 6.9251E-01 |
| 結論 (α=0.05) | 顯著變差 | 顯著變差 | 顯著提升 | 無顯著差異 |
| 結論 (α=0.01) | 顯著變差 | 顯著變差 | 顯著提升 | 無顯著差異 |

