# Paired T-Test Results (no-TL baseline vs ml_ibm TL)

## ml/bigru vs ml_ibm/bigru_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (bigru) | 1098.8679 | 1684.3895 | 0.8043 | 0.7225 |
| mean (bigru_TL_alignment) | 1143.1728 | 1748.1373 | 0.8480 | 0.7647 |
| p-value | 1.1769E-03 | 1.3737E-02 | 2.8502E-10 | 7.1755E-14 |
| 結論 (α=0.05) | 顯著變差 | 顯著變差 | 顯著提升 | 顯著提升 |
| 結論 (α=0.01) | 顯著變差 | 無顯著差異 | 顯著提升 | 顯著提升 |

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
| mean (gru_TL_alignment) | 1148.1668 | 1696.7077 | 0.8437 | 0.7408 |
| p-value | 1.0024E-09 | 3.5119E-12 | 1.0159E-02 | 2.1559E-01 |
| 結論 (α=0.05) | 顯著變差 | 顯著變差 | 顯著提升 | 無顯著差異 |
| 結論 (α=0.01) | 顯著變差 | 顯著變差 | 無顯著差異 | 無顯著差異 |

