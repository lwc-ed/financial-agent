# Paired T-Test Results (no-TL baseline vs ml_ibm TL)

## ml/bigru vs ml_ibm/bigru_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (bigru) | 1089.8457 | 1677.8508 | 0.8088 | 0.7235 |
| mean (bigru_TL_alignment) | 1059.2662 | 1701.4799 | 0.8543 | 0.7585 |
| p-value | 1.0217E-02 | 2.6992E-01 | 2.9887E-12 | 2.4334E-15 |
| 結論 (α=0.05) | 顯著提升 | 無顯著差異 | 顯著提升 | 顯著提升 |
| 結論 (α=0.01) | 無顯著差異 | 無顯著差異 | 顯著提升 | 顯著提升 |

## ml/bigru_exclude_user14 vs ml_ibm/bigru_TL_alignment_exclude_user14

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (bigru_exclude_user14) | 858.0620 | 1282.4871 | 0.8486 | 0.7522 |
| mean (bigru_TL_alignment_exclude_user14) | 806.4854 | 1160.1272 | 0.8856 | 0.7808 |
| p-value | 1.4176E-08 | 7.9769E-16 | 8.5084E-10 | 1.1393E-13 |
| 結論 (α=0.05) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |
| 結論 (α=0.01) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |

## ml/bilstm vs ml_ibm/bilstm_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (bilstm) | 1067.7855 | 1673.4847 | 0.8360 | 0.7414 |
| mean (bilstm_TL_alignment) | 1219.4890 | 1835.6250 | 0.8585 | 0.7463 |
| p-value | 3.8560E-10 | 9.7050E-06 | 8.6390E-09 | 6.4727E-02 |
| 結論 (α=0.05) | 顯著變差 | 顯著變差 | 顯著提升 | 無顯著差異 |
| 結論 (α=0.01) | 顯著變差 | 顯著變差 | 顯著提升 | 無顯著差異 |

## ml/gru vs ml_ibm/gru_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (gru) | 1065.3673 | 1551.9080 | 0.8343 | 0.7374 |
| mean (gru_TL_alignment) | 1134.4145 | 1654.5529 | 0.8549 | 0.7384 |
| p-value | 3.4326E-08 | 3.0912E-09 | 3.2205E-07 | 6.9251E-01 |
| 結論 (α=0.05) | 顯著變差 | 顯著變差 | 顯著提升 | 無顯著差異 |
| 結論 (α=0.01) | 顯著變差 | 顯著變差 | 顯著提升 | 無顯著差異 |

