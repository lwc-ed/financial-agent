# Paired T-Test Results (no-TL baseline vs ml_ibm TL)

## ml/bigru vs ml_ibm/bigru_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (bigru) | 1089.8457 | 1677.8508 | 0.8088 | 0.7235 |
| mean (bigru_TL_alignment) | 960.2968 | 1441.1494 | 0.8596 | 0.7616 |
| p-value | 5.2849E-12 | 5.7428E-12 | 1.6230E-12 | 8.9098E-16 |
| 結論 (α=0.05) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |
| 結論 (α=0.01) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |

## ml/bigru_exclude_user14 vs ml_ibm/bigru_TL_alignment_exclude_user14

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (bigru_exclude_user14) | 858.0620 | 1282.4871 | 0.8486 | 0.7522 |
| mean (bigru_TL_alignment_exclude_user14) | 779.5801 | 1141.1445 | 0.8916 | 0.7840 |
| p-value | 3.3290E-12 | 1.3968E-16 | 1.0180E-10 | 4.7750E-14 |
| 結論 (α=0.05) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |
| 結論 (α=0.01) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |

## ml/bilstm vs ml_ibm/bilstm_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (bilstm) | 1067.7855 | 1673.4847 | 0.8360 | 0.7414 |
| mean (bilstm_TL_alignment) | 1023.4844 | 1455.4378 | 0.8564 | 0.7481 |
| p-value | 1.0141E-02 | 3.1586E-07 | 2.1793E-07 | 2.3129E-02 |
| 結論 (α=0.05) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |
| 結論 (α=0.01) | 無顯著差異 | 顯著提升 | 顯著提升 | 無顯著差異 |

## ml/gru vs ml_ibm/gru_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (gru) | 1065.3673 | 1551.9080 | 0.8343 | 0.7374 |
| mean (gru_TL_alignment) | 1028.1397 | 1479.2812 | 0.8586 | 0.7574 |
| p-value | 1.2289E-03 | 9.5024E-07 | 2.3464E-08 | 3.4478E-09 |
| 結論 (α=0.05) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |
| 結論 (α=0.01) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |

