# Wilcoxon Signed-Rank Test Results (no-TL baseline vs ml_ibm TL)

## ml/bigru vs ml_ibm/bigru_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (bigru) | 1089.8457 | 1677.8508 | 0.8088 | 0.7235 |
| mean (bigru_TL_alignment) | 960.2968 | 1441.1494 | 0.8596 | 0.7616 |
| p-value | 1.8626E-09 | 1.8626E-09 | 3.7253E-09 | 1.8626E-09 |
| 結論 (α=0.05) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |
| 結論 (α=0.01) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |

## ml/bigru_exclude_user14 vs ml_ibm/bigru_TL_alignment_exclude_user14

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (bigru_exclude_user14) | 858.0620 | 1282.4871 | 0.8486 | 0.7522 |
| mean (bigru_TL_alignment_exclude_user14) | 779.5801 | 1141.1445 | 0.8916 | 0.7840 |
| p-value | 1.8626E-09 | 1.8626E-09 | 9.3132E-09 | 3.7253E-09 |
| 結論 (α=0.05) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |
| 結論 (α=0.01) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |

## ml/bilstm vs ml_ibm/bilstm_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (bilstm) | 1067.7855 | 1673.4847 | 0.8360 | 0.7414 |
| mean (bilstm_TL_alignment) | 1219.4890 | 1835.6250 | 0.8585 | 0.7463 |
| p-value | 5.5879E-09 | 2.6885E-05 | 9.3132E-09 | 7.3244E-02 |
| 結論 (α=0.05) | 顯著變差 | 顯著變差 | 顯著提升 | 無顯著差異 |
| 結論 (α=0.01) | 顯著變差 | 顯著變差 | 顯著提升 | 無顯著差異 |

## ml/gru vs ml_ibm/gru_TL_alignment

|  | MAE | RMSE | Binary_F1 | Weighted_F1 |
| --- | --- | --- | --- | --- |
| mean (gru) | 1065.3673 | 1551.9080 | 0.8343 | 0.7374 |
| mean (gru_TL_alignment) | 1028.1397 | 1479.2812 | 0.8586 | 0.7574 |
| p-value | 5.7765E-03 | 1.3039E-07 | 4.6566E-08 | 8.0094E-08 |
| 結論 (α=0.05) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |
| 結論 (α=0.01) | 顯著提升 | 顯著提升 | 顯著提升 | 顯著提升 |

