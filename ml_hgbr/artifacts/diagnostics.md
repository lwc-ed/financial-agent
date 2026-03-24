# Diagnostics

- target_column: `future_expense_7d_sum`
- feature_count: 25

## Test Target Distribution

- mean: 3048.780273
- median: 2336.000000
- p25: 1624.500000
- p75: 3620.500000
- p90: 6632.000000
- max: 11061.000000

## Baseline Vs Model Metrics

```text
model                             mae         rmse
--------------------------------------------------
mlp                       1853.903442  2701.936343
hgbr                      1165.793531  1650.570547
naive_7d_sum              1746.275757  2391.179416
moving_avg_30d_x7         1780.758911  4408.610666
```

## Feature Checks

- past_7d_sum: present=True matched_columns=['expense_7d_sum']
- past_30d_sum: present=True matched_columns=['expense_30d_sum']
- future-like columns in features: []
