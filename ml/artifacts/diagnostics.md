# Diagnostics

- target_column: `future_expense_7d_sum`
- feature_count: 22

## Test Target Distribution

- mean: 2146.619873
- median: 1848.500000
- p25: 1182.000000
- p75: 2816.500000
- p90: 3831.500000
- max: 7065.000000

## Baseline Vs Model Metrics

```text
model                             mae         rmse
--------------------------------------------------
mlp                       1693.216797  2112.878014
hgbr                       988.881727  1275.818003
naive_7d_sum              1479.155396  1942.856724
moving_avg_30d_x7         1217.206299  1662.021360
```

## Feature Checks

- past_7d_sum: present=True matched_columns=['expense_7d_sum']
- past_30d_sum: present=True matched_columns=['expense_30d_sum']
- future-like columns in features: []
