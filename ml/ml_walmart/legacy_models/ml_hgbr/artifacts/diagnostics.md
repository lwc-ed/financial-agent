# Diagnostics

- target_column: `future_expense_7d_sum`
- feature_count: 25

## Test Target Distribution

- mean: 2294.153809
- median: 2053.500000
- p25: 1497.750000
- p75: 2842.750000
- p90: 3811.000000
- max: 7860.000000

## Baseline Vs Model Metrics

```text
model                             mae         rmse
--------------------------------------------------
mlp                       1116.173828  1494.704653
hgbr                       890.803891  1271.809810
naive_7d_sum              1430.788696  1977.608467
moving_avg_30d_x7         1091.896729  1553.339467
```

## Feature Checks

- past_7d_sum: present=True matched_columns=['expense_7d_sum']
- past_30d_sum: present=True matched_columns=['expense_30d_sum']
- future-like columns in features: []
