# tax

台灣綜合所得稅試算（114 年度）。純本地計算，無外部 API、無資料庫。

- **對外接口**：`tax_calculator.calculate_taiwan_tax_2026(params: dict) -> str`
- **主要檔案**：tax_calculator.py
- **輸入**：`params` dict（gross_income、marital_status、扶養人數、列舉扣除額…）
- **相依**：無（linebot 以多輪補問收集參數後呼叫）
