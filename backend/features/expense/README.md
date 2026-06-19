# expense

記帳：記錄使用者的支出 / 收入，並查詢歷史消費。記帳寫入後會在背景觸發 risk 風險預測。

- **對外接口（Blueprint）**：
  - `expense_record_bp` → `POST /api/expense_record/save`（新增一筆收支）
  - `expense_history_bp` → `GET /api/expense_history/recent`、`/summary`（查詢/統計）
- **主要檔案**：expense_record.py、expense_history.py
- **資料 / 模型**：record.py（`Record` ORM，收支共用，以 type 區分）
- **相依**：core.database；寫入後由 linebot 觸發 risk、financial_status
