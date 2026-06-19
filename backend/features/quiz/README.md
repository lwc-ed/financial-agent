# quiz

保險風險承受度測驗：多輪問答，依使用者作答評估風險屬性。session 進度存在記憶體（伺服器重啟會清空）。

- **對外接口**：`quiz_handler.FullInsuranceQuizHandler`（由 linebot 的 quiz intent 管理 session）
- **主要檔案**：quiz_handler.py
- **狀態**：`user_sessions` in-memory dict，key = line_user_id
- **相依**：core.database（讀寫使用者）
