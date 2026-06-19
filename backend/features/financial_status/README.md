# financial_status

財務狀況指標：依記帳資料計算收支/結餘等指標與分級，並產生白話摘要（資產總覽頁使用）。純計算，不接 LLM。

- **對外接口**：
  - `financial_status_service.compute_and_store(...)`、`ensure_metrics(...)`、`refresh_for_user(...)`、`LEVEL_DISPLAY`
  - `financial_status_bp` → `GET /api/financial-status/`
- **主要檔案**：financial_status.py（route）、financial_status_service.py（計算邏輯）
- **資料 / 模型**：financial_status_model.py（`FinancialStatus` ORM）
- **相依**：core.database；資料不足（記帳 < 30 天）時改走引導文案
