# risk

財務風險預測（線上推論）：讀使用者近 30 天記帳，跑 BiGRU ensemble 預測未來 7 天消費並分級，必要時推播通知。記帳成功後由 linebot 在背景觸發。

- **對外接口**：
  - `bigru_service.predict_risk_for_user(line_user_id, db)` → 預測並寫入 `RiskPrediction`
  - `notification_service.check_and_notify(...)` → 判斷是否推播（含冷卻機制）
  - `ml_risk_bp` → `GET /api/ml/history`（查歷史預測）
- **主要檔案**：bigru_service.py、feature_schema.py、notification_service.py、ml_risk.py
- **資料 / 模型**：artifacts/（feature_config.json + risk_model.pkl）、risk_prediction.py、risk_notification.py
- **相依**：core.database；**模型權重在 `ml/ml_ibm/bigru_TL_alignment/artifacts_bigru_tl/`（離線訓練產物）**、torch
