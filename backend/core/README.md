# core

跨功能的基礎建設層：資料庫連線、token 計量、回應記錄、共用嵌入模型。各 feature 都依賴 core，但 core 不依賴任何 feature。

- **database.py**：建立兩個 MySQL 引擎與 session
  - `engine` / `SessionLocal` → `financial_agent` 主庫
  - `engine_benefit` / `SessionBenefit` → `credit_card_benefits` 信用卡回饋庫
  - `Base`：所有 ORM model 的共用 declarative base（`create_all` 用）
- **token_tracker.py**：每次 AI 呼叫後累加寫入 token 用量（`upsert_pipeline_tokens`）、檢查每日上限（`is_over_daily_limit`）、白名單豁免
- **token_log.py**：`UserTokenLog` ORM（token 用量明細表）
- **response_logger.py**：`log_response()`，把每次 pipeline 回覆耗時 append 到 jsonl
- **monthly_stats.py**：CLI 腳本，把 token 明細按月彙總（`python3 -m backend.core.monthly_stats`）
- **embedder.py**：HuggingFace 嵌入模型 singleton，被 news（RSS 向量搜尋）與 qa_rag（RAG 檢索）共用

**相依**：僅第三方套件（SQLAlchemy、pymysql、sentence-transformers…），不依賴 features。
