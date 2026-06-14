# qa_rag

金融知識問答（RAG）：以向量檢索知識庫 PDF + GPT 生成答案。

- **對外接口**：`rag_service.answer_financial_question(query: str) -> str`（由 linebot 的 financial_qa intent 呼叫）
- **主要檔案**：rag_service.py（檢索 + 生成）、rebuild_chroma.py（重建向量庫工具）
- **資料**：knowledge/（知識庫 PDF + ChromaDB 向量檔）；重建：`python -m backend.features.qa_rag.rebuild_chroma`
- **相依**：core.embedder（共用嵌入模型）、ChromaDB、OpenAI
