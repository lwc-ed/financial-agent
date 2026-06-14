# linebot

LINE Bot 樞紐：接收 webhook，用 GPT function calling 判斷意圖（intent），再分派給各 feature 處理。是整個系統的中央調度。

- **對外接口（Blueprint）**：`linebot_bp` → `POST /callback`（LINE webhook）
- **核心流程**：`orchestrate()` 判斷 intent → `handle_message()` 分派（expense / credit_card / tax / news / quiz / financial_qa / wishlist …）
- **主要檔案**：linebot.py
- **資料 / 模型**：conversation_memory.py（`ConversationMemory`，跨訊息短期記憶）
- **相依**：幾乎所有 feature（credit_card、tax、news、quiz、risk、financial_status、qa_rag）＋ core.database、core.token_tracker
