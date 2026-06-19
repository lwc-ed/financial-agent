# credit_card

信用卡回饋查詢：把使用者輸入解析成品牌候選 → 查信用卡回饋 DB → GPT 生成白話回覆。

- **對外接口**：`benefit_query.query_benefits()`、`ai_parser.normalize_input()`、`format_benefit_summary.build_summary()`、`ai_reply.generate_reply()`（由 linebot 的 credit_card intent 依序呼叫）
- **主要檔案**：ai_parser.py（品牌解析）、benefit_query.py（查 DB，FTS→LIKE fallback）、format_benefit_summary.py、ai_reply.py
- **資料 / 模型**：scrapers/（各銀行爬蟲 + 原始 json）、credit_card_benefit_model/（4 張信用卡回饋表 ORM）
- **相依**：core.database（`SessionBenefit`，連 credit_card_benefits 庫）、OpenAI
