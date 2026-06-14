# news

每日財經新聞 pipeline：識別主題 → 抓 RSS 文章（向量搜尋）→ 抓市場數據 → 文章不足時用 Perplexity fallback → OpenAI 生成摘要。

- **對外接口**：`daily_news_service.run_daily_news_pipeline(db, user_id, topic, user_msg)`（由 linebot 的 news intent 呼叫）
- **主要檔案**：intent_recognizer.py、rss_fetcher.py、market_data.py（yfinance）、perplexity_search.py、openai_news.py、daily_news_service.py（總流程）
- **資料 / 模型**：rss_config.json（RSS 來源設定）、raw_data/（debug 原始 JSON）、daily_news.py（`DailyNews` ORM）
- **相依**：core.embedder（向量搜尋）、OpenAI、Perplexity、yfinance
