import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytz

from backend.features.news.daily_news import DailyNews
from backend.features.news.intent_recognizer import recognize_intent
from backend.features.news.rss_fetcher import fetch_articles
from backend.features.news.market_data import fetch_market_data, fetch_historical_data
from backend.features.news.openai_news import summarize_news_with_openai
from backend.features.news.perplexity_search import (
    search_with_perplexity, FALLBACK_ARTICLE_THRESHOLD
)
from backend.core.token_tracker import upsert_pipeline_tokens

taipei    = pytz.timezone("Asia/Taipei")
RAW_DATA_DIR = Path(__file__).parent / "raw_data"
RAW_DATA_DIR.mkdir(exist_ok=True)


def get_taiwan_now():
    return datetime.now(taipei).replace(tzinfo=None)


def _save_raw_data(raw_data: dict, topic: str) -> Path:
    """把抓到的原始資料存成 JSON 檔，供事後人工檢查或 debug。"""
    ts       = datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M")
    safe_topic = topic.replace("/", "_") if topic else "general"
    filename = RAW_DATA_DIR / f"{ts}_{safe_topic}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
    print(f"[daily_news] raw data saved → {filename}")
    return filename


def run_daily_news_pipeline(db, user_id: int, topic: str, user_msg: str = "") -> str:
    """
    Pipeline：
      1. 意圖識別（recognize_intent）
      2. RSS 抓文章（broad_query 向量搜尋，限 24h）
      3. 抓今日市場數據
      4. 具體標的 → 歷史走勢（yfinance）
      5. 組合 raw_data，存 JSON
      6. 文章不足（< FALLBACK_ARTICLE_THRESHOLD）→ Perplexity fallback，直接回傳
      7. 存 DB（原始資料）
      8. OpenAI 產生摘要
      9. 更新 DB（摘要結果），回傳
    """
    normalized_topic = (topic or "").strip()

    try:
        # ── Step 1：意圖識別 ──────────────────────────────────────
        print(f"[daily_news] recognizing intent for: {normalized_topic!r}")
        intent = recognize_intent(normalized_topic)

        # ── Step 2：抓新聞文章（用 broad_query 做向量搜尋）────────
        search_query = intent.get("broad_query") or normalized_topic
        print(f"[daily_news] fetching articles, query={search_query!r}")
        articles = fetch_articles(query=search_query, hours_back=24)  # LINE 正式流程限 24h

        # ── Step 3：抓今日市場數據 ────────────────────────────────
        print("[daily_news] fetching market data")
        market_data = fetch_market_data()

        # ── Step 4：具體標的 → 抓歷史走勢 ────────────────────────
        historical_data = None
        if intent.get("type") == "specific" and intent.get("ticker"):
            ticker      = intent["ticker"]
            period_days = intent.get("period_days", 30)
            label       = intent.get("label", ticker)
            print(f"[daily_news] fetching historical data: {ticker} ({period_days}d)")
            historical_data = fetch_historical_data(ticker, period_days, label)

        # ── Step 5：組合 raw_data 並存 JSON ──────────────────────
        raw_data = {
            "topic":           normalized_topic or "綜合",
            "intent":          intent,
            "fetched_at":      market_data.get("fetched_at", ""),
            "market_data":     market_data,
            "historical_data": historical_data,
            "articles":        articles,
        }
        _save_raw_data(raw_data, normalized_topic or "general")

        # ── Step 6：文章不足 → Perplexity fallback ────────────────
        perplexity_tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        is_verification = intent.get("is_verification", False)
        # 查證型：文章 < 2 就觸發（即使有歷史數據也不例外）
        # 一般型：文章 < 3 且無歷史數據才觸發
        need_fallback = (
            (is_verification and len(articles) < 2)
            or (not is_verification and len(articles) < FALLBACK_ARTICLE_THRESHOLD and not historical_data)
        )
        if need_fallback:
            print(f"[daily_news] articles={len(articles)} < {FALLBACK_ARTICLE_THRESHOLD}, "
                  f"triggering Perplexity fallback (query={normalized_topic!r})")
            try:
                perplexity_content, perplexity_evidence, perplexity_tokens = search_with_perplexity(
                    query=user_msg or normalized_topic,
                    article_count=len(articles),
                )
                raw_data["perplexity_content"] = perplexity_content
                raw_data["perplexity_evidence"] = perplexity_evidence
                raw_data["perplexity_article_count"] = len(articles)
                print(f"[daily_news] perplexity content fetched, continuing to OpenAI")
            except Exception as pe:
                print(f"[daily_news] perplexity fallback error: {repr(pe)}")
                if not articles and not historical_data:
                    return "今日暫無符合主題的最新財經新聞，請稍後再試或換個主題。"

        if not articles and not historical_data:
            return "今日暫無符合主題的最新財經新聞，請稍後再試或換個主題。"

        # ── Step 7：存 DB（原始資料） ─────────────────────────────
        db_scraper = {"articles": articles, "market_data": market_data}
        if raw_data.get("perplexity_evidence"):
            db_scraper["perplexity_evidence"] = raw_data["perplexity_evidence"]
        # MySQL JSON 欄位不接受 NaN/Infinity（Python json.dumps 會直接輸出 NaN 字串）
        # 用 regex 在序列化後替換成 null
        import json, re
        _json_str = json.dumps(db_scraper, ensure_ascii=False)
        _json_str = re.sub(r'\bNaN\b', 'null', _json_str)
        _json_str = re.sub(r'\bInfinity\b', 'null', _json_str)
        _json_str = re.sub(r'\b-Infinity\b', 'null', _json_str)
        db_scraper = json.loads(_json_str)
        row = DailyNews(
            user_id=user_id,
            user_input=user_msg or normalized_topic,
            perplexity_scraper=db_scraper,
            gpt_response={"content": ""},
            created_at=get_taiwan_now(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        print(f"[daily_news] raw saved, no={row.no}")

        # ── Step 8：送 GPT 產生報告 ───────────────────────────────
        print("[daily_news] summarizing with openai")
        gpt_response, token_info = summarize_news_with_openai(raw_data, normalized_topic)
        total_k = round(token_info["total_tokens"] / 1000, 1)
        print(f"[daily_news] openai done, len={len(gpt_response)}")
        print(f"[daily_news] 本次token使用量：{total_k} k")

        # ── Step 9：更新 DB（摘要結果） ───────────────────────────
        row.gpt_response = {"content": gpt_response}
        row.created_at   = get_taiwan_now()
        db.commit()
        print(f"[daily_news] summary saved, no={row.no}")

        # ── Step 10：記錄 token 用量 ─────────────────────────────
        if user_id:
            upsert_pipeline_tokens(
                user_id=user_id,
                source="daily_news",
                model_openai="gpt-4o-mini",
                openai_prompt=token_info["prompt_tokens"],
                openai_completion=token_info["completion_tokens"],
                model_perplexity="sonar" if perplexity_tokens["prompt_tokens"] > 0 else None,
                perplexity_prompt=perplexity_tokens["prompt_tokens"],
                perplexity_completion=perplexity_tokens["completion_tokens"],
            )

        return gpt_response

    except Exception as e:
        db.rollback()
        print(f"[daily_news] pipeline error: {repr(e)}")
        return "每日產業新聞處理失敗，請稍後再試一次。"
