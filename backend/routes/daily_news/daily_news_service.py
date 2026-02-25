import os
from datetime import datetime
import pytz
from backend.models.daily_news import DailyNews
from backend.routes.daily_news.perplexity_news import fetch_perplexity_news
from backend.routes.daily_news.openai_news import summarize_news_with_openai

taipei = pytz.timezone("Asia/Taipei")


def get_taiwan_now():
    # DB 欄位是 DateTime（無 timezone），這裡寫入台灣當地時間的 naive datetime
    return datetime.now(taipei).replace(tzinfo=None)


def run_daily_news_pipeline(db, user_id: int, topic: str) -> str:
    """
    流程：Perplexity -> 存 DB -> OpenAI -> 更新 DB -> 回傳摘要
    """
    if not os.getenv("PERPLEXITY_API_KEY", "").strip():
        return "目前尚未設定 Perplexity API Key，暫時無法提供每日產業新聞。"

    try:
        perplexity_raw = fetch_perplexity_news(topic)
        print("[daily_news] perplexity fetched, len =", len(perplexity_raw))

        row = DailyNews(
            user_id=user_id,
            perplexity_scraper={"content": perplexity_raw},
            gpt_response=None,
            created_at=get_taiwan_now(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        print("[daily_news] raw saved, no =", row.no)

        gpt_response = summarize_news_with_openai(perplexity_raw, topic)
        print("[daily_news] openai summarized, len =", len(gpt_response))
        row.gpt_response = {"content": gpt_response}
        row.created_at = get_taiwan_now()
        db.commit()
        print("[daily_news] summary saved, no =", row.no)

        return gpt_response

    except Exception as e:
        db.rollback()
        print("[daily_news] pipeline error:", repr(e))
        return "每日產業新聞處理失敗，請稍後再試一次。"
