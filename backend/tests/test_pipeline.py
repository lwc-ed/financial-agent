"""
本地測試腳本：不需要 LINE，直接在終端機執行完整 daily news pipeline。

執行方式（從專案根目錄）：
  python3 -m backend.tests.test_pipeline
"""

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── 讓 import 找得到 backend 套件 ──────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DAILY_NEWS_DIR = ROOT / "backend" / "features" / "news"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from backend.features.news.intent_recognizer import recognize_intent
from backend.features.news.rss_fetcher        import fetch_articles
from backend.features.news.market_data         import fetch_market_data, fetch_historical_data
from backend.features.news.openai_news         import summarize_news_with_openai
from backend.features.news.perplexity_search   import (
    search_with_perplexity, FALLBACK_ARTICLE_THRESHOLD
)

TAIPEI_TZ = timezone(timedelta(hours=8))
SEP = "=" * 60


def fmt_time(seconds: float) -> str:
    return f"{seconds:.1f}s"


def print_section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def run_test(topic: str, skip_gpt: bool):
    total_start = time.time()

    # ────────────────────────────────────────────────────────────
    # Step 1：意圖識別
    # ────────────────────────────────────────────────────────────
    print_section(f"Step 1｜意圖識別  (query={topic!r})")
    t0 = time.time()
    intent = recognize_intent(topic)
    intent_elapsed = time.time() - t0

    print(f"\n✅ 意圖識別完成（耗時 {fmt_time(intent_elapsed)}）")
    print(f"   類型        : {intent['type']}")
    print(f"   查證型      : {intent.get('is_verification', False)}")
    print(f"   標的        : {intent['label']}")
    print(f"   Ticker      : {intent['ticker']}")
    print(f"   搜尋 query  : {intent['broad_query']}")
    print(f"   歷史天數    : {intent['period_days']} 天")

    # ────────────────────────────────────────────────────────────
    # Step 2：抓 RSS 文章（用 broad_query）
    # ────────────────────────────────────────────────────────────
    search_query = intent.get("broad_query") or topic
    print_section(f"Step 2｜抓 RSS 文章  (query={search_query!r}, 不限時間)")
    t0 = time.time()
    articles = fetch_articles(query=search_query)
    rss_elapsed = time.time() - t0

    print(f"\n✅ 共抓到 {len(articles)} 篇文章（耗時 {fmt_time(rss_elapsed)}）")
    if articles:
        print("\n--- 文章列表 ---")
        for i, a in enumerate(articles, 1):
            print(f"  [{i:02d}] score={a['score']:.3f} [{a['source']}] {a['published']}  {a['title'][:55]}")

    # ────────────────────────────────────────────────────────────
    # Step 3：抓市場數據
    # ────────────────────────────────────────────────────────────
    print_section("Step 3｜抓市場數據（yfinance）")
    t0 = time.time()
    market_data = fetch_market_data()
    mkt_elapsed = time.time() - t0

    print(f"\n✅ 市場數據（耗時 {fmt_time(mkt_elapsed)}）")
    for category, items in market_data.items():
        if category == "fetched_at":
            continue
        print(f"\n  【{category}】")
        for name, data in items.items():
            if "error" in data:
                print(f"    {name}: ❌ {data['error']}")
            else:
                sign = "+" if data["change"] >= 0 else ""
                print(
                    f"    {name}: {data['price']:>12}  "
                    f"{sign}{data['change']}  ({sign}{data['change_pct']}%)"
                )

    # ────────────────────────────────────────────────────────────
    # Step 4：具體標的 → 抓歷史走勢
    # ────────────────────────────────────────────────────────────
    historical_data = None
    hist_elapsed = 0.0
    if intent.get("type") == "specific" and intent.get("ticker"):
        ticker      = intent["ticker"]
        period_days = intent.get("period_days", 30)
        label       = intent.get("label", ticker)
        print_section(f"Step 4｜歷史走勢  ({label} / {ticker}, {period_days} 天)")
        t0 = time.time()
        historical_data = fetch_historical_data(ticker, period_days, label)
        hist_elapsed = time.time() - t0

        if "error" in historical_data:
            print(f"\n❌ 歷史數據取得失敗：{historical_data['error']}")
        else:
            chg  = historical_data["period_change_pct"]
            high = historical_data["period_high"]
            low  = historical_data["period_low"]
            recs = historical_data["records"]
            sign = "+" if chg >= 0 else ""
            print(f"\n✅ 歷史數據（耗時 {fmt_time(hist_elapsed)}）")
            print(f"   期間漲跌：{sign}{chg}%")
            print(f"   期間高點：{high['price']}（{high['date']}）")
            print(f"   期間低點：{low['price']}（{low['date']}）")
            print(f"   資料筆數：{len(recs)} 天")

    # ────────────────────────────────────────────────────────────
    # Step 5：組合 raw_data 並存 JSON
    # ────────────────────────────────────────────────────────────
    step_num = 5
    print_section(f"Step {step_num}｜存 JSON")
    raw_data = {
        "topic":           topic or "綜合",
        "intent":          intent,
        "fetched_at":      market_data.get("fetched_at", ""),
        "market_data":     market_data,
        "historical_data": historical_data,
        "articles":        articles,
    }

    ts         = datetime.now(tz=TAIPEI_TZ).strftime("%Y%m%d_%H%M")
    safe_topic = topic.replace("/", "_") if topic else "general"
    out_path   = DAILY_NEWS_DIR / "raw_data" / f"{ts}_{safe_topic}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 存檔完成：{out_path}")
    print(f"   文章數：{len(articles)}  市場標的：{sum(len(v) for k,v in market_data.items() if k != 'fetched_at')}")

    if skip_gpt:
        print("\n⏭  skip-gpt 模式，跳過 OpenAI 摘要")
        print(f"\n總耗時：{fmt_time(time.time() - total_start)}")
        return

    # ────────────────────────────────────────────────────────────
    # Step 6：Perplexity fallback（文章不足）
    # ────────────────────────────────────────────────────────────
    is_verification = intent.get("is_verification", False)
    use_perplexity = (
        (is_verification and len(articles) < 2)
        or (not is_verification and len(articles) < FALLBACK_ARTICLE_THRESHOLD and not historical_data)
    )
    perplexity_elapsed = 0.0

    if use_perplexity:
        print_section(
            f"Step 6｜Perplexity Fallback"
            f"  （文章數 {len(articles)} < {FALLBACK_ARTICLE_THRESHOLD}，改用即時搜尋）"
        )
        print(f"\n⚠️  RSS 文章不足（{len(articles)} 篇），觸發 Perplexity fallback")
        print(f"   原始查詢：{topic!r}")
        t0 = time.time()
        try:
            perplexity_response, perplexity_evidence = search_with_perplexity(
                query=topic,
                article_count=len(articles),
            )
            perplexity_elapsed = time.time() - t0
            source_names = perplexity_evidence.get("source_names", [])
            citations    = perplexity_evidence.get("citations", [])
            print(f"\n✅ Perplexity 完成（耗時 {fmt_time(perplexity_elapsed)}）")
            if source_names:
                print(f"   來源：{', '.join(source_names)}")
            if citations:
                print(f"   Citations（{len(citations)} 條）：")
                for c in citations:
                    print(f"     [{c['source']}]  {c['url']}")

            print_section("最終報告（Perplexity）")
            print(perplexity_response)

            print_section("整體耗時統計")
            total_elapsed = time.time() - total_start
            print(f"  意圖識別      : {fmt_time(intent_elapsed)}")
            print(f"  RSS 抓取      : {fmt_time(rss_elapsed)}")
            print(f"  市場數據      : {fmt_time(mkt_elapsed)}")
            if hist_elapsed:
                print(f"  歷史走勢      : {fmt_time(hist_elapsed)}")
            print(f"  Perplexity    : {fmt_time(perplexity_elapsed)}")
            print(f"  總計          : {fmt_time(total_elapsed)}")
        except Exception as e:
            print(f"❌ Perplexity 失敗：{e}")
        return

    if not articles and not historical_data:
        print("\n⚠️  無文章且無歷史數據，跳過 GPT")
        return

    # ────────────────────────────────────────────────────────────
    # Step 6：OpenAI 摘要
    # ────────────────────────────────────────────────────────────
    print_section("Step 6｜OpenAI 摘要")
    t0 = time.time()
    try:
        gpt_response, token_info = summarize_news_with_openai(raw_data, topic)
        gpt_elapsed = time.time() - t0
    except Exception as e:
        print(f"❌ OpenAI 失敗：{e}")
        return

    print(f"\n✅ GPT 完成（耗時 {fmt_time(gpt_elapsed)}）")
    print("\n--- Token 使用量 ---")
    print(f"  prompt tokens    : {token_info['prompt_tokens']:,}")
    print(f"  completion tokens: {token_info['completion_tokens']:,}")
    print(f"  total tokens     : {token_info['total_tokens']:,}")
    print(f"  預估費用 (USD)   : ${token_info['estimated_cost_usd']:.6f}")

    print_section("最終報告（LINE 推送內容）")
    print(gpt_response)

    print_section("整體耗時統計")
    total_elapsed = time.time() - total_start
    print(f"  意圖識別  : {fmt_time(intent_elapsed)}")
    print(f"  RSS 抓取  : {fmt_time(rss_elapsed)}")
    print(f"  市場數據  : {fmt_time(mkt_elapsed)}")
    if hist_elapsed:
        print(f"  歷史走勢  : {fmt_time(hist_elapsed)}")
    print(f"  OpenAI    : {fmt_time(gpt_elapsed)}")
    print(f"  總計      : {fmt_time(total_elapsed)}")


if __name__ == "__main__":
    DEFAULT_QUERY = "今日財經市場重點新聞"

    print("\n📰 Daily News Pipeline 本地測試")
    print("   支援任意自然語言輸入，例如：")
    print("   「今日股市焦點」「道瓊指數過去一季」「台積電最近走勢」「油價過去半年」")
    print(f"   （直接按 Enter = 使用預設：{DEFAULT_QUERY!r}）")

    raw_query = input("\n請輸入查詢：").strip()
    topic = raw_query if raw_query else DEFAULT_QUERY
    if not raw_query:
        print(f"ℹ️  使用預設查詢：{DEFAULT_QUERY!r}")

    raw_skip = input("跳過 OpenAI？省錢純測 RSS+數據 (y/N)：").strip().lower()
    skip_gpt = raw_skip == "y"

    print()
    run_test(topic=topic, skip_gpt=skip_gpt)
