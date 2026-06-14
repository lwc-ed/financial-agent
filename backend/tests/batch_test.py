"""
批次測試腳本：對 25 個預設 query 跑完整 pipeline（含 GPT），
結果存成 JSON + 可讀的 TXT 報告。

執行方式（從專案根目錄）：
  python3 -m backend.tests.batch_test
"""

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── 讓 import 找得到 backend 套件 ──────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DAILY_NEWS_DIR = ROOT / "backend" / "routes" / "daily_news"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from backend.routes.daily_news.intent_recognizer import recognize_intent
from backend.routes.daily_news.rss_fetcher        import fetch_articles
from backend.routes.daily_news.market_data         import fetch_market_data, fetch_historical_data
from backend.routes.daily_news.openai_news         import summarize_news_with_openai
from backend.routes.daily_news.perplexity_search   import (
    search_with_perplexity, FALLBACK_ARTICLE_THRESHOLD
)

TAIPEI_TZ = timezone(timedelta(hours=8))

# ── 測試題目 ──────────────────────────────────────────────────────
QUESTIONS = [
    # A. 市場情緒辨識（Market Sentiment）
    "最近市場偏向 fear 還是 greed？",
    "最近科技股氣氛偏多還是偏空？",
    "現在市場比較像牛市、震盪還是避險？",
    "最近資金是在追 AI 還是轉向防禦股？",
    "最近市場最擔心的風險是什麼？",
    # B. 因果解釋（Macro Causality）
    "為什麼美元變強有時候美股反而跌？",
    "為什麼外資買超台股時台幣常升值？",
    "為什麼油價上漲會影響科技股估值？",
    "為什麼聯準會升息會壓縮科技股本益比？",
    "AI 熱潮為什麼會影響台灣出口？",
    # C. 主題關聯（Theme Connection）
    "最近 AI、半導體跟聯準會之間有什麼關聯？",
    "最近哪些事件同時影響美股和台股？",
    "最近全球市場最重要的三件事是什麼？",
    "最近有哪些消息可能影響科技股估值？",
    "最近有哪些因素同時影響油價與美元？",
    # D. 公司 / 產業追蹤（Company Tracking）
    "最近有哪些新聞會影響 NVIDIA？",
    "老黃最近又講了什麼？",
    "蘇姿丰最近在推什麼？",
    "OpenAI 最近最大的新聞是什麼？",
    "Google 最近 AI 有什麼動作？",
    "Meta 最近為什麼被討論？",
    # E. 查證型（Fact Verification）
    "SpaceX 已經上市了嗎？",
    "聯準會確認今年會降息了嗎？",
    "NVIDIA 已經停止中國業務了嗎？",
    "台積電要去美國設總部是真的嗎？",
    "Apple 放棄 AI 計畫了嗎？",
    "中國昨天股市真的蒸發兩兆嗎？",
]

SEP  = "=" * 70
SEP2 = "-" * 70


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def run_single(query: str, market_data: dict) -> dict:
    """
    跑單一 query 的完整 pipeline，回傳結果 dict。
    market_data 已在外部抓好，不重複抓。
    """
    result = {
        "query":            query,
        "intent":           None,
        "article_count":    0,
        "articles":         [],       # 完整文章列表（debug 用）
        "historical_data":  None,     # 歷史走勢（debug 用）
        "has_historical":   False,
        "used_perplexity":  False,
        "raw_data":         None,     # 完整 raw_data dict（debug 用）
        "gpt_response":     None,
        "token_info":       None,
        "error":            None,
        "elapsed_sec":      0.0,
    }
    t0 = time.time()

    try:
        # Step 1：意圖識別
        intent = recognize_intent(query)
        result["intent"] = intent

        # Step 2：RSS 抓文章
        search_query = intent.get("broad_query") or query
        articles = fetch_articles(query=search_query)
        result["article_count"] = len(articles)
        result["articles"]      = articles   # 完整存，供 debug

        # Step 3：具體標的 → 歷史走勢
        historical_data = None
        if intent.get("type") == "specific" and intent.get("ticker"):
            historical_data = fetch_historical_data(
                intent["ticker"],
                intent.get("period_days", 30),
                intent.get("label", intent["ticker"]),
            )
            result["has_historical"]  = "error" not in historical_data
            result["historical_data"] = historical_data

        # Step 4：組合 raw_data
        raw_data = {
            "topic":           query,
            "intent":          intent,
            "fetched_at":      market_data.get("fetched_at", ""),
            "market_data":     market_data,
            "historical_data": historical_data,
            "articles":        articles,
        }
        result["raw_data"] = raw_data   # 存起來供 debug JSON 使用

        # Step 5：文章不足 → Perplexity fallback
        is_verification = intent.get("is_verification", False)
        use_perplexity = (
            (is_verification and len(articles) < 2)
            or (not is_verification and len(articles) < FALLBACK_ARTICLE_THRESHOLD and not historical_data)
        )
        if use_perplexity:
            perplexity_response, perplexity_evidence = search_with_perplexity(
                query=query,                # 使用原始輸入
                article_count=len(articles),
            )
            result["gpt_response"]        = perplexity_response
            result["used_perplexity"]     = True
            result["perplexity_evidence"] = perplexity_evidence   # 完整 evidence
            result["token_info"]          = {
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "estimated_cost_usd": 0.0,
            }
        else:
            # Step 5b：GPT 摘要
            gpt_response, token_info = summarize_news_with_openai(raw_data, query)
            result["gpt_response"] = gpt_response
            result["token_info"]   = token_info

    except Exception as e:
        result["error"] = repr(e)

    result["elapsed_sec"] = round(time.time() - t0, 1)
    return result


def main():
    total_start = time.time()
    ts = datetime.now(tz=TAIPEI_TZ).strftime("%Y%m%d_%H%M")
    out_dir = DAILY_NEWS_DIR / "batch_results"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / f"{ts}_batch_results.json"
    txt_path  = out_dir / f"{ts}_batch_report.txt"

    print(f"\n{'📋 批次測試開始':^70}")
    print(f"  題目數：{len(QUESTIONS)}")
    print(f"  結果將存至：{out_dir}")
    print(SEP)

    # ── 市場數據只抓一次（所有 query 共用）───────────────────────
    print("\n⏳ 抓取今日市場數據...")
    market_data = fetch_market_data()
    print(f"✅ 市場數據完成（{market_data.get('fetched_at', '')}）\n")

    results = []
    total_tokens  = 0
    total_cost    = 0.0
    success_count = 0
    fail_count    = 0

    for i, query in enumerate(QUESTIONS, 1):
        category = (
            "A. 市場情緒辨識"   if i <= 5  else
            "B. 因果解釋"       if i <= 10 else
            "C. 主題關聯"       if i <= 15 else
            "D. 公司/產業追蹤"  if i <= 21 else
            "E. 查證型"
        )

        print(f"\n[{i:02d}/{len(QUESTIONS)}] {category}")
        print(f"  Query: {query}")
        print(SEP2)

        result = run_single(query, market_data)
        results.append(result)

        if result["error"]:
            fail_count += 1
            print(f"  ❌ 失敗：{result['error']}")
        else:
            success_count += 1
            ti = result["token_info"]
            total_tokens += ti["total_tokens"]
            total_cost   += ti["estimated_cost_usd"]

            intent = result["intent"] or {}
            fallback_tag = "  [Perplexity↑]" if result["used_perplexity"] else ""
            verif_tag    = "  [查證型]" if intent.get("is_verification") else ""
            print(f"  ✅ 完成（{fmt_time(result['elapsed_sec'])}）{fallback_tag}{verif_tag}")
            print(f"     意圖      : {intent.get('type')} / {intent.get('label') or '—'}")
            print(f"     Ticker    : {intent.get('ticker') or '—'}")
            print(f"     broad_q   : {intent.get('broad_query', '')[:60]}")
            print(f"     文章      : {result['article_count']} 篇"
                  f"{'  + 歷史數據' if result['has_historical'] else ''}")
            if not result["used_perplexity"]:
                print(f"     Token     : {ti['total_tokens']:,}（prompt {ti['prompt_tokens']:,}"
                      f" + completion {ti['completion_tokens']:,}）")

            # ── debug：印出所有文章（score / source / title）────────
            arts = result.get("articles", [])
            if arts:
                print(f"\n  --- 抓到的文章（共 {len(arts)} 篇）---")
                for idx, a in enumerate(arts, 1):
                    score   = a.get("score", 0)
                    source  = a.get("source", "?")
                    title   = a.get("title", "")[:60]
                    pub     = a.get("published", "")[:16]
                    print(f"     [{idx:02d}] {score:.3f}  [{source}]  {pub}  {title}")
            else:
                print(f"\n  --- 抓到的文章：0 篇 ---")

            # ── debug：歷史數據摘要 ───────────────────────────────
            hist = result.get("historical_data")
            if hist and "error" not in (hist or {}):
                chg  = hist.get("period_change_pct", 0)
                high = hist.get("period_high", {})
                low  = hist.get("period_low", {})
                sign = "+" if chg >= 0 else ""
                print(f"\n  --- 歷史數據 ({hist.get('label','')}) ---")
                print(f"     期間漲跌：{sign}{chg}%")
                print(f"     高點：{high.get('price','N/A')}（{high.get('date','')}）  "
                      f"低點：{low.get('price','N/A')}（{low.get('date','')}）")
            elif hist and "error" in hist:
                print(f"\n  --- 歷史數據：❌ {hist['error']} ---")

            print()
            # 印出 GPT 回覆（前 400 字）
            preview = result["gpt_response"][:400].replace("\n", "\n     ")
            print(f"  --- 最終回覆預覽 ---\n     {preview}...")

        # 每題完成就即時存 JSON（防止中途中斷遺失）
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # ── 整體統計 ──────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{SEP}")
    print(f"  📊 批次測試完成")
    print(SEP)
    print(f"  總題數    : {len(QUESTIONS)}")
    print(f"  成功      : {success_count}")
    print(f"  失敗      : {fail_count}")
    print(f"  總 Token  : {total_tokens:,}")
    print(f"  總費用    : ${total_cost:.6f} USD")
    print(f"  總耗時    : {fmt_time(total_elapsed)}")
    print(f"  平均耗時  : {fmt_time(total_elapsed / len(QUESTIONS))}/題")

    # ── 存可讀 TXT 報告 ───────────────────────────────────────────
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"批次測試報告｜{ts}\n")
        f.write(f"{'=' * 70}\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"[{i:02d}] {r['query']}\n")
            f.write(f"{'-' * 70}\n")
            if r["error"]:
                f.write(f"❌ 失敗：{r['error']}\n\n")
                continue
            intent = r["intent"] or {}
            ti     = r["token_info"] or {}
            fallback_tag = "  [Perplexity fallback]" if r.get("used_perplexity") else ""
            verif_tag    = "  [查證型]" if intent.get("is_verification") else ""

            # ── 基本資訊 ──────────────────────────────────────────
            f.write(f"意圖：{intent.get('type')} | 標的：{intent.get('label') or '—'} "
                    f"| Ticker：{intent.get('ticker') or '—'}"
                    f"| 文章：{r['article_count']} 篇"
                    f"{'  + 歷史數據' if r['has_historical'] else ''}"
                    f"{fallback_tag}{verif_tag}\n")
            f.write(f"broad_query：{intent.get('broad_query', '')}\n")
            token_str = "Perplexity（不計 token）" if r.get("used_perplexity") else f"{ti.get('total_tokens', 0):,}"
            f.write(f"Token：{token_str}  耗時：{fmt_time(r['elapsed_sec'])}\n")

            # ── DEBUG：文章列表 ────────────────────────────────────
            arts = r.get("articles", [])
            if arts:
                f.write(f"\n【抓到的文章 {len(arts)} 篇】\n")
                for idx, a in enumerate(arts, 1):
                    score  = a.get("score", 0)
                    source = a.get("source", "?")
                    title  = a.get("title", "")
                    pub    = a.get("published", "")[:16]
                    f.write(f"  [{idx:02d}] {score:.3f}  [{source}]  {pub}  {title}\n")
            else:
                f.write(f"\n【抓到的文章：0 篇】\n")

            # ── DEBUG：Perplexity evidence（citation chain）─────────
            pev = r.get("perplexity_evidence")
            if pev:
                citations = pev.get("citations", [])
                f.write(f"\n【Perplexity Evidence】\n")
                f.write(f"  query              : {pev.get('query','')}\n")
                f.write(f"  rss_article_count  : {pev.get('rss_article_count', 0)}\n")
                f.write(f"  citations ({len(citations)} 條):\n")
                for c in citations:
                    f.write(f"    [{c.get('source','?')}]  {c.get('url','')}\n")

            # ── DEBUG：市場數據（GPT 實際拿到的數字）─────────────────
            raw = r.get("raw_data") or {}
            mkt = raw.get("market_data", {})
            if mkt:
                f.write(f"\n【市場數據（yfinance 實際值，fetched_at={mkt.get('fetched_at','')}）】\n")
                for category, items in mkt.items():
                    if category == "fetched_at":
                        continue
                    f.write(f"  {category}：")
                    parts = []
                    for name, data in items.items():
                        if "error" in data:
                            parts.append(f"{name}=ERROR")
                        else:
                            sign = "+" if data.get("change", 0) >= 0 else ""
                            parts.append(
                                f"{name}={data.get('price','N/A')}"
                                f"({sign}{data.get('change_pct','N/A')}%)"
                            )
                    f.write("  ".join(parts) + "\n")

            # ── DEBUG：歷史數據摘要 ────────────────────────────────
            hist = r.get("historical_data")
            if hist and "error" not in (hist or {}):
                chg  = hist.get("period_change_pct", 0)
                high = hist.get("period_high", {})
                low  = hist.get("period_low", {})
                sign = "+" if chg >= 0 else ""
                f.write(f"\n【歷史數據 {hist.get('label','')} / {hist.get('ticker','')}】\n")
                f.write(f"  期間漲跌：{sign}{chg}%  "
                        f"高點：{high.get('price','N/A')}（{high.get('date','')}）  "
                        f"低點：{low.get('price','N/A')}（{low.get('date','')}）\n")
            elif hist and "error" in hist:
                f.write(f"\n【歷史數據：❌ {hist['error']}】\n")

            # ── 最終回覆 ──────────────────────────────────────────
            f.write(f"\n【最終回覆】\n{r['gpt_response']}\n\n")
            f.write(f"{'=' * 70}\n\n")

        f.write(f"總結\n{'=' * 70}\n")
        f.write(f"成功：{success_count}/{len(QUESTIONS)}\n")
        f.write(f"總 Token：{total_tokens:,}\n")
        f.write(f"總費用：${total_cost:.6f} USD\n")
        f.write(f"總耗時：{fmt_time(total_elapsed)}\n")

    # ── 存 debug JSON（所有題的完整 raw_data）────────────────────
    debug_path = out_dir / f"{ts}_debug.json"
    debug_data = []
    for i, r in enumerate(results, 1):
        debug_data.append({
            "no":       i,
            "query":    r["query"],
            "raw_data": r.get("raw_data"),   # 完整 raw_data（含 market_data/articles/historical_data）
        })
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)

    print(f"\n  📄 JSON  ：{json_path}")
    print(f"  📄 TXT   ：{txt_path}")
    print(f"  📄 DEBUG ：{debug_path}")


if __name__ == "__main__":
    main()
