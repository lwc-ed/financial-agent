"""
Perplexity fallback 搜尋模組。

當 RSS 抓到的文章不足時啟用，使用使用者原始輸入做即時網路搜尋。
回傳搜尋結果文字 + 可讀的消息來源清單。
"""
import os
from urllib.parse import urlparse

from openai import OpenAI

# ── domain → 可讀來源名稱對照表 ──────────────────────────────────
DOMAIN_SOURCE_MAP: dict[str, str] = {
    "cna.com.tw":           "中央社",
    "cnbc.com":             "CNBC",
    "reuters.com":          "Reuters",
    "bloomberg.com":        "Bloomberg",
    "ft.com":               "Financial Times",
    "nikkei.com":           "Nikkei Asia",
    "asia.nikkei.com":      "Nikkei Asia",
    "wsj.com":              "Wall Street Journal",
    "yahoo.com":            "Yahoo Finance",
    "finance.yahoo.com":    "Yahoo Finance",
    "scmp.com":             "SCMP",
    "koreaherald.com":      "Korea Herald",
    "channelnewsasia.com":  "CNA Singapore",
    "euronews.com":         "Euronews",
    "businesstech.co.za":   "BusinessTech",
    "financialpost.com":    "Financial Post",
    "zaobao.com":           "聯合早報",
    "prnewswire.com":       "PR Newswire",
    "benzinga.com":         "Benzinga",
    "apnews.com":           "AP News",
    "bbc.com":              "BBC",
    "bbc.co.uk":            "BBC",
    "techcrunch.com":       "TechCrunch",
    "theverge.com":         "The Verge",
    "wired.com":            "Wired",
    "marketwatch.com":      "MarketWatch",
    "investing.com":        "Investing.com",
    "seekingalpha.com":     "Seeking Alpha",
    "thestreet.com":        "The Street",
    "fortune.com":          "Fortune",
    "businessinsider.com":  "Business Insider",
    "economist.com":        "The Economist",
    "washingtonpost.com":   "Washington Post",
    "nytimes.com":          "New York Times",
    "theguardian.com":      "The Guardian",
    "foxbusiness.com":      "Fox Business",
    "barrons.com":          "Barron's",
    "morningstar.com":      "Morningstar",
}

FALLBACK_ARTICLE_THRESHOLD = 3   # 文章數低於此值才觸發 fallback


def _url_to_source_name(url: str) -> str:
    """把 URL 轉成可讀的來源名稱。"""
    try:
        host = urlparse(url).netloc.lower()
        host = host.removeprefix("www.")
        # 先完全比對
        if host in DOMAIN_SOURCE_MAP:
            return DOMAIN_SOURCE_MAP[host]
        # 再用 endswith 比對（處理子域名）
        for domain, name in DOMAIN_SOURCE_MAP.items():
            if host.endswith(domain):
                return name
        # fallback：去掉 TLD，首字母大寫
        parts = host.split(".")
        return parts[-2].capitalize() if len(parts) >= 2 else host
    except Exception:
        return url


def search_with_perplexity(query: str, article_count: int) -> tuple[str, dict]:
    """
    使用 Perplexity 做即時網路搜尋。

    Args:
        query:         使用者原始輸入（不用 rewrite 過的 broad_query）
        article_count: 目前 RSS 找到的文章數（用於透明度聲明）

    Returns:
        (response_text, evidence)

        response_text  已包含透明度聲明與來源區塊，供直接推送給使用者
        evidence       結構化 dict，保存完整 citation chain：
            {
                "type":               "perplexity",
                "query":              str,         # 送出的原始查詢
                "rss_article_count":  int,         # fallback 前 RSS 的文章數
                "citations": [
                    {"url": str, "source": str},   # 每條引用的 URL + 可讀來源名
                    ...
                ],
                "source_names":       list[str],   # 去重後的來源名列表（方便顯示）
                "raw_response":       str,          # Perplexity 原始回覆（不含透明度聲明）
            }
    """
    api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing PERPLEXITY_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.perplexity.ai",
    )

    system_prompt = (
        "你是 LINE Bot 財經資訊查詢助理，專門根據最新網路資訊回答使用者問題。"
        "【語言規定】無論任何情況，你的回覆必須全程使用繁體中文，嚴禁使用日文、簡體中文或其他語言。"
        "語氣精確、冷靜。"
        "若問題是查證型（是否發生某事），請在第一句直接回答：是/否/目前尚未確認。"
        "只陳述有明確來源支持的事實，不推論、不捏造。"
        "準備中、計劃中、傳言、可能 ≠ 已發生，請明確區分。"
        "【重要限制】你是 LINE Bot，不可要求使用者提供截圖、連結、指數數值或任何額外資料。"
        "若即時搜尋結果不足以確認，請直接回答「目前資料不足以判斷」，"
        "並用既有資料給出最保守結論，可建議使用者自行查閱哪些資源，但不得要求對方提供。"
    )

    response = client.chat.completions.create(
        model="sonar",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ],
        temperature=0.2,
    )

    raw_content = (response.choices[0].message.content or "").strip()

    # ── 取 citations（Perplexity 特有欄位），建立結構化 evidence ──
    citation_urls: list[str] = getattr(response, "citations", []) or []
    structured_citations: list[dict] = []
    source_names: list[str] = []
    seen: set[str] = set()
    for url in citation_urls:
        name = _url_to_source_name(url)
        structured_citations.append({"url": url, "source": name})
        if name not in seen:
            seen.add(name)
            source_names.append(name)

    evidence = {
        "type":              "perplexity",
        "query":             query,
        "rss_article_count": article_count,
        "citations":         structured_citations,   # 完整 URL + 來源名
        "source_names":      source_names,           # 去重顯示用
        "raw_response":      raw_content,            # Perplexity 原始回覆
    }

    # ── 組合透明度聲明 + 內容 + 來源區塊 ─────────────────────
    notice = (
        f"⚠️ 近期相關報導不足（RSS 僅找到 {article_count} 篇），"
        f"以下結果來自即時網路搜尋：\n\n"
    )

    sources_block = ""
    if source_names:
        sources_block = (
            "\n─────────────────\n"
            "📰 消息來源（即時搜尋）\n"
            + "\n".join(f"・{s}" for s in source_names)
        )

    full_response = notice + raw_content + sources_block
    return full_response, evidence
