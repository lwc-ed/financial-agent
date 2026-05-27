import json
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import numpy as np
import requests
from bs4 import BeautifulSoup

from backend.routes.daily_news.embedder import encode_query, encode_documents, cosine_similarities

CONFIG_PATH = Path(__file__).parent / "rss_config.json"
TAIPEI_TZ = timezone(timedelta(hours=8))
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# cosine similarity 低於此閾值的文章視為與 query 無關，直接丟棄
SIMILARITY_THRESHOLD = 0.3


def _load_config() -> list[dict]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    return [s for s in cfg["sources"] if s.get("active", True)]


def _to_taipei(dt: datetime | None) -> str:
    """datetime → UTC+8 ISO 字串；無法解析時回傳空字串。"""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TAIPEI_TZ).strftime("%Y-%m-%d %H:%M")


def _parse_published(entry: dict) -> datetime | None:
    """從 feedparser entry 取出 published_parsed，轉成 datetime。"""
    t = entry.get("published_parsed")
    if t:
        try:
            return datetime(*t[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    return None


def _scrape_full_text(url: str) -> str:
    """爬單篇文章全文；失敗回傳空字串。"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        for sel in [
            "article",
            "[class*='article-body']",
            "[class*='post-content']",
            "[class*='entry-content']",
            "main",
        ]:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(" ", strip=True)
                if len(text) > 200:
                    return text
        body = soup.body
        if body:
            text = body.get_text(" ", strip=True)
            if len(text) > 200:
                return text
    except Exception as e:
        print(f"[rss_fetcher] scrape error {url}: {e}")
    return ""


def _fetch_source(
    source: dict,
    query_emb: np.ndarray,
    hours_back: int | None,
    result_list: list,
    lock: threading.Lock,
) -> None:
    """
    單一 source 的完整流程：
      抓 RSS → 篩時間（hours_back=None 時取全部）→ 向量打分 → 閾值過濾 → top 5 → 爬全文 → 寫入 result_list
    """
    name = source["name"]

    # ── 抓 RSS ──────────────────────────────────────────────────
    try:
        resp = requests.get(source["rss_url"], headers=HEADERS, timeout=15)
        feed = feedparser.parse(resp.content)
        entries = feed.entries
    except Exception as e:
        print(f"[rss_fetcher] RSS error [{name}]: {e}")
        return

    if not entries:
        print(f"[rss_fetcher] no entries [{name}]")
        return

    # ── 時間過濾：hours_back=None 時取全部 ──────────────────────
    if hours_back is None:
        recent = list(entries)
    else:
        cutoff = datetime.now(tz=TAIPEI_TZ) - timedelta(hours=hours_back)
        recent = []
        for entry in entries:
            pub_dt = _parse_published(entry)
            if pub_dt and pub_dt.astimezone(TAIPEI_TZ) < cutoff:
                continue
            recent.append(entry)

    if not recent:
        print(f"[rss_fetcher] no recent articles [{name}]")
        return

    # ── 向量打分：批次 encode 所有標題，算 cosine similarity ────
    titles = [(entry.get("title") or "").strip() for entry in recent]
    try:
        doc_embs = encode_documents(titles)
        sims = cosine_similarities(query_emb, doc_embs)   # 1-D array
    except Exception as e:
        print(f"[rss_fetcher] embedding error [{name}]: {e}")
        return

    # ── 閾值過濾（方案 A）───────────────────────────────────────
    scored = [
        (float(sims[i]), recent[i])
        for i in range(len(recent))
        if float(sims[i]) >= SIMILARITY_THRESHOLD
    ]

    if not scored:
        print(f"[rss_fetcher] no articles above threshold {SIMILARITY_THRESHOLD} [{name}] → skip")
        return

    # ── 取 top 5（依 similarity 由高到低）───────────────────────
    scored.sort(key=lambda x: x[0], reverse=True)
    top5 = scored[:5]

    # ── 爬全文 & 組 article dict ─────────────────────────────────
    articles = []
    should_scrape = source.get("scrape", True)

    for sim_score, entry in top5:
        title   = (entry.get("title") or "").strip()
        url     = (entry.get("link")  or "").strip()
        pub_dt  = _parse_published(entry)
        pub_str = _to_taipei(pub_dt)
        desc    = (entry.get("summary") or entry.get("description") or "").strip()

        if should_scrape and url:
            content = _scrape_full_text(url)
            if not content:
                content = desc      # fallback 到 RSS description
        else:
            content = desc

        articles.append({
            "source":    name,
            "title":     title,
            "url":       url,
            "published": pub_str,   # UTC+8
            "content":   content,
            "score":     round(sim_score, 4),
        })

    # ── Thread-safe 寫入共享 list ────────────────────────────────
    with lock:
        result_list.extend(articles)
        print(f"[rss_fetcher] [{name}] +{len(articles)} articles "
              f"(scores: {[a['score'] for a in articles]}, total: {len(result_list)})")


def fetch_articles(
    query: str,
    hours_back: int | None = None,
    max_workers: int = 16,
) -> list[dict]:
    """
    對所有 active source 並行抓取，回傳 article list。

    Args:
        query:      使用者原始輸入，直接做 encode_query（支援中英文自然語言）
        hours_back: 抓取過去幾小時內的文章；None = 不限時間，取 RSS 全部文章
        max_workers: 並行執行緒數

    Returns:
        list of {"source", "title", "url", "published", "content", "score"}
        依 score 由高到低排序
    """
    sources = _load_config()

    # ── encode query（只做一次，所有 source 共用）───────────────
    print(f"[rss_fetcher] encoding query: {query!r}")
    query_emb = encode_query(query)

    result_list: list[dict] = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_fetch_source, src, query_emb, hours_back, result_list, lock)
            for src in sources
        ]
        try:
            for f in as_completed(futures, timeout=60):
                exc = f.exception()
                if exc:
                    print(f"[rss_fetcher] worker exception: {exc}")
        except Exception:
            print("[rss_fetcher] overall timeout reached, using partial results")

    # 依 similarity score 排序（高 → 低）
    result_list.sort(key=lambda a: a["score"], reverse=True)
    print(f"[rss_fetcher] done. total articles: {len(result_list)}")
    return result_list
