import os
import re

from openai import OpenAI

# ── 查證型回覆品質檢查 ────────────────────────────────────────────
_VERDICT_PATTERN = re.compile(
    r"^(是[，。,、\s]|否[，。,、\s]|目前尚未確認|目前資料不足|尚無法確認|沒有確認|"
    r"目前無法|尚未有|並未|沒有宣布|尚未宣布)",
    re.MULTILINE,
)

def _has_clear_verdict(response: str) -> bool:
    """查證型回覆前 400 字是否含有明確 是/否/不足 結論。"""
    return bool(_VERDICT_PATTERN.search(response[:400]))


def _format_market_data(market_data: dict) -> str:
    """把 market_data dict 整理成可讀的文字區塊給 GPT。"""
    lines = []
    for category, items in market_data.items():
        if category == "fetched_at":
            continue
        lines.append(f"【{category}】")
        for name, data in items.items():
            if "error" in data:
                lines.append(f"  {name}: 資料取得失敗")
                continue
            price      = data.get("price", "N/A")
            change     = data.get("change", 0)
            change_pct = data.get("change_pct", 0)
            sign       = "+" if change >= 0 else ""
            lines.append(f"  {name}: {price}  ({sign}{change}, {sign}{change_pct}%)")
    return "\n".join(lines)


def _format_articles(articles: list[dict]) -> str:
    """把文章列表整理成給 GPT 的文字區塊。"""
    if not articles:
        return "（本次無相關新聞文章）"
    blocks = []
    for i, a in enumerate(articles, 1):
        block = (
            f"[{i}] 來源: {a['source']} | 時間: {a['published']}\n"
            f"標題: {a['title']}\n"
            f"內容: {a['content'][:800]}"   # 每篇最多 800 字，控制 token
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _format_historical_data(hist: dict) -> str:
    """把歷史走勢 dict 整理成給 GPT 的文字區塊。"""
    if "error" in hist:
        return f"（歷史數據取得失敗：{hist['error']}）"

    label       = hist.get("label", hist.get("ticker", ""))
    period_days = hist.get("period_days", 30)
    chg         = hist.get("period_change_pct", 0)
    high        = hist.get("period_high", {})
    low         = hist.get("period_low", {})
    records     = hist.get("records", [])

    sign = "+" if chg >= 0 else ""
    lines = [
        f"【{label} 近 {period_days} 日走勢】",
        f"期間漲跌：{sign}{chg}%",
        f"期間高點：{high.get('price', 'N/A')}（{high.get('date', '')}）",
        f"期間低點：{low.get('price', 'N/A')}（{low.get('date', '')}）",
        "",
        "日期          收盤價      漲跌幅",
    ]

    # 只顯示每週一筆（避免 token 太多），最後一筆一定顯示
    step = max(1, len(records) // 20)   # 最多顯示約 20 筆
    shown = records[::step]
    if records and records[-1] not in shown:
        shown.append(records[-1])

    for r in shown:
        sign_r = "+" if r["change_pct"] >= 0 else ""
        lines.append(f"  {r['date']}   {r['close']:>12}   {sign_r}{r['change_pct']}%")

    return "\n".join(lines)


def summarize_news_with_openai(raw_data: dict, topic: str) -> tuple[str, dict]:
    """
    從 raw_data dict 產生 LINE 推送格式的財經報告。

    raw_data 結構：
        articles:        list[dict]  — RSS 文章（依 score 排序）
        market_data:     dict        — 今日市場數據
        historical_data: dict | None — 具體標的歷史走勢（specific query 才有）
        intent:          dict | None — 意圖識別結果

    Args:
        raw_data: daily_news_service 組合的完整 dict
        topic:    使用者原始輸入
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    intent           = raw_data.get("intent") or {}
    query_type       = intent.get("type", "broad")
    is_verification  = intent.get("is_verification", False)
    display_topic    = intent.get("label") or topic or "綜合財經"
    all_articles     = raw_data.get("articles", [])
    market_data      = raw_data.get("market_data", {})
    historical_data  = raw_data.get("historical_data")

    # articles 已依 similarity score 排序，取 top 30
    articles = all_articles[:30]

    market_text   = _format_market_data(market_data)
    articles_text = _format_articles(articles)

    system_prompt = (

    "你是專業的國際金融市場編輯，負責把市場數據與新聞整理成 LINE 使用者容易閱讀的財經回覆。"

    "你的任務不是固定寫成晨報，而是根據使用者主題與新聞內容，選擇最適合的精簡格式。"

    "請使用繁體中文，語氣專業、冷靜、清楚，不誇大、不給保證獲利建議。"

    "市場數據是精確數字，必須如實呈現；新聞沒有支持的內容必須明確說明資料不足，不可捏造。"

    )

    # ── 依查詢類型決定 prompt 重心 ─────────────────────────────
    # 查證型優先：不管有沒有歷史數據，都強制走 D 格式
    if is_verification:
        focus_instruction = f"""
⚠️【查證型問題】使用者想確認某事件是否發生，必須使用格式 D（查證型回答）。
• 第一段「📌 初步結論」的第一句必須是：「是」、「否」或「目前尚未確認」
• 歷史股價與市場數據只能作為背景，不能用來證明事件是否發生
• 若新聞文章沒有直接支持使用者所問的事件，第一段就必須寫「目前資料不足以確認」

"""
    elif query_type == "specific" and historical_data and "error" not in historical_data:
        hist_text = _format_historical_data(historical_data)
        focus_instruction = f"""
本次使用者查詢的是具體標的「{display_topic}」，請以【歷史走勢分析】為報告主軸：
• 用歷史數據說明該標的的趨勢、高低點、整體漲跌幅
• 新聞文章作為補充背景，解釋可能的原因
• 今日市場數據作為最新定位參考

【{display_topic} 歷史走勢數據】
{hist_text}

"""
    else:
        focus_instruction = ""

    user_prompt = f"""
{focus_instruction}
請根據以下【市場數據】與【新聞文章】，回答使用者主題：「{display_topic}」。

請先判斷這次問題最適合哪一種回覆格式，並只輸出最終答案，不要說明你選了哪個格式。

【可選格式】

A. 綜合市場焦點：適合「今日股市焦點、最近市場怎麼看、財經重點」

B. 單一公司/產業整理：適合「台積電最近怎麼了、AI 股有什麼新聞」

C. 比較分析：適合「A 和 B 差在哪、台股和美股有什麼不同」

D. 查證型回答：適合「某事件是真的嗎、是否已經發生」

E. 事件追蹤：適合「某事件後續如何、目前進展到哪」

【共同輸出規則】

1. 總字數控制在 250–450 字，除非資料很多才可到 550 字

2. 使用短段落與條列，適合 LINE 閱讀

3. 最多使用 4 個主要段落

4. 每個段落標題要清楚，例如：

   • 重點摘要

   • 發生什麼事

   • 市場反應

   • 接下來看什麼

   • 投資人可留意

   • 資料不足處

5. 必須忠實使用市場數據中的具體數字；沒有的數字不要自行補

6. 新聞文章沒有支持的內容，不可推論成事實；「準備中」、「計劃中」、「傳言」≠「已發生」

7. 如果文章不足以回答問題，請明確寫：「目前資料不足以確認」

8. 不要出現資料來源編號，例如 [1]、[2]

9. 不要給出保證獲利、明確買賣指令或過度肯定的投資建議

10. emoji 可少量使用，但不要每行都有

11. 回覆必須像真正財經媒體的「重點整理」，而不是 AI 心得文

12. 如果多篇新聞其實在講同一事件，請整合成同一重點，不要重複描述

13. 查證型問題（含「是否/真的嗎/已經...了/宣布/確認/停止/放棄/上市了嗎/破產了嗎/設總部/蒸發」
    等詞）必須使用格式 D，不可改成一般公司近況整理，即使有歷史數據也不例外

14. 歷史股價、走勢數據只能作為背景脈絡，不能用來證明某事件是否發生

15. 若新聞文章為 0 篇，或無文章直接支持使用者所問的事件，第一段必須回答：
    「目前資料不足以確認」或「目前無新聞支持此說法」，不可用股價走勢填補

16. 查證型問題的第一段必須直接回答問題，不可先講股價、財報或公司近況

【格式細節】

如果是 A 綜合市場焦點，請輸出：

📌 重點摘要

📊 市場表現

🔎 主要驅動因素

👀 接下來看什麼

如果是 B 單一公司/產業整理，請輸出：

📌 重點摘要

📰 最新消息

📊 市場反應

👀 投資人可留意

如果是 C 比較分析，請輸出：

📌 一句話比較

🔍 共同點

⚖️ 差異點

👀 後續觀察

如果是 D 查證型回答，請輸出：

📌 初步結論（第一句必須是：「是」、「否」或「目前尚未確認」，再加一句說明）

📰 目前新聞支持什麼

⚠️ 尚未確認/資料不足處（「準備中」、「計劃中」、「傳言」≠「已發生」，請明確區分）

👀 後續觀察

如果是 E 事件追蹤，請輸出：

📌 目前進展

🧩 事件脈絡

📊 市場影響

👀 下一步觀察

【市場數據】（資料時間：{market_data.get('fetched_at', '')}）

{market_text}

【新聞文章】（共 {len(articles)} 篇，主題偏好：{display_topic}）

{articles_text}

"""

    total_chars = len(system_prompt) + len(user_prompt)
    print(f"[openai_news] prompt 總字數：{total_chars:,}  約 {total_chars//4:,} tokens（估算）")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.4,
    )

    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("OpenAI returned empty content")

    # ── 查證型品質保護：第一段沒有明確 verdict → 補提示 ──────────
    if is_verification and not _has_clear_verdict(content):
        print("[openai_news] verification query: no clear verdict detected, prepending notice")
        content = "⚠️ 目前資料不足以直接確認此事件是否發生。\n\n" + content

    # ── 附上消息來源（去重、保留出現順序）────────────────────────
    seen = []
    for a in articles:
        src = a.get("source", "").strip()
        if src and src not in seen:
            seen.append(src)
    if seen:
        sources_block = "\n─────────────────\n📰 消息來源\n" + "\n".join(f"・{s}" for s in seen)
        content = content + sources_block

    usage = response.usage
    token_info = {
        "prompt_tokens":      usage.prompt_tokens,
        "completion_tokens":  usage.completion_tokens,
        "total_tokens":       usage.total_tokens,
        "estimated_cost_usd": round(
            usage.prompt_tokens * 0.00000015 + usage.completion_tokens * 0.0000006, 6
        ),
    }

    return content, token_info
