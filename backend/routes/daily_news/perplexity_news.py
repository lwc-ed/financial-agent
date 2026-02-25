import os
import requests


PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


def fetch_perplexity_news(topic: str) -> str:
    """
    呼叫 Perplexity 取得每日產業新聞原始內容。
    topic 可為使用者指定產業；若為「無」則抓綜合產業重點。
    """
    api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing PERPLEXITY_API_KEY")

    normalized_topic = (topic or "").strip()
    focus_topic = "無特定主題（請以整體國際財金市場重點為主）"
    if normalized_topic and normalized_topic != "無":
        focus_topic = normalized_topic

    prompt = f"""
你是一名專業的國際金融市場分析師，負責產出「每天一次的國際財金新聞摘要」。
任務：整理「過去 24 小時」全球重要財經與金融市場新聞，重點優先放在：{focus_topic}
其次再放在：
1. 全球股市：美股、歐股、日股、港股等主要指數的漲跌情況與原因。
2. 匯率與利率：美元指數、主要貨幣匯率、重要央行利率動向。
3. 大宗商品：原油、黃金等價格變化與主要驅動因素。
4. 重大事件：重要的政策宣告、央行發言、地緣政治、企業財報、併購等，僅挑 3–5 則最重要的。

請依照以下輸出格式：
• Executive summary：3–5 句話，總結今天全球市場的關鍵走勢與情緒。
• 市場數據概覽：用條列方式列出主要股市指數、匯率、黃金、油價的大方向（上漲/下跌）與約略幅度。
• 重要新聞 TOP 3–5：每則用「標題 + 2–3 句說明 + 為何重要（1 句）」說明。
• 投資人關注重點：列出 3–5 個未來幾天需要追蹤的風險或機會。

要求：
• 文字簡潔、給投資人看的語氣，不要廢話。
• 所有時間一律換算為台灣時間並標註。
• 若資訊有不確定處，用「可能、尚待確認」等字眼註記，不要亂猜。
• 全文使用繁體中文。
• 一定要把篇幅控制在 500–800 字之間。
"""

    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": "你是專業的國際金融市場分析師，請嚴格遵守使用者要求格式。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(PERPLEXITY_API_URL, json=payload, headers=headers, timeout=60)
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()
    if not content:
        raise RuntimeError("Perplexity returned empty content")

    return content
