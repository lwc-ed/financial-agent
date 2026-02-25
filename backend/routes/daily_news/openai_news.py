import os
from openai import OpenAI


def summarize_news_with_openai(perplexity_raw: str, topic: str) -> str:
    """
    將 Perplexity 原始新聞整理成適合 LINE 推送的摘要內容。
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    normalized_topic = (topic or "").strip()
    display_topic = "綜合產業" if not normalized_topic or normalized_topic == "無" else normalized_topic

    system_prompt = (
        "你是專業的國際金融市場編輯，擅長將市場訊息整理成 LINE 可讀的投資晨報。"
        "請嚴格遵守指定格式與篇幅，使用繁體中文，語氣專業且冷靜。"
    )
    user_prompt = f"""
請幫我整理成「LINE 投資晨報格式」，規則如下：
1. 總字數控制在 250–400 字
2. 結構固定為：
• 標題
• 發生什麼事
• 股市
• 匯市
• 大宗商品
• 接下來看什麼
• 投資思路
• 一句話總結
3. 每個段落用短句條列
4. 換行適中，不要太碎
5. emoji 不要過度使用
6. 語氣專業、冷靜、不要過度渲染
7. 不要出現資料來源與引用編號

以下是新聞內容「
{perplexity_raw}
」

以下是參考範例（僅供語氣與呈現方式參考，不可照抄）：「
📊 今日市場懶人包（3 分鐘看懂）

🔥 發生什麼事？
美國最高法院推翻部分關稅後，川普再宣布全球 15% 新關稅。政策反覆，市場難以評估後續影響，風險情緒轉趨保守。

📉 股市
• 美股期貨下跌約 0.5–0.8%
• 歐股同步走弱
→ 科技與成長股壓力較大，本週聚焦 Nvidia 財報表現

💰 匯市
• 美元自高位回落
• 歐元、英鎊、日圓小幅走升
→ 市場重新評估降息時點與政策不確定性

🪙 大宗商品
• 黃金升至三週高點（避險需求支撐）
• 原油回落（關稅升溫帶來需求疑慮）

👀 接下來看什麼？
1️⃣ 關稅是否全面落地與是否有豁免細則
2️⃣ Nvidia 財報能否支撐 AI 高估值
3️⃣ 美國經濟數據與聯準會降息預期變化

💡 上班族投資思路（簡化版）

🛡 保守型
• 黃金或防禦型類股
• 提高現金比重，降低波動

📈 進取型
• 等 AI 財報結果再評估布局時點
• 若科技股明顯修正，採分批進場，不追高

🛢 能源
• 油價短線震盪為主，適合區間思維，避免重倉

一句話總結：
當前市場屬於政策不確定驅動的觀望階段，操作重點是控制風險，而非急於加碼。」

主題偏好：{display_topic}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("OpenAI returned empty content")

    return content
