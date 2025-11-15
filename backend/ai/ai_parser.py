from openai import OpenAI
import os, json, re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------
# ⚠️ NORMALIZE_MAP 只放
# 【不該丟給 GPT 的完全固定詞】
# ------------------------------
NORMALIZE_MAP = {
    "chatgpt": "ChatGPT",
    "gpt": "ChatGPT",
    "openai": "OpenAI",
    "gemni": "Gemini",
}

def normalize_input(user_input: str):
    """
    1. 先用 NORMALIZE_MAP 做最基本的正規化
    2. 其他通路／店名全部交給 GPT 做多候選推斷
    """

    text = user_input.strip()
    lower = text.lower()

    # ---- 只固定不可改的詞（如 ChatGPT / GPT） ----
    if lower in NORMALIZE_MAP:
        normalized = NORMALIZE_MAP[lower]
        return {
            "brand_name": normalized,
            "candidates": [
                {"brand_name": normalized, "score": 1.0}
            ]
        }

    # ===========================================================
    # 🔥 GPT 多候選解析 Prompt（強化版）
    # ===========================================================
    prompt = f"""
你是一個信用卡品牌判別器，負責把使用者輸入轉成多個「可能的品牌名稱候選」。

請輸出多組候選品牌，依相關性由高到低排序。
必要時請加入「通路分類」作為低分候選，例如：
- 餐廳 → 「國內餐飲」
- 火鍋店 → 「火鍋餐飲」
- 百貨 → 「百貨通路」

【示例】
使用者：「這一鍋」
候選：
1. 這一鍋（score 1.0）
2. 這一鍋餐飲（score 0.85）
3. 國內餐飲（score 0.6）

使用者：「巨城」
候選：
1. BIG CITY 遠東巨城購物中心（score 1.0）
2. 遠東巨城百貨（score 0.85）
3. 遠東百貨（score 0.7）
4. 百貨通路（score 0.4）

【使用者輸入】
"{text}"

請輸出純 JSON，不要加入額外說明：

{{
  "brand_name": <string or null>,
  "candidates": [
    {{"brand_name": <string>, "score": <0~1>}}
  ]
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
        )
        raw = response.choices[0].message.content.strip()

        # 取 JSON
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            raw = match.group(0)

        result = json.loads(raw)

        # ---- 防呆 ----
        if "candidates" not in result or not isinstance(result["candidates"], list):
            result["candidates"] = []

        if result["candidates"] and not result.get("brand_name"):
            result["brand_name"] = result["candidates"][0].get("brand_name")

        print("✅ Parser 結果：", result)
        return result

    except Exception as e:
        print("normalize_input error:", e)
        return {
            "brand_name": None,
            "candidates": []
        }