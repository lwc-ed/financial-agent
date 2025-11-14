from openai import OpenAI
import os, json, re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🔧 1. 使用者常見錯字/大小寫/簡稱 → 正規化成正式品牌名稱
NORMALIZE_MAP = {
    "chatgpt": "ChatGPT",
    "gpt": "ChatGPT",
    "openai": "OpenAI",
    "西提": "TASTY西堤牛排",   # 你也可以把常見通路加進來
}

def normalize_input(user_input: str):
    """
    Step 1：先在 LLM 前做「品牌正規化」處理簡稱/錯字
    Step 2：再把處理後的文字丟給 LLM 解析 brand_name / category / intent
    """

    # -------------------------
    # 🔧 Step 1：先做本地端品牌正規化（不經過 LLM）
    # -------------------------
    text = user_input.strip()
    lower = text.lower()

    if lower in NORMALIZE_MAP:
        normalized = NORMALIZE_MAP[lower]

        # 直接回傳解析結果（跳過 LLM）
        return {
            "brand_name": normalized,
            "category": None,     # 可自行定義，如 "數位服務"
            "intent": "查詢回饋",
            "candidates": [
                {
                    "brand_name": normalized,
                    "category": None,
                    "score": 1.0
                }
            ]
        }

    # ---------------------------------------
    # 🔧 Step 2：若無正規化項目 → 才丟給 LLM 解析
    # ---------------------------------------

    prompt = f"""
你是一個信用卡回饋查詢系統的「意圖與品牌解析」模組。

請將使用者輸入轉成標準結構化資訊，包含：
- 可能的品牌正式名稱（含簡稱、暱稱、錯字、縮寫的推斷）
- 所屬通路 / 大分類（例如："百貨通路"、"超商通路"、"國內餐飲"）
- 使用者意圖（查詢回饋 / 比較回饋 / 其他）
- 多個候選結果，按相關性排序（score 0~1）

使用者輸入：
"{text}"

請只輸出純 JSON，不得加入其他文字。

JSON 格式：
{{
  "brand_name": <string or null>,
  "category": <string or null>,
  "intent": <"查詢回饋"|"比較回饋"|"其他">,
  "candidates": [
    {{
      "brand_name": <string or null>,
      "category": <string or null>,
      "score": <0~1>
    }}
  ]
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()

        # 🧹 擷取 JSON
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)

        result = json.loads(raw)

        # 防呆
        if result.get("intent") not in ["查詢回饋", "比較回饋", "其他"]:
            result["intent"] = "查詢回饋"
        if "candidates" not in result or not isinstance(result["candidates"], list):
            result["candidates"] = []

        # 若最上層缺資料 → 補 candidates[0]
        if result["candidates"]:
            top = result["candidates"][0]
            if not result.get("brand_name"):
                result["brand_name"] = top.get("brand_name")
            if not result.get("category"):
                result["category"] = top.get("category")

        print("✅ 解析成功：", result)
        return result

    except Exception as e:
        print("normalize_input error:", e)
        return {
            "brand_name": None,
            "category": None,
            "intent": "查詢回饋",
            "candidates": []
        }