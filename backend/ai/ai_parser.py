from openai import OpenAI
import os, json, re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def normalize_input(user_input: str):
    """
    利用 GPT 解析使用者輸入，將簡稱、錯字、語意轉換成標準品牌或分類
    回傳格式：
    {
      "brand_name": "遠東百貨",
      "category": "百貨通路",
      "intent": "查詢回饋"
    }
    """
    prompt = f"""
    使用者輸入：「{user_input}」
    請幫我判斷並輸出「純 JSON 格式」，不要加任何說明文字：
    若無法確定意圖，請預設 intent 為 "查詢回饋"。
    {{
      "brand_name": <如果是品牌或商場請填正式名稱，否則 null>,
      "category": <如果是通路類別請填，如"餐飲通路"、"百貨通路"、"加油通路"，否則 null>,
      "intent": <"查詢回饋" 或 "比較回饋" 或 "其他">
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        text = response.choices[0].message.content.strip()

        # 🧹 用正則擷取出 { ... } 的 JSON 內容
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        else:
            print("⚠️ GPT 回傳非 JSON 格式:", text)

        result = json.loads(text)
        print("✅ 解析成功：", result)
        return result

    except Exception as e:
        print("normalize_input error:", e)
        print("原始文字回傳：", locals().get("text", "無內容"))
        return {"brand_name": None, "category": None, "intent": "查詢回饋"}