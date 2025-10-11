import openai
import os, json

openai.api_key = os.getenv("OPENAI_API_KEY")

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
    請幫我判斷並輸出 JSON 格式：
    {{
      "brand_name": <如果是品牌或商場請填正式名稱，否則 null>,
      "category": <如果是通路類別請填，如"餐飲通路"、"百貨通路"、"加油通路"，否則 null>,
      "intent": <"查詢回饋" 或 "比較回饋" 或 "其他">
    }}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        text = response.choices[0].message.content.strip()
        return json.loads(text)
    except Exception as e:
        print("normalize_input error:", e)
        return {"brand_name": None, "category": None, "intent": "查詢回饋"}