import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_reply(user_input: str, query_results: list):
    """
    將查詢結果轉成自然語言回覆
    """
    if not query_results:
        return f"找不到與「{user_input}」相關的信用卡回饋資訊 😢"

    # 將查詢結果轉為文字
    result_text = "\n".join([
        f"{r['display_name']} - {r['group_name']}（{r['brand']}）：{r['reward_rate']}"
        for r in query_results
    ])

    prompt = f"""
    使用者問：「{user_input}」
    查詢結果如下：
    {result_text}

    請用自然、親切的語氣回答使用者，清楚說明信用卡回饋。
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("generate_reply error:", e)
        return "伺服器忙碌中，請稍後再試 🌀"