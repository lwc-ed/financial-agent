from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_reply(user_input: str, query_results: list):
    if not query_results:
        return f"找不到與「{user_input}」相關的信用卡回饋資訊 💳"

    lines = []
    for r in query_results:
        brand_list = ', '.join(r.get('brands', []))
        lines.append(f"🏷️ {r.get('display_name', '未知活動')}｜分類：{r.get('group_name', '未分類')}｜品牌：{brand_list}")

    result_text = "\n".join(lines)

    prompt = f"""
    使用者問：「{user_input}」
    查詢結果如下：
    {result_text}

    請用自然、親切的語氣回答使用者，清楚說明相關信用卡回饋資訊。
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("generate_reply error:", e)
        return "伺服器忙碌中，請稍後再試 🌀"