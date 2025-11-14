from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_reply(user_input: str, query_results: list, summary: str):
    """
    query_results：從 DB 查出的結果（已排序）
    summary：build_summary(parsed, results) 產生的摘要
    """

    # ========= 無查詢結果 =========
    if not query_results:
        return f"找不到與「{user_input}」相關的信用卡回饋資訊 💳"

    # ========= 找最高回饋 =========
    def extract_reward_value(r):
        raw = r.get("reward_rate")
        if not raw:
            return -1
        cleaned = str(raw).replace("%", "").strip()
        try:
            return float(cleaned)
        except:
            return -1

    sorted_results = sorted(query_results, key=extract_reward_value, reverse=True)
    best = sorted_results[0]
    others = sorted_results[1:]

    # ========= 固定格式（LLM 不可修改） =========
    def format_item(r):
        bank = r.get("bank", "未知銀行")
        card = r.get("card_name", "未知卡片")
        rate = r.get("reward_rate") or "（尚未提供）"
        display = r.get("display_name") or "（無項目名稱）"
        return f"{bank} {card}：{display}（{rate}）"

    best_text = format_item(best)
    others_text = "\n".join(format_item(r) for r in others) if others else "無"

    # ========= LLM 系統規則 =========
    system_prompt = """
你是一個信用卡回饋查詢的小助手。

⚠【嚴格限制】⚠
你不能修改以下內容：
- 銀行名稱
- 卡片名稱
- 回饋 %
- display_name（項目名稱）
- 排序
- summary 的文字內容

你只能：
- 讓語氣更自然、親切
- 做輕量級語句修飾
- 加上舒適的段落過渡
- 不得添增新的優惠、不能猜測資訊
- 不得改動提供的數據與排序
"""

    # ========= LLM 輸入 =========
    user_prompt = f"""
使用者詢問：「{user_input}」

以下是 **不可修改的固定資料**：

【最高推薦】
{best_text}

【其他選項】
{others_text}

【完整回饋摘要（不可修改）】
{summary}

請用自然、不浮誇、正式但親切的語氣，把以上固定內容包裝成一段使用者能理解的回覆。
"""

    # ========= 呼叫 LLM =========
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("generate_reply error:", e)
        return "伺服器忙碌中，請稍後再試 🌀"