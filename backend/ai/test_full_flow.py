## 測試 ai 資料夾的整合流程

from ai_parser import normalize_input
from benefit_query import query_benefits
from format_benefit_summary import build_summary
from ai_reply import generate_reply


def run_test(user_input):
    print("\n==============================")
    print(f"💬 測試輸入：{user_input}")
    print("==============================")

    # ---------- Step 1. Parser ----------
    parsed = normalize_input(user_input)
    print("\n🧩 Parser 輸出：")
    print(parsed)

    # ---------- Step 2. Database Query ----------
    results = query_benefits(
        brand_name=parsed.get("brand_name"),
        category=parsed.get("category"),
        candidates=parsed.get("candidates"),
    )
    print("\n📊 DB 查詢結果：")
    for r in results:
        print(r)

    # ---------- Step 3. Summary ----------
    summary = build_summary(parsed, results)
    print("\n📄 Summary：")
    print(summary)

    # ---------- Step 4. AI Reply ----------
    reply = generate_reply(user_input, results, summary)
    print("\n🤖 AI 回覆：")
    print(reply)


# ======== 測試案例 ========

run_test("ChatGPT")
run_test("GPT")
run_test("西提")
run_test("巨城")
run_test("kkbox")
run_test("這一鍋")