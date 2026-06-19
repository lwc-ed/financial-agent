## 測試 ai 資料夾的整合流程
##   執行方式（從專案根目錄）：python3 -m backend.tests.test_full_flow

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.features.credit_card.ai_parser import normalize_input
from backend.features.credit_card.benefit_query import query_benefits
from backend.features.credit_card.format_benefit_summary import build_summary
from backend.features.credit_card.ai_reply import generate_reply


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
run_test("好樂迪")
"""
run_test("這一鍋")

"""
