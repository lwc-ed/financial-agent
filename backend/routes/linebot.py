from flask import Blueprint, request
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent, PostbackEvent
from linebot.v3.messaging import (
    MessagingApi, ReplyMessageRequest, PushMessageRequest,
    TextMessage, Configuration, ApiClient,
)
from backend.routes.quiz_handler import FullInsuranceQuizHandler
from backend.database import SessionLocal
from backend.models.user import User
from backend.models.wishlist import Wishlist
from backend.models.record import Record
from backend.routes.daily_news.daily_news_service import run_daily_news_pipeline
from backend.tax.tax_calculator import calculate_taiwan_tax_2026
from sqlalchemy import desc
from openai import OpenAI
from datetime import datetime
import threading
import re
import urllib.parse
import pytz
import json
import os
from dotenv import load_dotenv

taipei = pytz.timezone("Asia/Taipei")

linebot_bp = Blueprint("linebot", __name__)

load_dotenv()

CHANNEL_SECRET = os.getenv("CHANNEL_SECRET", "").strip()
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN", "").strip()

if not CHANNEL_SECRET or not CHANNEL_ACCESS_TOKEN:
    raise RuntimeError(
        "Missing LINE credentials. Please set CHANNEL_SECRET and CHANNEL_ACCESS_TOKEN (env or .env)."
    )

handler = WebhookHandler(CHANNEL_SECRET)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
quiz_engine = FullInsuranceQuizHandler()

# --------------------------------------------------
# 所得稅多輪對話 session（in-memory，伺服器重啟會清空）
# --------------------------------------------------
_tax_sessions: dict[str, dict] = {}

_TAX_QUESTIONS = [
    {
        "field": "gross_income",
        "question": "請問您今年（114年度）的「綜合所得總額」是多少元？\n（例：年薪80萬請輸入「800000」或「80萬」）",
        "type": "float",
    },
    {
        "field": "marital_status",
        "question": "請問是否與配偶「合併申報」？\n（請回覆「是」或「否」）",
        "type": "yn",
    },
    {
        "field": "num_under_70",
        "question": "申報戶中，未滿70歲共幾人？\n（含您本人、配偶及所有受扶養親屬，請輸入整數）",
        "type": "int",
    },
    {
        "field": "num_over_70",
        "question": "申報戶中，年滿70歲以上的直系尊親屬幾人？\n（若無請填「0」）",
        "type": "int",
    },
    {
        "field": "salary_earners",
        "question": "戶內有「薪資收入」的人數共幾人？\n（每人最高可扣除 218,000 元，若只有您本人請填「1」）",
        "type": "int",
    },
]


def _parse_tax_number(text: str) -> float | None:
    text = text.strip().replace(",", "").replace("，", "")
    m = re.search(r"(\d+(?:\.\d+)?)萬", text)
    if m:
        return float(m.group(1)) * 10000
    m = re.search(r"\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


def _parse_yn(text: str) -> str | None:
    t = text.strip().lower()
    if any(kw in t for kw in ["是", "有", "y", "yes", "合併"]):
        return "y"
    if any(kw in t for kw in ["否", "沒", "没", "无", "n", "no", "不"]):
        return "n"
    return None


def _get_next_tax_question(session: dict) -> dict | None:
    params = session["params"]
    for q in _TAX_QUESTIONS:
        if params.get(q["field"]) is None:
            return q
    return None


def _apply_tax_answer(session: dict, q: dict, text: str) -> tuple[bool, str]:
    """解析使用者回答，回傳 (成功, 錯誤提示)"""
    if q["type"] in ("float", "int"):
        val = _parse_tax_number(text)
        if val is None:
            return False, f"請輸入有效的數字，例如「800000」或「80萬」\n\n{q['question']}"
        session["params"][q["field"]] = int(val) if q["type"] == "int" else val
    elif q["type"] == "yn":
        val = _parse_yn(text)
        if val is None:
            return False, f"請回覆「是」或「否」\n\n{q['question']}"
        session["params"][q["field"]] = val
    return True, ""


# --------------------------------------------------
# 工具清單：新增功能時在此加入 tool 定義
# --------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "credit_card",
            "description": "信用卡回饋查詢，使用者詢問某商店或品牌要刷哪張卡、哪張卡回饋最高",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "商店或品牌名稱，從使用者輸入中抽取"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expense",
            "description": "記帳，使用者說要記錄消費、花費了多少錢",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "消費類別，例如：午餐、交通、飲料"},
                    "amount":   {"type": "integer", "description": "消費金額（純數字）"},
                },
                "required": ["category", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_expense",
            "description": "查詢消費紀錄，使用者說查紀錄、我花了多少錢、最近消費",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wishlist",
            "description": "新增欲望清單，使用者想記錄想買的東西",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "欲購買的品項清單",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item":  {"type": "string",  "description": "品項名稱"},
                                "price": {"type": "integer", "description": "價格"},
                            },
                            "required": ["item", "price"],
                        },
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tax",
            "description": "台灣綜合所得稅試算（114年度，2026年申報），使用者詢問要繳多少稅、所得稅怎麼算、幫我試算所得稅",
            "parameters": {
                "type": "object",
                "properties": {
                    "gross_income":       {"type": "number",  "description": "年度綜合所得總額（元）"},
                    "marital_status":     {"type": "string",  "enum": ["y", "n"], "description": "是否有配偶合併申報，y=是，n=否"},
                    "num_under_70":       {"type": "integer", "description": "申報戶70歲以下總人數（含本人、配偶、受扶養親屬）"},
                    "num_over_70":        {"type": "integer", "description": "70歲以上直系尊親屬人數"},
                    "salary_earners":     {"type": "integer", "description": "戶內有薪資收入的人數"},
                    "savings_interest":   {"type": "number",  "description": "儲蓄投資利息所得（元）"},
                    "num_disabled":       {"type": "integer", "description": "身心障礙者人數"},
                    "num_preschool":      {"type": "integer", "description": "6歲(含)以下幼兒人數"},
                    "num_college":        {"type": "integer", "description": "就讀大專院校子女人數"},
                    "num_ltc":            {"type": "integer", "description": "符合長期照顧扣除人數"},
                    "rent_amount":        {"type": "number",  "description": "年度房屋租金支出（元）"},
                    "itemized_deduction": {"type": "number",  "description": "列舉扣除額總和（若選用列舉而非標準扣除）"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "news",
            "description": "每日產業新聞，使用者想看新聞或指定某產業的新聞",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "使用者想查詢的財經主題，直接取用使用者的原始說法（例如「台股」「科技股」「美股」「比特幣」），若無特定主題填「綜合財經」"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "quiz",
            "description": "投資風險屬性評估測驗，使用者想了解自己的投資風險偏好、做風險評估、投資屬性測驗、理財風險測驗",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "financial_qa",
            "description": "金融知識問答，使用者詢問理財、投資、保險、股票、基金、ETF、複利、資產配置等金融知識相關問題",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "使用者的問題原文"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unknown",
            "description": "無法判斷意圖，使用者說的不屬於任何已知功能（例如打招呼、閒聊）",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # 預留：新增功能時在此解除註解並填入定義
    # {"type": "function", "function": {"name": "saving_challenge", "description": "...", "parameters": {...}}},
    # {"type": "function", "function": {"name": "ml_predict",       "description": "...", "parameters": {...}}},
    # {"type": "function", "function": {"name": "budget_alert",     "description": "...", "parameters": {...}}},
    # {"type": "function", "function": {"name": "report",           "description": "...", "parameters": {...}}},
]


def orchestrate(user_msg: str) -> dict:
    """GPT 判斷意圖並抽出參數，回傳 {"intent": str, "params": dict}"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_msg}],
            tools=TOOLS,
            tool_choice="required",
        )
        tool_call = response.choices[0].message.tool_calls[0]
        return {
            "intent": tool_call.function.name,
            "params": json.loads(tool_call.function.arguments),
        }
    except Exception as e:
        print("[orchestrate] error:", repr(e))
        return {"intent": "unknown", "params": {}}


def _reply(reply_token: str, text: str):
    try:
        line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text=text)])
        )
    except Exception as e:
        print("[linebot] reply failed:", repr(e))


def _reply_messages(reply_token: str, messages: list):
    try:
        line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=messages)
        )
    except Exception as e:
        print("[linebot] reply_messages failed:", repr(e))


def _push(line_user_id: str, text: str):
    try:
        line_bot_api.push_message(
            PushMessageRequest(to=line_user_id, messages=[TextMessage(text=text)])
        )
    except Exception as e:
        print("[linebot] push failed:", repr(e))


def process_credit_card_query(user_msg: str) -> str:
    """信用卡回饋查詢：GPT 解析品牌 → 查 DB → GPT 生成回覆"""
    from backend.ai.ai_parser import normalize_input
    from backend.ai.benefit_query import query_benefits
    from backend.ai.format_benefit_summary import build_summary
    from backend.ai.ai_reply import generate_reply

    parsed    = normalize_input(user_msg)
    results   = query_benefits(
        brand_name=parsed.get("brand_name"),
        category=parsed.get("category"),
        candidates=parsed.get("candidates", []),
    )
    summary   = build_summary(parsed, results)
    return generate_reply(user_msg, results, summary)


@linebot_bp.route("/callback", methods=["POST"])
def callback():
    body      = request.get_data(as_text=True)
    signature = request.headers.get("X-Line-Signature")
    try:
        handler.handle(body, signature)
    except Exception as e:
        print("[callback] handler.handle failed:", repr(e))
        raise
    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    line_user_id = event.source.user_id
    user_msg     = event.message.text
    print(f"🟢 收到 LINE 訊息：{user_msg}")
    db = SessionLocal()

    # ---------- Google 綁定檢查 ----------
    user = db.query(User).filter_by(line_user_id=line_user_id).first()
    if not user:
        user = db.query(User).filter(
            User.provider == "google",
            User.line_user_id == line_user_id,
        ).first()
        if not user:
            _reply(event.reply_token,
                   "⚠️ 您尚未綁定帳號，請先點擊下方連接進行 Google 登入並綁定 LINE\n"
                   "https://financial-agent.it.com/login_google\n"
                   "若綁定失敗可以參照以下步驟⭣\n"
                   "IPhone使用者：\n主頁\n  ⭣\n設定(右上角)\n  ⭣\nLINE Labs\n  ⭣\n關閉「使用預設瀏覽器開啟連結」")
            db.close()
            return
    # ---------- 綁定檢查完成 ----------

    # ---------- 所得稅多輪補問 ----------
    if line_user_id in _tax_sessions:
        if any(kw in user_msg for kw in ["取消", "算了", "停", "cancel"]):
            del _tax_sessions[line_user_id]
            _reply(event.reply_token, "已取消所得稅試算。")
        else:
            session = _tax_sessions[line_user_id]
            q = _get_next_tax_question(session)
            if q:
                ok, err = _apply_tax_answer(session, q, user_msg)
                if not ok:
                    _reply(event.reply_token, f"⚠️ {err}")
                else:
                    next_q = _get_next_tax_question(session)
                    if next_q:
                        _reply(event.reply_token, next_q["question"])
                    else:
                        del _tax_sessions[line_user_id]
                        try:
                            _reply(event.reply_token, calculate_taiwan_tax_2026(session["params"]))
                        except Exception as e:
                            print("[tax] calc error:", repr(e))
                            _reply(event.reply_token, "試算失敗，請稍後再試。")
        user.last_activity_time = datetime.now(taipei)
        db.commit()
        db.close()
        return
    # ---------- 所得稅多輪補問結束 ----------

    # ---------- 風險測驗進行中：文字訊息直接重送當前題目 ----------
    if line_user_id in quiz_engine.user_sessions:
        current_q = quiz_engine.user_sessions[line_user_id]["current_q"]
        _reply_messages(event.reply_token, [quiz_engine.build_question_message(line_user_id, current_q)])
        user.last_activity_time = datetime.now(taipei)
        db.commit()
        db.close()
        return
    # ---------- 風險測驗進行中結束 ----------

    # ---------- RR 等級查詢（RR1~RR5）----------
    rr_match = re.match(r"^(RR[1-5])$", user_msg.strip().upper())
    if rr_match:
        _reply(event.reply_token, quiz_engine.get_rr_level_description(rr_match.group(1)))
        user.last_activity_time = datetime.now(taipei)
        db.commit()
        db.close()
        return
    # ---------- RR 等級查詢結束 ----------

    # ---------- Orchestrator ----------
    result = orchestrate(user_msg)
    intent = result["intent"]
    params = result["params"]
    print(f"[orchestrate] intent={intent}, params={params}")

    if intent == "credit_card":
        _reply(event.reply_token, "🔍 正在為您查詢中，請稍候…")
        query = params.get("query", user_msg)
        threading.Thread(
            target=lambda: _push(line_user_id, process_credit_card_query(query)),
            daemon=True,
        ).start()

    elif intent == "expense":
        try:
            db.add(Record(
                line_user_id=line_user_id,
                type="支出",
                category=params["category"],
                amount=params["amount"],
                note="",
            ))
            db.commit()
            reply_text = f"已幫你記錄：{params['category']} {params['amount']} 元 ✅"
        except Exception as e:
            db.rollback()
            print("[linebot] expense write error:", repr(e))
            reply_text = "記帳失敗 QQ，等等再試試看。"
        _reply(event.reply_token, reply_text)

    elif intent == "query_expense":
        try:
            rows = (
                db.query(Record)
                .filter(Record.line_user_id == line_user_id)
                .order_by(desc(Record.timestamp), desc(Record.no))
                .limit(5)
                .all()
            )
            if not rows:
                reply_text = "你目前還沒有任何記帳紀錄喔～\n可以試試說：午餐 150"
            else:
                lines = ["你最近的記帳紀錄："]
                for r in rows:
                    line = f"- {r.category} {r.amount} 元"
                    if r.note:
                        line += f"（{r.note}）"
                    lines.append(line)
                reply_text = "\n".join(lines)
        except Exception as e:
            print("[linebot] query_expense error:", repr(e))
            reply_text = "查詢失敗，請稍後再試。"
        _reply(event.reply_token, reply_text)

    elif intent == "wishlist":
        try:
            added = []
            for item_data in params.get("items", []):
                db.add(Wishlist(user_id=user.id, item_name=item_data["item"], price=item_data["price"]))
                added.append(f"{item_data['item']} (${item_data['price']})")
            db.commit()
            if added:
                reply_text = f"已新增 {len(added)} 筆清單！\n" + "\n".join(f"✅ {i}" for i in added)
            else:
                reply_text = "沒有找到有效的品項，請重新輸入。"
        except Exception as e:
            db.rollback()
            print("[linebot] wishlist error:", repr(e))
            reply_text = f"新增失敗：{str(e)}"
        _reply(event.reply_token, reply_text)

    elif intent == "quiz":
        messages = quiz_engine.handle_start_quiz(line_user_id)
        _reply_messages(event.reply_token, messages)

    elif intent == "news":
        _reply(event.reply_token, "📰 正在整理今日產業新聞，請稍候…")
        final_reply = run_daily_news_pipeline(
            db=db, user_id=user.id, topic=params.get("topic", "一般")
        )
        _push(line_user_id, final_reply)

    elif intent == "tax":
        # 過濾 GPT 回傳的 None 值，只保留有實際內容的欄位
        collected = {k: v for k, v in params.items() if v is not None}
        missing = [q for q in _TAX_QUESTIONS if q["field"] not in collected]
        if not missing:
            try:
                _reply(event.reply_token, calculate_taiwan_tax_2026(collected))
            except Exception as e:
                print("[tax] calc error:", repr(e))
                _reply(event.reply_token, "試算失敗，請稍後再試。")
        else:
            _tax_sessions[line_user_id] = {"params": collected}
            _reply(event.reply_token, missing[0]["question"])

    elif intent == "financial_qa":
        from backend.ai.rag_service import answer_financial_question
        _reply(event.reply_token, "📚 正在查詢金融知識庫，請稍候…")
        query = params.get("query", user_msg)
        threading.Thread(
            target=lambda: _push(line_user_id, answer_financial_question(query)),
            daemon=True,
        ).start()

    else:  # unknown
        _reply(event.reply_token,
               "你好！我可以幫你：\n"
               "💳 查信用卡回饋（例如：星巴克刷哪張卡）\n"
               "🧾 記帳（例如：午餐 150）\n"
               "📋 查消費紀錄（例如：查紀錄）\n"
               "🛍 欲望清單（例如：幫我加 AirPods 35000）\n"
               "📰 每日金融新聞（例如：我想看科技產業新聞）\n"
               "🧮 所得稅試算（例如：幫我算所得稅，年收入80萬）\n"
               "📊 投資風險屬性測驗（例如：幫我做投資風險評估）\n"
               "📚 金融知識問答（例如：什麼是ETF？複利怎麼算？）")

    user.last_activity_time = datetime.now(taipei)
    db.commit()
    db.close()


@handler.add(PostbackEvent)
def handle_postback(event):
    line_user_id = event.source.user_id
    params = dict(urllib.parse.parse_qsl(event.postback.data))

    if params.get("action") != "full_quiz":
        return

    db = SessionLocal()
    try:
        user = db.query(User).filter_by(line_user_id=line_user_id).first()
        messages = quiz_engine.handle_quiz_postback(line_user_id, params)
        _reply_messages(event.reply_token, messages)
        if user:
            user.last_activity_time = datetime.now(taipei)
            db.commit()
    except Exception as e:
        print("[linebot] handle_postback error:", repr(e))
    finally:
        db.close()
