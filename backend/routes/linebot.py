from flask import Blueprint, request
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent, PostbackEvent
from linebot.v3.messaging import (
    MessagingApi, ReplyMessageRequest, PushMessageRequest,
    TextMessage, Configuration, ApiClient,
    QuickReply, QuickReplyItem, URIAction,
)
from backend.routes.quiz_handler import FullInsuranceQuizHandler
from backend.database import SessionLocal
from backend.models.user import User
from backend.models.conversation_memory import ConversationMemory
from backend.models.wishlist import Wishlist
from backend.models.record import Record
from backend.routes.daily_news.daily_news_service import run_daily_news_pipeline
from backend.tax.tax_calculator import calculate_taiwan_tax_2026
from backend.utils.token_tracker import upsert_pipeline_tokens, is_over_daily_limit
from backend.utils.response_logger import log_response
from sqlalchemy import desc
from openai import OpenAI
from datetime import datetime, timedelta
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
MEMORY_TTL_HOURS = 2
MEMORY_MAX_MESSAGES = 10
MEMORY_MAX_CHARS_PER_MESSAGE = 500

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
            "description": (
                "台灣綜合所得稅試算（114年度，2026年申報）。"
                "【只有】使用者明確提到「所得稅」、「報稅」、「繳稅」、「退稅」、「稅額試算」等稅務相關字眼才觸發。"
                "使用者只是提到薪水、收入、薪資金額，但沒有明確詢問稅務時，【不】觸發此工具。"
            ),
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


def _trim_memory_content(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= MEMORY_MAX_CHARS_PER_MESSAGE:
        return text
    return text[:MEMORY_MAX_CHARS_PER_MESSAGE] + "..."


def _remember_message(db, line_user_id: str, role: str, content: str) -> None:
    content = _trim_memory_content(content)
    if not line_user_id or not content:
        return
    now = datetime.utcnow()
    db.add(ConversationMemory(
        line_user_id=line_user_id,
        role=role,
        content=content,
        created_at=now,
        expires_at=now + timedelta(hours=MEMORY_TTL_HOURS),
    ))


def _load_recent_memory(db, line_user_id: str) -> list[dict]:
    now = datetime.utcnow()
    try:
        db.query(ConversationMemory).filter(
            ConversationMemory.line_user_id == line_user_id,
            ConversationMemory.expires_at.isnot(None),
            ConversationMemory.expires_at < now,
        ).delete(synchronize_session=False)

        rows = (
            db.query(ConversationMemory)
            .filter(
                ConversationMemory.line_user_id == line_user_id,
                (ConversationMemory.expires_at.is_(None)) | (ConversationMemory.expires_at >= now),
            )
            .order_by(desc(ConversationMemory.created_at))
            .limit(MEMORY_MAX_MESSAGES)
            .all()
        )
        return [
            {"role": row.role, "content": _trim_memory_content(row.content)}
            for row in reversed(rows)
            if row.role in ("user", "assistant")
        ]
    except Exception as e:
        print("[memory] load failed:", repr(e))
        db.rollback()
        return []


def _remember_message_standalone(line_user_id: str, role: str, content: str) -> None:
    db = SessionLocal()
    try:
        _remember_message(db, line_user_id, role, content)
        db.commit()
    except Exception as e:
        db.rollback()
        print("[memory] standalone write failed:", repr(e))
    finally:
        db.close()


def orchestrate(user_msg: str, memory_messages: list[dict] | None = None) -> dict:
    """GPT 判斷意圖並抽出參數，回傳 {"intent": str, "params": dict, "token_info": dict}"""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 LINE 理財助理的意圖判斷器。"
                    "你可以參考最近短期對話記憶來補全代名詞、省略的品項、商店、金額或主題，"
                    "但如果新訊息明確改變主題，以新訊息為準。"
                ),
            }
        ]
        if memory_messages:
            messages.extend(memory_messages)
        messages.append({"role": "user", "content": user_msg})
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="required",
        )
        usage = response.usage
        tool_call = response.choices[0].message.tool_calls[0]
        return {
            "intent": tool_call.function.name,
            "params": json.loads(tool_call.function.arguments),
            "token_info": {
                "prompt_tokens":     usage.prompt_tokens     if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            },
        }
    except Exception as e:
        print("[orchestrate] error:", repr(e))
        return {"intent": "unknown", "params": {}, "token_info": {"prompt_tokens": 0, "completion_tokens": 0}}


LIFF_URL = "https://liff.line.me/2008065321-vlAGLNjW"

_dashboard_qr = QuickReply(items=[
    QuickReplyItem(action=URIAction(label="📊 儀表板", uri=LIFF_URL))
])

def _reply(reply_token: str, text: str, line_user_id: str | None = None, db=None, remember: bool = True):
    in_quiz = line_user_id and line_user_id in quiz_engine.user_sessions
    try:
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(
                    text=text,
                    quick_reply=None if in_quiz else _dashboard_qr
                )]
            )
        )
        if remember and db is not None and line_user_id:
            _remember_message(db, line_user_id, "assistant", text)
    except Exception as e:
        print("[linebot] reply failed:", repr(e))


def _reply_messages(reply_token: str, messages: list):
    try:
        line_bot_api.reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=messages)
        )
    except Exception as e:
        print("[linebot] reply_messages failed:", repr(e))


def _push(line_user_id: str, text: str, remember: bool = True):
    try:
        line_bot_api.push_message(
            PushMessageRequest(
                to=line_user_id,
                messages=[TextMessage(text=text, quick_reply=_dashboard_qr)]
            )
        )
        if remember:
            _remember_message_standalone(line_user_id, "assistant", text)
    except Exception as e:
        print("[linebot] push failed:", repr(e))


def process_credit_card_query(user_msg: str, user_id: int | None = None) -> str:
    """信用卡回饋查詢：GPT 解析品牌 → 查 DB → GPT 生成回覆，pipeline 結束後統一記錄 token。"""
    from backend.ai.ai_parser import normalize_input
    from backend.ai.benefit_query import query_benefits
    from backend.ai.format_benefit_summary import build_summary
    from backend.ai.ai_reply import generate_reply

    parsed, parser_tokens = normalize_input(user_msg)
    results  = query_benefits(
        brand_name=parsed.get("brand_name"),
        category=parsed.get("category"),
        candidates=parsed.get("candidates", []),
    )
    summary  = build_summary(parsed, results)
    reply, reply_tokens = generate_reply(user_msg, results, summary)

    if user_id:
        upsert_pipeline_tokens(
            user_id=user_id,
            source="credit_card",
            model_openai="gpt-4o-mini",
            openai_prompt=parser_tokens["prompt_tokens"] + reply_tokens["prompt_tokens"],
            openai_completion=parser_tokens["completion_tokens"] + reply_tokens["completion_tokens"],
        )

    return reply


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
    t_start      = datetime.now()
    print(f"🟢 收到 LINE 訊息：{user_msg}")
    db = SessionLocal()

    user = db.query(User).filter_by(line_user_id=line_user_id).first()
    if not user:
        try:
            profile = line_bot_api.get_profile(line_user_id)
            display_name = profile.display_name
        except Exception:
            display_name = "LINE User"
        user = User(
            provider="line",
            provider_id=line_user_id,
            name=display_name,
            email=None,
            line_user_id=line_user_id,
        )
        db.add(user)
        try:
            db.commit()
            db.refresh(user)
        except Exception:
            db.rollback()
            user = db.query(User).filter_by(line_user_id=line_user_id).first()

    def reply(text: str, remember: bool = True):
        _reply(
            event.reply_token,
            text,
            line_user_id=line_user_id,
            db=db,
            remember=remember,
        )

    # ---------- Daily token 用量限制 ----------
    if is_over_daily_limit(user.id):
        reply("⚠️ 您今日的使用量已達上限，請明日再試。")
        db.close()
        return
    # ---------- Daily token 用量限制結束 ----------

    memory_messages = _load_recent_memory(db, line_user_id)
    _remember_message(db, line_user_id, "user", user_msg)

    # ---------- 所得稅多輪補問 ----------
    if line_user_id in _tax_sessions:
        if any(kw in user_msg for kw in ["取消", "算了", "停", "cancel"]):
            del _tax_sessions[line_user_id]
            reply("已取消所得稅試算。")
        else:
            session = _tax_sessions[line_user_id]
            q = _get_next_tax_question(session)
            if q:
                ok, err = _apply_tax_answer(session, q, user_msg)
                if not ok:
                    reply(f"⚠️ {err}")
                else:
                    next_q = _get_next_tax_question(session)
                    if next_q:
                        reply(next_q["question"])
                    else:
                        del _tax_sessions[line_user_id]
                        try:
                            reply(calculate_taiwan_tax_2026(session["params"]))
                        except Exception as e:
                            print("[tax] calc error:", repr(e))
                            reply("試算失敗，請稍後再試。")
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
        reply(quiz_engine.get_rr_level_description(rr_match.group(1)))
        user.last_activity_time = datetime.now(taipei)
        db.commit()
        db.close()
        return
    # ---------- RR 等級查詢結束 ----------

    # ---------- 風險測驗關鍵字觸發（不走 GPT，避免誤判）----------
    _QUIZ_KEYWORDS = ["風險測驗", "風險評估", "投資屬性", "風險屬性", "風險偏好測驗", "做測驗", "開始測驗"]
    if any(kw in user_msg for kw in _QUIZ_KEYWORDS):
        messages = quiz_engine.handle_start_quiz(line_user_id)
        _reply_messages(event.reply_token, messages)
        user.last_activity_time = datetime.now(taipei)
        db.commit()
        db.close()
        return
    # ---------- 風險測驗關鍵字觸發結束 ----------

    # ---------- Orchestrator ----------
    _UNKNOWN_REPLY = (
        "你好！我可以幫你：\n"
        "💳 查信用卡回饋（例如：星巴克刷哪張卡）\n"
        "🧾 記帳（例如：午餐 150）\n"
        "📋 查消費紀錄（例如：查紀錄）\n"
        "🛍 欲望清單（例如：幫我加 AirPods 35000）\n"
        "📰 每日金融新聞（例如：我想看科技產業新聞）\n"
        "🧮 所得稅試算（例如：幫我算所得稅，年收入80萬）\n"
        "📊 投資風險屬性測驗（例如：幫我做風險測驗）\n"
        "📚 金融知識問答（例如：什麼是ETF？複利怎麼算？）"
    )

    result = orchestrate(user_msg, memory_messages)
    intent = result["intent"]
    params = result["params"]
    orchestrate_tokens = result["token_info"]
    print(f"[orchestrate] intent={intent}, params={params}")

    if intent == "credit_card":
        reply("🔍 正在為您查詢中，請稍候…")
        query = params.get("query", user_msg)
        _uid, _input, _ts = user.id, user_msg, t_start
        def _run_credit_card():
            result = process_credit_card_query(query, user_id=_uid)
            _push(line_user_id, result)
            log_response(_uid, _input, "credit_card", (datetime.now() - _ts).total_seconds())
        threading.Thread(target=_run_credit_card, daemon=True).start()

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
        reply(reply_text)

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
        reply(reply_text)

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
        reply(reply_text)

    elif intent == "news":
        reply("📰 正在整理今日產業新聞，請稍候…")
        final_reply = run_daily_news_pipeline(
            db=db, user_id=user.id, topic=params.get("topic", "綜合財經"),
            user_msg=user_msg,
        )
        _push(line_user_id, final_reply)
        log_response(user.id, user_msg, "news", (datetime.now() - t_start).total_seconds())

    elif intent == "tax":
        # 關鍵字守衛：訊息裡沒有稅務字眼就視為 GPT 誤判，降為 unknown
        _TAX_KEYWORDS = ["所得稅", "報稅", "繳稅", "退稅", "稅額", "稅務", "節稅", "稅率", "綜所稅"]
        if not any(kw in user_msg for kw in _TAX_KEYWORDS):
            reply(_UNKNOWN_REPLY)
        else:
            # 過濾 GPT 回傳的 None 值，只保留有實際內容的欄位
            collected = {k: v for k, v in params.items() if v is not None}
            missing = [q for q in _TAX_QUESTIONS if q["field"] not in collected]
            if not missing:
                try:
                    reply(calculate_taiwan_tax_2026(collected))
                except Exception as e:
                    print("[tax] calc error:", repr(e))
                    reply("試算失敗，請稍後再試。")
            else:
                _tax_sessions[line_user_id] = {"params": collected}
                reply(missing[0]["question"])

    elif intent == "financial_qa":
        from backend.ai.rag_service import answer_financial_question
        reply("📚 正在查詢金融知識庫，請稍候…")
        query = params.get("query", user_msg)
        _uid, _input, _ts = user.id, user_msg, t_start
        def _run_financial_qa():
            result = answer_financial_question(query)
            _push(line_user_id, result)
            log_response(_uid, _input, "financial_qa", (datetime.now() - _ts).total_seconds())
        threading.Thread(target=_run_financial_qa, daemon=True).start()

    else:  # unknown
        reply(_UNKNOWN_REPLY)

    # ---------- Token 用量記錄 + 回應時間 ----------
    elapsed = (datetime.now() - t_start).total_seconds()
    if intent not in ("credit_card", "financial_qa"):
        log_response(user.id, user_msg, intent, elapsed)
    if intent not in ("credit_card", "news"):
        upsert_pipeline_tokens(
            user_id=user.id,
            source=intent,
            model_openai="gpt-4o-mini",
            openai_prompt=orchestrate_tokens["prompt_tokens"],
            openai_completion=orchestrate_tokens["completion_tokens"],
        )

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
