from flask import Blueprint, request
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import (
    MessagingApi, ReplyMessageRequest, PushMessageRequest,
    TextMessage, Configuration, ApiClient,
)
from backend.database import SessionLocal
from backend.models.user import User
from backend.models.wishlist import Wishlist
from backend.models.record import Record
from backend.routes.daily_news.daily_news_service import run_daily_news_pipeline
from sqlalchemy import desc
from openai import OpenAI
from datetime import datetime
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
            "name": "news",
            "description": "每日產業新聞，使用者想看新聞或指定某產業的新聞",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "產業主題，若無特定主題填「一般」"},
                },
                "required": ["topic"],
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

    # ---------- Orchestrator ----------
    result = orchestrate(user_msg)
    intent = result["intent"]
    params = result["params"]
    print(f"[orchestrate] intent={intent}, params={params}")

    if intent == "credit_card":
        _reply(event.reply_token, "🔍 正在為您查詢中，請稍候…")
        _push(line_user_id, process_credit_card_query(params.get("query", user_msg)))

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

    elif intent == "news":
        _reply(event.reply_token, "📰 正在整理今日產業新聞，請稍候…")
        final_reply = run_daily_news_pipeline(
            db=db, user_id=user.id, topic=params.get("topic", "一般")
        )
        _push(line_user_id, final_reply)

    else:  # unknown
        _reply(event.reply_token,
               "你好！我可以幫你：\n"
               "💳 查信用卡回饋（例如：星巴克刷哪張卡）\n"
               "🧾 記帳（例如：午餐 150）\n"
               "📋 查消費紀錄（例如：查紀錄）\n"
               "🛍 欲望清單（例如：幫我加 AirPods 35000）\n"
               "📰 每日產業新聞（例如：我想看科技產業新聞）")

    user.last_activity_time = datetime.now(taipei)
    db.commit()
    db.close()
