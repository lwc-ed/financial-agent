from flask import Blueprint, request
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import MessagingApi, ReplyMessageRequest, TextMessage, Configuration, ApiClient
from backend.database import SessionLocal
from backend.models.user import User
from datetime import datetime, timedelta
from backend.models.wishlist import Wishlist

from backend.models.record import Record   #  記帳資料表
from sqlalchemy import desc                #  查紀錄排序用
import re                                  #  解析「午餐 150」用

from datetime import datetime
import pytz
taipei = pytz.timezone("Asia/Taipei")
datetime.now(taipei)


linebot_bp = Blueprint("linebot", __name__)

# 這裡填入你的 LINE Secret 與 Access Token
handler = WebhookHandler("bde6ff24868fe4edeef87393ea9db525")
configuration = Configuration(
    access_token="4CtUYyGR0+ISjVhzcnGLmJmG8Qf/vzH5/gQM98g/jR2ZoMZguJPkvjiLvMXoSb3ctaKkMO7Onhe6Fa1bc3BHw6sF7coKlYy1dozA7/V6ZFOpt9S9wU8PXZhefQoOGtC2J6fj70vQzIqNktiQVx2MdAdB04t89/1O/w1cDnyilFU="
)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)


@linebot_bp.route("/callback", methods=["POST"])
def callback():
    body = request.get_data(as_text=True)
    signature = request.headers.get("X-Line-Signature")
    handler.handle(body, signature)
    return "OK"


def process_credit_card_query(user_msg):
    """
    信用卡回饋查詢主流程：
    1. GPT 解析意圖與正規化品牌
    2. DB Fulltext Search 查回饋
    3. 整理 summary
    4. AI Reply 模組生成回應
    """
    from backend.ai.ai_parser import normalize_input
    from backend.ai.benefit_query import query_benefits
    from backend.ai.format_benefit_summary import build_summary
    from backend.ai.ai_reply import generate_reply




    # Step 1：解析輸入
    parsed = normalize_input(user_msg)
    brand = parsed.get("brand_name")
    category = parsed.get("category")
    candidates = parsed.get("candidates", [])

    # Step 2：查 DB
    results = query_benefits(
        brand_name=brand,
        category=category,
        candidates=candidates
    )

    # Step 3：生成 summary
    summary = build_summary(parsed, results)

    # Step 4：AI 最終回覆
    reply_text = generate_reply(user_msg, results, summary)

    return reply_text

def parse_expense_text(text: str):
    """
    把像「午餐 150」這種訊息，拆成 (category, amount)
    回傳：(category:str, amount:int) 或 (None, None) 代表不是合法記帳格式
    """
    text = (text or "").strip()
    parts = text.split()

    # 只接受兩個字串：「類別 金額」
    if len(parts) != 2:
        return None, None

    category = parts[0]
    amount_raw = parts[1]

    # 把 "150"、"150元"、"$150"、"1,500" 都變成純數字字串
    s = re.sub(r"[,\s\$＄元圓]", "", amount_raw)
    if not re.fullmatch(r"\d+", s):
        return None, None

    return category, int(s)


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    line_user_id = event.source.user_id
    user_msg = event.message.text
    print(f"🟢 收到 LINE 訊息：{user_msg}")
    db = SessionLocal()

    # 從 LINE 取得 user id
    user = db.query(User).filter_by(line_user_id=line_user_id).first()

    # ---------- Google 綁定檢查（真正符合你需求的版本） ----------
    if not user:
        # 查詢是否有任何 Google 使用者綁定過這個 LINE user_id
        google_user = db.query(User).filter(
            User.provider == "google",
            User.line_user_id == line_user_id
        ).first()

        if google_user:
            # 找到 → 使用該 Google 綁定帳號
            user = google_user
        else:
            # 找不到 → 尚未綁定 → 禁止使用
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="⚠️ 您尚未綁定帳號，請先點擊下方連接進行 Google 登入並綁定 LINE\nhttps://financial-agent.it.com/login_google\n若綁定失敗可以參照以下步驟⭣\n" \
                    "IPhone使用者：\n主頁\n  ⭣\n設定(右上角)\n  ⭣\nLINE Labs\n  ⭣\n關閉「使用預設瀏覽器開啟連結」")]
                )
            )
            db.close()
            return
    # ---------- 綁定檢查完成 ----------

    # 超過 10 分鐘沒互動 → 重置
    if user.last_activity_time:
        db_time = user.last_activity_time.replace(tzinfo=taipei)
        if datetime.now(taipei) - db_time > timedelta(minutes=10):
            user.current_function = None
            db.commit()

    # 功能別名對應表
    function_alias = {
        "信用卡回饋查詢": "功能 A",
        "欲望清單": "功能 B",
        "紀錄消費": "功能 C",
        "其他": "功能 D",
        "儲蓄挑戰": "功能 E"
    }

    # 如果 user_msg 在 function_alias，則轉換為對應功能
    if user_msg in function_alias:
        user_msg = function_alias[user_msg]

    # 功能對應表
    function_map = {
        "功能 A": "💳 信用卡回饋查詢（AI+DB搜尋回饋",

        "功能 B": "📉 欲望清單（待接 DB）",
        "功能 C": "🧾 記帳功能：可輸入「午餐 150」或「查紀錄」",


        "功能 D": "其他功能",
        "功能 E": "⚠️ 儲蓄挑戰（待接分析功能）",
    }

    # 回覆文字
    if not user.current_function and user_msg not in function_map:
        reply_text = "請先點選功能"
    elif user_msg in ["功能 A", "功能 B", "功能 C", "功能 D", "功能 E"]:
        user.current_function = user_msg
        user.last_activity_time = datetime.now(taipei)
        db.commit()
        if user_msg == "功能 A":
            user.current_function = "信用卡回饋查詢"  # 強制轉為 信用卡回饋查詢 狀態
            reply_text = "💳 已進入信用卡回饋查詢模式，請輸入商店名稱（例如：遠百、星巴克）"
        elif user_msg == "功能 B":
            print("進入欲望清單模式")
            user.current_function = "wishlist"  # 強制轉為 wishlist 狀態
            db.commit()
            reply_text = "✍️ 請輸入欲望清單項目，格式：品項,價格\n例如：iPhone,35000"
        elif user_msg == "功能 C":
            print("進入記帳模式")
            user.current_function = "expense"   # 👈 記帳模式
            db.commit()
            reply_text = (
                "🧾 已進入記帳模式：\n"
                "・記帳：直接輸入「午餐 150」\n"
                "・查紀錄：輸入「查紀錄」會顯示最近 5 筆\n"
                "・離開記帳：輸入「離開」"
            )
        else:
            reply_text = f"✅ 你選擇了 {function_map[user_msg]}"
    elif user.current_function == "信用卡回饋查詢":

        print("👉 信用卡回饋查詢已啟動，收到使用者輸入 =", user_msg)

        # ⭐ 第 1 段：立即回覆避免 LINE Timeout
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text="🔍 正在為您查詢中，請稍候…")]
            )
        )

        # ⭐ 第 2 段：後台執行真正查詢
        final_reply = process_credit_card_query(user_msg)

        # ⭐ 第 3 段：push 第二段訊息（查詢結果）
        from linebot.v3.messaging import PushMessageRequest

        line_bot_api.push_message(
            PushMessageRequest(
                to=line_user_id,
                messages=[TextMessage(text=final_reply)]
            )
        )
        return   # ⚠️ 不要再往下執行

    elif user.current_function == "wishlist":
        print(f"處理欲望清單輸入: {user_msg}")
        try:
            # 處理中英文逗號
            separator = "," if "," in user_msg else "，"
            if separator not in user_msg:
                raise ValueError("缺少逗號")

            item_name, price_str = user_msg.split(separator, 1)
            
            # ORM 寫入
            new_item = Wishlist(
                user_id=user.id,
                item_name=item_name.strip(),
                price=int(price_str.strip())
            )
            db.add(new_item)
            db.commit()
            
            reply_text = f"✅ 已新增「{item_name.strip()}」價格 {price_str.strip()} 元到清單！"
            
            # 完成後重置
            user.current_function = None
            db.commit()
            
        except ValueError:
            reply_text = "格式錯誤！請確認使用逗號分隔，且價格為數字。\n範例：Switch, 10000"
        except Exception as e:
            reply_text = f"發生錯誤：{str(e)}"
            print(f"Error: {e}")

    elif user.current_function == "expense":
        text = user_msg.strip()

        if text == "離開":
            user.current_function = None
            db.commit()
            reply_text = "已離開記帳模式，之後可以再點「紀錄消費」回來記帳～"

        elif text == "查紀錄":
            try:
                q = (
                    db.query(Record)
                    .filter(Record.line_user_id == line_user_id)
                    .order_by(desc(Record.timestamp), desc(Record.id))
                    .limit(5)
                )
                rows = q.all()

                if not rows:
                    reply_text = "你目前還沒有任何記帳紀錄喔～\n可以試試輸入：午餐 150"
                else:
                    lines = ["你最近的記帳紀錄："]
                    for r in rows:
                        line = f"- {r.category} {r.amount} 元"
                        if r.note:
                            line += f"（{r.note}）"
                        lines.append(line)
                    reply_text = "\n".join(lines)
            except Exception as e:
                print("查紀錄錯誤：", e)
                reply_text = "查詢紀錄時發生錯誤 QQ，等等再試試。"

        else:
            # 嘗試把輸入當成「午餐 150」這種記帳格式
            category, amount = parse_expense_text(text)
            if category is None or amount is None:
                reply_text = (
                    "記帳格式是：「項目 金額」，例如：\n"
                    "・午餐 150\n"
                    "・飲料 60\n"
                    "也可以輸入「查紀錄」或「離開」。"
                )
            else:
                try:
                    rec = Record(
                        line_user_id=line_user_id,
                        type="支出",    # 目前全部先當支出
                        category=category,
                        amount=amount,
                        note="",
                    )
                    db.add(rec)
                    db.commit()
                    reply_text = f"已幫你記錄：{category} {amount} 元 ✅"
                except Exception as e:
                    print("記帳寫入錯誤：", e)
                    db.rollback()
                    reply_text = "記帳失敗 QQ，等等再試試看。"   
    
    
    else:
        reply_text = function_map.get(
            user_msg,
            f"你說的是：「{user_msg}」"
        )

    # 更新最後互動時間
    user.last_activity_time = datetime.now(taipei)
    db.commit()

    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=reply_text)]
        )
    )