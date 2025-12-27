from flask import Blueprint, request
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import MessagingApi, ReplyMessageRequest, TextMessage, Configuration, ApiClient
from backend.database import SessionLocal
from backend.models.user import User
from datetime import datetime, timedelta
from backend.models.wishlist import Wishlist

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

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    line_user_id = event.source.user_id
    user_msg = event.message.text
    print(f"🟢 收到 LINE 訊息：{user_msg}")
    db = SessionLocal()

    # 查詢使用者
    user = db.query(User).filter_by(line_user_id=line_user_id).first()
    if not user:
        user = User(
            line_user_id=line_user_id,
            current_function=None,
            last_activity_time=datetime.utcnow(),
            provider="line",
            provider_id=line_user_id,
            name="",
            email=""
        )
        db.add(user)
        db.commit()

    # 超過 10 分鐘沒互動 → 重置
    if user.last_activity_time and datetime.utcnow() - user.last_activity_time > timedelta(minutes=10):
        user.current_function = None
        db.commit()

    # 功能別名對應表
    function_alias = {
        "個人資料填寫": "功能 A",
        "慾望清單": "功能 B",
        "紀錄消費": "功能 C",
        "信用卡回饋查詢": "功能 D",
        "儲蓄挑戰": "功能 E"
    }

    # 如果 user_msg 在 function_alias，則轉換為對應功能
    if user_msg in function_alias:
        user_msg = function_alias[user_msg]

    # 功能對應表
    function_map = {
        "功能 A": "📊 個人資料填寫（待接後端）",
        "功能 B": "📉 慾望清單（待接 DB）",
        "功能 C": "🧾 記帳紀錄（待接 DB）",
        "功能 D": "💳 信用卡回饋查詢（AI+DB搜尋回饋）",
        "功能 E": "⚠️ 儲蓄挑戰（待接分析功能）",
    }

    # 回覆文字
    if not user.current_function and user_msg not in function_map:
        reply_text = "請先點選功能"
    elif user_msg in ["功能 A", "功能 B", "功能 C", "功能 D", "功能 E"]:
        user.current_function = user_msg
        user.last_activity_time = datetime.utcnow()
        db.commit()
        if user_msg == "功能 D":
            reply_text = "💳 已進入信用卡回饋查詢模式，請輸入商店名稱（例如：遠百、星巴克）"
        elif user_msg == "功能 B":
            print("進入慾望清單模式")
            user.current_function = "wishlist"  # 強制轉為 wishlist 狀態
            db.commit()
            reply_text =  (
                "請輸入欲望清單項目，格式：品項,價格\n"
                "支援一次輸入多筆 (用空白或換行隔開)！\n\n"
                "例如：\n"
                "iPhone,35000 滑鼠,1000"
            )
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
    elif user.current_function == "功能 D":

        print("👉 功能 D 已啟動，收到使用者輸入 =", user_msg)

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
        # 使用正規表達式抓取所有符合 "文字,數字" 的組合
        # 說明：
        #   ([^,，\n]+)  -> 抓取前面任何不是逗號或換行的文字 (品項)
        #   [,，]        -> 抓取半形或全形逗號
        #   \s* -> 允許逗號後有空白
        #   (\d+)        -> 抓取數字 (價格)
        pattern = r"([^,，\n]+)[,，]\s*(\d+)"
        matches = re.findall(pattern, user_msg)

        if not matches:
            # 如果完全沒有抓到任何一組符合的
            reply_text = (
                "格式看起來不太對喔 😅\n"
                "請確認格式為「品項,價格」，支援一次輸入多筆！\n\n"
                "範例：\n"
                "iPhone, 35000\n"
                "滑鼠, 1000"
            )
        else:
            try:
                added_items = []
                for item_name, price_str in matches:
                    # 去除多餘空白
                    clean_name = item_name.strip()
                    clean_price = int(price_str)
                    
                    # 忽略空字串 (避免有些 user 輸入造成誤判)
                    if not clean_name: 
                        continue

                    # 建立物件
                    new_item = Wishlist(
                        user_id=user.id,
                        item_name=clean_name,
                        price=clean_price
                    )
                    db.add(new_item)
                    added_items.append(f"{clean_name} (${clean_price})")
                
                if not added_items:
                    reply_text = "找不到有效的品項，請重新輸入。"
                else:
                    db.commit() # 全部加完再一次 commit，效能較好
                    
                    # 組合回覆訊息
                    items_str = "\n".join([f"✅ {item}" for item in added_items])
                    reply_text = f"已為您一口氣新增 {len(added_items)} 筆清單！\n{items_str}"

                    # 完成後重置狀態 (如果想讓使用者連續輸入，這兩行可以註解掉)
                    user.current_function = None
                    db.commit()

            except Exception as e:
                db.rollback()
                reply_text = f"寫入資料庫時發生錯誤 😭：{str(e)}"
                print(f"Wishlist Batch Add Error: {e}")

    elif user_msg == "紀錄消費":
        from backend.routes import expense_record
        reply_text = expense_record.get_expense_summary(user_id=line_user_id)
    else:
        reply_text = function_map.get(user_msg, f"你說的是：「{user_msg}」")

    # 更新最後互動時間
    user.last_activity_time = datetime.utcnow()
    db.commit()

    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=reply_text)]
        )
    )
