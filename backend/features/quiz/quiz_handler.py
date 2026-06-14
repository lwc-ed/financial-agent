import datetime
from linebot.v3.messaging import (
    TextMessage,
    QuickReply,
    QuickReplyItem,
    PostbackAction
)
from backend.core.database import SessionLocal
from backend.features.web.user import User

class FullInsuranceQuizHandler:
    # 問卷題目
    def __init__(self):
        # 記憶體 Session 快取
        # 結構: user_id -> {"current_q": 0, "score": 0, "q2_selected": []}
        self.user_sessions = {}
        
        # 問卷內容
        self.QUESTIONS = [
            {
                "type": "single", # Q1
                "text": "【Q1/12】您的年齡落在哪個區間？",
                "options": [
                    ("18歲以下", "5"), ("19歲~35歲", "5"), ("36歲~45歲", "4"),
                    ("46歲~55歲", "3"), ("56歲~64歲", "2"), ("65歲以上", "1")
                ]
            },
            {
                "type": "multiple", # Q2
                "text": "【Q2/12】（可複選）您曾經使用過哪些投資理財工具？（請逐一調整勾選，完成後滑至最右點選確認按鈕）",
                "options": [
                    ("A. 無", "A"),
                    ("B. 債券型基金/ETF/保單連結", "B"),
                    ("C. 股票型/其他類型基金", "C"),
                    ("D. 股票", "D"),
                    ("E. 外匯交易", "E"),
                    ("F. 期貨/選擇權/衍生性商品", "F")
                ]
            },
            {
                "type": "single", # Q3
                "text": "【Q3/12】您投資「與債券類型相關商品」的理財工具經驗？",
                "options": [
                    ("無經驗", "1"), ("1年以下", "2"),  ("1年(含)~3年", "3"), ("3年(含)~5年", "4"), ("5年(含)以上", "5") 
                ]
            },
            {
                "type": "single", # Q4
                "text": "【Q4/12】您投資「與其他類型相關商品(如股票)」的理財工具經驗？",
                "options": [
                    ("無經驗", "1"), ("1年以下", "2"), ("1年(含)~3年", "3"), ("3年(含)~5年", "4"),("5年(含)以上", "5")
                ]
            },
            {
                "type": "single", # Q5
                "text": "【Q5/12】下列何者最符合您對投資理財工具的理解？",
                "options": [
                    ("A. 不熟悉，但有興趣瞭解", "1"),
                    ("B. 瞭解基本知識，例如股票與基金的分別", "2"),
                    ("C. 瞭解基本知識，並明白分散投資與資產配置的重要性", "3"),
                    ("D. 對投資理財工具及其投資風險有進一步的認識", "4"),
                    ("E. 非常熟悉大部分投資理財工具與風險", "5")
                ]
            },
            {
                "type": "single", # Q6
                "text": "【Q6/12】您每年可用於購買投資理財工具之金額(新台幣)？",
                "options": [
                    ("A. 未滿50萬", "1"),
                    ("B. 50萬~未滿100萬", "2"),
                    ("C. 100萬~未滿300萬", "4"),
                    ("D. 300萬以上", "5")
                ]
            },
            {
                "type": "single", # Q7
                "text": "【Q7/12】您的備用金(現金及存款)相當於您幾個月的生活開銷？",
                "options": [
                    ("A. 無備用金或不需負擔開銷", "1"),
                    ("B. 3個月以下", "1"),
                    ("C. 超過3個月未達6個月", "2"),
                    ("D. 超過6個月未達1年", "3"),
                    ("E. 超過1年", "4"),
                    ("F. 超過3年以上", "5")
                ]
            },
            {
                "type": "single", # Q8
                "text": "【Q8/12】購買外幣計價投資標的，您每年可承受的價格損失(含匯率風險)？",
                "options": [
                    ("A. 無法接受虧損", "1"),
                    ("B. 可承受 -5%", "2"),
                    ("C. 可承受 -10%", "3"),
                    ("D. 可承受 -15%", "4"),
                    ("E. 可承受 -20%", "5")
                ]
            },
            {
                "type": "single", # Q9
                "text": "【Q9/12】投資達到預計期間時(如3、5年)，您可承受的價格損失(含匯率風險)？",
                "options": [
                    ("A. 無法接受虧損", "1"),
                    ("B. 可承受 -5%", "2"),
                    ("C. 可承受 -10%", "3"),
                    ("D. 可承受 -15%", "4"),
                    ("E. 可承受 -20%", "5")
                ]
            },
            {
                "type": "single", # Q10
                "text": "【Q10/12】您的投資回報期望？",
                "options": [
                    ("A. 避免資產損失", "1"),
                    ("B. 資產每年穩定成長", "3"),
                    ("C. 資產短期快速成長", "5")
                ]
            },
            {
                "type": "single", # Q11
                "text": "【Q11/12】就長期投資而言，您期望每年平均投資報酬率？",
                "options": [
                    ("A. 1%(含)~5%", "1"),
                    ("B. 5%(含)~10%", "3"),
                    ("C. 10%(含)~15%", "4"),
                    ("D. 15%(含)~20%", "5")
                ]
            },
            {
                "type": "single", # Q12
                "text": "【Q12/12】當投資發生虧損或達到停損點時，您會採取的處理方式？",
                "options": [
                    ("A. 立即賣出", "1"),
                    ("B. 先賣出一半", "1"),
                    ("C. 虧損未達6個月就賣掉", "2"),
                    ("D. 虧損達6個月以上才考慮出售", "3"),
                    ("E. 持有1年以上", "4"),
                    ("F. 持有至回本", "4")
                ]
            }
        ]
    # line bot 按鈕選單建立 
    def build_question_message(self, user_id, q_index):
        """根據題型動態生成對應的 Quick Reply 訊息卡片"""
        question = self.QUESTIONS[q_index]
        items = []
        
        # 處理複選題邏輯 (Q2)
        if question["type"] == "multiple":
            selected = self.user_sessions[user_id].get("q2_selected", [])
            for label, code in question["options"]:
                # 如果該選項已經被選過，在畫面上打勾提示使用者
                display_label = f"✅ {label}" if code in selected else label
                items.append(QuickReplyItem(
                    action=PostbackAction(
                        label=display_label[:20],
                        data=f"action=full_quiz&q={q_index}&type=toggle&code={code}",
                        display_text=f"{label}"
                    )
                ))
            # 額外新增一個完成複選的按鈕
            items.append(QuickReplyItem(
                action=PostbackAction(
                    label="👉【我確認都選好了，可以進行下一題】",
                    data=f"action=full_quiz&q={q_index}&type=submit",
                    display_text="我已選好 Q2 的所有經驗工具"
                )
            ))
        else:
            # 處理一般單選題邏輯
            for label, score in question["options"]:
                items.append(QuickReplyItem(
                    action=PostbackAction(
                        label=label[:20],
                        data=f"action=full_quiz&q={q_index}&type=single&score={score}",
                        display_text=label
                    )
                ))

        # 每題都附上退出按鈕
        items.append(QuickReplyItem(
            action=PostbackAction(
                label="❌ 退出測驗",
                data=f"action=full_quiz&q={q_index}&type=exit",
                display_text="退出測驗"
            )
        ))

        return TextMessage(text=question["text"], quick_reply=QuickReply(items=items))
    # 複選題（Q2）分數計算
    def calculate_q2_score(self, selected_codes):
        """計算複選題分數：根據勾選工具的風險等級，給予 1~5 分的權重點數"""
        if not selected_codes or "A" in selected_codes: # 無經驗
            return 1
        score = 1
        if "B" in selected_codes: score += 1 # 債券型商品
        if "C" in selected_codes: score += 1 # 股票型基金
        if "D" in selected_codes: score += 1 # 股票
        if "E" in selected_codes or "F" in selected_codes: score += 2 # 外匯/衍生性高風險
        return min(score, 5) # 最高上限 5 分
    # 總分數計算->結果分析
    def get_result_analysis(self, score):
        """根據 12 題總分（配分區間 12~59 分）對照 PDF 官方評級說明"""
        if score <= 25:
            return (
                "【保守型】",
                "🛡️ 合適投資標的風險等級：低風險(RR1) 及 中低風險(RR2)\n"
                "📊 建議配置：股票 20% / 債券 80%\n"
                "💡 官方評級說明：您屬於風險趨避者，通常期望避免投資資本金之損失，但仍願意承受少量風險以增加投資報酬。投資主要為風險等級較低之商品。\n"
                "⭐️ 如果不了解風險等級，可直接至對話框輸入欲查詢的等級，如：RR1"
            )
        elif score <= 43:
            return (
                "【穩健型】",
                "⚖️ 合適投資標的風險等級：低風險(RR1) 至 中度風險(RR3)\n"
                "📊 建議配置：股票 50% / 債券 50%\n"
                "💡 官方評級說明：您屬於風險中立者，願意承擔部分風險以增加投資報酬；為了獲得提高投資報酬之機會，可以接受投資包含不同風險等級之商品。\n"
                "⭐️ 如果不了解風險等級，可直接至對話框輸入欲查詢的等級，如：RR3"
            )
        else:
            return (
                "【積極型】",
                "🔥 合適投資標的風險等級：低風險(RR1) 至 高風險(RR5)\n"
                "📊 建議配置：股票 80% / 債券 20%\n"
                "💡 官方評級說明：您屬於風險追求者，願意承擔相當程度風險以增加投資報酬；可以接受將所有資金投資於風險較高之商品（例如股票型基金），藉以獲取較高投資報酬。本金可能造成全部虧損且價值頻繁劇烈波動。\n"
                "⭐️ 如果不了解風險等級，可直接至對話框輸入欲查詢的等級，如：RR5"
            )
    # 風險說明（RR1-RR5），需要自行手動輸入
    def get_rr_level_description(self, rr_level: str) -> str:
        """
        根據傳入的 RR 等級（'RR1' 到 'RR5'），回傳對應的官方定義與詳細說明。
        支援大小寫，若輸入錯誤則回傳提示訊息。
        """
        # 統一轉成大寫，避免大小寫不一致的問題
        level = rr_level.upper().strip()
        
        rr_database = {
            "RR1": (
                "🟢【RR1 - 低風險】\n"
                "📌 官方定義：主要投資於具有高流動性、收益穩定且本金損失風險極低之金融商品。\n"
                "💡 實務說明：例如台灣新台幣定存、短期票券、國內外貨幣市場基金。這種標的幾乎沒有本金虧損風險，波動度極低，適合極度保守、只想打敗通膨的投資人。"
            ),
            "RR2": (
                "🔵【RR2 - 中低風險】\n"
                "📌 官方定義：主要投資於固定收益商品，且整體本金損失風險較低之金融商品。\n"
                "💡 實務說明：例如全球政府公債、高評級公司債券基金、債券型 ETF。這類標的以追求穩定配息為主，雖然債券價格會隨著市場利率波動，但整體本金波動幅度相對較小。"
            ),
            "RR3": (
                "🟡【RR3 - 中度風險】\n"
                "📌 官方定義：投資以追求資產長期增值為目的，通常包含不同風險等級商品，本金可能產生部分虧損，且價值可能頻繁波動之金融商品。\n"
                "💡 實務說明：例如全球平衡型基金、高殖利率股票、區域型公債基金。這類標的既能跟上股市的成長，又有債券部位做防護，本金會隨市場行情上下高低頻繁震盪。"
            ),
            "RR4": (
                "🟠【RR4 - 中高風險】\n"
                "📌 官方定義：主要投資於單一國家、特定成熟市場或傳統產業股票，本金可能產生顯著虧損，且價值可能劇烈波動之金融商品。\n"
                "💡 實務說明：例如美國股市 ETF、標普 500、全球科技股基金、或是特定大型市值股票。當遇到地緣政治或經濟修正時，本金會出現明顯的跌幅，但長期來看具有較高的資產增值潛力。"
            ),
            "RR5": (
                "🔴【RR5 - 高風險】\n"
                "📌 官方定義：主要投資於高波動的單一新興市場、高科技/生技等單一產業股票，或期貨、選擇權等衍生性商品，本金可能造成全部虧損，且價值可能頻繁且劇烈波動之金融商品。\n"
                "💡 實務說明：例如加密貨幣、台股當沖期貨、單一新興市場基金、或是生技/原物料槓桿型商品。這類標的屬於「高風險、高回報」，本金可能會在短時間內暴跌或翻倍，心臟不夠大顆千萬別碰。"
            )
        }
        
        # 回傳對應的說明，如果傳入不合法的字串（例如 'RR6'），則給予防呆提示
        return rr_database.get(
            level, 
            f"❌ 找不到查詢的風險等級 '{rr_level}'。請輸入 RR1、RR2、RR3、RR4 或 RR5 進行查詢。"
        )
    # 初始化題目
    def handle_start_quiz(self, user_id):
        """初始化全 12 題測驗狀態"""
        self.user_sessions[user_id] = {"current_q": 0, "score": 0, "q2_selected": []}
        welcome_msg = TextMessage(text="📋 您好，現在開始進行風險屬性評估測驗，共 12 題。請依畫面按鈕進行作答：")
        first_q_msg = self.build_question_message(user_id, 0)
        return [welcome_msg, first_q_msg] 
    # State Machine
    def handle_quiz_postback(self, user_id, params):
        """處理 12 題狀態機轉移邏輯，兼容複選與單選行為"""
        if user_id not in self.user_sessions:
            return self.handle_start_quiz(user_id)
            
        q_index = int(params.get("q"))
        click_type = params.get("type")

        # 處理退出測驗
        if click_type == "exit":
            self.user_sessions.pop(user_id, None)
            return [TextMessage(text="已退出測驗。如需重新開始，請說「幫我做風險測驗」。")]

        # 防止重複點擊舊按鈕
        if q_index != self.user_sessions[user_id]["current_q"]:
            return [self.build_question_message(user_id, self.user_sessions[user_id]["current_q"])]

        # 處理複選題點擊切換 (Q2)
        if click_type == "toggle":
            code = params.get("code")
            selected = self.user_sessions[user_id]["q2_selected"]
            if code == "A": # 如果選了無，清空其他
                self.user_sessions[user_id]["q2_selected"] = ["A"]
                q2_score = self.calculate_q2_score(["A"])
                self.user_sessions[user_id]["score"] += q2_score
                self.user_sessions[user_id]["current_q"] += 1  # 進度直接 +1
                
                # 自動抓取 Q3 (下一題) 的題目訊息送回前台
                next_q = self.user_sessions[user_id]["current_q"]
                return [self.build_question_message(user_id, next_q)]
            else:
                if "A" in selected: selected.remove("A")
                if code in selected:
                    selected.remove(code)
                else:
                    selected.append(code)
            self.user_sessions[user_id]["q2_selected"] = selected
            # 複選中，刷新同一個問題卡片讓使用者繼續勾選
            return [self.build_question_message(user_id, q_index)]
            
        # 處理複選題點擊送出
        elif click_type == "submit":
            q2_score = self.calculate_q2_score(self.user_sessions[user_id]["q2_selected"])
            self.user_sessions[user_id]["score"] += q2_score
            self.user_sessions[user_id]["current_q"] += 1
            
        # 處理一般單選題點擊送出
        elif click_type == "single":
            score_added = int(params.get("score"))
            self.user_sessions[user_id]["score"] += score_added
            self.user_sessions[user_id]["current_q"] += 1
            
        # 推進到下一步
        next_q = self.user_sessions[user_id]["current_q"]
        if next_q < len(self.QUESTIONS):
            return [self.build_question_message(user_id, next_q)]
        else:
            # 12 題全部作答完畢
            final_score = self.user_sessions[user_id]["score"]
            label, advice = self.get_result_analysis(final_score)

            # 將測驗結果寫入 DB
            risk_type_clean = label.strip("【】")
            try:
                db = SessionLocal()
                user = db.query(User).filter(User.line_user_id == user_id).first()
                if user:
                    user.risk_score = final_score
                    user.risk_type = risk_type_clean
                    db.commit()
            except Exception as e:
                print(f"[quiz] 儲存風險測驗結果失敗: {e}")
            finally:
                db.close()

            result_text = (
                f"您的風險偏好已評估完成！\n"
                f"您的評估總分為：{final_score} 分\n"
                f"您的風險屬性類型為：{label}\n\n"
                f"{advice}"
            )
            self.user_sessions.pop(user_id, None)
            return [TextMessage(text=result_text)]


