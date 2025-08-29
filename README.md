# financial-agent

## 事前作業
1.  git pull -> 更新repo
2.  cd financial-agent
3.  **打開兩個終端機**  
    **終端機A:**  
    ```bash
    python3 app.py
    ```  
    **終端機B:**  
    ```bash
    ngrok http 8000
    ```  
4.  複製 `https://13d62ea100e6.ngrok-free.app` (在終端機B裡，要找一下)  
    範例：
    ```
    ngrok                                                                               (Ctrl+C to quit)
                                                                                                        
    🧱 Block threats before they reach your services with new WAF actions →  https://ngrok.com/r/waf    
                                                                                                        
    Session Status                online                                                                
    Account                       supergreatfinancialagent@gmail.com (Plan: Free)                       
    Version                       3.26.0                                                                
    Region                        Japan (jp)                                                            
    Latency                       40ms                                                                  
    Web Interface                 http://127.0.0.1:4040                                                 
    Forwarding                    https://13d62ea100e6.ngrok-free.app (這個網址) -> http://localhost:8000          
                                                                                                        
    Connections                   ttl     opn     rt1     rt5     p50     p90                           
                                0       0       0.00    0.00    0.00    0.00                          
                                                                                
    ```
5.  去[Line官方帳號](https://developers.line.biz/console/channel/2007892068/messaging-api)更改Webhook URL  
   `https://81f3d5915d67.ngrok-free.app/callback`（記得加）/callback
## setup_rich_menu 執行方式(only fot lwc)
```
/opt/homebrew/bin/python3.10 setup_rich_menu.py
```
## Line畫面
```
┌────────────┬────────────┬────────────┐
│   Area A   │   Area B   │            │  ← 上半部 (y=0 ~ 421)
├────────────┼────────────│   Area C   │
│   Area D   │   Area E   │            │  ← 下半部 (y=421 ~ 843)
└────────────┴────────────┴────────────┘
```
## 功能架構
```
📌 功能統整表

| 功能  | 目的 | 實作方式 | 資料處理位置 | 展示方式 |
|------|------|----------|--------------|----------|
| **1. 個人資料填寫** | 讓使用者建立個人檔案（姓名、收入、目標等） | 提供表單（LIFF 小網頁 / 外部網站） | 伺服器資料庫（DB） | LINE Bot 回覆「填寫完成」，或在 LIFF 直接看到 |
| **2. 紀錄消費（記帳）** | 讓使用者輸入日常消費 | 使用 LINE Bot 指令（ex:「午餐 120」）或 LIFF 表單 | 後端伺服器接收 → 存入 DB | LINE Bot 即時回覆「已紀錄：午餐 120 元」 |
| **3. 消費紀錄（查看狀況）** | 查看消費分類、剩餘可用金額 | 後端讀取 DB，計算統計 | DB 運算 + 伺服器整理成報表 | LINE Bot 傳回清單 / 圖表圖片（或開 LIFF 網頁看完整表格） |
| **4. 慾望清單** | 使用者輸入想買的東西＋金額 | LINE Bot 輸入「想買 Switch 10000」或 LIFF 表單 | 存在 DB，標記是否已達成 | LINE Bot 回傳目前清單；或在 LIFF 做更漂亮的清單展示 |
| **5. 儲蓄挑戰** | 幫使用者依照清單規劃存錢 | 後端演算法計算最佳分配 + OpenAI API 生成自然語言建議 | 演算法（DB → 儲蓄計畫表），AI（生成對話） | LINE Bot 文字 / 圖表呈現；或開 LIFF 看「進度條 + 成就系統」 |
```
⸻

🔹 整體架構運作
	1.	LINE Bot
	•	基本互動（輸入消費、查詢紀錄、呼叫功能）
	•	傳文字 / 按鈕 / Quick Reply
	2.	伺服器 + 資料庫（核心大腦）
	•	Flask / FastAPI / Node.js
	•	MySQL / PostgreSQL / MongoDB
	•	演算法處理（消費統計、儲蓄規劃）
	3.	AI API（選配）
	•	負責把演算法結果轉成自然語言對話
	•	例如「依照你的清單，我建議先完成耳機，再挑戰 Switch！」
	4.	LIFF / 外部網站
	•	適合表單（資料填寫）和進度視覺化（圖表、清單、進度條）
	•	讓使用者「感覺還在 LINE 內操作」，體驗無縫

⸻

✅ 這樣你的 五大功能 就能 互相串連：
	•	個人資料 → 提供基礎數據
	•	消費紀錄 → 瞭解財務狀況
	•	慾望清單 → 建立存錢目標
	•	儲蓄挑戰 → 幫使用者設計計畫
	•	LINE Bot + LIFF → 簡單事用 Bot、複雜事用網頁

⸻
