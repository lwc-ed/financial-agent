# financial-agent

## 事前作業
1. 更新repo
   ```
   git pull
   ```
2. 進入repo
   ```
   cd financial-agent
   ```
3. **打開兩個終端機**  
   **終端機A:**  
   ```bash
   python3 app.py
   ```  
   **終端機B:**  
   ```bash
   ngrok http 8000
   ```  
4. 複製 `https://13d62ea100e6.ngrok-free.app` (在終端機B裡，要找一下)  
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
# 📌 功能統整表

| 功能 | 目的 | 實作方式 | 資料處理位置 | 展示方式 |
|------|------|----------|--------------|----------|
| **1. 個人資料填寫** | 讓使用者建立個人檔案（姓名、收入、目標等） | 提供表單（LIFF 小網頁 / 外部網站） | 存在伺服器的資料庫（DB） | LINE Bot 回覆「填寫完成」，或在 LIFF 直接看到 |
| **2. 紀錄消費（記帳）** | 讓使用者輸入日常消費 | 使用 LINE Bot 指令（ex:「午餐 120」）或 LIFF 表單 | 後端伺服器接收 → 存入 DB | LINE Bot 即時回覆「已紀錄：午餐 120 元」 |
| **3. 消費紀錄（查看狀況）** | 查看消費分類、剩餘可用金額 | 後端讀取 DB，計算統計 | DB 運算 + 伺服器整理成報表 | LINE Bot 傳回清單 / 圖表圖片（或開 LIFF 網頁看完整表格） |
| **4. 慾望清單** | 使用者輸入想買的東西＋金額 | LINE Bot 輸入「想買 Switch 10000」或 LIFF 表單 | 存在 DB，標記是否已達成 | LINE Bot 回傳目前清單；或在 LIFF 展示更漂亮的清單 |
| **5. 儲蓄挑戰** | 幫使用者依照清單規劃存錢 | 後端演算法計算最佳分配 + OpenAI API 生成自然語言建議 | 演算法（DB → 儲蓄計畫表），AI（生成對話） | LINE Bot 文字 / 圖表呈現；或 LIFF 展示「進度條 + 成就系統」 |

---

# 🔹 整體架構運作

1. **LINE Bot**  
   - 基本互動（輸入消費、查詢紀錄、呼叫功能）  
   - 傳文字 / 按鈕 / Quick Reply  

2. **伺服器 + 資料庫（核心大腦）**  
   - Flask / FastAPI / Node.js  
   - MySQL / PostgreSQL / MongoDB  
   - 演算法處理（消費統計、儲蓄規劃）  

3. **AI API（選配）**  
   - 把演算法結果轉成自然語言建議  
   - 例如「依照你的清單，我建議先完成耳機，再挑戰 Switch！」  

4. **LIFF / 外部網站**  
   - 適合表單（資料填寫）和進度視覺化（圖表、清單、進度條）  
   - 讓使用者「感覺還在 LINE 內操作」，體驗無縫 

## 事後作業
1. 分批新增檔案上來
   ```
   git add (檔案名稱)
   git commit -m"備注內容"
   ```
   或一次上傳
   ```
   git add .
   git commit -m"一次備注所有內容"
   ```
2. 上傳資料夾
   ```
   git push
   ```

# 📂 專案結構與檔案說明

```
financial-agent/
├── backend/
│   ├── app.py                 # Flask 主程式，啟動後端伺服器
│   ├── linebot_handler.py     # 處理 LINE Bot 事件的邏輯
│   ├── routes/                # 使用 Blueprint 模組管理 API 路由，依功能分離（如 expense_record、wishlist 等），各模組皆暴露一個 _bp 方便註冊與維護
│   ├── models/                # 資料模型定義（ORM 或 schema），負責資料庫資料結構與操作
│   ├── setup_rich_menu.py     # 設定 LINE Rich Menu 的腳本
└── frontend/
    ├── public/                # React 公開資源
    ├── src/                   # React 程式碼
    ├── .env                   # 前端環境變數設定（API 端點等）
    └── package.json
├── picture/
│   └── rich_menu/             # 儲存 LINE Rich Menu 使用的圖片
├── README.md
└── .gitignore
```

- **backend/**：後端使用 Flask 框架，`app.py` 是主入口，`linebot_handler.py` 負責處理 LINE Bot 事件，`routes/` 用於分模組管理 API 路由，依功能分離（如 expense_record、wishlist 等），每個 Blueprint 模組皆暴露 `_bp` 方便註冊與維護。`models/` 則定義資料模型（ORM 或 schema），處理資料庫的資料結構與操作。`setup_rich_menu.py` 用於部署 Rich Menu。環境變數可透過專案根目錄或系統環境設定管理。

- **frontend/**：React 應用程式，包含前端頁面與互動邏輯，`.env` 用來設定前端環境變數（如 API 伺服器 URL），確保前後端分離。

- **picture/**：存放圖片資源，尤其是 LINE Rich Menu 使用的圖片素材。

- **命名規範與提交規則**：
  - Blueprint 命名皆以 `_bp` 結尾，例如 `user_bp`、`expense_bp`，以利識別。
  - Git commit 訊息請清楚描述改動內容，避免使用模糊字眼，方便多人協作與版本追蹤。

- **啟動提醒**：
  - 後端啟動請在 `backend/` 目錄下執行：
    ```
    python3 app.py
    ```
  - 前端啟動請在 `frontend/` 目錄下執行：
    ```
    npm install
    npm start
    ```
  - 確認 `.env` 設定正確，避免啟動錯誤。

請依此專案結構與規範進行開發與維護，確保團隊合作順暢。