以下整理為 可直接使用、排版完整、專業等級的最終版 README.md。
你只需要 複製→貼進你的 README.md 即可。

我已依照你的專案實際結構、部署流程、後端/前端架構、AWS RDS、EC2、FastAPI、LIFF、LINE Bot、爬蟲測試檔等全部內容整合成正式文件。

⸻

📘 financial-agent — 完整專案說明文件

本專案將 LINE Bot、Flask/FastAPI、React、MySQL（AWS RDS）、EC2 與 AI 技術整合為一個金融助理（Financial Agent）。
提供記帳、慾望清單、儲蓄規劃、個人化分析、Rich Menu 操作介面與 LIFF 視覺化頁面。

⸻

📑 目錄
	•	📦 必要安裝套件與工具￼
	•	🚀 專案啟動方式￼
	•	📌 功能統整表￼
	•	🔹 整體架構運作￼
	•	📂 專案結構與檔案說明￼
	•	🖼 Rich Menu 設定方式￼
	•	⚙️ FastAPI 與 MySQL 使用方式￼
	•	🌐 WSL 連接 AWS RDS 教學￼
	•	🐥 EC2 Branch 測試流程￼
	•	🔄 Branch 與 Main 同步教學￼
	•	🧪 虛擬環境使用方式￼
	•	📌 目前仍需修改的項目￼

⸻

📦 必要安裝套件與工具

Python（後端）

pip install --upgrade pip
pip install -r requirements.txt

檢查是否成功：

pip list


⸻

Node.js / npm（前端）

使用 React + Vite。

安裝方式：

cd frontend
npm install

可選套件：
	•	liff（LINE 前端）
	•	axios（API）
	•	openai（AI 功能）

⸻

🚀 專案啟動方式

1. 更新專案

cd financial-agent
git pull

2. 啟動後端（Flask）

cd backend
python3 app.py

前端：

cd frontend
npm start


⸻

📌 功能統整表

功能	用途	實作方式	資料處理位置	展示方式
個人資料填寫	建立使用者資訊	LIFF 表單	DB	LINE 回覆 / LIFF
紀錄消費	日常記帳	LINE 指令或 LIFF	DB	LINE 即時回覆
消費紀錄分析	檢視分類與統計	後端分析	DB	LINE 圖表 / LIFF
慾望清單	管理想買物品	LINE / LIFF	DB	列表 / 進度條
儲蓄挑戰	自動產生儲蓄計畫	演算法 + AI	DB + 分析	LINE / LIFF


⸻

🔹 整體架構運作

1. LINE Bot

接收文字、按鈕、Quick Reply，呼叫後端 API。

2. 伺服器（Flask / FastAPI）
	•	處理邏輯
	•	儲存數據
	•	回傳分析結果

3. AI 模組
	•	整理消費
	•	生成人性化建議

4. LIFF / Web（React）
	•	表單填寫
	•	儀表板、進度條、視覺化圖表

⸻

📂 專案結構與檔案說明

financial-agent/
├── backend/
│   ├── app.py                 # Flask 主程式
│   ├── linebot_handler.py     # 處理 LINE Bot 事件
│   ├── routes/                # API Blueprint 模組（記帳/願望清單等）
│   ├── models/                # 資料庫 ORM 模型
│   ├── setup_rich_menu.py     # Rich Menu 設定腳本
├── frontend/
│   ├── public/
│   ├── src/                   # React 程式碼
│   ├── .env                   # API 位置設定
│   └── package.json
├── picture/
│   └── rich_menu/             # Rich Menu 圖片
├── README.md
└── .gitignore

命名規範：
	•	Blueprint 統一以 _bp 結尾（如 user_bp）
	•	Commit message 需具體描述變更內容

⸻

🖼 Rich Menu 設定方式

/opt/homebrew/bin/python3.10 setup_rich_menu.py

區域配置示意：

┌────────────┬────────────┬────────────┐
│   Area A   │   Area B   │            │  ← (y=0~421)
├────────────┼────────────│   Area C   │
│   Area D   │   Area E   │            │  ← (y=421~843)
└────────────┴────────────┴────────────┘


⸻

⚙️ FastAPI 與 MySQL 使用方式

安裝：

pip install fastapi uvicorn sqlalchemy pymysql python-dotenv

啟動：

uvicorn main:app --reload

測試介面：
👉 http://127.0.0.1:8000/docs#/

常用 API（範例）：

建立使用者

POST /users/
{
  "username": "alice"
}

查詢文章

GET /posts/{post_id}


⸻

🌐 WSL 連接 AWS RDS 教學

1. DNS 測試

nslookup financial-agent.cpwk2ce8cqyu.us-east-2.rds.amazonaws.com

2. 測試 MySQL 連線

mysql -h financial-agent.cpwk2ce8cqyu.us-east-2.rds.amazonaws.com -P 3306 -u nycuiemagent -p

3. 修正 DNS 設定

編輯設定：

[network]
generateResolvConf = false


⸻

🐥 EC2 Branch 測試流程

Step 1：本地完成程式

git push

Step 2：登入 EC2

ssh ubuntu@3.21.167.93

Step 3：切換到自己的 branch

cd financial-agent
git checkout feature-login
git pull

Step 4：啟動後端

cd backend
python3 app.py

確認無誤後再在 GitHub merge 到 main。

⸻

🔄 Branch 與 Main 同步教學

git fetch origin
git checkout main
git pull origin main
git checkout feature-login
git merge main


⸻

🧪 虛擬環境使用方式

建立

python3 -m venv venv

啟動

source venv/bin/activate

安裝套件

pip install -r requirements.txt

執行爬蟲與測試

python3 -m backend.routes.credit_card.cube_benefits_scraper
python3 backend/ai/test_full_flow.py


⸻

📌 目前仍需修改的項目
	•	ai/ai_query：搜尋與匹配邏輯需重新調整
	•	ai/ai_parser：品牌清洗、FTS 匹配、rewrite 還需調整
	•	信用卡 datasets 更新後需重新整理 reward rate split
	•	Blueprints 與 DB 結構可能需要重構以利擴充

⸻

如果你需要，我也可以幫你：

✅ 自動產生 API 規格書（Swagger / OpenAPI）
✅ 製作 ERD（資料庫架構圖）
✅ 繪出系統架構圖（Architecture Diagram）
✅ 幫 README 加上 badge、logo、架構流程圖

只要告訴我即可。