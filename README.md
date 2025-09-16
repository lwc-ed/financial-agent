# financial-agent

# 📦 必要安裝套件與工具

本專案開發與執行需要安裝以下套件與工具，以下為 macOS / Linux 平台的安裝範例。
[Line官方帳號管理後台](https://developers.line.biz/console/channel/2007892068/messaging-api)

### Python (後端)
安裝依賴套件
```
pip install --upgrade pip
pip install -r requirements.txt
```
檢查是否安裝成功
```
pip list
```

### Node.js / npm (前端)

- 使用 React 與 Vite 建構前端
- 主要套件包括：
  - react
  - react-dom
  - vite
- 可選套件（視專案需求）：
  - liff（LINE Front-end Framework）
  - axios（HTTP 請求）
  - openai（OpenAI API 客戶端）
- 安裝方式：
  ```bash
  cd frontend
  npm install
  ```
  會根據 `package.json` 自動安裝所有依賴套件。



# 🚀 啟動方式

1. **更新並進入專案目錄**
   ```bash
   cd financial-agent
   git pull
   ```

2. **啟動後端伺服器（於 `backend/` 目錄）**
   ```bash
   cd backend
   python3 app.py
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

---

# 📂 專案結構與檔案說明

```
financial-agent/
├── backend/
│   ├── app.py                 # Flask 主程式，啟動後端伺服器
│   ├── linebot_handler.py     # 處理 LINE Bot 事件的邏輯
│   ├── routes/                # 使用 Blueprint 模組管理 API 路由，依功能分離（如 expense_record、wishlist 等），各模組皆暴露一個 _bp 方便註冊與維護
│   ├── models/                # 資料模型定義（ORM 或 schema），負責資料庫資料結構與操作
│   ├── setup_rich_menu.py     # 設定 LINE Rich Menu 的腳本
├─── frontend/
│   ├── public/                # React 公開資源
│   ├── src/                   # React 程式碼
│   ├── .env                   # 前端環境變數設定（API 端點等）
│   └── package.json
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

---

# 補充說明

## setup_rich_menu 執行方式 (only for lwc)
```bash
/opt/homebrew/bin/python3.10 setup_rich_menu.py
```

## Line畫面區域配置示意
```
┌────────────┬────────────┬────────────┐
│   Area A   │   Area B   │            │  ← 上半部 (y=0 ~ 421)
├────────────┼────────────│   Area C   │
│   Area D   │   Area E   │            │  ← 下半部 (y=421 ~ 843)
└────────────┴────────────┴────────────┘
```

---

# 遇到問題可以先問GPT大神或Claude，把error貼給他們看

# Fast API + MySQL 說明

## 安裝必要套件
請先安裝依賴套件：
```bash
pip install fastapi uvicorn sqlalchemy pymysql python-dotenv
```

# WSL 連接 AWS RDS (MySQL) 教學
## 1. 檢查 DNS 是否能解析
```bash
nslookup financial-agent.cpwk2ce8cqyu.us-east-2.rds.amazonaws.com
```
若正確，會解析出一個 Public IP (例如 3.129.xx.xx)。

## 2. 測試能否連到 RDS
安裝 MySQL client：
```bash
sudo apt update
sudo apt install mysql-client -y
```
測試連線：
```bash
mysql -h financial-agent.cpwk2ce8cqyu.us-east-2.rds.amazonaws.com -P 3306 -u nycuiemagent -p
```
輸入密碼後，若成功會進到 MySQL prompt (mysql>)，表示網路跟帳號都 OK。

## 3. 設定 DNS (避免 WSL DNS 問題)
有時候 WSL 會用錯 DNS，需要手動設定。
```bash
sudo nano /etc/wsl.conf
```
內容加上：
```ini
[network]
generateResolvConf = false
```
然後修改 DNS：
```bash
sudo rm /etc/resolv.conf
echo -e "nameserver 8.8.8.8\nnameserver 1.1.1.1" | sudo tee /etc/resolv.conf
```
重啟 WSL：
```powershell
wsl --shutdown
```
## 4.常見錯誤排查
Unknown MySQL server host

代表 DNS 無法解析 → 檢查 /etc/resolv.conf，確認有 8.8.8.8 或 1.1.1.1。

❌ Access denied for user

帳號或密碼錯誤。

或者 RDS user 沒有對外權限，檢查 IAM / MySQL user 權限。

❌ Timeout

RDS 安全群組沒有開放你的 IP。

在 AWS console → Security group → Inbound rules，加上：

Type: MySQL/Aurora

Port: 3306

Source: 你的 IP (或測試用 0.0.0.0/0)

## 連接fastapi

進入 backend/ 資料夾，執行：
```bash
uvicorn main:app --reload
```
啟動後開啟瀏覽器連到：
👉 http://127.0.0.1:8000/docs#/

這裡可以直接測試 API 功能。

API 功能

POST /posts/ : 新增一篇貼文

GET /posts/{post_id} : 讀取貼文

DELETE /posts/{post_id} : 刪除貼文

POST /users/ : 新增使用者

GET /users/{user_id} : 讀取使用者

#建立使用者
```bash
POST /users/
{
  "username": "alice"
}
```

#查詢使用者
```bash
GET /users/{user_id}
```

#建立文章
```bash
POST /posts/
{
  "title": "First Post",
  "content": "Hello FastAPI!",
  "user_id": 1
}
```

#查詢文章
```bash
GET /posts/{post_id}
```

#刪除文章
```bash
DELETE /posts/{post_id}
<<<<<<< HEAD
```


#啟動後端伺服器

進入 backend/ 資料夾，執行：
```bash
uvicorn main:app --reload
```
<<<<<<< HEAD
啟動後開啟瀏覽器連到：
👉 http://127.0.0.1:8000/docs#/

這裡可以直接測試 API 功能。

## financial_agent 資料庫
```
mysql> SELECT * FROM users;
=======
>>>>>>> 3e6afbe (刪除無用內容)
+----+----------+-------------+--------+------------------+---------+---------------------+
| id | provider | provider_id | name   | email            | picture | created_at          |
+----+----------+-------------+--------+------------------+---------+---------------------+
|  1 | line     | U1234567890 | 小明   | test@example.com  | NULL    | 2025-09-16 01:45:11 |
+----+----------+-------------+--------+------------------+---------+---------------------+
1 row in set (0.203 sec)

mysql> INSERT INTO messages (user_id, content, reply)
    -> VALUES (1, '哈囉', '你好呀！'); -- 測試內容
Query OK, 1 row affected (0.206 sec)

mysql> SELECT * FROM messages;
+----+---------+---------------------+-----------------+---------------------+
| id | user_id | content(使用者輸入的) | reply(我們回覆的) | created_at          |
+----+---------+---------------------+-----------------+---------------------+
|  1 |       1 | 哈囉                 | 你好呀！         | 2025-09-16 01:45:34 |
+----+---------+---------------------+-----------------+---------------------+
```
跑EC2(lwc)
```
ssh -i ~/desktop/劉建良專題/financial-agent-key.pem ubuntu@3.21.167.93
```
<<<<<<< HEAD
test
=======
```
-- 選擇你要用的資料庫
USE financial_agent;

-- 確認有哪些資料表
SHOW TABLES;

-- 查使用者資料
SELECT * FROM users;
```
>>>>>>> 2910413 (hi)
