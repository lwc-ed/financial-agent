# financial-agent

LINE Bot 理財助手，支援記帳、消費查詢、慾望清單與儲蓄挑戰等功能。

- [LINE 官方帳號管理後台](https://developers.line.biz/console/channel/2007892068/messaging-api)

---

# 📦 必要安裝套件

### Python（後端）
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Node.js / npm（前端）
主要套件：`react`、`react-dom`、`vite`、`liff`、`axios`、`openai`
```bash
cd frontend
npm install
```

---

# 🚀 啟動方式

### 後端
```bash
cd backend
python3 app.py
```
或使用 FastAPI：
```bash
cd backend
uvicorn main:app --reload
```
啟動後可至 http://127.0.0.1:8000/docs 測試 API。

### 前端
```bash
cd frontend
npm install
npm start
```

> 確認 `.env` 設定正確，避免啟動錯誤。

---

# 📌 功能統整表

| 功能 | 目的 | 實作方式 | 資料處理位置 | 展示方式 |
|------|------|----------|--------------|----------|
| **1. 個人資料填寫** | 讓使用者建立個人檔案（姓名、收入、目標等） | 提供表單（LIFF 小網頁 / 外部網站） | 存在伺服器的資料庫（DB） | LINE Bot 回覆「填寫完成」，或在 LIFF 直接看到 |
| **2. 紀錄消費（記帳）** | 讓使用者輸入日常消費 | 使用 LINE Bot 指令（ex:「午餐 120」）或 LIFF 表單 | 後端伺服器接收 → 存入 DB | LINE Bot 即時回覆「已紀錄：午餐 120 元」 |
| **3. 消費紀錄（查看狀況）** | 查看消費分類、剩餘可用金額 | 後端讀取 DB，計算統計 | DB 運算 + 伺服器整理成報表 | LINE Bot 傳回清單 / 圖表圖片（或開 LIFF 網頁看完整表格） |
| **4. 慾望清單** | 使用者輸入想買的東西＋金額 | LINE Bot 輸入「想買 Switch 10000」或 LIFF 表單 | 存在 DB，標記是否已達成 | LINE Bot 回傳目前清單；或在 LIFF 展示更漂亮的清單 |
| **5. 儲蓄挑戰** | 幫使用者依照清單規劃存錢 | 後端演算法計算最佳分配 + OpenAI API 生成自然語言建議 | 演算法（DB → 儲蓄計畫表），AI（生成對話） | LINE Bot 文字 / 圖表呈現；或 LIFF 展示「進度條 + 成就系統」 |

---

# 🏗️ 整體架構

1. **LINE Bot** — 基本互動（輸入消費、查詢紀錄、呼叫功能），傳文字 / 按鈕 / Quick Reply
2. **伺服器 + 資料庫** — Flask / FastAPI、MySQL / PostgreSQL，演算法處理（消費統計、儲蓄規劃）
3. **AI API（選配）** — 把演算法結果轉成自然語言建議
4. **LIFF / 外部網站** — 適合表單與進度視覺化，讓使用者感覺還在 LINE 內操作

---

# 📂 專案結構

```
financial-agent/
├── backend/
│   ├── app.py                 # Flask 主程式，啟動後端伺服器
│   ├── linebot_handler.py     # 處理 LINE Bot 事件的邏輯
│   ├── routes/                # Blueprint 模組，依功能分離（expense_record、wishlist 等）
│   ├── models/                # 資料模型定義（ORM / schema）
│   └── setup_rich_menu.py     # 設定 LINE Rich Menu 的腳本
├── frontend/
│   ├── public/                # React 公開資源
│   ├── src/                   # React 程式碼
│   ├── .env                   # 前端環境變數設定
│   └── package.json
├── picture/
│   └── rich_menu/             # LINE Rich Menu 使用的圖片
├── README.md
└── .gitignore
```

**命名規範**：Blueprint 皆以 `_bp` 結尾（例如 `user_bp`、`expense_bp`）。

---

# 🗄️ 資料庫（AWS RDS MySQL）

### WSL 連線 RDS

**1. 確認 DNS 解析**
```bash
nslookup financial-agent.cpwk2ce8cqyu.us-east-2.rds.amazonaws.com
```

**2. 安裝 MySQL client 並測試連線**
```bash
sudo apt update && sudo apt install mysql-client -y
mysql -h financial-agent.cpwk2ce8cqyu.us-east-2.rds.amazonaws.com -P 3306 -u nycuiemagent -p
```

**3. 修正 WSL DNS 問題（如無法解析）**
```bash
sudo nano /etc/wsl.conf
# 加入：
# [network]
# generateResolvConf = false

sudo rm /etc/resolv.conf
echo -e "nameserver 8.8.8.8\nnameserver 1.1.1.1" | sudo tee /etc/resolv.conf
# 重啟 WSL：wsl --shutdown
```

**常見錯誤**
- `Unknown MySQL server host` → 檢查 `/etc/resolv.conf`，確認有 8.8.8.8 或 1.1.1.1
- `Access denied for user` → 帳號或密碼錯誤，或 RDS user 無對外權限
- `Timeout` → AWS Console → Security group → Inbound rules，開放 MySQL/3306

### 常用 SQL 指令
```sql
USE financial_agent;
SHOW TABLES;
SELECT * FROM users;
SELECT * FROM messages;
```

### FastAPI 常用 API
| 方法 | 路徑 | 說明 |
|------|------|------|
| POST | `/users/` | 新增使用者 |
| GET | `/users/{user_id}` | 讀取使用者 |
| POST | `/posts/` | 新增貼文 |
| GET | `/posts/{post_id}` | 讀取貼文 |
| DELETE | `/posts/{post_id}` | 刪除貼文 |

---

# ☁️ EC2 部署

**連線 EC2**
```bash
# lwc 專用
ssh -i ~/desktop/劉建良專題/financial-agent-key.pem ubuntu@3.21.167.93

# 所有人
ssh ubuntu@3.21.167.93
```

**在 EC2 測試自己的 branch**
```bash
# Step 1：本地寫完後推上去
git push

# Step 2：進 EC2 切換 branch
ssh ubuntu@3.21.167.93
cd financial-agent
git checkout 自己的branch  # 例如：git checkout feature-login

# Step 3：更新並啟動
git pull
cd backend
python3 app.py

# Step 4：測試無誤後，回 GitHub 發 PR merge 到 main
```

**更改 EC2 時區**
```bash
sudo timedatectl set-timezone Asia/Taipei
timedatectl  # 確認時區
```

---

# 🔀 Git 工作流程

**讓 feature branch 與 main 同步**
```bash
git fetch origin
git checkout main
git pull origin main
git checkout feature-login
git merge main
```

---

# 🐍 虛擬環境

```bash
# 建立
python3 -m venv venv

# 啟動
source venv/bin/activate

# 安裝套件
pip install -r requirements.txt

# 啟動後端
python3 -m backend.app
```

**其他指令**
```bash
python3 -m backend.routes.credit_card.cube_benefits_scraper  # 爬蟲
python3 backend/ai/test_full_flow.py                          # 測試檔
```

---

# 🔧 補充說明

**setup_rich_menu 執行方式（only for lwc）**
```bash
/opt/homebrew/bin/python3.10 setup_rich_menu.py
```

**LINE 畫面區域配置示意**
```
┌────────────┬────────────┬────────────┐
│   Area A   │   Area B   │            │  ← 上半部 (y=0 ~ 421)
├────────────┼────────────│   Area C   │
│   Area D   │   Area E   │            │  ← 下半部 (y=421 ~ 843)
└────────────┴────────────┴────────────┘
```

**待修改項目**
- `ai/` 資料夾裡的 `ai_query` 尋找邏輯還要再改
- `ai_parser` 也還要再改

---

# 📊 實驗結果

| 日期 | 訓練模型 | 訓練結果 | 備注 |
|------|----------|----------|------|
|      |          |          |      |
