# 系統架構圖
```mermaid
flowchart TD
    subgraph FE["前台（使用者端）"]
        direction LR
        USER([LINE 使用者])
        LIFF([LIFF 前端<br/>React + Vite])
    end

    CB[Flask /callback]
    ORCH{GPT-4o-mini<br/>意圖判斷}

    subgraph INTENT["意圖分流"]
        direction LR
        ACC[記帳模組<br/>expense / income]
        CC[信用卡查詢<br/>FTS + LIKE]
        RAG[金融問答<br/>RAG]
        NEWS[新聞 Pipeline<br/>RSS + Perplexity]
        OTHER[稅務試算<br/>保險測驗]
    end

    BIGRU[BiGRU 風險預測<br/>背景 Thread]
    REPLY[GPT 生成回覆<br/>& LINE 回覆]
    TOKEN[Token 用量追蹤]

    subgraph BE["後台（資料儲存端）"]
        direction LR
        MYSQL[(MySQL<br/>financial agent)]
        CCDB[(MySQL<br/>credit card benefits)]
        CHROMA[(ChromaDB<br/>金融知識向量庫)]
    end

    USER        -->|傳送訊息| CB
    CB          --> ORCH

    ORCH        --> ACC
    ORCH        --> CC
    ORCH        --> RAG
    ORCH        --> NEWS
    ORCH        --> OTHER
    ORCH        -->|query expense| MYSQL

    ACC         -->|寫入紀錄| MYSQL
    ACC         --> BIGRU
    BIGRU       -->|風險通知 Push| USER
    BIGRU       -->|寫入預測結果| MYSQL

    CC          -->|FTS / LIKE| CCDB
    RAG         -->|向量搜尋| CHROMA

    ACC         --> REPLY
    CC          --> REPLY
    RAG         --> REPLY
    NEWS        --> REPLY
    OTHER       --> REPLY
    MYSQL       -->|查詢紀錄| REPLY

    REPLY       -->|LINE 回覆| USER
    REPLY       --> TOKEN
    TOKEN       -->|累加寫入| MYSQL
    LIFF        -->|Google OAuth / 個人資料| MYSQL

    classDef orange fill:#E8916A,stroke:#C97A52,color:#fff
    classDef blue   fill:#4A6FA5,stroke:#3A5A8A,color:#fff
    classDef teal   fill:#2E8B7A,stroke:#1E7A69,color:#fff
    classDef green  fill:#5A8A5A,stroke:#3A6A3A,color:#fff
    classDef purple fill:#7A5AA8,stroke:#5A3A88,color:#fff
    classDef gray   fill:#6B7280,stroke:#4B5260,color:#fff
    classDef diamond fill:#3D5A9E,stroke:#2D4A8E,color:#fff

    class USER,LIFF orange
    class CB,ACC,CC,NEWS,OTHER blue
    class RAG,REPLY teal
    class BIGRU green
    class TOKEN purple
    class MYSQL,CCDB,CHROMA gray
    class ORCH diamond

    style FE     fill:#FDF3EC,stroke:#E8916A,color:#333
    style INTENT fill:#EDF2FA,stroke:#4A6FA5,color:#333
    style BE     fill:#F0F0F2,stroke:#6B7280,color:#333
```










# 分層架構圖
## 第一層 
### 總覽
```mermaid
flowchart LR
    USER([LINE 使用者]):::orange
    LIFF([LIFF 前端<br/>React + Vite]):::orange

    FLASK[Flask 後端<br/>意圖判斷 & 路由分派]:::blue

    subgraph AI["AI 模組群"]
        direction TB
        A1[記帳 & ML 風險預測]:::green
        A2[信用卡 & 金融問答]:::teal
        A3[新聞 & 稅務 & 保險測驗]:::teal
    end

    subgraph DB["資料儲存層"]
        direction TB
        D1[(MySQL<br/>主資料庫)]:::gray
        D2[(MySQL<br/>信用卡回饋)]:::gray
        D3[(ChromaDB<br/>金融知識)]:::gray
    end

    USER -->|LINE 訊息| FLASK
    LIFF -->|Google 登入 / 個人資料| FLASK
    FLASK -->|意圖分流| AI
    AI -->|讀寫| DB
    FLASK -->|查詢 / 寫入| DB
    AI -->|回覆| USER

    classDef orange fill:#E8916A,stroke:#C97A52,color:#fff
    classDef blue   fill:#4A6FA5,stroke:#3A5A8A,color:#fff
    classDef teal   fill:#2E8B7A,stroke:#1E7A69,color:#fff
    classDef green  fill:#5A8A5A,stroke:#3A6A3A,color:#fff
    classDef gray   fill:#6B7280,stroke:#4B5260,color:#fff

    style AI fill:#EDF2FA,stroke:#4A6FA5,color:#333
    style DB fill:#F0F0F2,stroke:#6B7280,color:#333
```

## 第二層 
### Flask
```mermaid
flowchart LR
    USER([LINE 使用者]):::orange
    LIFF([LIFF 前端]):::orange

    subgraph FLASK["Flask 後端"]
        direction TB
        CB["callback<br/>Webhook 接收"]:::blue
        ORCH{GPT-4o-mini<br/>意圖判斷}:::diamond
        AUTH["LINE 帳號綁定"]:::blue
    end

    subgraph INTENT["意圖分流"]
        direction LR

        subgraph COL1[" "]
            direction TB
            I1[expense / income<br/>記帳]:::green
            I2[query expense<br/>查帳]:::blue
        end

        subgraph COL2[" "]
            direction TB
            I3[credit card<br/>信用卡查詢]:::teal
            I4[financial qa<br/>金融問答]:::teal

        end

        subgraph COL3[" "]
            direction TB
            I5[news<br/>每日新聞]:::teal
            I6[tax / quiz<br/>稅務 & 保險測驗]:::blue
            I7[remember context<br/>短期背景記憶]:::purple
        end

    end

    TOKEN[Token 用量追蹤<br/>每日上限 50,000]:::purple
    LOGGER[Response Logger<br/>紀錄回覆與耗時]:::purple

    USER   -->|LINE 訊息| CB
    LIFF   -->|登入 / 個人資料| AUTH
    CB     --> ORCH
    ORCH   --> INTENT
    INTENT --> TOKEN
    TOKEN  --> LOGGER
    LOGGER -->|LINE 回覆| USER

    classDef orange  fill:#E8916A,stroke:#C97A52,color:#fff
    classDef blue    fill:#4A6FA5,stroke:#3A5A8A,color:#fff
    classDef teal    fill:#2E8B7A,stroke:#1E7A69,color:#fff
    classDef green   fill:#5A8A5A,stroke:#3A6A3A,color:#fff
    classDef purple  fill:#7A5AA8,stroke:#5A3A88,color:#fff
    classDef diamond fill:#3D5A9E,stroke:#2D4A8E,color:#fff

    style FLASK  fill:#EDF2FA,stroke:#4A6FA5,color:#333
    style INTENT fill:#F0F7F4,stroke:#2E8B7A,color:#333
    style COL1   fill:none,stroke:none
    style COL2   fill:none,stroke:none
    style COL3   fill:none,stroke:none
```
---

### DB
```mermaid
flowchart LR
    subgraph MAIN["MySQL — financial agent"]
        direction TB
        T1[users<br/>使用者基本資料 & LINE 綁定]:::gray
        T2[records<br/>收支紀錄]:::gray
        T3[wishlist<br/>欲望清單]:::gray
        T4[risk predictions<br/>BiGRU 風險預測結果]:::gray
        T5[user token logs<br/>Token 用量累計]:::gray
        T6[daily news<br/>每日新聞摘要快取]:::gray
        T7[conversation memory<br/>短期對話記憶 TTL 2h]:::gray
        T1 ~~~ T2 ~~~ T3 ~~~ T4 ~~~ T5 ~~~ T6 ~~~ T7
    end

    subgraph CC["MySQL | 信用卡優惠"]
        direction TB
        C1[各銀行信用卡回饋資料表<br/>FTS 全文檢索 + LIKE fallback]:::ccblue
    end

    subgraph CHROMA["ChromaDB — 金融知識向量庫"]
        direction TB
        V1[金融知識 PDF 向量化<br/>Financial knowledge final.pdf]:::teal
        V2[HuggingFace Embedding<br/>google/embedding-gemma-300m]:::teal
        V1 ~~~ V2
    end

    W1[記帳 / 查帳]:::blue
    W2[BiGRU 風險預測]:::green
    W3[Token 追蹤]:::purple
    W4[信用卡查詢]:::blue
    W5[金融問答 RAG]:::teal
    W6[每日新聞]:::blue
    W7[Google OAuth / LIFF]:::orange

    W8[短期記憶模組]:::purple

    W7  -->|寫入 / 更新| T1
    W1  -->|寫入| T2
    W1  -->|讀取| T2
    W2  -->|寫入| T4
    W3  -->|累加寫入| T5
    W6  -->|快取寫入| T6
    W8  -->|讀寫| T7
    W4  -->|FTS / LIKE 查詢| C1
    W5  -->|向量搜尋| V1

    classDef gray   fill:#6B7280,stroke:#4B5260,color:#fff
    classDef ccblue fill:#4A6FA5,stroke:#3A5A8A,color:#fff
    classDef teal   fill:#2E8B7A,stroke:#1E7A69,color:#fff
    classDef blue   fill:#4A6FA5,stroke:#3A5A8A,color:#fff
    classDef green  fill:#5A8A5A,stroke:#3A6A3A,color:#fff
    classDef purple fill:#7A5AA8,stroke:#5A3A88,color:#fff
    classDef orange fill:#E8916A,stroke:#C97A52,color:#fff

    style MAIN   fill:#F0F0F2,stroke:#6B7280,color:#333
    style CC     fill:#EDF2FA,stroke:#4A6FA5,color:#333
    style CHROMA fill:#F0F7F4,stroke:#2E8B7A,color:#333
```
---

### 記帳 & ML 風險預測
```mermaid
flowchart LR
    A([使用者傳訊]):::orange
    B[GPT 解析<br/>金額 / 類別]:::blue
    C[(records 表<br/>寫入收支)]:::gray
    D[BiGRU 風險預測<br/>背景 Thread]:::green
    E{風險等級}:::diamond
    F([LINE Push 通知]):::orange
    G[(risk predictions 表)]:::gray

    A --> B --> C --> D --> E
    E -->|高風險| F
    E -->|正常| G
    D --> G

    classDef orange fill:#E8916A,stroke:#C97A52,color:#fff
    classDef blue   fill:#4A6FA5,stroke:#3A5A8A,color:#fff
    classDef green  fill:#5A8A5A,stroke:#3A6A3A,color:#fff
    classDef gray   fill:#6B7280,stroke:#4B5260,color:#fff
    classDef diamond fill:#3D5A9E,stroke:#2D4A8E,color:#fff
```

---

### 信用卡查詢
```mermaid
flowchart LR
    A([使用者傳訊]):::orange
    B[ai parser<br/>GPT 抽出品牌候選名稱]:::blue
    C[benefit query<br/>FTS 全文檢索]:::blue
    D{有結果?}:::diamond
    E[LIKE fallback<br/>模糊查詢]:::blue
    F[format benefit summary<br/>整理回饋摘要]:::blue
    G[ai reply<br/>GPT 生成回覆]:::teal
    H([LINE Push 回覆]):::orange

    A --> B --> C --> D
    D -->|是| F
    D -->|否| E --> F
    F --> G --> H

    classDef orange  fill:#E8916A,stroke:#C97A52,color:#fff
    classDef blue    fill:#4A6FA5,stroke:#3A5A8A,color:#fff
    classDef teal    fill:#2E8B7A,stroke:#1E7A69,color:#fff
    classDef diamond fill:#3D5A9E,stroke:#2D4A8E,color:#fff
```
---


### 金融問答 RAG
```mermaid
flowchart LR
    A([使用者傳訊]):::orange
    B[HuggingFace Embedding<br/>問題向量化]:::teal
    C[(ChromaDB<br/>向量搜尋相關段落)]:::gray
    D[GPT 結合上下文<br/>生成回答]:::teal
    E([LINE 回覆]):::orange

    A --> B --> C --> D --> E

    classDef orange fill:#E8916A,stroke:#C97A52,color:#fff
    classDef teal   fill:#2E8B7A,stroke:#1E7A69,color:#fff
    classDef gray   fill:#6B7280,stroke:#4B5260,color:#fff
```

---
### 每日新聞 Pipeline
```mermaid
flowchart LR
    A([使用者傳訊]):::orange
    B[intent recognizer<br/>識別主題 broad query]:::blue
    C[rss fetcher<br/>抓取 RSS 向量搜尋<br/>限 24 小時內]:::blue
    D{文章足夠?}:::diamond
    E[perplexity search<br/>fallback 補充]:::blue
    F[market data<br/>yfinance 今日行情]:::blue
    G[openai news<br/>GPT 生成新聞摘要]:::teal
    H[(daily news 表<br/>快取結果)]:::gray
    I([LINE 回覆]):::orange

    A --> B --> C --> D
    D -->|否| E --> G
    D -->|是| G
    F --> G
    G --> H --> I

    classDef orange  fill:#E8916A,stroke:#C97A52,color:#fff
    classDef blue    fill:#4A6FA5,stroke:#3A5A8A,color:#fff
    classDef teal    fill:#2E8B7A,stroke:#1E7A69,color:#fff
    classDef gray    fill:#6B7280,stroke:#4B5260,color:#fff
    classDef diamond fill:#3D5A9E,stroke:#2D4A8E,color:#fff
```

---

### 所得稅試算
```mermaid
flowchart LR
    A([使用者傳訊]):::orange
    B{tax sessions<br/>有進行中 session?}:::diamond
    C[開新 session<br/>逐題補問]:::blue
    D[繼續補問<br/>收集缺少參數]:::blue
    E[calculate taiwan tax 2026<br/>本地試算]:::teal
    F([LINE 回覆試算結果]):::orange

    A --> B
    B -->|否| C --> D
    B -->|是| D
    D -->|參數齊全| E --> F

    classDef orange  fill:#E8916A,stroke:#C97A52,color:#fff
    classDef blue    fill:#4A6FA5,stroke:#3A5A8A,color:#fff
    classDef teal    fill:#2E8B7A,stroke:#1E7A69,color:#fff
    classDef diamond fill:#3D5A9E,stroke:#2D4A8E,color:#fff
```

---

### 保險風險測驗
```mermaid
flowchart LR
    A([使用者傳訊]):::orange
    B{quiz engine<br/>有進行中 session?}:::diamond
    C[Full Insurance Quiz Handler<br/>開始測驗 / 發題]:::blue
    D[使用者作答<br/>儲存進度至 user sessions]:::blue
    E[計算風險承受等級<br/>RR Level 1–5]:::teal
    F([LINE 回覆測驗結果]):::orange

    A --> B
    B -->|否| C --> D
    B -->|是| D
    D -->|全部答完| E --> F

    classDef orange  fill:#E8916A,stroke:#C97A52,color:#fff
    classDef blue    fill:#4A6FA5,stroke:#3A5A8A,color:#fff
    classDef teal    fill:#2E8B7A,stroke:#1E7A69,color:#fff
    classDef diamond fill:#3D5A9E,stroke:#2D4A8E,color:#fff
```
---

### 短期對話記憶
```mermaid
flowchart LR
    A([使用者傳訊]):::orange
    B[載入近期記憶<br/>_load_recent_memory<br/>最多 10 則 / TTL 2h]:::purple
    C[刪除已過期記錄<br/>lazy cleanup]:::purple
    D[orchestrate<br/>記憶作為 GPT context]:::blue
    E{intent}:::diamond
    F[remember_context<br/>回覆「我記住了」]:::purple
    G[其他功能處理]:::blue
    H[寫入使用者訊息<br/>_remember_message role=user]:::purple
    I[寫入 Bot 回覆<br/>_remember_message role=assistant]:::purple
    J[(conversation_memory 表<br/>MySQL)]:::gray

    A --> B
    B --> C
    B --> D
    D --> E
    E -->|remember_context| F --> I
    E -->|其他| G --> I
    A --> H
    H --> J
    I --> J
    B --> J

    classDef orange  fill:#E8916A,stroke:#C97A52,color:#fff
    classDef blue    fill:#4A6FA5,stroke:#3A5A8A,color:#fff
    classDef purple  fill:#7A5AA8,stroke:#5A3A88,color:#fff
    classDef gray    fill:#6B7280,stroke:#4B5260,color:#fff
    classDef diamond fill:#3D5A9E,stroke:#2D4A8E,color:#fff
```

---

## 第三層
### BiGRU 風險預測 — 訓練 Pipeline
```mermaid
flowchart LR
    subgraph OFFLINE["離線訓練"]
        direction LR
        subgraph PRE["資料前處理"]
            direction TB
            P1[IBM 消費資料<br>1_preprocess_ibm.py]:::gray
            P2[個人記帳資料<br>2_preprocess_personal.py]:::gray
            P3[產生風險標籤<br>2b_generate_labels.py]:::gray
            P1 ~~~ P2 ~~~ P3
        end
        subgraph TRAIN["模型訓練"]
            direction TB
            T1[BiGRU 預訓練<br>3_pretrain_bigru.py]:::blue
            T2[個人資料 Fine-tune<br>4_finetune_bigru.py]:::blue
            T3[Ensemble 推論<br>5_predict_bigru.py]:::blue
            T1 ~~~ T2 ~~~ T3
        end
        ART[(artifacts_bigru_tl<br>模型權重 .pkl)]:::gray
        PRE --> TRAIN --> ART
    end

    subgraph ONLINE["線上推論"]
        direction TB
        L1[載入模型權重<br>_load_assets]:::green
        L2[讀取近 30 天記帳紀錄]:::green
        L3[特徵工程<br>ALIGNED_FEATURE_COLS]:::green
        L4[BiGRU 推論<br>預測未來 7 天消費]:::green
        L5[計算風險等級<br>Risk Level 1–5]:::green
    end

    ART --> L1
    L1 --> L2 --> L3 --> L4 --> L5

    classDef gray  fill:#6B7280,stroke:#4B5260,color:#fff
    classDef blue  fill:#4A6FA5,stroke:#3A5A8A,color:#fff
    classDef green fill:#5A8A5A,stroke:#3A6A3A,color:#fff

    style OFFLINE fill:#F5F5F5,stroke:#9CA3AF,color:#333
    style ONLINE  fill:#F0F7F0,stroke:#5A8A5A,color:#333
    style PRE     fill:none,stroke:none
    style TRAIN   fill:none,stroke:none
```

---


### 金融問答 RAG
```mermaid
flowchart LR
    subgraph OFFLINE["離線建庫"]
        direction TB
        O1[載入 PDF<br>Financial_knowledge_final.pdf]:::gray
        O2[切段 Chunking]:::blue
        O3[HuggingFace Embedding<br>google/embedding-gemma-300m]:::teal
        O4[(ChromaDB<br>儲存向量)]:::gray
        O1 --> O2 --> O3 --> O4
    end

    subgraph ONLINE["線上查詢（rag_service.py）"]
        direction TB
        Q1([使用者提問]):::orange
        Q2[問題向量化<br>同一 Embedding 模型]:::teal
        Q3[ChromaDB 向量搜尋<br>取 Top-K 相關段落]:::teal
        Q4[GPT 結合段落<br>生成回答]:::teal
        Q5([LINE 回覆]):::orange
        Q1 --> Q2 --> Q3 --> Q4 --> Q5
    end

    O4 --> Q3

    classDef orange fill:#E8916A,stroke:#C97A52,color:#fff
    classDef blue   fill:#4A6FA5,stroke:#3A5A8A,color:#fff
    classDef teal   fill:#2E8B7A,stroke:#1E7A69,color:#fff
    classDef gray   fill:#6B7280,stroke:#4B5260,color:#fff

    style OFFLINE fill:#F5F5F5,stroke:#9CA3AF,color:#333
    style ONLINE  fill:#F0F7F4,stroke:#2E8B7A,color:#333
```

---


### 信用卡查詢 — FTS 機制
```mermaid
flowchart LR
    A([使用者傳訊]):::orange
    B[ai_parser<br>GPT 抽出品牌候選<br>含信心分數 score]:::blue

    subgraph QUERY["benefit_query.py"]
        direction TB
        Q1[BANK_CARD_MAP<br>table_name → 銀行 & 卡名對應]:::blue
        Q2[fts_search<br>對所有資料表執行<br>FULLTEXT 全文檢索]:::blue
        Q3{有命中結果?}:::diamond
        Q4[like_search<br>LIKE 模糊查詢 fallback]:::blue
        Q1 --> Q2 --> Q3
        Q3 -->|否| Q4
    end

    C[format_benefit_summary<br>整理成結構化摘要]:::blue
    D[ai_reply<br>GPT 生成自然語言回覆]:::teal
    E([LINE Push 回覆]):::orange

    A --> B --> QUERY
    Q3 -->|是| C
    Q4 --> C
    C --> D --> E

    classDef orange  fill:#E8916A,stroke:#C97A52,color:#fff
    classDef blue    fill:#4A6FA5,stroke:#3A5A8A,color:#fff
    classDef teal    fill:#2E8B7A,stroke:#1E7A69,color:#fff
    classDef diamond fill:#3D5A9E,stroke:#2D4A8E,color:#fff

    style QUERY fill:#EDF2FA,stroke:#4A6FA5,color:#333
```
