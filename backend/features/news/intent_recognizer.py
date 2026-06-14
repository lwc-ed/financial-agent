"""
意圖識別模組：分析使用者輸入，判斷是廣泛主題還是具體金融標的。

回傳 dict 範例：
  {
    "type": "specific",           # "broad" | "specific"
    "label": "道瓊指數",           # 標的中文名，broad 時為 null
    "ticker": "^DJI",             # yfinance ticker，broad 時為 null
    "broad_query": "美股走勢...",  # 適合向量搜尋的廣泛 query
    "period_days": 30,            # 歷史數據天數（使用者指定或預設 30）
    "is_verification": False,     # 是否為查證型問題
  }
"""
import json
import os
import re

from openai import OpenAI

# ── 查證型關鍵詞（本地預判，GPT 之前先過一次）────────────────────
_VERIFY_PATTERNS = re.compile(
    r"真的嗎|是否|有沒有|已經.{0,8}了|宣布|確認|"
    r"上市了嗎|退休了嗎|破產了嗎|停止|放棄|設總部|"
    r"還沒|有在|真的有|蒸發|倒閉|合併|併購.{0,4}嗎|"
    r"降息了嗎|加息了嗎|裁員了嗎"
)

# ── 人名關鍵詞（問人名 → 一律 broad，不抓歷史股價）──────────────
_PERSON_PATTERNS = re.compile(
    r"老黃|黃仁勳|Jensen Huang|"
    r"蘇姿丰|Lisa Su|"
    r"祖克柏|Zuckerberg|"
    r"馬斯克|Elon Musk|"
    r"黃敏雄|魏哲家|劉德音|"
    r"Sam Altman|奧特曼|"
    r"皮采|Pichai|Sundar"
)

_SYSTEM_PROMPT = """\
你是金融資料分析助理。分析使用者輸入，回傳一個 JSON 物件，格式如下：
{
  "type": "broad" 或 "specific",
  "label": "標的中文名稱（broad 時填 null）",
  "ticker": "yfinance 代號（broad 時填 null）",
  "broad_query": "適合向量搜尋的廣泛 query，中英文混合",
  "period_days": 整數（使用者指定的天數，未指定時預設 30）,
  "is_verification": true 或 false
}

判斷規則：
- specific：使用者問的是特定指數、個股、商品、匯率（必須能對應到 yfinance ticker）
- broad：使用者問的是廣泛主題、人物、因果、事件，或是查證某件事是否發生
- is_verification：只要問題含有「是否/真的嗎/已經...了/宣布/確認/停止/放棄/上市了嗎/
  破產了嗎/設總部/蒸發/合併」等查證意圖，就設為 true

【重要】以下情況一律 type=broad：
- 問某人（CEO、創辦人、分析師）的言論或動向 → broad，ticker=null
- 問事件是否發生（is_verification=true）→ broad，ticker=null
- 問因果、比較、趨勢等廣泛問題 → broad

常見 ticker 對照（只有問標的本身才用）：
  ^DJI=道瓊  ^GSPC=S&P500  ^IXIC=Nasdaq  ^TWII=台股加權  ^N225=日經  ^KS11=韓股
  2330.TW=台積電  2454.TW=聯發科  NVDA=輝達/Nvidia  AAPL=蘋果  TSLA=特斯拉  MSFT=微軟
  AMD=超微  META=Meta  GOOGL=Google  AMZN=Amazon
  GC=F=黃金  CL=F=原油  BTC-USD=比特幣  ETH-USD=以太坊
  DX-Y.NYB=美元指數  USDTWD=X=美元/台幣  USDJPY=X=美元/日圓  EURUSD=X=歐元/美元

時間範圍對照（period_days）：
  一週=7  一個月=30  一季/三個月=90  半年=180  一年=365

只回傳 JSON，不要其他文字。
"""

_EXAMPLES = [
    # 具體標的
    ("道瓊指數最近情況",
     '{"type":"specific","label":"道瓊指數","ticker":"^DJI","broad_query":"美股市場走勢 Dow Jones US stock market","period_days":30,"is_verification":false}'),
    ("台積電過去一季表現",
     '{"type":"specific","label":"台積電","ticker":"2330.TW","broad_query":"台積電 TSMC 半導體科技股","period_days":90,"is_verification":false}'),
    ("黃金過去一年走勢",
     '{"type":"specific","label":"黃金","ticker":"GC=F","broad_query":"黃金 gold 貴金屬 避險資產","period_days":365,"is_verification":false}'),
    # 廣泛主題
    ("今日股市摘要",
     '{"type":"broad","label":null,"ticker":null,"broad_query":"今日股市市場動態 stock market today","period_days":30,"is_verification":false}'),
    ("最近科技股動態",
     '{"type":"broad","label":null,"ticker":null,"broad_query":"科技股 technology stocks AI semiconductor","period_days":30,"is_verification":false}'),
    # 人名 → broad
    ("老黃最近又講了什麼？",
     '{"type":"broad","label":null,"ticker":null,"broad_query":"黃仁勳 Jensen Huang NVIDIA CEO 最新言論","period_days":30,"is_verification":false}'),
    ("蘇姿丰最近在推什麼？",
     '{"type":"broad","label":null,"ticker":null,"broad_query":"蘇姿丰 Lisa Su AMD CEO 最新動態","period_days":30,"is_verification":false}'),
    # 查證型
    ("OpenAI 已經上市了嗎？",
     '{"type":"broad","label":null,"ticker":null,"broad_query":"OpenAI IPO 上市 股票發行","period_days":30,"is_verification":true}'),
    ("台積電宣布併購 Intel 了嗎？",
     '{"type":"broad","label":null,"ticker":null,"broad_query":"台積電 TSMC Intel 併購 合併","period_days":30,"is_verification":true}'),
    ("聯準會確認今年會降息了嗎？",
     '{"type":"broad","label":null,"ticker":null,"broad_query":"聯準會 Fed 降息 利率決策 2025","period_days":30,"is_verification":true}'),
]


def _local_precheck(query: str) -> dict:
    """
    本地規則預判（不呼叫 GPT），回傳需要強制覆蓋的欄位。
    回傳空 dict 表示不需要覆蓋，讓 GPT 自己判斷。
    """
    overrides: dict = {}

    # 查證型：優先偵測
    if _VERIFY_PATTERNS.search(query):
        overrides["is_verification"] = True
        overrides["type"] = "broad"
        overrides["label"] = None
        overrides["ticker"] = None

    # 人名：一律 broad，不拉歷史股價
    if _PERSON_PATTERNS.search(query):
        overrides["type"] = "broad"
        overrides["label"] = None
        overrides["ticker"] = None
        # 若同時是查證型，is_verification 已在上面設好

    return overrides


def recognize_intent(query: str) -> dict:
    """
    分析使用者輸入，回傳意圖 dict。

    Args:
        query: 使用者原始輸入

    Returns:
        {
            "type": "broad" | "specific",
            "label": str | None,
            "ticker": str | None,
            "broad_query": str,
            "period_days": int,
            "is_verification": bool,
        }
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    # ── 本地預判（deterministic，不花 API）────────────────────────
    local_overrides = _local_precheck(query)
    if local_overrides:
        print(f"[intent_recognizer] local precheck overrides: {local_overrides}")

    client = OpenAI(api_key=api_key)

    # 組 few-shot messages
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for user_ex, asst_ex in _EXAMPLES:
        messages.append({"role": "user",      "content": user_ex})
        messages.append({"role": "assistant", "content": asst_ex})
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        max_tokens=200,
    )

    raw = (response.choices[0].message.content or "").strip()

    try:
        intent = json.loads(raw)
    except json.JSONDecodeError:
        # fallback：當 broad 處理
        print(f"[intent_recognizer] JSON parse failed: {raw!r}, fallback to broad")
        intent = {
            "type": "broad",
            "label": None,
            "ticker": None,
            "broad_query": query,
            "period_days": 30,
            "is_verification": False,
        }

    # 補全缺失欄位
    intent.setdefault("type", "broad")
    intent.setdefault("label", None)
    intent.setdefault("ticker", None)
    intent.setdefault("broad_query", query)
    intent.setdefault("period_days", 30)
    intent.setdefault("is_verification", False)

    # ── 本地預判覆蓋 GPT 結果（確保 verification/person 邏輯正確）──
    intent.update(local_overrides)

    print(
        f"[intent_recognizer] type={intent['type']!r}  "
        f"label={intent['label']!r}  ticker={intent['ticker']!r}  "
        f"period_days={intent['period_days']}  "
        f"is_verification={intent['is_verification']}  "
        f"broad_query={intent['broad_query']!r}"
    )
    return intent
