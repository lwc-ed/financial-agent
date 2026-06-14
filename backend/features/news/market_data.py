from datetime import datetime, timezone, timedelta, date

import yfinance as yf

TAIPEI_TZ = timezone(timedelta(hours=8))

# ── 追蹤標的定義 ──────────────────────────────────────────────────
TICKERS = {
    "台股": {
        "加權指數":  "^TWII",
        "台積電":   "2330.TW",
        "聯發科":   "2454.TW",
    },
    "美股": {
        "S&P 500":  "^GSPC",
        "道瓊":     "^DJI",
        "Nasdaq":   "^IXIC",
    },
    "匯率": {
        "美元指數":  "DX-Y.NYB",
        "美元/台幣": "USDTWD=X",
        "美元/日圓": "USDJPY=X",
        "歐元/美元": "EURUSD=X",
    },
    "大宗商品": {
        "黃金":  "GC=F",
        "原油":  "CL=F",
    },
}


def _fetch_ticker(symbol: str, display_name: str) -> dict:
    """
    抓單一標的最新數據，回傳：
      name, symbol, price, change, change_pct, currency, updated_at
    """
    try:
        tk   = yf.Ticker(symbol)
        hist = tk.history(period="2d")   # 抓兩天以便計算漲跌

        if hist.empty or len(hist) < 1:
            return {"name": display_name, "symbol": symbol, "error": "no data"}

        latest = hist.iloc[-1]
        price  = round(float(latest["Close"]), 4)

        if len(hist) >= 2:
            prev_close = round(float(hist.iloc[-2]["Close"]), 4)
            change     = round(price - prev_close, 4)
            change_pct = round(change / prev_close * 100, 2) if prev_close else 0.0
        else:
            change     = 0.0
            change_pct = 0.0

        updated_at = datetime.now(tz=TAIPEI_TZ).strftime("%Y-%m-%d %H:%M")

        return {
            "name":       display_name,
            "symbol":     symbol,
            "price":      price,
            "change":     change,
            "change_pct": change_pct,   # 正值 = 上漲，負值 = 下跌
            "updated_at": updated_at,
        }

    except Exception as e:
        return {"name": display_name, "symbol": symbol, "error": str(e)}


def fetch_historical_data(ticker_symbol: str, period_days: int, label: str = "") -> dict:
    """
    抓取特定標的的歷史每日收盤價。

    Args:
        ticker_symbol: yfinance 代號，如 "^DJI"
        period_days:   天數，如 30、90、365
        label:         顯示用中文名稱，如 "道瓊指數"

    Returns:
        {
          "ticker": "^DJI",
          "label": "道瓊指數",
          "period_days": 30,
          "period_change_pct": 3.9,
          "period_high": {"price": 50500.0, "date": "2026-05-10"},
          "period_low":  {"price": 47200.0, "date": "2026-04-25"},
          "records": [{"date": "...", "close": ..., "change_pct": ...}, ...]
        }
        失敗時回傳 {"error": "..."}
    """
    try:
        tk    = yf.Ticker(ticker_symbol)
        start = (datetime.now() - timedelta(days=period_days)).strftime("%Y-%m-%d")
        hist  = tk.history(start=start)

        if hist.empty:
            return {"ticker": ticker_symbol, "label": label, "error": "no data"}

        records = []
        prev_close = None
        for dt, row in hist.iterrows():
            close = round(float(row["Close"]), 4)
            if prev_close is not None and prev_close != 0:
                chg_pct = round((close - prev_close) / prev_close * 100, 2)
            else:
                chg_pct = 0.0
            records.append({
                "date":       dt.strftime("%Y-%m-%d"),
                "close":      close,
                "change_pct": chg_pct,
            })
            prev_close = close

        closes    = [r["close"] for r in records]
        max_close = max(closes)
        min_close = min(closes)
        max_date  = records[closes.index(max_close)]["date"]
        min_date  = records[closes.index(min_close)]["date"]
        period_chg = round((closes[-1] - closes[0]) / closes[0] * 100, 2) if closes[0] else 0.0

        return {
            "ticker":            ticker_symbol,
            "label":             label or ticker_symbol,
            "period_days":       period_days,
            "period_change_pct": period_chg,
            "period_high":       {"price": max_close, "date": max_date},
            "period_low":        {"price": min_close, "date": min_date},
            "records":           records,
        }

    except Exception as e:
        return {"ticker": ticker_symbol, "label": label, "error": str(e)}


def fetch_market_data() -> dict:
    """
    抓取所有追蹤標的，回傳巢狀 dict：
    {
      "台股": {
        "加權指數": {"price": ..., "change": ..., "change_pct": ...},
        ...
      },
      "美股": { ... },
      ...
      "fetched_at": "2026-05-21 08:00"
    }
    """
    result: dict = {}

    for category, items in TICKERS.items():
        result[category] = {}
        for display_name, symbol in items.items():
            data = _fetch_ticker(symbol, display_name)
            result[category][display_name] = data

    result["fetched_at"] = datetime.now(tz=TAIPEI_TZ).strftime("%Y-%m-%d %H:%M")
    return result
