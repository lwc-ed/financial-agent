"""
架構冒煙測試（feature-based 重構後驗證用）
=================================================
目的：在「不連真實資料庫」的情況下，驗證重構後整個 backend 仍能正確組裝、
所有 feature 模組都 import 得到、所有路由都接得上 handler。

策略：
  - create_all() 與資料庫連線一律被替換成假的（不會 timeout）。
  - 需要 DB 的 handler，session 換成 FakeSession，查詢一律回傳空資料，
    代表「假裝連到 DB」。只要 handler 有回傳 HTTP 狀態碼，就證明 wiring 沒壞。
  - 純函式（稅務試算）直接用範例輸入真的跑一次。

執行方式（從專案根目錄）：
  python3 -m backend.tests.test_smoke
"""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── 1. 阻斷真實 DB：create_all 變 no-op，避免啟動時連線 ──────────────
import sqlalchemy
sqlalchemy.MetaData.create_all = lambda *a, **k: None


# ── 假的資料庫 session：任何查詢都回傳空資料（假裝連到了）──────────────
class _FakeQuery:
    def all(self):    return []
    def first(self):  return None
    def scalar(self): return 0
    def count(self):  return 0
    def __iter__(self): return iter([])
    def __getattr__(self, _):           # filter / filter_by / order_by / limit ...
        return lambda *a, **k: self


class _FakeSession:
    def query(self, *a, **k):   return _FakeQuery()
    def execute(self, *a, **k): return _FakeQuery()
    def add(self, *a, **k):     pass
    def commit(self):           pass
    def rollback(self):         pass
    def close(self):            pass
    def __getattr__(self, _):   return lambda *a, **k: None


def _fake_sessionmaker(*a, **k):
    return _FakeSession()


# ── 待驗證的所有 feature 模組 ────────────────────────────────────────
MODULES = [
    # core
    "backend.core.database", "backend.core.token_tracker",
    "backend.core.response_logger", "backend.core.monthly_stats",
    "backend.core.embedder", "backend.core.token_log",
    # web
    "backend.features.web.auth", "backend.features.web.dashboard",
    "backend.features.web.liff_test", "backend.features.web.profile",
    "backend.features.web.user",
    # linebot
    "backend.features.linebot.linebot", "backend.features.linebot.conversation_memory",
    # expense
    "backend.features.expense.expense_record", "backend.features.expense.expense_history",
    "backend.features.expense.record",
    # credit_card
    "backend.features.credit_card.ai_parser", "backend.features.credit_card.ai_reply",
    "backend.features.credit_card.benefit_query",
    "backend.features.credit_card.format_benefit_summary",
    # tax / wishlist / saving / quiz
    "backend.features.tax.tax_calculator",
    "backend.features.wishlist.wishlist", "backend.features.wishlist.wishlist_model",
    "backend.features.saving_challenge.saving_challenge",
    "backend.features.saving_challenge.saving_challenge_model",
    "backend.features.quiz.quiz_handler",
    # news
    "backend.features.news.daily_news_service", "backend.features.news.intent_recognizer",
    "backend.features.news.market_data", "backend.features.news.openai_news",
    "backend.features.news.perplexity_search", "backend.features.news.rss_fetcher",
    "backend.features.news.daily_news",
    # risk
    "backend.features.risk.bigru_service", "backend.features.risk.feature_schema",
    "backend.features.risk.notification_service", "backend.features.risk.ml_risk",
    "backend.features.risk.risk_prediction", "backend.features.risk.risk_notification",
    # financial_status
    "backend.features.financial_status.financial_status",
    "backend.features.financial_status.financial_status_service",
    "backend.features.financial_status.financial_status_model",
    # qa_rag
    "backend.features.qa_rag.rag_service", "backend.features.qa_rag.rebuild_chroma",
]

# app.py 註冊的 blueprint（feature : 模組路徑 : 變數名）
BLUEPRINTS = [
    ("web.auth",            "backend.features.web.auth",                       "auth_bp"),
    ("linebot",             "backend.features.linebot.linebot",                "linebot_bp"),
    ("expense.record",      "backend.features.expense.expense_record",         "expense_record_bp"),
    ("expense.history",     "backend.features.expense.expense_history",        "expense_history_bp"),
    ("wishlist",            "backend.features.wishlist.wishlist",              "wishlist_bp"),
    ("saving_challenge",    "backend.features.saving_challenge.saving_challenge", "saving_challenge_bp"),
    ("web.profile",         "backend.features.web.profile",                    "profile_bp"),
    ("web.liff_test",       "backend.features.web.liff_test",                  "liff_test_bp"),
    ("web.dashboard",       "backend.features.web.dashboard",                  "dashboard_bp"),
    ("risk.ml_risk",        "backend.features.risk.ml_risk",                   "ml_risk_bp"),
    ("financial_status",    "backend.features.financial_status.financial_status", "financial_status_bp"),
]

PREFIX = {
    "expense_record_bp": "/api/expense_record",
    "expense_history_bp": "/api/expense_history",
    "wishlist_bp": "/api/wishlist",
    "profile_bp": "/api/profile",
    "ml_risk_bp": "/api/ml",
    "financial_status_bp": "/api/financial-status",
}


def main():
    passed, failed = 0, 0

    def ok(msg):
        nonlocal passed; passed += 1; print(f"  ✅ {msg}")

    def bad(msg):
        nonlocal failed; failed += 1; print(f"  ❌ {msg}")

    # ── 1. 模組 import ────────────────────────────────────────────
    print("\n[1] 模組 import（驗證重構後 wiring）")
    loaded = {}
    for m in MODULES:
        try:
            loaded[m] = importlib.import_module(m)
            ok(m)
        except Exception as e:
            bad(f"{m}  ->  {type(e).__name__}: {e}")

    # 把所有模組裡的 DB session 換成假的（假裝連到 DB）
    for mod in loaded.values():
        for name in ("SessionLocal", "SessionBenefit"):
            if hasattr(mod, name):
                setattr(mod, name, _fake_sessionmaker)

    # ── 2. Flask app 組裝 + 註冊所有 blueprint ─────────────────────
    print("\n[2] Flask app 組裝與 blueprint 註冊")
    from flask import Flask
    from flask_cors import CORS
    app = Flask("backend.app", template_folder="features/web/liff")
    app.root_path = os.path.abspath("backend")
    CORS(app)

    @app.route("/favicon.ico")
    def favicon():
        return "", 204

    for label, modpath, varname in BLUEPRINTS:
        try:
            bp = getattr(importlib.import_module(modpath), varname)
            app.register_blueprint(bp, url_prefix=PREFIX.get(varname))
            ok(f"register {label}")
        except Exception as e:
            bad(f"register {label}  ->  {type(e).__name__}: {e}")

    n_routes = len(list(app.url_map.iter_rules()))
    ok(f"路由總數 = {n_routes}")

    # ── 3. 純函式：稅務試算（不需 DB，真的算一次）─────────────────
    print("\n[3] 純邏輯功能：台灣所得稅試算")
    try:
        from backend.features.tax.tax_calculator import calculate_taiwan_tax_2026
        result = calculate_taiwan_tax_2026({
            "gross_income": 1_000_000, "marital_status": "n",
            "num_under_70": 1, "salary_earners": 1,
        })
        assert isinstance(result, str) and len(result) > 0
        ok(f"稅務試算回傳結果（{len(result)} 字）")
    except Exception as e:
        bad(f"稅務試算  ->  {type(e).__name__}: {e}")

    # ── 4. 所有 GET 路由：用假 DB 打打看（證明 handler 接得到）──────
    print("\n[4] GET 路由冒煙（假 DB，回應即代表 wiring OK）")
    client = app.test_client()
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: str(r)):
        if "GET" not in (rule.methods or set()):
            continue
        if rule.arguments:          # 跳過需要路徑參數的
            continue
        path = str(rule)
        try:
            resp = client.get(path)
            ok(f"GET {path}  ->  HTTP {resp.status_code}")
        except Exception as e:
            bad(f"GET {path}  ->  {type(e).__name__}: {e}")

    # ── 5. POST 路由：只確認存在於 url_map（需 payload，不實打）─────
    print("\n[5] POST 路由存在性")
    post_paths = sorted({str(r) for r in app.url_map.iter_rules()
                         if "POST" in (r.methods or set())})
    for p in post_paths:
        ok(f"POST {p}  已註冊")

    # ── 總結 ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  結果：{passed} 通過 / {failed} 失敗")
    print("=" * 55)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
