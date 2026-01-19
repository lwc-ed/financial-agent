from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from urllib.parse import quote_plus

DB_USER = "nycuiemagent"
DB_PASSWORD = quote_plus("SUPERidol$")  # 密碼含 $
DB_HOST = "financial-agent-rescued.cpwk2ce8cqyu.us-east-2.rds.amazonaws.com"
DB_PORT = "3306"

# --- 主應用資料庫 financial_agent ---
MAIN_DB_NAME = "financial_agent"
MAIN_DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{MAIN_DB_NAME}"

engine = create_engine(
    MAIN_DB_URL,
    echo=True,
    future=True,
    pool_pre_ping=True,
    pool_recycle=280,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)

# --- 信用卡回饋專用資料庫 credit_card_benefits ---
BENEFIT_DB_NAME = "credit_card_benefits"
BENEFIT_DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{BENEFIT_DB_NAME}"

engine_benefit = create_engine(
    BENEFIT_DB_URL,
    echo=True,
    future=True,
    pool_pre_ping=True,
    pool_recycle=280,
)

# 給 cube_benefits 爬蟲用（寫入 credit_card_benefits DB）
SessionBenefit = sessionmaker(bind=engine_benefit, autocommit=False, autoflush=False, future=True)

# Base
Base = declarative_base()

# --- 啟動時顯示目前連線狀況 ---
try:
    with engine.connect() as conn:
        print(">>> MAIN DATABASE:", conn.execute(text("SELECT DATABASE()")).scalar())
except Exception as e:
    print(">>> MAIN DATABASE CONNECTION FAILED:", e)

try:
    with engine_benefit.connect() as conn:
        print(">>> BENEFIT DATABASE:", conn.execute(text("SELECT DATABASE()")).scalar())
except Exception as e:
    print(">>> BENEFIT DATABASE CONNECTION FAILED:", e)