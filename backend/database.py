from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from urllib.parse import quote_plus

DB_USER = "nycuiemagent"
DB_PASSWORD = quote_plus("SUPERidol$")  # 密碼含 $，需 URL encode
DB_HOST = "financial-agent.cpwk2ce8cqyu.us-east-2.rds.amazonaws.com"
DB_PORT = "3306"
DB_NAME = "financial_agent"             # 指向你的應用資料庫

URL_DATABASE = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(
    URL_DATABASE,
    echo=True,            # 開發期可觀察 SQL；上線可關掉
    future=True,
    pool_pre_ping=True,   # 避免連線閒置斷線
    pool_recycle=280,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
Base = declarative_base()

# 啟動時簡單檢查目前連到哪個 DB（可留著做除錯）
try:
    with engine.connect() as conn:
        cur_db = conn.execute(text("SELECT DATABASE()")).scalar()
        print(">>> CURRENT DATABASE:", cur_db)
except Exception as e:
    print(">>> DATABASE CONNECTION TEST FAILED:", e)
