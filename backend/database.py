from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from urllib.parse import quote_plus

<<<<<<< HEAD
URL_DATABASE = "mysql+pymysql://nycuiemagent:SUPERidol$@financial-agent.cpwk2ce8cqyu.us-east-2.rds.amazonaws.com:3306/financial_agent"
=======
DB_USER = "nycuiemagent"
DB_PASSWORD = quote_plus("SUPERidol$")  # 特殊字元要 encode
DB_HOST = "financial-agent.cpwk2ce8cqyu.us-east-2.rds.amazonaws.com"
DB_PORT = "3306"
DB_NAME = "financial_agent"             # ⚠️ 改這裡，不要用 mysql
>>>>>>> 7090550 (hh)

URL_DATABASE = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(URL_DATABASE, echo=True, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
Base = declarative_base()