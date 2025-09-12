from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base 

URL_DATABASE = "mysql+pymysql://test:SUPERidol$0907@172.20.225.159:3306/linebotfast"

engine=create_engine(URL_DATABASE, echo=True, future=True)

SessionLocal=sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base=declarative_base() 