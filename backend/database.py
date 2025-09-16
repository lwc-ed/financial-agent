from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base 

URL_DATABASE = "mysql+pymysql://nycuiemagent:SUPERidol$@financial-agent.cpwk2ce8cqyu.us-east-2.rds.amazonaws.com:3306/mysql"

engine=create_engine(URL_DATABASE, echo=True, future=True)

SessionLocal=sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base=declarative_base()