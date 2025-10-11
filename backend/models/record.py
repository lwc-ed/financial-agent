# backend/models/record.py
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from backend.database import Base   # 從 database.py 匯入 Base

class Record(Base):
    __tablename__ = "records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64))
    type = Column(String(10), nullable=False)       # '支出' / '收入'
    category = Column(String(64), nullable=False)
    amount = Column(Integer, nullable=False)
    note = Column(String(255))
    timestamp = Column(DateTime, server_default=func.now())

