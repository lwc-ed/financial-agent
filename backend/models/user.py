from sqlalchemy import Column, Integer, String, DateTime
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)  # 系統用流水號
    provider = Column(String(50), nullable=False)
    provider_id = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    line_user_id = Column(String(64), unique=True, nullable=True)  # LINE 對應 ID

    # 新增的欄位
    current_function = Column(String, nullable=True)  
    last_activity_time = Column(DateTime, default=datetime.utcnow)