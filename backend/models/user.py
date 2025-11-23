from sqlalchemy import Column, Integer, String
from database import Base
from sqlalchemy import Column, Integer, String, DateTime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String(50), nullable=False)
    provider_id = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    current_function = Column(String(50), nullable=True)
    last_activity_time = Column(DateTime, nullable=True)
    line_user_id = Column(String(64), unique=True, nullable=True)
