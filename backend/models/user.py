from sqlalchemy import Column, Integer, String, DateTime
from backend.database import Base  # 注意：如果您的環境是從根目錄跑，建議加 backend.

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String(50), nullable=False)
    provider_id = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)

    # 這裡是解決衝突後的結果
    line_user_id = Column(String(64), unique=True, nullable=True)
    current_function = Column(String(50), nullable=True)
    last_activity_time = Column(DateTime, nullable=True)
