from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from backend.core.database import Base


class RiskNotification(Base):
    __tablename__ = "risk_notifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    risk_level = Column(Integer, nullable=False)
    direction = Column(String(16), nullable=False)  # upgrade / downgrade / same
    notified_at = Column(DateTime, server_default=func.now())
