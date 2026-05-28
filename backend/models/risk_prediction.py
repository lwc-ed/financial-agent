from sqlalchemy import Column, Integer, Float, DateTime, String
from sqlalchemy.sql import func
from backend.database import Base


class RiskPrediction(Base):
    __tablename__ = "risk_predictions"

    user_id = Column(Integer, primary_key=True)
    predicted_expense_7d = Column(Float)
    monthly_income_avg = Column(Float)
    risk_ratio = Column(Float)
    risk_level = Column(Integer)         # 1=safe, 2=caution, 3=alert, 4=critical
    alarm = Column(String(16))           # low_risk / high_risk
    data_days = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())
    last_notified_at = Column(DateTime, nullable=True)
    last_notified_level = Column(Integer, nullable=True)
