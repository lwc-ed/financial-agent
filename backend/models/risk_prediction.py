from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from backend.database import Base


class RiskPrediction(Base):
    __tablename__ = "risk_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    line_user_id = Column(String(64), nullable=False)
    predicted_expense_7d = Column(Float)
    monthly_income_avg = Column(Float)
    risk_ratio = Column(Float)
    risk_level = Column(Integer)   # 1=safe, 2=caution, 3=alert, 4=critical
    alarm = Column(String(16))     # low_risk / high_risk
    data_days = Column(Integer)    # 用了幾天的消費資料
    created_at = Column(DateTime, server_default=func.now())
