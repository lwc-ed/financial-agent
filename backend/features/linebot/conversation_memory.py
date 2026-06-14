from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, Index

from backend.core.database import Base


class ConversationMemory(Base):
    __tablename__ = "conversation_memory"
    __table_args__ = (
        Index("idx_conversation_memory_user_created", "line_user_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    line_user_id = Column(String(64), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True, index=True)
