import uuid as uuid_pkg
from datetime import UTC, datetime
from typing import List, TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text, Enum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.db.database import Base

if TYPE_CHECKING:
    from .user import User


class Conversation(Base):
    """
    Model for storing chat conversations.
    Each conversation contains multiple messages between user and assistant.
    """
    __tablename__ = "conversations"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default_factory=uuid_pkg.uuid4, 
        unique=True,
        init=False
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default_factory=lambda: datetime.now(UTC),
        init=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default_factory=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        init=False
    )

    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="conversations",
        init=False
    )
    messages: Mapped[List["Message"]] = relationship(
        "Message", 
        back_populates="conversation", 
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        init=False
    )


class Message(Base):
    """
    Model for storing individual messages in a conversation.
    Includes support for agent reasoning steps and tool calls for audit trail.
    """
    __tablename__ = "messages"

    id: Mapped[uuid_pkg.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default_factory=uuid_pkg.uuid4, 
        unique=True,
        init=False
    )
    conversation_id: Mapped[uuid_pkg.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("conversations.id"), 
        nullable=False, 
        index=True
    )
    role: Mapped[str] = mapped_column(
        Enum("user", "assistant", name="message_role"), 
        nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Agent-specific fields for audit trail
    reasoning_steps: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)
    tool_calls: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)
    citations: Mapped[dict | None] = mapped_column(JSONB, nullable=True, default=None)
    
    # Metadata
    processing_time: Mapped[float | None] = mapped_column(nullable=True, default=None)
    confidence_score: Mapped[float | None] = mapped_column(nullable=True, default=None)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default_factory=lambda: datetime.now(UTC),
        init=False
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", 
        back_populates="messages",
        init=False
    )