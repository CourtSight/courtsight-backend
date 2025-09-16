"""
CRUD operations for chatbot models.
"""

import uuid
from typing import List, Optional
from datetime import datetime

from sqlalchemy import and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.future import select

from ..models.chatbot import Conversation, Message


class ChatbotCRUD:
    """
    Custom CRUD operations for chatbot conversations and messages.
    Uses async SQLAlchemy operations following CourtSight patterns.
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    # Conversation CRUD operations
    
    async def create_conversation(
        self, 
        user_id: int, 
        title: Optional[str] = None
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            user_id=user_id,
            title=title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)
        
        return conversation
    
    async def get_conversation_by_id(self, conversation_id: uuid.UUID) -> Optional[Conversation]:
        """Get conversation by ID."""
        query = select(Conversation).where(Conversation.id == conversation_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_user_conversations(
        self, 
        user_id: int, 
        limit: int = 20, 
        offset: int = 0
    ) -> List[Conversation]:
        """Get all conversations for a user."""
        query = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(desc(Conversation.updated_at))
            .limit(limit)
            .offset(offset)
        )
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def update_conversation(
        self, 
        conversation_id: uuid.UUID, 
        title: Optional[str] = None
    ) -> Optional[Conversation]:
        """Update conversation title."""
        conversation = await self.get_conversation_by_id(conversation_id)
        if not conversation:
            return None
        
        if title is not None:
            conversation.title = title
            conversation.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(conversation)
        
        return conversation
    
    async def delete_conversation(self, conversation_id: uuid.UUID) -> bool:
        """Delete a conversation and all its messages."""
        conversation = await self.get_conversation_by_id(conversation_id)
        if not conversation:
            return False
        
        # Delete all messages first (due to foreign key constraint)
        await self.db.execute(
            select(Message).where(Message.conversation_id == conversation_id)
        )
        
        # Delete conversation
        await self.db.delete(conversation)
        await self.db.commit()
        
        return True
    
    # Message CRUD operations
    
    async def create_message(
        self,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        reasoning_steps: Optional[dict] = None,
        tool_calls: Optional[dict] = None,
        citations: Optional[dict] = None,
        processing_time: Optional[float] = None,
        confidence_score: Optional[float] = None
    ) -> Message:
        """Create a new message in a conversation."""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            reasoning_steps=reasoning_steps,
            tool_calls=tool_calls,
            citations=citations,
            processing_time=processing_time,
            confidence_score=confidence_score
        )
        
        self.db.add(message)
        
        # Update conversation's updated_at timestamp
        conversation = await self.get_conversation_by_id(conversation_id)
        if conversation:
            conversation.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(message)
        
        return message
    
    async def get_message_by_id(self, message_id: uuid.UUID) -> Optional[Message]:
        """Get message by ID."""
        query = select(Message).where(Message.id == message_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_conversation_messages(
        self, 
        conversation_id: uuid.UUID,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Message]:
        """Get all messages for a conversation."""
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
            .offset(offset)
        )
        
        if limit:
            query = query.limit(limit)
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def get_last_message(self, conversation_id: uuid.UUID) -> Optional[Message]:
        """Get the last message in a conversation."""
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(desc(Message.created_at))
            .limit(1)
        )
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_message_count(self, conversation_id: uuid.UUID) -> int:
        """Get total number of messages in a conversation."""
        query = (
            select(func.count(Message.id))
            .where(Message.conversation_id == conversation_id)
        )
        
        result = await self.db.execute(query)
        return result.scalar() or 0
    
    async def delete_message(self, message_id: uuid.UUID) -> bool:
        """Delete a specific message."""
        message = await self.get_message_by_id(message_id)
        if not message:
            return False
        
        await self.db.delete(message)
        await self.db.commit()
        
        return True
    
    # Analytics and reporting
    
    async def get_user_stats(self, user_id: int) -> dict:
        """Get usage statistics for a user."""
        try:
            # Total conversations
            conv_count_query = (
                select(func.count(Conversation.id))
                .where(Conversation.user_id == user_id)
            )
            conv_count_result = await self.db.execute(conv_count_query)
            total_conversations = conv_count_result.scalar() or 0
            
            # Total messages
            msg_count_query = (
                select(func.count(Message.id))
                .join(Conversation)
                .where(Conversation.user_id == user_id)
            )
            msg_count_result = await self.db.execute(msg_count_query)
            total_messages = msg_count_result.scalar() or 0
            
            # Average processing time
            avg_time_query = (
                select(func.avg(Message.processing_time))
                .join(Conversation)
                .where(
                    and_(
                        Conversation.user_id == user_id,
                        Message.processing_time.isnot(None)
                    )
                )
            )
            avg_time_result = await self.db.execute(avg_time_query)
            avg_processing_time = avg_time_result.scalar() or 0.0
            
            # Average confidence score
            avg_conf_query = (
                select(func.avg(Message.confidence_score))
                .join(Conversation)
                .where(
                    and_(
                        Conversation.user_id == user_id,
                        Message.confidence_score.isnot(None)
                    )
                )
            )
            avg_conf_result = await self.db.execute(avg_conf_query)
            avg_confidence = avg_conf_result.scalar() or 0.0
            
            return {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "average_processing_time": float(avg_processing_time),
                "average_confidence_score": float(avg_confidence)
            }
            
        except Exception as e:
            # Return default stats on error
            return {
                "total_conversations": 0,
                "total_messages": 0,
                "average_processing_time": 0.0,
                "average_confidence_score": 0.0
            }
    
    async def get_conversation_with_messages(
        self, 
        conversation_id: uuid.UUID
    ) -> Optional[Conversation]:
        """Get conversation with all messages loaded."""
        query = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()