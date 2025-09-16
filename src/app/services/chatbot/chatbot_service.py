"""
Chatbot Service Layer for Sprint 1.
Handles business logic for chatbot operations including conversation management.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ...crud.crud_chatbot import ChatbotCRUD
from ...models.chatbot import Conversation, Message
from ...schemas.chatbot import (
    ChatRequest, 
    ChatResponse, 
    ConversationSummary,
    ConversationCreate,
    ConversationRead
)
from .agent import LegalChatbotAgentV1

logger = logging.getLogger(__name__)


class ChatbotService:
    """
    Service layer for chatbot operations.
    
    Handles:
    - Chat message processing
    - Conversation management
    - Agent orchestration
    - Database persistence
    """
    
    def __init__(self, db_session: AsyncSession, user_id: int):
        self.db = db_session
        self.user_id = user_id
        self.crud = ChatbotCRUD(db_session)
        self._agent_cache: Dict[str, LegalChatbotAgentV1] = {}
    
    async def process_chat_request(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request and return response.
        
        Args:
            request: Chat request from user
            
        Returns:
            Chat response with agent output
        """
        try:
            # Get or create conversation
            conversation = await self._get_or_create_conversation(request.conversation_id)
            
            # Save user message
            user_message = await self._save_user_message(conversation.id, request.message)
            
            # Get agent for this conversation
            agent = self._get_agent_for_conversation(str(conversation.id))
            
            # Process the question with the agent
            agent_result = agent.process_question(request.message, self.user_id)
            
            if not agent_result.get("success", False):
                # Handle agent errors
                error_response = ChatResponse(
                    answer="Maaf, terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi.",
                    conversation_id=conversation.id,
                    message_id=user_message.id,
                    processing_time=agent_result.get("processing_time", 0.0),
                    timestamp=datetime.utcnow()
                )
                return error_response
            
            # Save assistant message with agent metadata
            assistant_message = await self._save_assistant_message(
                conversation.id,
                agent_result["answer"],
                agent_result.get("reasoning_steps", []),
                agent_result.get("tool_calls", []),
                agent_result.get("processing_time", 0.0)
            )
            
            # Build response
            response = ChatResponse(
                answer=agent_result["answer"],
                conversation_id=conversation.id,
                message_id=assistant_message.id,
                reasoning_steps=self._format_reasoning_steps(agent_result.get("reasoning_steps", [])),
                tool_calls=self._format_tool_calls(agent_result.get("tool_calls", [])),
                processing_time=agent_result.get("processing_time", 0.0),
                timestamp=datetime.utcnow()
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing chat request: {str(e)}")
            # Return error response
            return ChatResponse(
                answer="Maaf, terjadi kesalahan sistem. Silakan coba lagi nanti.",
                conversation_id=request.conversation_id or uuid.uuid4(),
                message_id=uuid.uuid4(),
                processing_time=0.0,
                timestamp=datetime.utcnow()
            )
    
    async def get_user_conversations(self, limit: int = 20, offset: int = 0) -> List[ConversationSummary]:
        """
        Get list of user's conversations.
        
        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            
        Returns:
            List of conversation summaries
        """
        try:
            conversations = await self.crud.get_user_conversations(
                user_id=self.user_id,
                limit=limit,
                offset=offset
            )
            
            summaries = []
            for conv in conversations:
                # Get last message for preview
                last_message = await self.crud.get_last_message(conv.id)
                message_count = await self.crud.get_message_count(conv.id)
                
                summary = ConversationSummary(
                    id=conv.id,
                    title=conv.title or "New Conversation",
                    last_message=last_message.content if last_message else "",
                    message_count=message_count,
                    created_at=conv.created_at,
                    updated_at=conv.updated_at
                )
                summaries.append(summary)
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error getting user conversations: {str(e)}")
            return []
    
    async def get_conversation_detail(self, conversation_id: uuid.UUID) -> Optional[ConversationRead]:
        """
        Get full conversation details with messages.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Full conversation details or None if not found
        """
        try:
            # Verify user owns this conversation
            conversation = await self.crud.get_conversation_by_id(conversation_id)
            if not conversation or conversation.user_id != self.user_id:
                return None
            
            # Get all messages for this conversation
            messages = await self.crud.get_conversation_messages(conversation_id)
            
            # Format messages
            formatted_messages = []
            for msg in messages:
                formatted_msg = {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "reasoning_steps": msg.reasoning_steps,
                    "tool_calls": msg.tool_calls,
                    "citations": msg.citations,
                    "processing_time": msg.processing_time,
                    "confidence_score": msg.confidence_score,
                    "created_at": msg.created_at
                }
                formatted_messages.append(formatted_msg)
            
            return ConversationRead(
                id=conversation.id,
                user_id=conversation.user_id,
                title=conversation.title,
                messages=formatted_messages,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at
            )
            
        except Exception as e:
            logger.error(f"Error getting conversation detail: {str(e)}")
            return None
    
    async def create_conversation(self, request: ConversationCreate) -> ConversationRead:
        """
        Create a new conversation.
        
        Args:
            request: Conversation creation request
            
        Returns:
            Created conversation details
        """
        try:
            conversation = await self.crud.create_conversation(
                user_id=self.user_id,
                title=request.title
            )
            
            # If initial message provided, process it
            if request.initial_message:
                chat_request = ChatRequest(
                    message=request.initial_message,
                    conversation_id=conversation.id
                )
                await self.process_chat_request(chat_request)
            
            # Return conversation details
            return await self.get_conversation_detail(conversation.id)
            
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise
    
    async def delete_conversation(self, conversation_id: uuid.UUID) -> bool:
        """
        Delete a conversation (soft delete).
        
        Args:
            conversation_id: ID of conversation to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify user owns this conversation
            conversation = await self.crud.get_conversation_by_id(conversation_id)
            if not conversation or conversation.user_id != self.user_id:
                return False
            
            # Delete conversation (cascade will handle messages)
            await self.crud.delete_conversation(conversation_id)
            
            # Clear agent cache for this conversation
            if str(conversation_id) in self._agent_cache:
                del self._agent_cache[str(conversation_id)]
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting conversation: {str(e)}")
            return False
    
    def _get_agent_for_conversation(self, conversation_id: str) -> LegalChatbotAgentV1:
        """
        Get or create agent instance for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Agent instance for the conversation
        """
        if conversation_id not in self._agent_cache:
            self._agent_cache[conversation_id] = LegalChatbotAgentV1(conversation_id)
        
        return self._agent_cache[conversation_id]
    
    async def _get_or_create_conversation(self, conversation_id: Optional[uuid.UUID]) -> Conversation:
        """Get existing conversation or create new one."""
        if conversation_id:
            conversation = await self.crud.get_conversation_by_id(conversation_id)
            if conversation and conversation.user_id == self.user_id:
                return conversation
        
        # Create new conversation
        return await self.crud.create_conversation(
            user_id=self.user_id,
            title=None  # Will be auto-generated
        )
    
    async def _save_user_message(self, conversation_id: uuid.UUID, content: str) -> Message:
        """Save user message to database."""
        return await self.crud.create_message(
            conversation_id=conversation_id,
            role="user",
            content=content
        )
    
    async def _save_assistant_message(
        self,
        conversation_id: uuid.UUID,
        content: str,
        reasoning_steps: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        processing_time: float
    ) -> Message:
        """Save assistant message with metadata to database."""
        return await self.crud.create_message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            reasoning_steps=reasoning_steps,
            tool_calls=tool_calls,
            processing_time=processing_time
        )
    
    def _format_reasoning_steps(self, steps: List[Dict[str, Any]]) -> List:
        """Format reasoning steps for response."""
        formatted_steps = []
        for step in steps:
            formatted_steps.append({
                "step_number": step.get("step_number"),
                "action": step.get("action"),
                "tool_used": step.get("tool_used"),
                "result": step.get("result"),
                "timestamp": datetime.fromtimestamp(step.get("timestamp", 0))
            })
        return formatted_steps
    
    def _format_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List:
        """Format tool calls for response."""
        formatted_calls = []
        for call in tool_calls:
            formatted_calls.append({
                "tool_name": call.get("tool_name"),
                "input_data": call.get("input_data", {}),
                "output_data": call.get("output_data"),
                "execution_time": call.get("execution_time", 0.0),
                "success": call.get("success", True)
            })
        return formatted_calls