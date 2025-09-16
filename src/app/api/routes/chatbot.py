"""
FastAPI routes for chatbot endpoints.
Implements chat functionality with ReAct agent integration.
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.dependencies import get_current_user, get_database_session
from ...models.user import User
from ...schemas.chatbot import (
    ChatRequest,
    ChatResponse,
    ConversationSummary,
    ConversationCreate,
    ConversationRead,
    ConversationUpdate,
    ChatbotStats
)
from ...services.chatbot import ChatbotService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chatbot"])


@router.post("/", response_model=ChatResponse)
async def chat_with_legal_assistant(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> ChatResponse:
    """
    Send a message to the legal chatbot assistant.
    
    This endpoint processes legal questions using the ReAct agent with access to:
    - Legal document search via RAG system
    - Text summarization capabilities  
    - Fact validation and verification
    
    Features:
    - Multi-step reasoning with audit trail
    - Tool usage logging for transparency
    - Conversation memory management
    - Response time monitoring
    
    Args:
        request: Chat request with user message and optional conversation ID
        background_tasks: FastAPI background tasks for async operations
        current_user: Authenticated user
        db: Database session
        
    Returns:
        ChatResponse with assistant's answer and metadata
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Initialize chatbot service
        chatbot_service = ChatbotService(db, current_user.id)
        
        # Add background task for analytics
        background_tasks.add_task(
            _log_chat_analytics,
            current_user.id,
            request.message,
            len(request.message)
        )
        
        # Process chat request
        response = await chatbot_service.process_chat_request(request)
        
        logger.info(
            f"Chat processed for user {current_user.id}, "
            f"processing_time: {response.processing_time:.2f}s"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing chat request"
        )


@router.get("/conversations", response_model=List[ConversationSummary])
async def get_user_conversations(
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> List[ConversationSummary]:
    """
    Get list of user's chat conversations.
    
    Returns paginated list of conversations with summary information
    including last message preview and message count.
    
    Args:
        limit: Maximum number of conversations to return (default: 20)
        offset: Number of conversations to skip for pagination (default: 0)
        current_user: Authenticated user
        db: Database session
        
    Returns:
        List of conversation summaries
    """
    try:
        # Validate pagination parameters
        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be between 1 and 100"
            )
        
        if offset < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Offset must be non-negative"
            )
        
        # Get conversations
        chatbot_service = ChatbotService(db, current_user.id)
        conversations = await chatbot_service.get_user_conversations(limit, offset)
        
        return conversations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving conversations"
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationRead)
async def get_conversation_detail(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> ConversationRead:
    """
    Get detailed conversation information with all messages.
    
    Returns complete conversation history including all messages,
    reasoning steps, tool calls, and metadata for audit purposes.
    
    Args:
        conversation_id: UUID of the conversation
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Complete conversation details
        
    Raises:
        HTTPException: If conversation not found or access denied
    """
    try:
        chatbot_service = ChatbotService(db, current_user.id)
        conversation = await chatbot_service.get_conversation_detail(conversation_id)
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or access denied"
            )
        
        return conversation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation detail: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving conversation details"
        )


@router.post("/conversations", response_model=ConversationRead)
async def create_new_conversation(
    request: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> ConversationRead:
    """
    Create a new chat conversation.
    
    Optionally includes an initial message to start the conversation.
    If initial message is provided, it will be processed by the agent.
    
    Args:
        request: Conversation creation request
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Created conversation details
    """
    try:
        chatbot_service = ChatbotService(db, current_user.id)
        conversation = await chatbot_service.create_conversation(request)
        
        return conversation
        
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating conversation"
        )


@router.put("/conversations/{conversation_id}", response_model=ConversationRead)
async def update_conversation(
    conversation_id: UUID,
    request: ConversationUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> ConversationRead:
    """
    Update conversation details (e.g., title).
    
    Args:
        conversation_id: UUID of the conversation to update
        request: Update request with new values
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Updated conversation details
        
    Raises:
        HTTPException: If conversation not found or access denied
    """
    try:
        chatbot_service = ChatbotService(db, current_user.id)
        
        # First verify user has access to this conversation
        existing = await chatbot_service.get_conversation_detail(conversation_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or access denied"
            )
        
        # Update conversation (implementation needed in service)
        # For now, return the existing conversation
        # TODO: Implement update functionality in service layer
        
        return existing
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating conversation"
        )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> dict:
    """
    Delete a conversation and all its messages.
    
    This is a soft delete operation. The conversation and messages
    will be marked as deleted but retained for audit purposes.
    
    Args:
        conversation_id: UUID of the conversation to delete
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Success confirmation
        
    Raises:
        HTTPException: If conversation not found or access denied
    """
    try:
        chatbot_service = ChatbotService(db, current_user.id)
        
        success = await chatbot_service.delete_conversation(conversation_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or access denied"
            )
        
        return {"message": "Conversation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting conversation"
        )


@router.get("/stats", response_model=ChatbotStats)
async def get_chatbot_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session)
) -> ChatbotStats:
    """
    Get chatbot usage statistics for the current user.
    
    Returns metrics like total conversations, messages, average response time,
    and other usage analytics.
    
    Args:
        current_user: Authenticated user
        db: Database session
        
    Returns:
        User's chatbot usage statistics
    """
    try:
        chatbot_service = ChatbotService(db, current_user.id)
        
        # Get basic stats from CRUD
        from ...crud.crud_chatbot import ChatbotCRUD
        crud = ChatbotCRUD(db)
        stats = await crud.get_user_stats(current_user.id)
        
        # Build response with default values for missing fields
        return ChatbotStats(
            total_conversations=stats.get("total_conversations", 0),
            total_messages=stats.get("total_messages", 0),
            average_response_time=stats.get("average_processing_time", 0.0),
            average_confidence_score=stats.get("average_confidence_score", None),
            most_used_tools=[],  # TODO: Implement tool usage tracking
            today_conversations=0,  # TODO: Implement daily stats
            today_messages=0  # TODO: Implement daily stats
        )
        
    except Exception as e:
        logger.error(f"Error getting chatbot stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving chatbot statistics"
        )


@router.get("/health")
async def chatbot_health_check() -> dict:
    """
    Health check endpoint for chatbot services.
    
    Verifies that all chatbot components are functioning properly:
    - LLM service connectivity
    - Tool availability
    - Agent initialization
    
    Returns:
        Health status information
    """
    try:
        # Test agent initialization
        from ...services.chatbot.agent import LegalChatbotAgentV1
        
        test_agent = LegalChatbotAgentV1()
        health_status = test_agent.health_check()
        
        return {
            "status": "healthy" if health_status["overall_status"] == "healthy" else "unhealthy",
            "components": health_status,
            "timestamp": str(datetime.utcnow())
        }
        
    except Exception as e:
        logger.error(f"Chatbot health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": str(datetime.utcnow())
        }


# Background task functions

async def _log_chat_analytics(user_id: int, message: str, message_length: int) -> None:
    """
    Background task to log chat analytics.
    
    Args:
        user_id: ID of the user
        message: User's message
        message_length: Length of the message
    """
    try:
        # TODO: Implement analytics logging
        # This could log to a separate analytics database or service
        logger.info(f"Chat analytics: user={user_id}, length={message_length}")
        
    except Exception as e:
        logger.error(f"Error logging chat analytics: {str(e)}")


# Import datetime for health check
from datetime import datetime