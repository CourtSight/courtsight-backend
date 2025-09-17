import logging
from datetime import datetime
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

from ...services.chatbot.service import (
    get_chatbot_service,
    ChatbotService,
    ChatRequest as ChatbotRequest,
    ChatResponse as ChatbotResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chatbot"])


@router.post("/", response_model=ChatResponse)
async def chat_with_legal_assistant(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    chat_service: ChatbotService = Depends(get_chatbot_service),
    # current_user: User = Depends(get_current_user),
    # db: AsyncSession = Depends(get_database_session)
) -> ChatResponse:
    """
    Send a message to the enhanced legal chatbot assistant (Sprint 2).
    
    This endpoint processes legal questions using enhanced features:
    - Langraph workflow orchestration for complex reasoning
    - Advanced tools (case comparison, precedent exploration, citation generation)
    - Enhanced memory system with entity tracking and topic clustering
    - Redis caching for performance optimization
    - Intelligent routing between simple and complex queries
    
    Features:
    - Multi-step reasoning with audit trail
    - Tool usage logging for transparency
    - Conversation memory management
    - Response time monitoring
    - Entity and topic tracking
    - Performance metrics
    
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
        # Get enhanced chatbot service
        
        # Ensure we have a conversation_id for memory continuity
        if not request.conversation_id:
            import uuid
            request.conversation_id = str(uuid.uuid4())
            logger.info(f"Created new conversation ID: {request.conversation_id}")
        else:
            logger.info(f"Using existing conversation ID: {request.conversation_id}")
            
        # Convert to enhanced request format
        enhanced_request = ChatbotRequest(
            query=request.message,
            conversation_id=request.conversation_id,
            use_workflow=True,  # Enable Langraph workflow
            use_advanced_tools=True,  # Enable advanced tools
            memory_enabled=True,  # Enable enhanced memory
            cache_enabled=True  # Enable Redis caching
        )
        
        # Add background task for analytics
        background_tasks.add_task(
            _log_chat_analytics,
            1,
            request.message,
            len(request.message)
        )
        
        # Process with enhanced service
        enhanced_response = await chat_service.chat(enhanced_request)
        
        # Convert back to standard response format
        import uuid
        from datetime import datetime
        
        response = ChatResponse(
            answer=enhanced_response.response,
            conversation_id=UUID(enhanced_response.conversation_id),
            message_id=request.conversation_id,  # Use existing conversation ID
            processing_time=enhanced_response.response_time,
            timestamp=datetime.utcnow(),
            confidence_score=enhanced_response.confidence_score,
        )
        
        logger.info(
            f"Enhanced chat processed for user 1, "
            f"processing_time: {response.processing_time:.2f}s, "
            f"workflow_used: {enhanced_response.workflow_used}, "
            f"tools_used: {enhanced_response.tools_used}"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in enhanced chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing chat request"
        )


@router.post("/stream")
async def chat_with_legal_assistant_stream(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    chat_service: ChatbotService = Depends(get_chatbot_service),
):
    """
    Send a message to the enhanced legal chatbot assistant with streaming response.
    
    This endpoint provides real-time streaming responses for better user experience:
    - Server-Sent Events (SSE) for real-time updates
    - Progress indicators for long-running operations
    - Step-by-step workflow visibility
    - Tool usage notifications
    - Error handling with graceful degradation
    
    Response format:
    - Each chunk is sent as Server-Sent Events
    - JSON data with different message types:
      - 'status': Processing status updates
      - 'partial_response': Incremental response content
      - 'final_response': Complete response
      - 'complete': Final metadata
      - 'error': Error information
    
    Args:
        request: Chat request with user message and optional conversation ID
        background_tasks: FastAPI background tasks for async operations
        chat_service: Enhanced chatbot service instance
        
    Returns:
        StreamingResponse with Server-Sent Events
    """
    from fastapi.responses import StreamingResponse
    
    try:
        # Ensure we have a conversation_id for memory continuity  
        if not request.conversation_id:
            import uuid
            request.conversation_id = str(uuid.uuid4())
            logger.info(f"Created new conversation ID for streaming: {request.conversation_id}")
        else:
            logger.info(f"Using existing conversation ID for streaming: {request.conversation_id}")
        
        # Convert to enhanced request format
        enhanced_request = ChatbotRequest(
            query=request.message,
            conversation_id=request.conversation_id,
            use_workflow=True,  # Enable Langraph workflow
            use_advanced_tools=True,  # Enable advanced tools
            memory_enabled=True,  # Enable enhanced memory
            cache_enabled=True  # Enable Redis caching
        )
        
        # Add background task for analytics
        background_tasks.add_task(
            _log_chat_analytics,
            1,
            request.message,
            len(request.message)
        )
        
        # Create streaming response
        return StreamingResponse(
            chat_service.chat_stream(enhanced_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"Error in streaming chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing streaming chat request"
        )


@router.get("/demo")
async def streaming_demo():
    """
    Serve a simple HTML demo page for testing the streaming chat functionality.
    
    Returns:
        HTML page with streaming chat interface
    """
    from fastapi.responses import HTMLResponse
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Streaming Chatbot Demo</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .chat-container {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .message-area {
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ddd;
                padding: 15px;
                margin-bottom: 20px;
                background: #fafafa;
                border-radius: 5px;
            }
            .message {
                margin-bottom: 15px;
                padding: 10px;
                border-radius: 5px;
            }
            .user-message {
                background: #007bff;
                color: white;
                text-align: right;
            }
            .bot-message {
                background: #e9ecef;
                color: #333;
            }
            .status-message {
                background: #fff3cd;
                color: #856404;
                font-style: italic;
                font-size: 0.9em;
            }
            .input-area {
                display: flex;
                gap: 10px;
            }
            #messageInput {
                flex: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
            }
            #sendButton {
                padding: 10px 20px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            #sendButton:hover {
                background: #0056b3;
            }
            #sendButton:disabled {
                background: #6c757d;
                cursor: not-allowed;
            }
            .progress-bar {
                width: 100%;
                height: 4px;
                background: #e9ecef;
                border-radius: 2px;
                margin: 5px 0;
                overflow: hidden;
            }
            .progress-fill {
                height: 100%;
                background: #007bff;
                transition: width 0.3s ease;
            }
            .status-message {
                background: #e8f4fd;
                border: 1px solid #bee5eb;
                color: #0c5460;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                font-size: 0.9em;
            }
            .metadata {
                font-size: 0.8em;
                color: #666;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <h1>Supreme Court Legal Assistant - Streaming Demo</h1>
            <p>Ask legal questions and see real-time streaming responses!</p>
            <div id="messageArea" class="message-area"></div>
            <div class="input-area">
                <input type="text" id="messageInput" placeholder="Ask a legal question..." />
                <button id="sendButton" onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            const messageArea = document.getElementById('messageArea');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            
            let currentBotMessage = null;
            
            // Get conversation ID from localStorage or create new one
            let currentConversationId = localStorage.getItem('conversationId');
            if (!currentConversationId) {
                currentConversationId = 'chat_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
                localStorage.setItem('conversationId', currentConversationId);
                console.log('Created new conversation ID:', currentConversationId);
            } else {
                console.log('Using existing conversation ID:', currentConversationId);
            }
            
            // Add conversation info to page
            messageArea.innerHTML = `<div class="message status-message">🔗 Conversation ID: ${currentConversationId}<br/>💡 Your conversation history will be remembered across page reloads</div>`;
            
            // Add button to clear conversation
            const clearButton = document.createElement('button');
            clearButton.textContent = 'Clear Conversation';
            clearButton.style.marginLeft = '10px';
            clearButton.style.padding = '5px 10px';
            clearButton.style.fontSize = '14px';
            clearButton.onclick = function() {
                if (confirm('Are you sure you want to clear the conversation history?')) {
                    localStorage.removeItem('conversationId');
                    location.reload();
                }
            };
            sendButton.parentNode.appendChild(clearButton);
            
            // Add Enter key support
            messageInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !sendButton.disabled) {
                    sendMessage();
                }
            });
            
            // Focus input on page load
            messageInput.focus();

            function addMessage(content, type = 'user') {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}-message`;
                messageDiv.innerHTML = content;
                messageArea.appendChild(messageDiv);
                messageArea.scrollTop = messageArea.scrollHeight;
                return messageDiv;
            }

            function updateBotMessage(content) {
                if (currentBotMessage) {
                    currentBotMessage.innerHTML = content;
                } else {
                    currentBotMessage = addMessage(content, 'bot');
                }
                messageArea.scrollTop = messageArea.scrollHeight;
            }

            function addStatusMessage(content) {
                const statusDiv = addMessage(content, 'status');
                setTimeout(() => {
                    if (statusDiv.parentNode) {
                        statusDiv.parentNode.removeChild(statusDiv);
                    }
                }, 3000);
            }

            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;

                sendButton.disabled = true;
                messageInput.disabled = true;

                addMessage(message, 'user');
                messageInput.value = '';
                currentBotMessage = null;

                try {
                    const response = await fetch('/api/v1/chat/stream', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            conversation_id: currentConversationId
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\\n');

                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.substring(6));
                                    
                                    switch (data.type) {
                                        case 'status':
                                            addStatusMessage(`🔄 ${data.message}`);
                                            break;
                                        
                                        case 'partial_response':
                                            updateBotMessage(data.content);
                                            break;
                                        
                                        case 'final_response':
                                            updateBotMessage(data.content);
                                            break;
                                        
                                        case 'complete':
                                            if (data.conversation_id && data.conversation_id !== currentConversationId) {
                                                currentConversationId = data.conversation_id;
                                                localStorage.setItem('conversationId', currentConversationId);
                                                console.log('Updated conversation ID:', currentConversationId);
                                            }
                                            const metadata = `
                                                <div class="metadata">
                                                    ⏱️ ${data.response_time.toFixed(2)}s | 
                                                    🔧 ${data.tools_used.length} tools | 
                                                    ${data.workflow_used ? '🔀 Workflow' : '🤖 Agent'}
                                                </div>
                                            `;
                                            if (currentBotMessage) {
                                                currentBotMessage.innerHTML += metadata;
                                            }
                                            break;
                                        
                                        case 'error':
                                            updateBotMessage(`❌ Error: ${data.message}`);
                                            break;
                                    }
                                } catch (e) {
                                    console.error('Error parsing SSE data:', e);
                                }
                            }
                        }
                    }
                } catch (error) {
                    console.error('Error:', error);
                    addMessage(`❌ Connection error: ${error.message}`, 'bot');
                } finally {
                    sendButton.disabled = false;
                    messageInput.disabled = false;
                    messageInput.focus();
                }
            }

            messageInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !sendButton.disabled) {
                    sendMessage();
                }
            });

            messageInput.focus();
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


# @router.get("/conversations", response_model=List[ConversationSummary])
# async def get_user_conversations(
#     limit: int = 20,
#     offset: int = 0,
#     current_user: User = Depends(get_current_user),
#     db: AsyncSession = Depends(get_database_session)
# ) -> List[ConversationSummary]:
#     """
#     Get list of user's chat conversations.
    
#     Returns paginated list of conversations with summary information
#     including last message preview and message count.
    
#     Args:
#         limit: Maximum number of conversations to return (default: 20)
#         offset: Number of conversations to skip for pagination (default: 0)
#         current_user: Authenticated user
#         db: Database session
        
#     Returns:
#         List of conversation summaries
#     """
#     try:
#         # Validate pagination parameters
#         if limit < 1 or limit > 100:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Limit must be between 1 and 100"
#             )
        
#         if offset < 0:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Offset must be non-negative"
#             )
        
#         # Get conversations
#         chatbot_service = ChatbotService(db, current_user.id)
#         conversations = await chatbot_service.get_user_conversations(limit, offset)
        
#         return conversations
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error getting conversations: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error retrieving conversations"
#         )


# @router.get("/conversations/{conversation_id}", response_model=ConversationRead)
# async def get_conversation_detail(
#     conversation_id: UUID,
#     current_user: User = Depends(get_current_user),
#     db: AsyncSession = Depends(get_database_session)
# ) -> ConversationRead:
#     """
#     Get detailed conversation information with all messages.
    
#     Returns complete conversation history including all messages,
#     reasoning steps, tool calls, and metadata for audit purposes.
    
#     Args:
#         conversation_id: UUID of the conversation
#         current_user: Authenticated user
#         db: Database session
        
#     Returns:
#         Complete conversation details
        
#     Raises:
#         HTTPException: If conversation not found or access denied
#     """
#     try:
#         chatbot_service = ChatbotService(db, current_user.id)
#         conversation = await chatbot_service.get_conversation_detail(conversation_id)
        
#         if not conversation:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="Conversation not found or access denied"
#             )
        
#         return conversation
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error getting conversation detail: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error retrieving conversation details"
#         )


# @router.post("/conversations", response_model=ConversationRead)
# async def create_new_conversation(
#     request: ConversationCreate,
#     current_user: User = Depends(get_current_user),
#     db: AsyncSession = Depends(get_database_session)
# ) -> ConversationRead:
#     """
#     Create a new chat conversation.
    
#     Optionally includes an initial message to start the conversation.
#     If initial message is provided, it will be processed by the agent.
    
#     Args:
#         request: Conversation creation request
#         current_user: Authenticated user
#         db: Database session
        
#     Returns:
#         Created conversation details
#     """
#     try:
#         chatbot_service = ChatbotService(db, current_user.id)
#         conversation = await chatbot_service.create_conversation(request)
        
#         return conversation
        
#     except Exception as e:
#         logger.error(f"Error creating conversation: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error creating conversation"
#         )


# @router.put("/conversations/{conversation_id}", response_model=ConversationRead)
# async def update_conversation(
#     conversation_id: UUID,
#     request: ConversationUpdate,
#     current_user: User = Depends(get_current_user),
#     db: AsyncSession = Depends(get_database_session)
# ) -> ConversationRead:
#     """
#     Update conversation details (e.g., title).
    
#     Args:
#         conversation_id: UUID of the conversation to update
#         request: Update request with new values
#         current_user: Authenticated user
#         db: Database session
        
#     Returns:
#         Updated conversation details
        
#     Raises:
#         HTTPException: If conversation not found or access denied
#     """
#     try:
#         chatbot_service = ChatbotService(db, current_user.id)
        
#         # First verify user has access to this conversation
#         existing = await chatbot_service.get_conversation_detail(conversation_id)
#         if not existing:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="Conversation not found or access denied"
#             )
        
#         # Update conversation (implementation needed in service)
#         # For now, return the existing conversation
#         # TODO: Implement update functionality in service layer
        
#         return existing
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error updating conversation: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error updating conversation"
#         )


# @router.delete("/conversations/{conversation_id}")
# async def delete_conversation(
#     conversation_id: UUID,
#     current_user: User = Depends(get_current_user),
#     db: AsyncSession = Depends(get_database_session)
# ) -> dict:
#     """
#     Delete a conversation and all its messages.
    
#     This is a soft delete operation. The conversation and messages
#     will be marked as deleted but retained for audit purposes.
    
#     Args:
#         conversation_id: UUID of the conversation to delete
#         current_user: Authenticated user
#         db: Database session
        
#     Returns:
#         Success confirmation
        
#     Raises:
#         HTTPException: If conversation not found or access denied
#     """
#     try:
#         chatbot_service = ChatbotService(db, current_user.id)
        
#         success = await chatbot_service.delete_conversation(conversation_id)
        
#         if not success:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="Conversation not found or access denied"
#             )
        
#         return {"message": "Conversation deleted successfully"}
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error deleting conversation: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error deleting conversation"
#         )


# @router.get("/stats", response_model=ChatbotStats)
# async def get_chatbot_stats(
#     current_user: User = Depends(get_current_user),
#     db: AsyncSession = Depends(get_database_session)
# ) -> ChatbotStats:
#     """
#     Get chatbot usage statistics for the current user.
    
#     Returns metrics like total conversations, messages, average response time,
#     and other usage analytics.
    
#     Args:
#         current_user: Authenticated user
#         db: Database session
        
#     Returns:
#         User's chatbot usage statistics
#     """
#     try:
#         chatbot_service = ChatbotService(db, current_user.id)
        
#         # Get basic stats from CRUD
#         from ...crud.crud_chatbot import ChatbotCRUD
#         crud = ChatbotCRUD(db)
#         stats = await crud.get_user_stats(current_user.id)
        
#         # Build response with default values for missing fields
#         return ChatbotStats(
#             total_conversations=stats.get("total_conversations", 0),
#             total_messages=stats.get("total_messages", 0),
#             average_response_time=stats.get("average_processing_time", 0.0),
#             average_confidence_score=stats.get("average_confidence_score", None),
#             most_used_tools=[],  # TODO: Implement tool usage tracking
#             today_conversations=0,  # TODO: Implement daily stats
#             today_messages=0  # TODO: Implement daily stats
#         )
        
#     except Exception as e:
#         logger.error(f"Error getting chatbot stats: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error retrieving chatbot statistics"
#         )


# @router.get("/health")
# async def chatbot_health_check() -> dict:
#     """
#     Health check endpoint for chatbot services.
    
#     Verifies that all chatbot components are functioning properly:
#     - LLM service connectivity
#     - Tool availability
#     - Agent initialization
#     - Enhanced services (Sprint 2)
    
#     Returns:
#         Health status information
#     """
#     try:
#         # Test enhanced service
#         enhanced_service = get_enhanced_chatbot_service()
#         performance_metrics = enhanced_service.get_performance_metrics()
        
#         # Test legacy agent initialization
#         from ...services.chatbot.agent import LegalChatbotAgentV1
        
#         test_agent = LegalChatbotAgentV1()
#         health_status = test_agent.health_check()
        
#         return {
#             "status": "healthy" if health_status["overall_status"] == "healthy" else "unhealthy",
#             "components": health_status,
#             "enhanced_service": {
#                 "status": "healthy" if performance_metrics.get("service_status", {}).get("llm_service") else "unhealthy",
#                 "performance": performance_metrics
#             },
#             "sprint_version": "2.0",
#             "timestamp": str(datetime.utcnow())
#         }
        
#     except Exception as e:
#         logger.error(f"Chatbot health check failed: {str(e)}")
#         return {
#             "status": "unhealthy",
#             "error": str(e),
#             "timestamp": str(datetime.utcnow())
#         }


# @router.get("/memory/entities")
# async def get_entity_insights(
#     current_user: User = Depends(get_current_user)
# ) -> dict:
#     """
#     Get insights about tracked entities in conversation memory.
    
#     Returns information about legal entities mentioned in conversations,
#     including people, organizations, cases, and legal principles.
    
#     Args:
#         current_user: Authenticated user
        
#     Returns:
#         Entity insights and statistics
#     """
#     try:
#         enhanced_service = get_enhanced_chatbot_service()
#         insights = enhanced_service.get_entity_insights()
        
#         return {
#             "status": "success",
#             "data": insights,
#             "user_id": current_user.id
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting entity insights: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error retrieving entity insights"
#         )


# @router.get("/memory/history/{conversation_id}")
# async def get_enhanced_conversation_history(
#     conversation_id: str,
#     limit: int = 20,
#     current_user: User = Depends(get_current_user)
# ) -> dict:
#     """
#     Get enhanced conversation history with memory metadata.
    
#     Returns conversation messages with entity and topic information
#     from the enhanced memory system.
    
#     Args:
#         conversation_id: Conversation identifier
#         limit: Maximum number of messages to return
#         current_user: Authenticated user
        
#     Returns:
#         Enhanced conversation history
#     """
#     try:
#         enhanced_service = get_enhanced_chatbot_service()
#         history = enhanced_service.get_conversation_history(conversation_id, limit)
        
#         return {
#             "status": "success",
#             "conversation_id": conversation_id,
#             "messages": history,
#             "total_messages": len(history)
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting enhanced conversation history: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error retrieving conversation history"
#         )


# @router.get("/performance")
# async def get_chatbot_performance_metrics(
#     current_user: User = Depends(get_current_user)
# ) -> dict:
#     """
#     Get detailed performance metrics for the enhanced chatbot.
    
#     Returns metrics including response times, memory usage,
#     tool usage statistics, and service health.
    
#     Args:
#         current_user: Authenticated user
        
#     Returns:
#         Performance metrics and statistics
#     """
#     try:
#         enhanced_service = get_enhanced_chatbot_service()
#         metrics = enhanced_service.get_performance_metrics()
        
#         return {
#             "status": "success",
#             "metrics": metrics,
#             "user_id": current_user.id,
#             "sprint_version": "2.0"
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting performance metrics: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error retrieving performance metrics"
#         )


# @router.post("/memory/clear")
# async def clear_conversation_memory(
#     conversation_id: Optional[str] = None,
#     current_user: User = Depends(get_current_user)
# ) -> dict:
#     """
#     Clear conversation memory for enhanced chatbot.
    
#     Clears entity tracking, topic clustering, and conversation history
#     from the enhanced memory system.
    
#     Args:
#         conversation_id: Specific conversation to clear (optional)
#         current_user: Authenticated user
        
#     Returns:
#         Clear operation result
#     """
#     try:
#         enhanced_service = get_enhanced_chatbot_service()
#         success = enhanced_service.clear_conversation(conversation_id)
        
#         return {
#             "status": "success" if success else "failed",
#             "message": f"Memory cleared{'for conversation ' + conversation_id if conversation_id else ' completely'}",
#             "conversation_id": conversation_id
#         }
        
#     except Exception as e:
#         logger.error(f"Error clearing conversation memory: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error clearing conversation memory"
#         )


# @router.post("/tools/case-comparison")
# async def compare_legal_cases(
#     cases: List[str],
#     current_user: User = Depends(get_current_user)
# ) -> dict:
#     """
#     Compare multiple legal cases using the advanced case comparator tool.
    
#     This endpoint provides direct access to the case comparison functionality
#     without requiring a full conversation context.
    
#     Args:
#         cases: List of case identifiers or descriptions to compare
#         current_user: Authenticated user
        
#     Returns:
#         Detailed case comparison analysis
#     """
#     try:
#         enhanced_service = get_enhanced_chatbot_service()
        
#         # Use the case comparator tool directly
#         case_comparator = enhanced_service.case_comparator
        
#         # Convert list to comma-separated string for tool input
#         cases_input = ", ".join(cases)
        
#         # Execute comparison
#         comparison_result = await case_comparator._arun(
#             case_identifiers=cases_input,
#             comparison_criteria="comprehensive",
#             include_similarities=True,
#             include_differences=True
#         )
        
#         return {
#             "status": "success",
#             "comparison_result": comparison_result,
#             "cases_compared": len(cases),
#             "user_id": current_user.id
#         }
        
#     except Exception as e:
#         logger.error(f"Error comparing legal cases: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error performing case comparison"
#         )


# @router.post("/tools/precedent-search")
# async def search_legal_precedents(
#     legal_issue: str,
#     similarity_threshold: float = 0.7,
#     max_results: int = 5,
#     current_user: User = Depends(get_current_user)
# ) -> dict:
#     """
#     Search for legal precedents using the advanced precedent explorer tool.
    
#     This endpoint provides direct access to precedent search functionality
#     for finding relevant case law and legal precedents.
    
#     Args:
#         legal_issue: Legal issue or question to find precedents for
#         similarity_threshold: Minimum similarity threshold (0.0-1.0)
#         max_results: Maximum number of precedents to return
#         current_user: Authenticated user
        
#     Returns:
#         Precedent search results with analysis
#     """
#     try:
#         enhanced_service = get_enhanced_chatbot_service()
        
#         # Use the precedent explorer tool directly
#         precedent_explorer = enhanced_service.precedent_explorer
        
#         # Execute precedent search
#         precedent_result = await precedent_explorer._arun(
#             legal_issue=legal_issue,
#             similarity_threshold=similarity_threshold,
#             max_results=max_results
#         )
        
#         return {
#             "status": "success",
#             "precedent_analysis": precedent_result,
#             "legal_issue": legal_issue,
#             "parameters": {
#                 "similarity_threshold": similarity_threshold,
#                 "max_results": max_results
#             },
#             "user_id": current_user.id
#         }
        
#     except Exception as e:
#         logger.error(f"Error searching legal precedents: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error performing precedent search"
#         )


# @router.post("/tools/citation-generation")
# async def generate_legal_citation(
#     case_reference: str,
#     citation_format: str = "indonesian_legal",
#     include_page_numbers: bool = False,
#     include_pinpoint: bool = False,
#     current_user: User = Depends(get_current_user)
# ) -> dict:
#     """
#     Generate properly formatted legal citations using the citation generator tool.
    
#     This endpoint provides direct access to citation generation functionality
#     for creating proper legal references in various formats.
    
#     Args:
#         case_reference: Case information or document reference
#         citation_format: Citation format (indonesian_legal, apa, bluebook, oscola, custom)
#         include_page_numbers: Include page numbers in citation
#         include_pinpoint: Include pinpoint citations
#         current_user: Authenticated user
        
#     Returns:
#         Generated citation with formatting details
#     """
#     try:
#         enhanced_service = get_enhanced_chatbot_service()
        
#         # Use the citation generator tool directly
#         citation_generator = enhanced_service.citation_generator
        
#         # Execute citation generation
#         citation_result = await citation_generator._arun(
#             case_reference=case_reference,
#             citation_format=citation_format,
#             include_page_numbers=include_page_numbers,
#             include_pinpoint=include_pinpoint
#         )
        
#         return {
#             "status": "success",
#             "citation": citation_result,
#             "case_reference": case_reference,
#             "parameters": {
#                 "citation_format": citation_format,
#                 "include_page_numbers": include_page_numbers,
#                 "include_pinpoint": include_pinpoint
#             },
#             "user_id": current_user.id
#         }
        
#     except Exception as e:
#         logger.error(f"Error generating legal citation: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error generating citation"
#         )


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


async def _store_conversation_async(
    chatbot_service: ChatbotService,
    request: ChatRequest,
    response: ChatResponse
) -> None:
    """
    Background task to store conversation in database.
    
    Args:
        chatbot_service: Chatbot service instance
        request: Chat request
        response: Chat response
    """
    try:
        # Store conversation using legacy service for database consistency
        await chatbot_service.store_conversation(
            message=request.message,
            response=response.response,
            conversation_id=response.conversation_id,
            processing_time=response.processing_time,
            tools_used=response.tools_used,
            metadata=response.metadata
        )
        
        logger.debug(f"Conversation {response.conversation_id} stored successfully")
        
    except Exception as e:
        logger.error(f"Error storing conversation: {str(e)}")


