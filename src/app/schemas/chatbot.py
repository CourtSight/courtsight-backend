from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from ..core.schemas import TimestampSchema, UUIDSchema


class ReasoningStep(BaseModel):
    """Schema for individual reasoning steps in the agent's thought process."""
    step_number: int = Field(..., description="Sequential step number")
    action: str = Field(..., description="Action taken in this step")
    tool_used: Optional[str] = Field(None, description="Tool used in this step")
    tool_input: Optional[Dict[str, Any]] = Field(None, description="Input parameters for the tool")
    result: str = Field(..., description="Result of this reasoning step")
    timestamp: datetime = Field(..., description="When this step was executed")


class ToolCall(BaseModel):
    """Schema for tool call information."""
    tool_name: str = Field(..., description="Name of the tool that was called")
    input_data: Dict[str, Any] = Field(..., description="Input parameters passed to the tool")
    output_data: str = Field(..., description="Output returned by the tool")
    execution_time: float = Field(..., description="Time taken to execute the tool in seconds")
    success: bool = Field(True, description="Whether the tool call was successful")
    error_message: Optional[str] = Field(None, description="Error message if tool call failed")


class Citation(BaseModel):
    """Schema for legal document citations."""
    source_type: str = Field(..., description="Type of source (putusan_ma, uu, peraturan)")
    source_id: str = Field(..., description="Unique identifier of the source document")
    title: str = Field(..., description="Title of the source document")
    url: Optional[str] = Field(None, description="URL to access the source document")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score between 0 and 1")
    excerpt: Optional[str] = Field(None, description="Relevant excerpt from the source")


class ChatRequest(BaseModel):
    """Schema for incoming chat requests."""
    model_config = ConfigDict(extra="forbid")
    
    message: Annotated[str, Field(
        min_length=5, 
        max_length=2000, 
        description="User's question or message",
        examples=["Apa dasar hukum putusan tentang sengketa tanah?"]
    )]
    conversation_id: Optional[UUID] = Field(
        None, 
        description="ID of existing conversation, if continuing a chat"
    )
    include_reasoning: bool = Field(
        True, 
        description="Whether to include detailed reasoning steps in response"
    )
    max_tokens: Optional[int] = Field(
        None, 
        ge=100, 
        le=4000, 
        description="Maximum tokens for the response"
    )


class ChatResponse(BaseModel):
    """Schema for chat responses from the assistant."""
    answer: str = Field(..., description="The assistant's response to the user's question")
    conversation_id: UUID = Field(..., description="ID of the conversation")
    message_id: UUID = Field(..., description="ID of this specific message")
    
    # Agent execution details
    reasoning_steps: List[ReasoningStep] = Field(
        default_factory=list, 
        description="Detailed reasoning steps taken by the agent"
    )
    tool_calls: List[ToolCall] = Field(
        default_factory=list, 
        description="Tools that were called during processing"
    )
    citations: List[Citation] = Field(
        default_factory=list, 
        description="Citations for sources used in the response"
    )
    
    # Performance metrics
    processing_time: float = Field(..., description="Time taken to process the request in seconds")
    confidence_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score of the response"
    )
    
    # Metadata
    timestamp: datetime = Field(..., description="When the response was generated")
    model_used: Optional[str] = Field(None, description="LLM model used for generation")


class ConversationSummary(BaseModel):
    """Schema for conversation summaries in list views."""
    id: UUID = Field(..., description="Conversation ID")
    title: Optional[str] = Field(None, description="Conversation title")
    last_message: str = Field(..., description="Preview of the last message")
    message_count: int = Field(..., description="Total number of messages in conversation")
    created_at: datetime = Field(..., description="When the conversation was started")
    updated_at: datetime = Field(..., description="When the conversation was last updated")


class MessageBase(BaseModel):
    """Base schema for messages."""
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")


class MessageRead(MessageBase, UUIDSchema, TimestampSchema):
    """Schema for reading messages with full details."""
    conversation_id: UUID = Field(..., description="ID of the conversation this message belongs to")
    
    # Agent-specific fields (only for assistant messages)
    reasoning_steps: Optional[Dict[str, Any]] = Field(None, description="Agent reasoning steps")
    tool_calls: Optional[Dict[str, Any]] = Field(None, description="Tool calls made")
    citations: Optional[Dict[str, Any]] = Field(None, description="Citations used")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    confidence_score: Optional[float] = Field(None, description="Confidence score")


class ConversationCreate(BaseModel):
    """Schema for creating new conversations."""
    model_config = ConfigDict(extra="forbid")
    
    title: Optional[str] = Field(
        None, 
        max_length=255, 
        description="Optional title for the conversation"
    )
    initial_message: Optional[str] = Field(
        None, 
        min_length=5, 
        max_length=2000, 
        description="Optional initial message to start the conversation"
    )


class ConversationRead(UUIDSchema, TimestampSchema):
    """Schema for reading full conversation details."""
    user_id: int = Field(..., description="ID of the user who owns this conversation")
    title: Optional[str] = Field(None, description="Conversation title")
    messages: List[MessageRead] = Field(
        default_factory=list, 
        description="All messages in the conversation"
    )


class ConversationUpdate(BaseModel):
    """Schema for updating conversation details."""
    model_config = ConfigDict(extra="forbid")
    
    title: Optional[str] = Field(
        None, 
        max_length=255, 
        description="New title for the conversation"
    )


class ChatbotStats(BaseModel):
    """Schema for chatbot usage statistics."""
    total_conversations: int = Field(..., description="Total number of conversations")
    total_messages: int = Field(..., description="Total number of messages")
    average_response_time: float = Field(..., description="Average response time in seconds")
    average_confidence_score: Optional[float] = Field(None, description="Average confidence score")
    most_used_tools: List[str] = Field(default_factory=list, description="Most frequently used tools")
    today_conversations: int = Field(..., description="Conversations started today")
    today_messages: int = Field(..., description="Messages sent today")