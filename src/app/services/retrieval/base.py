"""
Base retrieval interfaces and strategy pattern for Court RAG system.
Provides a flexible architecture for different retrieval methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    VECTOR_SEARCH = "vector_search"
    PARENT_CHILD = "parent_child"
    HYBRID = "hybrid"
    SEMANTIC_SEARCH = "semantic_search"
    KEYWORD_SEARCH = "keyword_search"

class RetrievalRequest(BaseModel):
    """Unified request model for all retrieval strategies."""
    query: str = Field(..., description="Search query")
    strategy: RetrievalStrategy = Field(RetrievalStrategy.HYBRID, description="Retrieval strategy to use")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional search filters")
    collection_name: Optional[str] = Field(None, description="Vector collection name")
    include_scores: bool = Field(True, description="Include similarity scores")
    include_metadata: bool = Field(True, description="Include document metadata")
    context_window: int = Field(2000, description="Context window size")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "putusan narkotika",
                "strategy": "hybrid",
                "top_k": 5,
                "filters": {
                    "court": "PN Denpasar",
                    "year": "2025"
                },
                "collection_name": "ma_putusan_pc_chunks",
                "include_scores": True,
                "include_metadata": True,
                "context_window": 2000
            }
        }

class RetrievedDocument(BaseModel):
    """Retrieved document with metadata and scores."""
    id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    score: Optional[float] = Field(None, description="Similarity score")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")
    parent_id: Optional[str] = Field(None, description="Parent document ID")
    source: str = Field(..., description="Source information")
    
class RetrievalResponse(BaseModel):
    """Unified response model for all retrieval strategies."""
    query: str = Field(..., description="Original query")
    strategy_used: RetrievalStrategy = Field(..., description="Strategy that was used")
    documents: List[RetrievedDocument] = Field(..., description="Retrieved documents")
    total_found: int = Field(..., description="Total documents found")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")

class BaseRetriever(ABC):
    """Abstract base class for all retrieval strategies."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
    
    @abstractmethod
    def retrieve(self, request: RetrievalRequest) -> List[Document]:
        """Retrieve documents based on the request."""
        pass
    
    @abstractmethod
    def supports_filters(self) -> bool:
        """Check if this retriever supports filtering."""
        pass
    
    def get_strategy(self) -> RetrievalStrategy:
        """Get the strategy this retriever implements."""
        return RetrievalStrategy.VECTOR_SEARCH  # Default
