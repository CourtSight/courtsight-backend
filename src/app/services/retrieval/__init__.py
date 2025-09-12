"""
Retrieval services package.
Provides unified interface for document retrieval with multiple strategies.
"""

from .base import BaseRetriever, RetrievalRequest, RetrievalStrategy
from .vector_search import VectorSearchRetriever
from .parent_child import ParentChildRetriever
from .hybrid import HybridRetriever
from .service import (
    RetrievalService, 
    create_retrieval_service, 
    get_retrieval_service,
    reset_retrieval_service,
    get_retrieval_service_status,
    search_with_retrieval_service,
    multi_strategy_search
)

__all__ = [
    # Base interfaces
    "BaseRetriever",
    "RetrievalRequest", 
    "RetrievalStrategy",
    
    # Retriever implementations
    "VectorSearchRetriever",
    "ParentChildRetriever", 
    "HybridRetriever",
    
    # Main service
    "RetrievalService",
    "create_retrieval_service",
    "get_retrieval_service",
    "reset_retrieval_service",
    "get_retrieval_service_status",
    
    # Utility functions
    "search_with_retrieval_service",
    "multi_strategy_search"
]
