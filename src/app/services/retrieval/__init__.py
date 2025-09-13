"""
Retrieval services package.
Provides unified interface for document retrieval with multiple strategies.
"""

from .base import BaseRetriever, RetrievalRequest, RetrievalStrategy
from .hybrid import HybridRetriever
from .parent_child import ParentChildRetriever
from .service import (
    RetrievalService,
    create_retrieval_service,
    get_retrieval_service,
    get_retrieval_service_status,
    multi_strategy_search,
    reset_retrieval_service,
    search_with_retrieval_service,
)
from .vector_search import VectorSearchRetriever

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
