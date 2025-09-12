"""
Vector search retriever implementation.
Uses direct vector store similarity search.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_postgres import PGVector

from .base import BaseRetriever, RetrievalRequest, RetrievalStrategy

logger = logging.getLogger(__name__)

class VectorSearchRetriever(BaseRetriever):
    """Direct vector store similarity search retriever."""
    
    def __init__(self, vector_store: PGVector, **kwargs):
        super().__init__("vector_search", **kwargs)
        self.vector_store = vector_store
    
    def retrieve(self, request: RetrievalRequest) -> List[Document]:
        """Retrieve documents using vector similarity search."""
        try:
            # Prepare search kwargs
            search_kwargs = {"k": request.top_k}
            
            # Add filters if supported
            if request.filters and self.supports_filters():
                search_kwargs["filter"] = self._prepare_filters(request.filters)
            
            # Execute similarity search
            if request.include_scores:
                # Get documents with scores (not async)
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    request.query, 
                    **search_kwargs
                )
                
                
                # Add scores to document metadata
                documents = []
                for doc, score in docs_with_scores:
                    doc.metadata["similarity_score"] = float(score)
                    doc.metadata["retrieval_strategy"] = "vector_search"
                    documents.append(doc)
                
                return documents
            else:
                # Get documents without scores (not async)
                documents = self.vector_store.similarity_search(
                    request.query,
                    **search_kwargs
                )
                
                
                # Add metadata
                for doc in documents:
                    doc.metadata["retrieval_strategy"] = "vector_search"
                
                return documents
                
        except Exception as e:
            logger.error(f"Vector search retrieval failed: {e}")
            return []
    
    def supports_filters(self) -> bool:
        """Vector store supports filtering."""
        return True
    
    def get_strategy(self) -> RetrievalStrategy:
        """Get the strategy this retriever implements."""
        return RetrievalStrategy.VECTOR_SEARCH
    
    def _prepare_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare filters for vector store."""
        # Convert filters to vector store format
        prepared_filters = {}
        
        for key, value in filters.items():
            if key == "court":
                prepared_filters["court"] = value
            elif key == "year":
                prepared_filters["year"] = str(value)
            elif key == "case_type":
                prepared_filters["crime_type"] = value
            elif key == "case_number":
                prepared_filters["case_number"] = value
            else:
                # Pass through other filters
                prepared_filters[key] = value
        
        return prepared_filters
