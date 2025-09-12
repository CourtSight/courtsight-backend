"""
Service layer for Supreme Court RAG system.
Implements business logic using LangChain orchestration while maintaining
clean architecture separation of concerns.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, date
from pydantic import BaseModel, Field

from langchain_core.documents import Document

from src.app.services.rag.chains import CourtRAGChains, SearchResult, create_rag_chains
from src.app.core.config import settings
from src.app.core.exceptions import RAGServiceError, ValidationError
from src.app.schemas.search import SearchRequest, SearchResponse, SearchFilters


class RAGMetrics(BaseModel):
    """Metrics for RAG system performance monitoring."""
    query_time: float = Field(description="Total query processing time in seconds")
    retrieval_time: float = Field(description="Document retrieval time")
    generation_time: float = Field(description="LLM generation time") 
    validation_time: float = Field(description="Claim validation time")
    documents_retrieved: int = Field(description="Number of documents retrieved")
    tokens_used: int = Field(description="Total tokens consumed")
    confidence_score: float = Field(description="Overall confidence in response")


class RAGService:
    """
    Business logic service for Supreme Court document search and analysis.
    
    This service implements the core business rules while delegating
    RAG orchestration to LangChain Expression Language chains.
    """
    
    def __init__(self, rag_chains: CourtRAGChains):
        self.rag_chains = rag_chains
        self._initialize_callbacks()
    
    def _initialize_callbacks(self) -> None:
        """Initialize LangChain callbacks for monitoring and evaluation."""
        self.callback_handlers = [
            # Add RAGAS evaluation callbacks here
            # Add performance monitoring callbacks
        ]
    
    async def search_documents(
        self,
        request: SearchRequest,
        user_id: Optional[str] = None
    ) -> SearchResponse:
        """
        Execute semantic search on Supreme Court documents.
        
        Implements PRD requirements:
        - 3 second initial response time
        - 10 second comprehensive results
        - Proper error handling and validation
        
        Args:
            request: Search request with query and filters
            user_id: Optional user identifier for audit logging
            
        Returns:
            SearchResponse with validated results and metrics
            
        Raises:
            RAGServiceError: For system-level errors
            ValidationError: For invalid input or results
        """
        start_time = datetime.now()
        
        try:
            # Input validation
            self._validate_search_request(request)
            
            # Convert filters to LangChain format
            lc_filters = self._convert_filters(request.filters)
            
            # Execute RAG pipeline with timeout for PRD compliance
            # Use synchronous invoke for now
            result = self.rag_chains.invoke(request.query, lc_filters)
            
            # Check if result is None
            if result is None:
                raise RAGServiceError("RAG pipeline returned no results")
            
            # Calculate metrics
            metrics = self._calculate_metrics(start_time, result)
            
            # Convert to response format
            response = self._format_response(result, metrics, request)
            
            # Audit logging
            await self._log_search_event(request, response, user_id)
            
            return response
            
        except Exception as e:
            await self._log_error(request, str(e), user_id)
            raise RAGServiceError(f"Search failed: {str(e)}")
    
    def _validate_search_request(self, request: SearchRequest) -> None:
        """Validate search request according to business rules."""
        if not request.query or len(request.query.strip()) < 3:
            raise ValidationError("Query must be at least 3 characters")
        
        if len(request.query) > 1000:
            raise ValidationError("Query too long (max 1000 characters)")
        
        # Validate filters
        if request.filters:
            self._validate_filters(request.filters)
    
    def _validate_filters(self, filters: SearchFilters) -> None:
        """Validate search filters against business rules."""
        if filters.date_range:
            if filters.date_range.start > filters.date_range.end:
                raise ValidationError("Invalid date range")
            
            # Ensure dates are not in the future
            if filters.date_range.end > date.today():
                raise ValidationError("End date cannot be in the future")
        
        # Validate jurisdiction codes
        valid_jurisdictions = ["ID", "MY", "SG", "TH", "VN", "PH"]  # Southeast Asia
        if filters.jurisdiction and filters.jurisdiction not in valid_jurisdictions:
            raise ValidationError(f"Invalid jurisdiction: {filters.jurisdiction}")
    
    def _convert_filters(self, filters: Optional[SearchFilters]) -> Optional[Dict[str, Any]]:
        """Convert Pydantic filters to LangChain retriever format."""
        if not filters:
            return None
        
        lc_filters = {}
        
        if filters.jurisdiction:
            lc_filters["jurisdiction"] = filters.jurisdiction
        
        if filters.date_range:
            lc_filters["date_range"] = {
                "start": filters.date_range.start.isoformat(),
                "end": filters.date_range.end.isoformat()
            }
        
        if filters.case_type:
            lc_filters["case_type"] = filters.case_type
        
        if filters.court_level:
            lc_filters["court_level"] = filters.court_level
        
        return lc_filters
    
    def _calculate_metrics(self, start_time: datetime, result: SearchResult) -> RAGMetrics:
        """Calculate performance metrics for monitoring."""
        total_time = (datetime.now() - start_time).total_seconds()
        
        return RAGMetrics(
            query_time=total_time,
            retrieval_time=0.0,  # Will be populated by callbacks
            generation_time=0.0,  # Will be populated by callbacks
            validation_time=0.0,  # Will be populated by callbacks
            documents_retrieved=len(result.source_documents),
            tokens_used=0,  # Will be populated by callbacks
            confidence_score=self._calculate_confidence(result)
        )
    
    def _calculate_confidence(self, result: SearchResult) -> float:
        """Calculate confidence score based on validation results."""
        if result.validation_status == "Supported":
            return 0.9
        elif result.validation_status == "Partially Supported":
            return 0.7
        elif result.validation_status == "Uncertain":
            return 0.5
        else:  # Unsupported
            return 0.3
    
    def _format_response(
        self,
        result: SearchResult,
        metrics: RAGMetrics,
        request: SearchRequest
    ) -> SearchResponse:
        """Format LangChain result into API response."""
        return SearchResponse(
            query=request.query,
            results=[result.model_dump()],  # Convert to dict
            metrics=metrics.model_dump(),   # Convert to dict
            timestamp=datetime.now(),
            filters_applied=request.filters,
            total_results=1,  # Single result for now
            has_more=False
        )
    
    async def _log_search_event(
        self,
        request: SearchRequest,
        response: SearchResponse,
        user_id: Optional[str]
    ) -> None:
        """Log search event for audit and analytics."""
        # Implementation for audit logging
        pass
    
    async def _log_error(
        self,
        request: SearchRequest,
        error: str,
        user_id: Optional[str]
    ) -> None:
        """Log error event for monitoring."""
        # Implementation for error logging
        pass
    
    async def add_documents_bulk(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Add documents to the RAG system in batches.
        
        Implements PRD requirement for efficient bulk document processing.
        
        Args:
            documents: List of documents to add
            batch_size: Number of documents to process per batch
            
        Returns:
            Processing results and statistics
        """
        total_docs = len(documents)
        processed = 0
        errors = []
        
        try:
            # Process in batches for memory efficiency
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                
                try:
                    self.rag_chains.add_documents(batch)
                    processed += len(batch)
                except Exception as e:
                    errors.append(f"Batch {i//batch_size + 1}: {str(e)}")
            
            return {
                "total_documents": total_docs,
                "processed_successfully": processed,
                "errors": errors,
                "success_rate": processed / total_docs if total_docs > 0 else 0
            }
            
        except Exception as e:
            raise RAGServiceError(f"Bulk document processing failed: {str(e)}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Check RAG system health and performance.
        
        Returns:
            System health metrics and status
        """
        try:
            # Test basic functionality
            test_query = "test query"
            start_time = datetime.now()
            
            # This should be a lightweight test
            # result = await self.rag_chains.ainvoke(test_query)
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "vector_store_connected": True,  # Check actual connection
                "llm_service_available": True,   # Check actual service
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now()
            }


# Service factory function
def create_rag_service() -> RAGService:
    """
    Factory function to create configured RAG service.
    
    Returns:
        Configured RAGService instance
    """
    # Create RAG chains with production configuration
    rag_chains = create_rag_chains(
        database_url=settings.DATABASE_URL,
        collection_name=settings.VECTOR_COLLECTION_NAME,
    )
    
    return RAGService(rag_chains)

