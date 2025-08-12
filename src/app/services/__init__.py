from typing import Optional
from datetime import datetime, timedelta
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from .search_orchestrator import SearchOrchestrator
from ..schemas.legal_search import SearchRequest, SearchResponse
from ..core.config import settings

logger = logging.getLogger(__name__)


class SearchServiceFactory:
    """
    Factory for creating search service instances.
    Implements service layer abstraction and dependency injection.
    """
    
    _orchestrator_instance: Optional[SearchOrchestrator] = None
    
    @classmethod
    def get_search_orchestrator(cls) -> SearchOrchestrator:
        """
        Get singleton instance of SearchOrchestrator.
        Implements singleton pattern for service efficiency.
        """
        if cls._orchestrator_instance is None:
            cls._orchestrator_instance = SearchOrchestrator()
        return cls._orchestrator_instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._orchestrator_instance = None


class SearchService:
    """
    High-level service layer for legal document search.
    This service provides the main business logic interface
    and implements cross-cutting concerns like logging and monitoring.
    """
    
    def __init__(self):
        self.orchestrator = SearchServiceFactory.get_search_orchestrator()
    
    async def search_documents(
        self,
        request: SearchRequest,
        db: AsyncSession,
        user_id: Optional[int] = None
    ) -> SearchResponse:
        """
        Execute legal document search with full orchestration.
        
        Args:
            request: Search request parameters
            db: Database session
            user_id: Optional user ID for analytics
            
        Returns:
            Complete search response with documents and analysis
        """
        logger.info(f"Starting search for query: '{request.query[:100]}...' by user {user_id}")
        
        try:
            # Validate request
            self._validate_search_request(request)
            
            # Execute search through orchestrator
            response = await self.orchestrator.search(request, db, user_id)
            
            # Log search results
            logger.info(
                f"Search completed. Query ID: {response.query_id}, "
                f"Results: {response.total_count}, "
                f"Time: {response.processing_time_ms}ms"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Search service error: {e}")
            raise
    
    async def get_search_analytics(self, db: AsyncSession) -> dict:
        """Get search analytics and system health."""
        return await self.orchestrator.get_search_analytics(db)
    
    def _validate_search_request(self, request: SearchRequest) -> None:
        """Validate search request parameters."""
        if not request.query or len(request.query.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")
        
        if request.max_results > 100:
            raise ValueError("Maximum results cannot exceed 100")
        
        if request.filters and request.filters.date_from and request.filters.date_to:
            if request.filters.date_from > request.filters.date_to:
                raise ValueError("Date from cannot be after date to")
    
    def health_check(self) -> dict:
        """Check service health status."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "legal_search",
            "version": "1.0.0"
        }


# Global service instance
search_service = SearchService()
