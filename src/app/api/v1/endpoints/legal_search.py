from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.db.session import get_async_session
from ...api.dependencies import get_current_user
from ...models.users import Users
from ...schemas.legal_search import SearchRequest, SearchResponse
from ...services import search_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search Supreme Court Legal Documents",
    description="""
    Execute AI-powered search across Supreme Court legal documents.
    
    This endpoint implements the complete CourtSight search workflow:
    - AI embedding generation for semantic search
    - Vector similarity search in document database  
    - LLM-powered analysis and summarization
    - Citation validation and verification
    
    Features:
    - Semantic search using multilingual embeddings
    - Advanced filtering by jurisdiction, case type, legal area
    - AI-generated summaries and key points extraction
    - Citation validation against source documents
    - Search analytics and caching
    
    Search phases:
    1. **Immediate Response**: Document retrieval and ranking
    2. **AI Analysis** (optional): Summary and legal themes
    3. **Validation** (optional): Citation accuracy verification
    """,
    responses={
        200: {
            "description": "Search completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "documents": [
                            {
                                "id": 1,
                                "case_number": "123/PK/2023",
                                "court_name": "Mahkamah Agung RI",
                                "title": "Putusan tentang Hak Asasi Manusia",
                                "relevance_score": 0.92
                            }
                        ],
                        "total_count": 1,
                        "query_id": "550e8400-e29b-41d4-a716-446655440000",
                        "processing_time_ms": 1250,
                        "ai_summary": {
                            "summary": "Kasus ini membahas interpretasi HAM...",
                            "key_points": ["Hak fundamental", "Interpretasi konstitusi"]
                        }
                    }
                }
            }
        },
        400: {"description": "Invalid search parameters"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)
async def search_legal_documents(
    search_request: SearchRequest,
    current_user: Users = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> SearchResponse:
    """
    Search legal documents using AI-powered semantic search.
    Maps to CourtSight Feature F1.1-F1.13 sequence diagram.
    """
    try:
        # Execute search through service layer
        response = await search_service.search_documents(
            request=search_request,
            db=db,
            user_id=current_user.id
        )
        
        # Add background task for analytics (optional)
        background_tasks.add_task(
            _log_search_metrics,
            query=search_request.query,
            user_id=current_user.id,
            results_count=response.total_count,
            processing_time=response.processing_time_ms
        )
        
        return response
        
    except ValueError as e:
        logger.warning(f"Invalid search request from user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid search parameters: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Search failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search service temporarily unavailable"
        )


@router.get(
    "/analytics",
    summary="Get Search Analytics",
    description="""
    Retrieve search analytics and system statistics.
    
    Provides insights into:
    - Popular search queries and trends
    - Document collection statistics
    - System performance metrics
    - Service health status
    
    Useful for administrators and system monitoring.
    """,
    responses={
        200: {
            "description": "Analytics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "popular_queries": [
                            {"query": "hak asasi manusia", "count": 45},
                            {"query": "perdata", "count": 32}
                        ],
                        "document_statistics": {
                            "total_documents": 15420,
                            "by_jurisdiction": {"Supreme Court": 8500, "High Court": 6920}
                        },
                        "service_status": "operational"
                    }
                }
            }
        }
    }
)
async def get_search_analytics(
    current_user: Users = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
) -> Any:
    """Get search analytics and statistics."""
    try:
        analytics = await search_service.get_search_analytics(db)
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analytics service temporarily unavailable"
        )


@router.get(
    "/health",
    summary="Service Health Check",
    description="Check the health status of the legal search service.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "service": "legal_search",
                        "version": "1.0.0"
                    }
                }
            }
        }
    }
)
async def health_check() -> Any:
    """Check service health status."""
    return search_service.health_check()


@router.get(
    "/search/suggestions",
    summary="Get Search Suggestions",
    description="""
    Get search query suggestions based on popular searches and legal terminology.
    
    Helps users discover relevant search terms and legal concepts.
    """,
    responses={
        200: {
            "description": "Suggestions retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "suggestions": [
                            "hak asasi manusia",
                            "perdata kontrak",
                            "pidana korupsi",
                            "tata usaha negara"
                        ]
                    }
                }
            }
        }
    }
)
async def get_search_suggestions(
    q: Optional[str] = Query(None, description="Partial query for autocomplete"),
    limit: int = Query(10, ge=1, le=20, description="Maximum number of suggestions"),
    current_user: Users = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
) -> Any:
    """Get search suggestions for autocomplete."""
    try:
        # This would typically query a suggestions index or popular searches
        # For now, return static suggestions based on common legal terms
        legal_suggestions = [
            "hak asasi manusia",
            "perdata kontrak",
            "pidana korupsi",
            "tata usaha negara",
            "hukum administrasi",
            "peradilan agama",
            "hukum perburuhan",
            "hak milik intelektual",
            "lingkungan hidup",
            "perlindungan konsumen"
        ]
        
        if q:
            # Filter suggestions based on query
            filtered = [s for s in legal_suggestions if q.lower() in s.lower()]
            return {"suggestions": filtered[:limit]}
        
        return {"suggestions": legal_suggestions[:limit]}
        
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        return {"suggestions": []}


# Background task functions
async def _log_search_metrics(
    query: str,
    user_id: int,
    results_count: int,
    processing_time: int
) -> None:
    """Log search metrics for analytics (background task)."""
    try:
        # This would typically send metrics to analytics service
        logger.info(
            f"Search metrics: user_id={user_id}, query_length={len(query)}, "
            f"results={results_count}, time={processing_time}ms"
        )
    except Exception as e:
        logger.error(f"Failed to log search metrics: {e}")


# Export router
legal_search_router = router
