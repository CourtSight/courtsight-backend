"""
FastAPI routes for Supreme Court RAG search endpoints.
Implements presentation layer following clean architecture principles.
"""

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...core.dependencies import get_current_user
from ...core.exceptions import RAGServiceError, ValidationError
from ...models.user import User
from ...schemas.search import HealthResponse, SearchRequest, SearchResponse
from ...services.rag_service import RAGService, create_rag_service

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.post("/", response_model=SearchResponse)
async def search_supreme_court_documents(
    request: SearchRequest,
    # current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(create_rag_service),
    # rate_limiter = Depends(get_rate_limiter)
) -> SearchResponse:
    """
    Search Supreme Court documents using AI-powered semantic search.

    Implements PRD requirements:
    - Natural language query processing
    - Semantic similarity search
    - Response generation with citations
    - Claim validation and filtering
    - 3-second initial response target

    Args:
        request: Search request with query and optional filters
        background_tasks: FastAPI background tasks for async operations
        current_user: Authenticated user (from dependency)
        rag_service: RAG service instance (from dependency)
        rate_limiter: Rate limiting handler

    Returns:
        SearchResponse with validated results and citations

    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Rate limiting check
        # await rate_limiter.check_rate_limit(current_user.id, "search")

        # Execute search through service layer
        result = await rag_service.search_documents(
            request=request,
            user_id=1
        )

        # # Add background analytics task
        # background_tasks.add_task(
        #     _track_search_analytics,
        #     user_id=1,
        #     query=request.query,
        #     result_count=len(result.results),
        #     response_time=result.metrics.query_time
        # )

        return result

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RAGServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/stream")
async def search_with_streaming(
    request: SearchRequest,
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(create_rag_service)
) -> StreamingResponse:
    """
    Stream search results for real-time user experience.

    Implements progressive result delivery to meet PRD performance requirements:
    - Initial results in 3 seconds
    - Comprehensive results within 10 seconds

    Args:
        request: Search request
        current_user: Authenticated user
        rag_service: RAG service instance

    Returns:
        StreamingResponse with progressive results
    """
    async def generate_streaming_response():
        """Generate streaming response with progressive results."""
        try:
            # Send initial acknowledgment
            yield f"data: {{'status': 'searching', 'query': '{request.query}'}}\n\n"

            # Start search process
            result = await rag_service.search_documents(
                request=request,
                user_id=current_user.id
            )

            # Send final results
            import json
            yield f"data: {json.dumps(result.dict())}\n\n"
            yield "data: {'status': 'complete'}\n\n"

        except Exception as e:
            yield f"data: {{'status': 'error', 'message': '{str(e)}'}}\n\n"

    return StreamingResponse(
        generate_streaming_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )



@router.get("/health", response_model=HealthResponse)
async def get_search_health(
    rag_service: RAGService = Depends(create_rag_service)
) -> HealthResponse:
    """
    Get RAG system health status.

    Implements PRD monitoring requirements for 99.5% uptime.

    Returns:
        HealthResponse with system status and metrics
    """
    try:
        health_data =  rag_service.get_system_health()

        return HealthResponse(
            status=health_data["status"],
            timestamp=health_data["timestamp"],
            response_time=health_data.get("response_time", 0),
            services={
                "vector_store": "healthy" if health_data.get("vector_store_connected") else "unhealthy",
                "llm_service": "healthy" if health_data.get("llm_service_available") else "unhealthy",
                "validation_service": "healthy"  # Add actual check
            },
            error=health_data.get("error")
        )

    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            error=str(e),
            services={
                "vector_store": "unknown",
                "llm_service": "unknown",
                "validation_service": "unknown"
            }
        )


# Background task functions
async def _track_search_analytics(
    user_id: str,
    query: str,
    result_count: int,
    response_time: float
) -> None:
    """Track search analytics for business intelligence."""
    # Implementation for analytics tracking
    pass


async def _process_documents_background(
    documents: List[dict],
    user_id: str,
    rag_service: RAGService
) -> None:
    """Process documents in background for large batches."""
    try:
        from langchain_core.documents import Document

        lc_documents = [
            Document(
                page_content=doc["content"],
                metadata=doc.get("metadata", {})
            )
            for doc in documents
        ]

        result = await rag_service.add_documents_bulk(lc_documents)

        # Notify user of completion (could use websockets, email, etc.)
        await _notify_processing_complete(user_id, result)

    except Exception as e:
        # Log error and notify user
        await _notify_processing_error(user_id, str(e))


async def _notify_processing_complete(user_id: str, result: dict) -> None:
    """Notify user of background processing completion."""
    # Implementation for user notification
    pass


async def _notify_processing_error(user_id: str, error: str) -> None:
    """Notify user of background processing error."""
    # Implementation for error notification
    pass
