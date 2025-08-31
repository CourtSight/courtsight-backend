"""
FastAPI routes for Supreme Court RAG search endpoints.
Implements presentation layer following clean architecture principles.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from typing import Optional, List
import asyncio
from datetime import datetime

from ...services.rag_service import RAGService, create_rag_service
from ...core.dependencies import get_current_user, get_rate_limiter
from ...core.exceptions import RAGServiceError, ValidationError
from ...schemas.search import (
    SearchRequest,
    SearchResponse, 
    SearchFilters,
    BulkDocumentRequest,
    BulkDocumentResponse,
    HealthResponse
)
from ...models.user import User

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.post("/", response_model=SearchResponse)
async def search_supreme_court_documents(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(create_rag_service),
    rate_limiter = Depends(get_rate_limiter)
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
        await rate_limiter.check_rate_limit(current_user.id, "search")
        
        # Execute search through service layer
        result = await rag_service.search_documents(
            request=request,
            user_id=current_user.id
        )
        
        # Add background analytics task
        background_tasks.add_task(
            _track_search_analytics,
            user_id=current_user.id,
            query=request.query,
            result_count=len(result.results),
            response_time=result.metrics.query_time
        )
        
        return result
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RAGServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
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


@router.get("/suggestions")
async def get_search_suggestions(
    query: str = Query(..., min_length=1, max_length=100),
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(create_rag_service)
) -> List[str]:
    """
    Get search query suggestions based on document content.
    
    Args:
        query: Partial query for suggestions
        current_user: Authenticated user
        rag_service: RAG service instance
        
    Returns:
        List of suggested query completions
    """
    try:
        # Implementation for query suggestions
        # Could use vector similarity search on indexed terms
        suggestions = [
            f"{query} hukum pidana",
            f"{query} mahkamah agung",
            f"{query} putusan kasasi",
            f"{query} yurisprudensi"
        ]
        
        return suggestions[:5]  # Limit to 5 suggestions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get suggestions")


@router.post("/documents/bulk", response_model=BulkDocumentResponse)
async def add_documents_bulk(
    request: BulkDocumentRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(create_rag_service)
) -> BulkDocumentResponse:
    """
    Add multiple documents to the search index in bulk.
    
    Implements PRD requirement for efficient document ingestion pipeline.
    Requires admin privileges.
    
    Args:
        request: Bulk document addition request
        background_tasks: For async processing
        current_user: Authenticated admin user
        rag_service: RAG service instance
        
    Returns:
        BulkDocumentResponse with processing status
        
    Raises:
        HTTPException: If user lacks permissions or processing fails
    """
    # Check admin permissions
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Start bulk processing in background for large batches
        if len(request.documents) > 50:
            background_tasks.add_task(
                _process_documents_background,
                documents=request.documents,
                user_id=current_user.id,
                rag_service=rag_service
            )
            
            return BulkDocumentResponse(
                status="processing",
                message="Large batch processing started in background",
                total_documents=len(request.documents),
                processed_count=0,
                timestamp=datetime.now()
            )
        else:
            # Process smaller batches synchronously
            from langchain_core.documents import Document
            
            lc_documents = [
                Document(
                    page_content=doc.content,
                    metadata=doc.metadata
                )
                for doc in request.documents
            ]
            
            result = await rag_service.add_documents_bulk(lc_documents)
            
            return BulkDocumentResponse(
                status="completed",
                message="Documents processed successfully",
                total_documents=result["total_documents"],
                processed_count=result["processed_successfully"],
                errors=result.get("errors", []),
                timestamp=datetime.now()
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk processing failed: {str(e)}")


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
        health_data = await rag_service.get_system_health()
        
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
