"""
FastAPI routes for multimodal retrieval API endpoints.
Provides retrieval-as-a-service for RAG, REACT agents, and multimodal models.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from ...services.rag_service import RAGService, create_rag_service
from ...core.dependencies import get_current_user
from ...schemas.search import SearchRequest, SearchFilters, DateRange
from ...models.user import User

router = APIRouter(prefix="/api/v1/retrieval", tags=["retrieval"])


class RetrievalRequest(BaseModel):
    """Request model for multimodal retrieval API."""
    query: str = Field(..., description="The search query")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters")
    include_metadata: bool = Field(True, description="Include document metadata")
    include_scores: bool = Field(True, description="Include similarity scores")
    context_window: int = Field(1000, description="Context window size in characters")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "putusan mahkamah agung tentang korupsi",
                "top_k": 5,
                "filters": {
                    "jurisdiction": "ID",
                    "case_type": "criminal"
                },
                "include_metadata": True,
                "include_scores": True,
                "context_window": 1000
            }
        }


class RetrievedDocument(BaseModel):
    """Retrieved document model for API response."""
    id: str = Field(..., description="Document unique identifier")
    content: str = Field(..., description="Document content/chunk")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    score: Optional[float] = Field(None, description="Similarity score")
    source: str = Field(..., description="Source document information")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_123",
                "content": "Dalam putusan Mahkamah Agung No. 123/Pid/2023...",
                "metadata": {
                    "jurisdiction": "ID",
                    "case_type": "criminal",
                    "court_level": "supreme"
                },
                "score": 0.85,
                "source": "putusan_123_pid_2023.pdf"
            }
        }


class RetrievalResponse(BaseModel):
    """Response model for multimodal retrieval API."""
    query: str = Field(..., description="Original query")
    documents: List[RetrievedDocument] = Field(..., description="Retrieved documents")
    total_found: int = Field(..., description="Total documents found")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "putusan mahkamah agung tentang korupsi",
                "documents": [
                    {
                        "id": "doc_123",
                        "content": "Dalam putusan Mahkamah Agung No. 123/Pid/2023...",
                        "metadata": {"jurisdiction": "ID", "case_type": "criminal"},
                        "score": 0.85,
                        "source": "putusan_123_pid_2023.pdf"
                    }
                ],
                "total_found": 1,
                "processing_time": 0.45,
                "timestamp": "2025-09-01T10:00:00Z"
            }
        }


@router.post("/", response_model=RetrievalResponse)
async def retrieve_documents(
    request: RetrievalRequest,
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(create_rag_service)
) -> RetrievalResponse:
    """
    Multimodal retrieval API endpoint.

    Retrieves relevant documents for RAG, REACT agents, and multimodal models.
    Optimized for low-latency, high-throughput retrieval operations.

    Args:
        request: Retrieval request with query and parameters
        current_user: Authenticated user
        rag_service: RAG service instance

    Returns:
        RetrievalResponse with relevant documents and metadata
    """
    try:
        import time
        start_time = time.time()

        # Convert retrieval request to search request
        search_request = SearchRequest(
            query=request.query,
            max_results=request.top_k,
            filters=SearchFilters(**request.filters) if request.filters else None,
            include_summary=False,
            include_validation=False
        )

        # Execute search
        search_response = await rag_service.search_documents(
            request=search_request,
            user_id=current_user.id
        )

        # Convert search results to retrieval format
        documents = []
        for result in search_response.results:
            # Extract content from source documents
            for source_doc in result.source_documents:
                doc = RetrievedDocument(
                    id=f"{source_doc.get('id', 'unknown')}",
                    content=source_doc.get('content', '')[:request.context_window],
                    metadata={
                        "title": source_doc.get('title', ''),
                        "source": source_doc.get('source', ''),
                        "page": source_doc.get('page', ''),
                        "chunk_id": source_doc.get('chunk_id', '')
                    },
                    score=source_doc.get('score'),
                    source=source_doc.get('source', '')
                )
                documents.append(doc)

        processing_time = time.time() - start_time

        return RetrievalResponse(
            query=request.query,
            documents=documents[:request.top_k],  # Limit to requested top_k
            total_found=len(documents),
            processing_time=round(processing_time, 3)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}"
        )


@router.get("/health")
async def retrieval_health_check():
    """Health check endpoint for retrieval service."""
    return {
        "status": "healthy",
        "service": "multimodal-retrieval-api",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@router.post("/batch", response_model=List[RetrievalResponse])
async def batch_retrieve_documents(
    requests: List[RetrievalRequest],
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(create_rag_service)
) -> List[RetrievalResponse]:
    """
    Batch retrieval endpoint for multiple queries.

    Processes multiple retrieval requests in parallel for efficiency.
    Useful for multimodal agents that need to retrieve context for multiple queries.

    Args:
        requests: List of retrieval requests
        current_user: Authenticated user
        rag_service: RAG service instance

    Returns:
        List of retrieval responses
    """
    try:
        # Process requests in parallel
        tasks = [
            retrieve_documents(req, current_user, rag_service)
            for req in requests
        ]

        responses = await asyncio.gather(*tasks)
        return responses

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch retrieval failed: {str(e)}"
        )
