"""
Enhanced retrieval API routes using the new driven architecture service.
Provides access to multiple retrieval strategies with unified interface.
"""

import logging
from enum import Enum
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...services.retrieval import RetrievalStrategy, get_retrieval_service
from ...services.retrieval.service import (
    multi_strategy_search,
    search_with_retrieval_service,
)

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1/retrieval", tags=["retrieval"])

# Request/Response Models
class StrategyEnum(str, Enum):
    """Available retrieval strategies."""
    VECTOR_SEARCH = "vector_search"
    PARENT_CHILD = "parent_child"
    HYBRID = "hybrid"


class EnhancedRetrievalRequest(BaseModel):
    """Enhanced retrieval request model."""
    query: str = Field(..., description="Search query")
    strategy: StrategyEnum = Field(
        StrategyEnum.PARENT_CHILD,
        description="Retrieval strategy to use"
    )
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    filters: Dict[str, Any] | None = Field(None, description="Optional filters")
    include_scores: bool = Field(True, description="Include similarity scores")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "sanksi pidana korupsi",
                "strategy": "parent_child",
                "top_k": 5,
                "filters": {"year": "2023"},
                "include_scores": True
            }
        }


class MultiStrategyRequest(BaseModel):
    """Multi-strategy comparison request."""
    query: str = Field(..., description="Search query")
    strategies: List[StrategyEnum] = Field(
        default=[StrategyEnum.VECTOR_SEARCH, StrategyEnum.PARENT_CHILD, StrategyEnum.HYBRID],
        description="Strategies to compare"
    )
    top_k: int = Field(3, description="Documents per strategy", ge=1, le=10)
    merge_results: bool = Field(True, description="Merge and deduplicate results")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "putusan pidana pencucian uang",
                "strategies": ["vector_search", "parent_child", "hybrid"],
                "top_k": 3,
                "merge_results": True
            }
        }
class DocumentResponse(BaseModel):
    """Document response model."""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    score: float | None = Field(None, description="Similarity score")


class RetrievalResponse(BaseModel):
    """Standard retrieval response."""
    query: str = Field(..., description="Original query")
    strategy: str = Field(..., description="Strategy used")
    documents: List[DocumentResponse] = Field(..., description="Retrieved documents")
    total_found: int = Field(..., description="Total documents found")
    execution_time: float = Field(..., description="Execution time in seconds")


class MultiStrategyResponse(BaseModel):
    """Multi-strategy comparison response."""
    query: str = Field(..., description="Original query")
    results: Dict[str, List[DocumentResponse]] = Field(..., description="Results by strategy")
    execution_time: float = Field(..., description="Execution time in seconds")


class ServiceStatusResponse(BaseModel):
    """Service status response."""
    default_strategy: str = Field(..., description="Default retrieval strategy")
    available_strategies: List[str] = Field(..., description="Available strategies")
    retrievers: Dict[str, Dict[str, Any]] = Field(..., description="Retriever status details")


# Helper Functions
def _convert_strategy(strategy: StrategyEnum) -> RetrievalStrategy:
    """Convert API strategy enum to service strategy enum."""
    mapping = {
        StrategyEnum.VECTOR_SEARCH: RetrievalStrategy.VECTOR_SEARCH,
        StrategyEnum.PARENT_CHILD: RetrievalStrategy.PARENT_CHILD,
        StrategyEnum.HYBRID: RetrievalStrategy.HYBRID
    }
    return mapping[strategy]


def _format_documents(documents) -> List[DocumentResponse]:
    """Format documents for API response."""
    return [
        DocumentResponse(
            content=doc.page_content,
            metadata=doc.metadata,
            score=doc.metadata.get("similarity_score") or doc.metadata.get("hybrid_score")
        )
        for doc in documents
    ]


# API Endpoints
@router.post("/", response_model=RetrievalResponse)
async def enhanced_search(
    request: EnhancedRetrievalRequest,
    # current_user: User = Depends(get_current_user)
):
    """
    Enhanced document search with multiple strategies.
    Supports vector search, parent-child chunking, and hybrid approaches.
    """
    import time
    start_time = time.time()

    try:
        # Convert strategy
        strategy = _convert_strategy(request.strategy)

        # Perform search
        documents = await search_with_retrieval_service(
            query=request.query,
            strategy=strategy,
            top_k=request.top_k,
            filters=request.filters,
            include_scores=request.include_scores
        )

        execution_time = time.time() - start_time

        return RetrievalResponse(
            query=request.query,
            strategy=request.strategy.value,
            documents=_format_documents(documents),
            total_found=len(documents),
            execution_time=execution_time
        )

    except Exception as e:
        logger.error(f"Enhanced search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/multi-strategy", response_model=MultiStrategyResponse)
async def multi_strategy_comparison(
    request: MultiStrategyRequest,
    # current_user: User = Depends(get_current_user)
):
    """
    Compare multiple retrieval strategies for the same query.
    Useful for strategy evaluation and optimization.
    """
    import time
    start_time = time.time()

    try:
        # Convert strategies
        strategies = [_convert_strategy(s) for s in request.strategies]

        # Perform multi-strategy search
        results = await multi_strategy_search(
            query=request.query,
            strategies=strategies,
            top_k=request.top_k,
            merge_results=request.merge_results
        )

        # Format results
        formatted_results = {}
        for strategy_name, documents in results.items():
            formatted_results[strategy_name] = _format_documents(documents)

        execution_time = time.time() - start_time

        return MultiStrategyResponse(
            query=request.query,
            results=formatted_results,
            execution_time=execution_time
        )

    except Exception as e:
        logger.error(f"Multi-strategy search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-strategy search failed: {str(e)}")


@router.get("/status", response_model=ServiceStatusResponse)
async def get_service_status(
    # current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive status of the retrieval service.
    Shows available strategies and their health status.
    """
    try:
        service = await get_retrieval_service()
        status = service.get_service_status()

        return ServiceStatusResponse(
            default_strategy=status["default_strategy"],
            available_strategies=status["available_strategies"],
            retrievers=status["retrievers"]
        )

    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
