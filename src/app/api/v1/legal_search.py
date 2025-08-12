"""
Legal Document Search API endpoints for CourtSight Feature 1.
Provides REST API for AI-powered legal document search.
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.database import async_get_db
from app.schemas.legal_search import (
    SearchRequest, 
    SearchResponse, 
    LegalDocumentRead,
)
from app.services.search_orchestrator import SearchOrchestrator
from app.crud.crud_legal_documents import crud_legal_documents
from app.api.dependencies import get_current_user, get_optional_user
from app.models.user import User

router = APIRouter(prefix="/legal-search", tags=["legal-search"])

# Initialize search orchestrator
search_orchestrator = SearchOrchestrator()


@router.post("/", response_model=SearchResponse)
async def search_legal_documents(
    request: SearchRequest,
    req: Request,
    db: AsyncSession = Depends(async_get_db),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user)
) -> SearchResponse:
    """
    Perform AI-powered search of legal documents.
    
    Maps to CourtSight Feature 1: AI-based Search Engine for Supreme Court Decisions
    - F1.1: Natural language query processing
    - F1.2: Semantic search with embeddings
    - F1.3: Advanced filtering
    - F1.4: Relevance ranking
    """
    try:
        # Validate request
        if not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query cannot be empty"
            )
        
        if request.max_results > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum results limit is 100"
            )
        
        # Perform search using orchestrator
        search_response = await search_orchestrator.search(request, db, current_user.get("id") if current_user else None)
        
        return search_response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/documents/{document_id}", response_model=LegalDocumentRead)
async def get_legal_document(
    document_id: int,
    db: AsyncSession = Depends(async_get_db),
    current_user: User = Depends(get_optional_user)
):
    """Get a specific legal document by ID."""
    document = await crud_legal_documents.get_model(db, id=document_id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Legal document not found"
        )
    
    if not document.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document is not available"
        )
    
    return LegalDocumentRead.model_validate(document)


@router.get("/documents", response_model=List[LegalDocumentRead])
async def list_legal_documents(
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of documents to return"),
    is_active: bool = Query(True, description="Filter by active status"),
    db: AsyncSession = Depends(async_get_db),
    current_user: User = Depends(get_optional_user)
) -> List[LegalDocumentRead]:
    """
    List legal documents with optional filtering.
    """
    try:
        # Get documents using FastCRUD get_multi
        result = await crud_legal_documents.get_multi(
            db, 
            offset=skip, 
            limit=limit
        )
        
        # Extract the data from FastCRUD response
        documents = result["data"] if isinstance(result, dict) else result
        
        return documents
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve documents: {str(e)}"
        )


# @router.post("/documents", response_model=LegalDocumentRead)
# async def create_legal_document(
#     document: LegalDocumentCreate,
#     db: AsyncSession = Depends(async_get_db),
#     current_user: Dict[str, Any] = Depends(get_current_user)
# ) -> LegalDocumentRead:
#     """
#     Create a new legal document (admin only).
#     """
#     # Check if user has admin privileges
#     if not current_user.get("is_superuser", False):
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Only administrators can create legal documents"
#         )
    
#     try:
#         # Create document
#         db_document = await crud_legal_documents.create(db, obj_in=document)
        
#         # TODO: Trigger background task for embedding generation
#         # await generate_document_embedding.delay(db_document.id)
        
#         return LegalDocumentRead.model_validate(db_document)
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to create document: {str(e)}"
#         )




@router.get("/stats")
async def get_search_stats(
    db: AsyncSession = Depends(async_get_db)
) -> dict:
    """
    Get search and document statistics.
    """
    try:
        stats = await crud_legal_documents.get_statistics(db)
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )
