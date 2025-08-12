from typing import Optional, List, Dict, Any
from fastcrud import FastCRUD
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, text
from sqlalchemy.orm import selectinload

from ..models.legal_document import LegalDocument, DocumentCitation, SearchQuery
from ..schemas.legal_search import (
    LegalDocumentCreate, LegalDocumentUpdate, LegalDocumentRead,
    SearchQueryCreate, SearchQueryUpdate,
    SearchFilters
)


class CRUDLegalDocument(FastCRUD[LegalDocument, LegalDocumentCreate, LegalDocumentUpdate, LegalDocumentUpdate, None, LegalDocumentRead]):
    """
    CRUD operations for Legal Documents.
    Implements document storage and retrieval (F1.7).
    """
    
    async def get_model(self, db: AsyncSession, id: int) -> Optional[LegalDocument]:
        """Get a legal document by ID returning the model object."""
        query = select(LegalDocument).options(
            selectinload(LegalDocument.citations)
        ).where(LegalDocument.id == id)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by_case_number(
        self, 
        db: AsyncSession, 
        case_number: str,
        jurisdiction: Optional[str] = None
    ) -> Optional[LegalDocument]:
        """Get document by case number and optionally jurisdiction."""
        query = select(LegalDocument).where(
            and_(
                LegalDocument.case_number == case_number,
                LegalDocument.is_active == True
            )
        )
        
        if jurisdiction:
            query = query.where(LegalDocument.jurisdiction == jurisdiction)
            
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by_content_hash(
        self, 
        db: AsyncSession, 
        content_hash: str
    ) -> Optional[LegalDocument]:
        """Get document by content hash for deduplication."""
        query = select(LegalDocument).where(
            LegalDocument.content_hash == content_hash
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def search_with_filters(
        self,
        db: AsyncSession,
        filters: SearchFilters,
        limit: int = 10,
        offset: int = 0
    ) -> List[LegalDocument]:
        """
        Search documents with metadata filters.
        Used as fallback or pre-filter for vector search.
        """
        query = select(LegalDocument).where(LegalDocument.is_active == True)
        
        # Apply filters
        if filters.jurisdiction:
            query = query.where(LegalDocument.jurisdiction.ilike(f"%{filters.jurisdiction}%"))
            
        if filters.case_type:
            query = query.where(LegalDocument.case_type.ilike(f"%{filters.case_type}%"))
            
        if filters.legal_area:
            query = query.where(LegalDocument.legal_area.ilike(f"%{filters.legal_area}%"))
            
        if filters.court_name:
            query = query.where(LegalDocument.court_name.ilike(f"%{filters.court_name}%"))
            
        if filters.date_from:
            query = query.where(LegalDocument.decision_date >= filters.date_from)
            
        if filters.date_to:
            query = query.where(LegalDocument.decision_date <= filters.date_to)
        
        # Apply pagination and ordering
        query = query.order_by(LegalDocument.decision_date.desc()).offset(offset).limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_documents_by_ids(
        self,
        db: AsyncSession,
        document_ids: List[int],
        include_citations: bool = False
    ) -> List[LegalDocument]:
        """
        Retrieve multiple documents by IDs.
        Maps to F1.7 - fetching full document content after vector search.
        """
        query = select(LegalDocument).where(
            and_(
                LegalDocument.id.in_(document_ids),
                LegalDocument.is_active == True
            )
        )
        
        if include_citations:
            query = query.options(selectinload(LegalDocument.citations))
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def update_processing_status(
        self,
        db: AsyncSession,
        document_id: int,
        status: str,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """Update document processing status and vector ID."""
        update_data = {"processing_status": status}
        if embedding:
            update_data["embedding"] = embedding
            
        await self.update(db=db, object=update_data, id=document_id)
        return True
    
    async def get_statistics(self, db: AsyncSession) -> Dict[str, Any]:
        """Get document collection statistics."""
        # Count by jurisdiction
        jurisdiction_query = text("""
            SELECT jurisdiction, COUNT(*) as count 
            FROM legal_document 
            WHERE is_active = true 
            GROUP BY jurisdiction
            ORDER BY count DESC
        """)
        
        # Count by case type
        case_type_query = text("""
            SELECT case_type, COUNT(*) as count 
            FROM legal_document 
            WHERE is_active = true AND case_type IS NOT NULL
            GROUP BY case_type
            ORDER BY count DESC
        """)
        
        # Total counts
        total_query = text("""
            SELECT 
                COUNT(*) as total_documents,
                COUNT(CASE WHEN processing_status = 'processed' THEN 1 END) as processed_documents,
                COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as vectorized_documents
            FROM legal_document 
            WHERE is_active = true
        """)
        
        jurisdiction_result = await db.execute(jurisdiction_query)
        case_type_result = await db.execute(case_type_query)
        total_result = await db.execute(total_query)
        
        return {
            "total_statistics": total_result.fetchone()._asdict(),
            "by_jurisdiction": [row._asdict() for row in jurisdiction_result.fetchall()],
            "by_case_type": [row._asdict() for row in case_type_result.fetchall()]
        }


class CRUDDocumentCitation(FastCRUD[DocumentCitation, None, None, None, None, None]):
    """CRUD operations for Document Citations."""
    
    async def create_citations_batch(
        self,
        db: AsyncSession,
        document_id: int,
        citations_data: List[Dict[str, Any]]
    ) -> List[DocumentCitation]:
        """Create multiple citations for a document."""
        citations = []
        for citation_data in citations_data:
            citation = DocumentCitation(
                document_id=document_id,
                **citation_data
            )
            db.add(citation)
            citations.append(citation)
        
        await db.flush()
        return citations
    
    async def update_validation_status(
        self,
        db: AsyncSession,
        citation_id: int,
        is_validated: bool,
        validation_score: Optional[float] = None,
        validation_notes: Optional[str] = None
    ) -> bool:
        """Update citation validation status (F1.11)."""
        update_data = {
            "is_validated": is_validated,
            "validation_score": validation_score,
            "validation_notes": validation_notes
        }
        await self.update(db=db, object=update_data, id=citation_id)
        return True


class CRUDSearchQuery(FastCRUD[SearchQuery, SearchQueryCreate, SearchQueryUpdate, SearchQueryUpdate, None, None]):
    """
    CRUD operations for Search Queries.
    Used for analytics and caching.
    """
    
    async def get_by_hash(
        self, 
        db: AsyncSession, 
        query_hash: str,
        max_age_hours: int = 24
    ) -> Optional[SearchQuery]:
        """Get recent search by hash for caching."""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        query = select(SearchQuery).where(
            and_(
                SearchQuery.query_hash == query_hash,
                SearchQuery.created_at >= cutoff_time
            )
        ).order_by(SearchQuery.created_at.desc())
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_popular_queries(
        self,
        db: AsyncSession,
        limit: int = 10,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get most popular search queries for analytics."""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        query = text("""
            SELECT 
                query_text,
                COUNT(*) as search_count,
                AVG(response_time_ms) as avg_response_time,
                AVG(results_count) as avg_results_count
            FROM search_query 
            WHERE created_at >= :cutoff_time
            GROUP BY query_text
            ORDER BY search_count DESC
            LIMIT :limit
        """)
        
        result = await db.execute(query, {"cutoff_time": cutoff_time, "limit": limit})
        return [row._asdict() for row in result.fetchall()]
    
    async def get_user_search_history(
        self,
        db: AsyncSession,
        user_id: int,
        limit: int = 20
    ) -> List[SearchQuery]:
        """Get user's search history."""
        query = select(SearchQuery).where(
            SearchQuery.user_id == user_id
        ).order_by(SearchQuery.created_at.desc()).limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()


# Create CRUD instances
crud_legal_documents = CRUDLegalDocument(LegalDocument)
crud_document_citations = CRUDDocumentCitation(DocumentCitation)
crud_search_queries = CRUDSearchQuery(SearchQuery)
