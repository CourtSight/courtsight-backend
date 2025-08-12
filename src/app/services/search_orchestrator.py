import uuid as uuid_pkg
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, func
import logging

from ..schemas.legal_search import (
    SearchRequest, SearchResponse, SearchResultSummary,
    LegalDocumentRead, EmbeddingRequest, 
    LLMAnalysisRequest, ValidationRequest,
    SearchQueryCreate, SearchQueryUpdate
)
from ..crud.crud_legal_documents import crud_legal_documents, crud_search_queries
from ..models.legal_document import LegalDocument
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from ..core.config import settings

logger = logging.getLogger(__name__)


class SearchOrchestrator:
    """
    Main orchestrator for legal document search using Supabase pgvector.
    Implements the complete search workflow from F1.1 to F1.13.
    
    This class coordinates between different services:
    - Embedding Service (F1.3-F1.4) 
    - Supabase pgvector for similarity search (F1.5-F1.6)
    - Document Store (F1.7)
    - LLM Analysis Service (F1.8-F1.9)
    - Validation Service (F1.10-F1.11)
    """
    
    def __init__(self):
        """Initialize the search orchestrator with service clients."""
        self.embedding_client = embedding_service
        self.llm_client = llm_service
        logger.info("SearchOrchestrator initialized")
    
    async def search(
        self,
        request: SearchRequest,
        db: AsyncSession,
        user_id: Optional[int] = None
    ) -> SearchResponse:
        """
        Execute complete search workflow using Supabase pgvector.
        Maps to the full sequence F1.1 through F1.13.
        """
        start_time = time.time()
        query_id = uuid_pkg.uuid4()
        
        try:
            # F1.1-F1.2: Process search request and create query record
            query_hash = self._generate_query_hash(request)
            search_query_record = await self._create_search_record(
                db, request, query_hash, user_id, query_id
            )
            
            # Check cache first
            cached_result = await self._check_cache(db, query_hash)
            if cached_result:
                return cached_result
            
            # F1.3-F1.4: Generate embedding for search query
            query_embedding = await self._generate_query_embedding(request.query)
            
            # F1.5-F1.6: Perform similarity search using Supabase pgvector
            documents = await self._perform_supabase_vector_search(
                db, query_embedding, request
            )
            
            # Prepare base response (Phase 1)
            response = SearchResponse(
                documents=documents,
                total_count=len(documents),
                query_id=query_id,
                processing_time_ms=int((time.time() - start_time) * 1000),
                search_strategy="supabase_pgvector"
            )
            
            # F1.8-F1.9: Generate AI analysis (Phase 2)
            if request.include_summary and documents:
                ai_summary = await self._generate_ai_analysis(request.query, documents)
                response.ai_summary = ai_summary
            
            # F1.10-F1.11: Validate citations (Phase 3)
            if request.include_validation and response.ai_summary:
                validation_status = await self._validate_citations(
                    response.ai_summary, documents
                )
                response.validation_status = validation_status["status"]
                response.validated_citations_count = validation_status["validated_count"]
            
            # Update search record with results
            await self._update_search_record(
                db, search_query_record.id, len(documents), 
                int((time.time() - start_time) * 1000)
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Search failed for query {query_id}: {e}")
            # Return empty response with error info
            return SearchResponse(
                documents=[],
                total_count=0,
                query_id=query_id,
                processing_time_ms=int((time.time() - start_time) * 1000),
                search_strategy="failed"
            )
    
    async def _generate_query_embedding(self, query_text: str) -> List[float]:
        """
        Generate embedding for search query.
        Maps to F1.3-F1.4.
        """
        try:
            embedding_request = EmbeddingRequest(
                text=query_text,
                model_name="multilingual-e5-large"  # Optimized for multilingual legal text
            )
            
            async with self.embedding_client as client:
                embedding_response = await client.generate_embedding(embedding_request)
                return embedding_response.embedding
                
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}")
    
    async def _perform_supabase_vector_search(
        self,
        db: AsyncSession,
        query_embedding: List[float],
        request: SearchRequest
    ) -> List[LegalDocumentRead]:
        """
        Perform similarity search using Supabase pgvector RPC function.
        Maps to F1.5-F1.6-F1.7 combined.
        """
        try:
            # Use the Supabase RPC function for vector search
            rpc_params = {
                "query_embedding": query_embedding,
                "match_threshold": 0.7,
                "match_count": request.max_results
            }
            
            # Add optional filters
            if request.filters:
                if request.filters.jurisdiction:
                    rpc_params["filter_jurisdiction"] = request.filters.jurisdiction
                if request.filters.case_type:
                    rpc_params["filter_case_type"] = request.filters.case_type
                if request.filters.legal_area:
                    rpc_params["filter_legal_area"] = request.filters.legal_area
                if request.filters.date_from:
                    rpc_params["filter_date_from"] = request.filters.date_from
                if request.filters.date_to:
                    rpc_params["filter_date_to"] = request.filters.date_to
            
            # Call the Supabase RPC function
            rpc_query = "SELECT * FROM match_legal_documents("
            param_parts = []
            for key, value in rpc_params.items():
                if key == "query_embedding":
                    param_parts.append(f"{key} => :query_embedding")
                else:
                    param_parts.append(f"{key} => :{key}")
            
            rpc_query += ", ".join(param_parts) + ")"
            
            # Execute the RPC call
            result = await db.execute(text(rpc_query), rpc_params)
            rows = result.fetchall()
            
            # Convert to response format
            documents = []
            for row in rows:
                doc_read = LegalDocumentRead(
                    id=row.id,
                    uuid=row.uuid,
                    case_number=row.case_number,
                    court_name=row.court_name,
                    jurisdiction=row.jurisdiction,
                    title=row.title,
                    summary=row.summary,
                    decision_date=row.decision_date,
                    case_type=row.case_type,
                    legal_area=row.legal_area,
                    relevance_score=float(row.similarity),
                    matched_snippets=self._extract_relevant_snippets(
                        f"{row.title}. {row.summary or ''}", max_snippets=3
                    )
                )
                documents.append(doc_read)
            
            return documents
            
        except Exception as e:
            logger.error(f"Supabase vector search failed: {e}")
            # Fallback to metadata-only search
            return await self._fallback_metadata_search(db, request)
    
    async def _fallback_metadata_search(
        self,
        db: AsyncSession,
        request: SearchRequest
    ) -> List[LegalDocumentRead]:
        """Fallback search using metadata filters only."""
        try:
            if request.filters:
                documents = await crud_legal_documents.search_with_filters(
                    db, request.filters, limit=request.max_results
                )
                return [self._convert_to_read_schema(doc) for doc in documents]
            return []
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []
    
    async def _generate_ai_analysis(
        self,
        query: str,
        documents: List[LegalDocumentRead]
    ) -> SearchResultSummary:
        """
        Generate AI analysis and summary of search results.
        Maps to F1.8-F1.9.
        """
        try:
            # Prepare context documents for LLM
            context_documents = [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "case_number": doc.case_number,
                    "court_name": doc.court_name,
                    "jurisdiction": doc.jurisdiction,
                    "summary": doc.summary or doc.title,
                    "relevance_score": doc.relevance_score
                }
                for doc in documents[:5]  # Limit context to top 5 documents
            ]
            
            analysis_request = LLMAnalysisRequest(
                query=query,
                context_documents=context_documents,
                analysis_type="summary_and_analysis",
                max_tokens=1500
            )
            
            async with self.llm_client as client:
                analysis_response = await client.analyze_search_results(analysis_request)
            
            return SearchResultSummary(
                summary=analysis_response.summary,
                key_points=analysis_response.key_points,
                legal_themes=analysis_response.legal_themes,
                confidence_score=analysis_response.confidence_score
            )
            
        except Exception as e:
            logger.error(f"AI analysis generation failed: {e}")
            # Return basic summary
            return SearchResultSummary(
                summary=f"Found {len(documents)} relevant legal documents for the query.",
                key_points=[f"Document {i+1}: {doc.title}" for i, doc in enumerate(documents[:3])],
                legal_themes=list(set([doc.legal_area for doc in documents if doc.legal_area])),
                confidence_score=0.5
            )
    
    async def _validate_citations(
        self,
        ai_summary: SearchResultSummary,
        documents: List[LegalDocumentRead]
    ) -> Dict[str, Any]:
        """
        Validate citations in AI-generated content.
        Maps to F1.10-F1.11.
        """
        try:
            # Extract citations from summary and key points
            text_to_validate = ai_summary.summary + " " + " ".join(ai_summary.key_points)
            
            # Use LLM to extract citations first
            async with self.llm_client as client:
                citations = await client.extract_citations(text_to_validate)
            
            if not citations:
                return {"status": "no_citations", "validated_count": 0}
            
            # Prepare source documents for validation
            source_documents = [
                {
                    "id": doc.id,
                    "case_number": doc.case_number,
                    "court_name": doc.court_name,
                    "title": doc.title
                }
                for doc in documents
            ]
            
            validation_request = ValidationRequest(
                citations=citations,
                source_documents=source_documents,
                validation_method="fuzzy_match"
            )
            
            async with self.validation_client as client:
                validation_response = await client.validate_citations(validation_request)
            
            validated_count = sum(1 for result in validation_response.results if result.is_valid)
            
            return {
                "status": "validated",
                "validated_count": validated_count,
                "total_citations": len(citations),
                "validity_score": validation_response.overall_validity_score
            }
            
        except Exception as e:
            logger.error(f"Citation validation failed: {e}")
            return {"status": "validation_failed", "validated_count": 0}
    
    def _generate_query_hash(self, request: SearchRequest) -> str:
        """Generate hash for query caching."""
        query_string = f"{request.query}|{request.filters}|{request.max_results}"
        return hashlib.sha256(query_string.encode()).hexdigest()
    
    async def _create_search_record(
        self,
        db: AsyncSession,
        request: SearchRequest,
        query_hash: str,
        user_id: Optional[int],
        query_id: uuid_pkg.UUID
    ) -> Any:
        """Create search query record for analytics."""
        search_query = SearchQueryCreate(
            query_text=request.query,
            query_hash=query_hash,
            jurisdiction_filter=request.filters.jurisdiction if request.filters else None,
            case_type_filter=request.filters.case_type if request.filters else None,
            date_from=request.filters.date_from if request.filters else None,
            date_to=request.filters.date_to if request.filters else None,
            user_id=user_id,
            session_id=str(query_id)
        )
        
        return await crud_search_queries.create(db=db, object=search_query)
    
    async def _update_search_record(
        self,
        db: AsyncSession,
        record_id: int,
        results_count: int,
        response_time_ms: int
    ) -> None:
        """Update search record with results."""
        update_data = SearchQueryUpdate(
            results_count=results_count,
            response_time_ms=response_time_ms
        )
        await crud_search_queries.update(db=db, object=update_data, id=record_id)
    
    async def _check_cache(self, db: AsyncSession, query_hash: str) -> Optional[SearchResponse]:
        """Check if we have cached results for this query."""
        cached_query = await crud_search_queries.get_by_hash(db, query_hash, max_age_hours=6)
        if cached_query and cached_query.results_count and cached_query.results_count > 0:
            # For demo purposes, return None (no cache)
            # In production, you'd implement proper result caching
            pass
        return None
    
    def _convert_to_read_schema(self, document: LegalDocument) -> LegalDocumentRead:
        """Convert SQLAlchemy model to Pydantic schema."""
        return LegalDocumentRead(
            id=document.id,
            uuid=document.uuid,
            case_number=document.case_number,
            court_name=document.court_name,
            jurisdiction=document.jurisdiction,
            title=document.title,
            summary=document.summary,
            decision_date=document.decision_date,
            case_type=document.case_type,
            legal_area=document.legal_area,
            citations=[
                {
                    "id": citation.id,
                    "cited_case_number": citation.cited_case_number,
                    "cited_court": citation.cited_court,
                    "citation_text": citation.citation_text,
                    "page_reference": citation.page_reference,
                    "is_validated": citation.is_validated,
                    "validation_score": citation.validation_score,
                    "validation_notes": citation.validation_notes
                }
                for citation in (document.citations or [])
            ] if hasattr(document, 'citations') and document.citations else None
        )
    
    def _extract_relevant_snippets(self, full_text: str, max_snippets: int = 3) -> List[str]:
        """Extract relevant text snippets (simplified implementation)."""
        # This is a simplified implementation
        # In production, you'd use more sophisticated text extraction
        sentences = full_text.split('. ')
        if len(sentences) <= max_snippets:
            return sentences
        
        # Return first, middle, and last parts
        step = len(sentences) // max_snippets
        return [
            sentences[i * step] + "."
            for i in range(max_snippets)
            if i * step < len(sentences)
        ]
    
    async def get_search_analytics(self, db: AsyncSession) -> Dict[str, Any]:
        """Get search analytics and statistics."""
        try:
            popular_queries = await crud_search_queries.get_popular_queries(db)
            doc_stats = await crud_legal_documents.get_statistics(db)
            
            return {
                "popular_queries": popular_queries,
                "document_statistics": doc_stats,
                "service_status": "operational"
            }
            
        except Exception as e:
            logger.error(f"Failed to get search analytics: {e}")
            return {"error": str(e)}
