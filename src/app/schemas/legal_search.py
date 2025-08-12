import uuid as uuid_pkg
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# ============= Search Request/Response Schemas =============

class SearchFilters(BaseModel):
    """Search filters for legal document queries (F1.2)"""
    jurisdiction: Optional[str] = Field(None, description="Filter by jurisdiction (e.g., 'Indonesia', 'Singapore')")
    case_type: Optional[str] = Field(None, description="Filter by case type (e.g., 'Criminal', 'Civil')")
    legal_area: Optional[str] = Field(None, description="Filter by legal area (e.g., 'Corporate Law')")
    date_from: Optional[datetime] = Field(None, description="Filter decisions from this date")
    date_to: Optional[datetime] = Field(None, description="Filter decisions until this date")
    court_name: Optional[str] = Field(None, description="Filter by specific court")


class SearchRequest(BaseModel):
    """Main search request schema (F1.1-F1.2)"""
    query: str = Field(..., min_length=3, max_length=500, description="Natural language search query")
    filters: Optional[SearchFilters] = Field(default_factory=SearchFilters, description="Search filters")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum number of results to return")
    include_summary: bool = Field(default=True, description="Whether to generate AI summary (Phase 2)")
    include_validation: bool = Field(default=False, description="Whether to validate citations (Phase 3)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "corruption cases in West Java, Indonesia",
                "filters": {
                    "jurisdiction": "Indonesia",
                    "case_type": "Criminal",
                    "date_from": "2020-01-01T00:00:00Z"
                },
                "max_results": 10,
                "include_summary": True,
                "include_validation": False
            }
        }
    )


class DocumentCitationRead(BaseModel):
    """Schema for reading document citations (F1.10-F1.11)"""
    id: int
    cited_case_number: str
    cited_court: str
    citation_text: str
    page_reference: Optional[str] = None
    is_validated: bool
    validation_score: Optional[float] = None
    validation_notes: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class LegalDocumentRead(BaseModel):
    """Schema for reading legal documents (F1.7)"""
    id: int
    uuid: uuid_pkg.UUID
    case_number: str
    court_name: str
    jurisdiction: str
    title: str
    summary: Optional[str] = None
    decision_date: Optional[datetime] = None
    case_type: Optional[str] = None
    legal_area: Optional[str] = None
    
    # Search-specific fields
    relevance_score: Optional[float] = Field(None, description="Similarity score from vector search")
    matched_snippets: Optional[List[str]] = Field(None, description="Relevant text snippets")
    
    # Citations (if requested)
    citations: Optional[List[DocumentCitationRead]] = None
    
    model_config = ConfigDict(from_attributes=True)


class SearchResultSummary(BaseModel):
    """AI-generated summary of search results (F1.8-F1.9)"""
    summary: str = Field(..., description="AI-generated summary of the search results")
    key_points: List[str] = Field(..., description="Key legal points extracted from results")
    legal_themes: List[str] = Field(..., description="Main legal themes identified")
    case_law_trends: Optional[str] = Field(None, description="Trends analysis across time periods")
    confidence_score: float = Field(..., ge=0, le=1, description="AI confidence in the summary accuracy")


class SearchResponse(BaseModel):
    """Complete search response schema (F1.12-F1.13)"""
    # Core results (Phase 1)
    documents: List[LegalDocumentRead] = Field(..., description="Matching legal documents")
    total_count: int = Field(..., description="Total number of matching documents")
    
    # Enhanced results (Phase 2)
    ai_summary: Optional[SearchResultSummary] = Field(None, description="AI-generated analysis")
    
    # Query metadata
    query_id: uuid_pkg.UUID = Field(..., description="Unique identifier for this search")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    search_strategy: str = Field(..., description="Search strategy used (semantic, hybrid, etc.)")
    
    # Quality metrics (Phase 3)
    validation_status: Optional[str] = Field(None, description="Citation validation status")
    validated_citations_count: Optional[int] = Field(None, description="Number of validated citations")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": [
                    {
                        "id": 1,
                        "uuid": "123e4567-e89b-12d3-a456-426614174000",
                        "case_number": "123/Pid.Sus/2023/PN Jkt.Sel",
                        "court_name": "Pengadilan Negeri Jakarta Selatan",
                        "jurisdiction": "Indonesia",
                        "title": "Putusan Tindak Pidana Korupsi",
                        "relevance_score": 0.95
                    }
                ],
                "total_count": 42,
                "ai_summary": {
                    "summary": "The search results show a pattern of corruption cases...",
                    "key_points": ["Point 1", "Point 2"],
                    "legal_themes": ["Corruption", "Public Officials"],
                    "confidence_score": 0.87
                },
                "query_id": "123e4567-e89b-12d3-a456-426614174000",
                "processing_time_ms": 1250,
                "search_strategy": "semantic_search"
            }
        }
    )


# ============= Internal Service Communication Schemas =============

class EmbeddingRequest(BaseModel):
    """Request schema for embedding service (F1.3)"""
    text: str = Field(..., description="Text to convert to embedding")
    model_name: str = Field(default="multilingual-e5-large", description="Embedding model to use")


class EmbeddingResponse(BaseModel):
    """Response schema from embedding service (F1.4)"""
    embedding: List[float] = Field(..., description="Vector representation of the text")
    model_name: str = Field(..., description="Model used for embedding")
    dimension: int = Field(..., description="Vector dimension")


class SimilaritySearchRequest(BaseModel):
    """Request schema for vector similarity search (F1.5)"""
    query_vector: List[float] = Field(..., description="Query vector for similarity search")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of similar documents to retrieve")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    threshold: float = Field(default=0.7, ge=0, le=1, description="Minimum similarity threshold")


class SimilaritySearchResult(BaseModel):
    """Individual result from similarity search"""
    document_id: str = Field(..., description="Document identifier")
    similarity_score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")


class SimilaritySearchResponse(BaseModel):
    """Response schema from similarity search (F1.6)"""
    results: List[SimilaritySearchResult] = Field(..., description="Similar documents")
    total_count: int = Field(..., description="Total number of results")
    search_time_ms: int = Field(..., description="Search execution time")


class LLMAnalysisRequest(BaseModel):
    """Request schema for LLM analysis service (F1.8)"""
    query: str = Field(..., description="Original user query")
    context_documents: List[Dict[str, Any]] = Field(..., description="Retrieved documents for context")
    analysis_type: str = Field(default="summary_and_analysis", description="Type of analysis requested")
    max_tokens: int = Field(default=1500, description="Maximum tokens for response")


class LLMAnalysisResponse(BaseModel):
    """Response schema from LLM analysis service (F1.9)"""
    summary: str = Field(..., description="Generated summary")
    key_points: List[str] = Field(..., description="Extracted key points")
    legal_themes: List[str] = Field(..., description="Identified legal themes")
    citations: List[str] = Field(..., description="Extracted citations")
    confidence_score: float = Field(..., description="Analysis confidence score")
    model_used: str = Field(..., description="LLM model used for analysis")


class ValidationRequest(BaseModel):
    """Request schema for citation validation service (F1.10)"""
    citations: List[str] = Field(..., description="Citations to validate")
    source_documents: List[Dict[str, Any]] = Field(..., description="Source documents for validation")
    validation_method: str = Field(default="fuzzy_match", description="Validation method to use")


class ValidationResult(BaseModel):
    """Individual citation validation result"""
    citation: str = Field(..., description="Original citation")
    is_valid: bool = Field(..., description="Whether citation is valid")
    confidence_score: float = Field(..., description="Validation confidence")
    matched_document: Optional[Dict[str, Any]] = Field(None, description="Matched source document")
    validation_notes: Optional[str] = Field(None, description="Additional validation notes")


class ValidationResponse(BaseModel):
    """Response schema from validation service (F1.11)"""
    results: List[ValidationResult] = Field(..., description="Validation results")
    overall_validity_score: float = Field(..., description="Overall validity score")
    validation_time_ms: int = Field(..., description="Validation processing time")


# ============= Database CRUD Schemas =============

class LegalDocumentCreate(BaseModel):
    """Schema for creating legal documents"""
    case_number: str
    court_name: str
    jurisdiction: str
    title: str
    full_text: str
    summary: Optional[str] = None
    decision_date: Optional[datetime] = None
    case_type: Optional[str] = None
    legal_area: Optional[str] = None
    content_hash: str


class LegalDocumentUpdate(BaseModel):
    """Schema for updating legal documents"""
    title: Optional[str] = None
    summary: Optional[str] = None
    case_type: Optional[str] = None
    legal_area: Optional[str] = None
    processing_status: Optional[str] = None
    embedding_vector_id: Optional[str] = None


class SearchQueryCreate(BaseModel):
    """Schema for creating search query records"""
    query_text: str
    query_hash: str
    jurisdiction_filter: Optional[str] = None
    case_type_filter: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    user_id: Optional[int] = None
    session_id: Optional[str] = None


class SearchQueryUpdate(BaseModel):
    """Schema for updating search query records"""
    results_count: Optional[int] = None
    response_time_ms: Optional[int] = None
