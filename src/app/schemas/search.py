"""
Pydantic schemas for Supreme Court RAG search API.
Defines the API contracts and data validation models.
"""

from datetime import date, datetime
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field, validator


class CaseType(str, Enum):
    """Enumeration of legal case types."""
    CRIMINAL = "criminal"
    CIVIL = "civil"
    ADMINISTRATIVE = "administrative"
    CONSTITUTIONAL = "constitutional"
    COMMERCIAL = "commercial"


class CourtLevel(str, Enum):
    """Enumeration of court levels."""
    SUPREME = "supreme"
    HIGH = "high"
    DISTRICT = "district"
    SPECIALIZED = "specialized"


class Jurisdiction(str, Enum):
    """Southeast Asian jurisdiction codes."""
    INDONESIA = "ID"
    MALAYSIA = "MY"
    SINGAPORE = "SG"
    THAILAND = "TH"
    VIETNAM = "VN"
    PHILIPPINES = "PH"


class DateRange(BaseModel):
    """Date range filter for search queries."""
    start: date = Field(..., description="Start date (inclusive)")
    end: date = Field(..., description="End date (inclusive)")

    @validator('end')
    def end_after_start(cls, v, values):
        """Ensure end date is after start date."""
        if 'start' in values and v < values['start']:
            raise ValueError('End date must be after start date')
        return v

    @validator('end')
    def end_not_future(cls, v):
        """Ensure end date is not in the future."""
        if v > date.today():
            raise ValueError('End date cannot be in the future')
        return v


class SearchFilters(BaseModel):
    """Search filters for refining query results."""
    jurisdiction: Jurisdiction | None = Field(None, description="Jurisdiction code")
    date_range: DateRange | None = Field(None, description="Date range filter")
    case_type: CaseType | None = Field(None, description="Type of legal case")
    court_level: CourtLevel | None = Field(None, description="Court level")
    case_number: str | None = Field(None, description="Specific case number")

    class Config:
        json_schema_extra = {
            "example": {
                "jurisdiction": "ID",
                "date_range": {
                    "start": "2020-01-01",
                    "end": "2023-12-31"
                },
                "case_type": "criminal",
                "court_level": "supreme"
            }
        }


class SearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language search query"
    )
    filters: SearchFilters | None = Field(None, description="Optional search filters")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results")
    include_summary: bool = Field(True, description="Include AI-generated summary")
    include_validation: bool = Field(True, description="Include claim validation")

    @validator('query')
    def query_not_empty(cls, v):
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "query": "putusan mahkamah agung tentang korupsi",
                "filters": {
                    "jurisdiction": "ID",
                    "case_type": "criminal"
                },
                "max_results": 10,
                "include_summary": True,
                "include_validation": True
            }
        }


class SourceDocument(BaseModel):
    """Source document citation information."""
    title: str = Field(..., description="Judul dokumen lengkap")
    case_number: str = Field(..., description="Nomor perkara resmi (contoh: 123/PID/2023)")
    excerpt: str = Field(..., description="Kutipan relevan langsung dari dokumen, maksimal 300 kata, dengan highlighting bagian penting")
    source: str = Field(..., description="Sumber dokumen dari metadata retrieved")
    link_pdf: str = Field(..., description="URL atau path ke dokumen sumber dari metadata retrieved")
    validation_status: str = Field(..., description="Supported/Partially Supported/Unsupported/Uncertain")
    relevance_score: float = Field(0.95, ge=0.0, le=1.0, description="Relevance score")
    legal_areas: List[str] = Field(default_factory=list, description="Relevant legal areas/topics")

    class Config:
        json_schema_extra = {
            "example": {
            "title": "Judul dokumen lengkap",
            "case_number": "Nomor perkara resmi (contoh: 123/PID/2023)",
            "excerpt": "Kutipan relevan langsung dari dokumen, maksimal 300 kata, dengan highlighting bagian penting",
            "source": "Sumber dokumen dari metadata retrieved",
            "link_pdf": "URL atau path ke dokumen sumber dari metadata retrieved",
            "validation_status": "Supported",
            "relevance_score": 0.95,
            "legal_areas": ["Area hukum 1", "Area hukum 2"]
            }
        }


class ValidationStatus(str, Enum):
    """Claim validation status categories."""
    SUPPORTED = "Supported"
    PARTIALLY_SUPPORTED = "Partially Supported"
    UNSUPPORTED = "Unsupported"
    UNCERTAIN = "Uncertain"


class SearchResult(BaseModel):
    """Individual search result with AI analysis."""
    summary: str = Field(..., description="AI-generated summary of relevant information")
    key_points: List[str] = Field(default_factory=list, description="Key legal points extracted")
    source_documents: List[SourceDocument] = Field(default_factory=list, description="Source citations")
    validation_status: str = Field(..., description="Overall claim validation status")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence in result")
    legal_areas: List[str] = Field(default_factory=list, description="Relevant legal areas/topics")

    class Config:
        json_schema_extra = {
            "example": {
                "summary": "Berdasarkan putusan yang ditemukan, korupsi dana desa...",
                "key_points": [
                    "Unsur korupsi dana desa terpenuhi",
                    "Sanksi pidana 5 tahun penjara",
                    "Ganti rugi wajib dibayar"
                ],
                "source_documents": [
                   {                
                    "title": "Judul dokumen lengkap",
                    "case_number": "Nomor perkara resmi (contoh: 123/PID/2023)",
                    "excerpt": "Kutipan relevan langsung dari dokumen, maksimal 300 kata, dengan highlighting bagian penting",
                    "source": "Sumber dokumen dari metadata retrieved",
                    "link_pdf": "URL atau path ke dokumen sumber dari metadata retrieved",
                    "validation_status": "Supported/Partially Supported/Unsupported/Uncertain",
                    "relevance_score": 0.95,
                    "legal_areas": ["Area hukum 1", "Area hukum 2"]
                    }
                ],
                "validation_status": "Supported",
                "confidence_score": 0.82,
                "legal_areas": ["Hukum Pidana", "Tindak Pidana Korupsi"]
            }
        }


class RAGMetrics(BaseModel):
    """Performance metrics for RAG system monitoring."""
    query_time: float = Field(..., description="Total query processing time in seconds")
    retrieval_time: float = Field(0.0, description="Document retrieval time")
    generation_time: float = Field(0.0, description="AI generation time")
    validation_time: float = Field(0.0, description="Claim validation time")
    documents_retrieved: int = Field(..., description="Number of documents retrieved")
    tokens_used: int = Field(0, description="Total tokens consumed")
    cache_hit: bool = Field(False, description="Whether result was cached")

    class Config:
        json_schema_extra = {
            "example": {
                "query_time": 2.5,
                "retrieval_time": 0.8,
                "generation_time": 1.2,
                "validation_time": 0.5,
                "documents_retrieved": 10,
                "tokens_used": 1500,
                "cache_hit": False
            }
        }


class SearchResponse(BaseModel):
    """Complete search response with results and metadata."""
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    metrics: RAGMetrics = Field(..., description="Performance metrics")
    timestamp: datetime = Field(..., description="Response timestamp")
    filters_applied: SearchFilters | None = Field(None, description="Filters that were applied")
    total_results: int = Field(..., description="Total number of results found")
    has_more: bool = Field(False, description="Whether more results are available")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "putusan mahkamah agung tentang korupsi",
                "results": [
                    {
                        "summary": "Berdasarkan putusan yang ditemukan...",
                        "key_points": ["Unsur korupsi terpenuhi"],
                        "source_documents": [],
                        "validation_status": "Supported",
                        "confidence_score": 0.82
                    }
                ],
                "metrics": {
                    "query_time": 2.5,
                    "documents_retrieved": 10,
                    "tokens_used": 1500
                },
                "timestamp": "2023-12-01T10:30:00Z",
                "total_results": 25,
                "has_more": True
            }
        }


# Document ingestion schemas
class DocumentMetadata(BaseModel):
    """Metadata for a legal document."""
    case_number: str | None = Field(None, description="Case number")
    title: str | None = Field(None, description="Document title")
    court: str | None = Field(None, description="Court name")
    jurisdiction: Jurisdiction | None = Field(None, description="Jurisdiction")
    date: datetime | None = Field(None, description="Document date")
    case_type: CaseType | None = Field(None, description="Case type")
    court_level: CourtLevel | None = Field(None, description="Court level")
    language: str = Field("id", description="Document language code")
    page_count: int | None = Field(None, description="Number of pages")
    file_format: str = Field("pdf", description="Original file format")

    class Config:
        json_schema_extra = {
            "example": {
                "case_number": "123/K/Pid/2023",
                "title": "Putusan Kasasi Tindak Pidana Korupsi",
                "court": "Mahkamah Agung Republik Indonesia",
                "jurisdiction": "ID",
                "date": "2023-05-15",
                "case_type": "criminal",
                "court_level": "supreme",
                "language": "id",
                "page_count": 25,
                "file_format": "pdf"
            }
        }


class DocumentInput(BaseModel):
    """Input document for bulk ingestion."""
    content: str = Field(..., description="Document text content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")

    @validator('content')
    def content_not_empty(cls, v):
        """Ensure content is not empty."""
        if not v.strip():
            raise ValueError('Document content cannot be empty')
        return v.strip()


class BulkDocumentRequest(BaseModel):
    """Request for bulk document ingestion."""
    documents: List[DocumentInput] = Field(..., description="List of documents to ingest")
    batch_size: int = Field(50, ge=1, le=200, description="Processing batch size")
    overwrite_existing: bool = Field(False, description="Whether to overwrite existing documents")

    @validator('documents')
    def documents_not_empty(cls, v):
        """Ensure documents list is not empty."""
        if not v:
            raise ValueError('Documents list cannot be empty')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "content": "Putusan Mahkamah Agung...",
                        "metadata": {
                            "case_number": "123/K/Pid/2023",
                            "title": "Putusan Kasasi Korupsi"
                        }
                    }
                ],
                "batch_size": 50,
                "overwrite_existing": False
            }
        }


class BulkDocumentResponse(BaseModel):
    """Response for bulk document ingestion."""
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    total_documents: int = Field(..., description="Total documents submitted")
    processed_count: int = Field(..., description="Number of documents processed")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    timestamp: datetime = Field(..., description="Response timestamp")
    processing_id: str | None = Field(None, description="Background processing ID")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "completed",
                "message": "Documents processed successfully",
                "total_documents": 100,
                "processed_count": 95,
                "errors": ["Document 45: Invalid format"],
                "timestamp": "2023-12-01T10:30:00Z"
            }
        }


# Health and monitoring schemas
class ServiceStatus(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class HealthResponse(BaseModel):
    """System health response."""
    status: ServiceStatus = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    response_time: float = Field(0.0, description="System response time in seconds")
    services: Dict[str, ServiceStatus] = Field(..., description="Individual service statuses")
    error: str | None = Field(None, description="Error message if unhealthy")
    version: str = Field("1.0.0", description="API version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2023-12-01T10:30:00Z",
                "response_time": 0.15,
                "services": {
                    "vector_store": "healthy",
                    "llm_service": "healthy",
                    "validation_service": "healthy"
                },
                "version": "1.0.0"
            }
        }
