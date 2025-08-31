import uuid as uuid_pkg
from datetime import UTC, datetime
from typing import Optional, List

from sqlalchemy import DateTime, String, Text, Integer, Float, ForeignKey, Index, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import text
from pgvector.sqlalchemy import Vector

# Create a separate declarative base without dataclass functionality
class LegalDocumentBase(DeclarativeBase):
    """Base class for legal document models without dataclass issues."""
    pass


class LegalDocument(LegalDocumentBase):
    """
    Model for storing legal court documents/decisions with vector embeddings.
    Maps to F1.7 - Document Store functionality with Supabase pgvector.
    """
    __tablename__ = "legal_document"

    id: Mapped[int] = mapped_column("id", autoincrement=True, nullable=False, unique=True, primary_key=True)
    uuid: Mapped[uuid_pkg.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, unique=True, index=True, server_default=text('gen_random_uuid()'))
    
    # Document identification
    case_number: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    court_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    jurisdiction: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    
    # Document content
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    full_text: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    
    # Vector embeddings for semantic search (using Supabase pgvector)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(384), nullable=True)  # 384 dimensions for gte-small model
    
    # Legal metadata
    decision_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    case_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, default=None)
    legal_area: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, default=None)
    
    # Processing metadata
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    
    # Langchain PGStore integration
    collection_id: Mapped[Optional[uuid_pkg.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    embedding_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    
    # Audit fields
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text('now()'))
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    
    # Status tracking
    processing_status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    
    # Relations
    citations: Mapped[List["DocumentCitation"]] = relationship("DocumentCitation", back_populates="document", cascade="all, delete-orphan")
    
    # Langchain PGStore relationships (optional - may not be fully functional without langchain models)
    # collection: Mapped[Optional["LangchainCollection"]] = relationship("LangchainCollection", foreign_keys=[collection_id])
    # langchain_embedding: Mapped[Optional["LangchainEmbedding"]] = relationship("LangchainEmbedding", foreign_keys=[embedding_id])


class DocumentCitation(LegalDocumentBase):
    """
    Model for storing citations within legal documents.
    Maps to validation functionality in F1.10-F1.11.
    """
    __tablename__ = "document_citation"

    id: Mapped[int] = mapped_column("id", autoincrement=True, nullable=False, unique=True, primary_key=True)
    
    # Parent document
    document_id: Mapped[int] = mapped_column(ForeignKey("legal_document.id"), nullable=False, index=True)
    
    # Citation details
    cited_case_number: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    cited_court: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, default=None)
    citation_text: Mapped[str] = mapped_column(Text, nullable=False)
    page_reference: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default=None)
    
    # Validation status
    is_validated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    validation_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    validation_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    
    # Audit
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text('now()'))
    validated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    
    # Relations
    document: Mapped["LegalDocument"] = relationship("LegalDocument", back_populates="citations")


class SearchQuery(LegalDocumentBase):
    """
    Model for tracking search queries for analytics.
    Maps to F1.1-F1.2 and F1.12-F1.13 functionality.
    """
    __tablename__ = "search_query"

    id: Mapped[int] = mapped_column("id", autoincrement=True, nullable=False, unique=True, primary_key=True)
    
    # Query details
    query_text: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    # Filters applied
    jurisdiction_filter: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, default=None)
    case_type_filter: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, default=None)
    date_from: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    date_to: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
    
    # Results
    results_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=None)
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=None)
    
    # User tracking
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=None, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, default=None)
    
    # Audit
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text('now()'))


# Database indexes for performance
Index('idx_legal_document_jurisdiction_type', LegalDocument.jurisdiction, LegalDocument.case_type)
Index('idx_legal_document_date', LegalDocument.decision_date)
Index('idx_citation_case_number', DocumentCitation.cited_case_number)
Index('idx_search_query_hash_date', SearchQuery.query_hash, SearchQuery.created_at)
