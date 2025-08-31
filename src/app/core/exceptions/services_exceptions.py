"""
Custom exceptions for Supreme Court RAG system.
Implements domain-specific error handling with proper HTTP status mapping.
"""

from typing import Optional, Dict, Any


class RAGSystemException(Exception):
    """Base exception for RAG system errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class RAGServiceError(RAGSystemException):
    """Raised when RAG service operations fail."""
    pass


class ValidationError(RAGSystemException):
    """Raised when input validation fails."""
    pass


class DocumentProcessingError(RAGSystemException):
    """Raised when document processing fails."""
    pass


class EmbeddingError(RAGSystemException):
    """Raised when embedding operations fail."""
    pass


class LLMError(RAGSystemException):
    """Raised when LLM operations fail."""
    pass


class VectorStoreError(RAGSystemException):
    """Raised when vector store operations fail."""
    pass


class GuardrailsError(RAGSystemException):
    """Raised when guardrails validation fails."""
    pass


class EvaluationError(RAGSystemException):
    """Raised when evaluation operations fail."""
    pass


class ConfigurationError(RAGSystemException):
    """Raised when configuration is invalid."""
    pass


class AuthenticationError(RAGSystemException):
    """Raised when authentication fails."""
    pass


class AuthorizationError(RAGSystemException):
    """Raised when authorization fails."""
    pass


class RateLimitError(RAGSystemException):
    """Raised when rate limits are exceeded."""
    pass


class TimeoutError(RAGSystemException):
    """Raised when operations timeout."""
    pass


class ResourceNotFoundError(RAGSystemException):
    """Raised when requested resources are not found."""
    pass


class ServiceUnavailableError(RAGSystemException):
    """Raised when external services are unavailable."""
    pass
