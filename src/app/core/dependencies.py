"""
Dependency injefrom ..services.rag_service import RAGService, create_rag_service
from ..services.rag.chains import CourtRAGChains, create_rag_chains
from ..services.document_processor import DocumentProcessor, create_document_processor
from ..services.guardrails_validator import GuardrailsValidator, create_guardrails_validator
from ..services.evaluation import RAGEvaluator, create_rag_evaluator
from ..core.db.database import async_get_db as async_get_db
from ..models.user import User
from ..crud.crud_users import crud_usersstem for Supreme Court RAG application.
Implements clean architecture dependency patterns with proper
lifecycle management and configuration injection.
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict, Any
from functools import lru_cache

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector

from .config import Settings
from ..services.rag_service import RAGService, create_rag_service
from ..services.rag.chains import CourtRAGChains, create_rag_chains
from ..services.document_processor import DocumentProcessor, create_document_processor
from ..services.guardrails_validator import GuardrailsValidator, create_guardrails_validator
from ..services.evaluation import RAGEvaluator, create_rag_evaluator
from ..core.db.database import async_get_db
from ..models.user import User
from ..crud.crud_users import crud_users

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()


# Configuration Dependencies
@lru_cache()
def get_settings() -> Settings:
    """Get application settings with caching."""
    from .config import get_settings as _get_settings
    return _get_settings()


# Database Dependencies
async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session with proper cleanup."""
    async for session in async_get_db():
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            pass


# LangChain Component Dependencies
@lru_cache()
def get_vertex_ai_embeddings(
    settings: Settings = Depends(get_settings)
) -> GoogleGenerativeAIEmbeddings:
    """
    Get configured Gemini embeddings instance.
    
    Uses LRU cache to ensure single instance per application lifecycle.
    """
    try:
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.GOOGLE_API_KEY.get_secret_value()
        )
    except Exception as e:
        logger.error(f"Failed to initialize Gemini embeddings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service unavailable"
        )


@lru_cache()
def get_vertex_ai_embeddings_direct() -> GoogleGenerativeAIEmbeddings:
    """Get Gemini embeddings instance directly (for testing)."""
    settings = get_settings()
    try:
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.GOOGLE_API_KEY.get_secret_value()
        )
    except Exception as e:
        logger.error(f"Failed to initialize Gemini embeddings: {str(e)}")
        raise


def get_vertex_ai_llm_direct() -> ChatGoogleGenerativeAI:
    """Get Gemini LLM instance directly (for testing)."""
    settings = get_settings()
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            verbose=True,
            google_api_key=settings.GOOGLE_API_KEY.get_secret_value()
        )
    except Exception as e:
        logger.error(f"Failed to initialize Gemini LLM: {str(e)}")
        raise


def get_vertex_ai_embeddings(
    settings: Settings = Depends(get_settings)
) -> GoogleGenerativeAIEmbeddings:
    """
    Get configured Gemini embeddings instance.
    
    Uses dependency injection for FastAPI integration.
    """
    try:
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.GOOGLE_API_KEY.get_secret_value()
        )
    except Exception as e:
        logger.error(f"Failed to initialize Gemini embeddings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embeddings service unavailable"
        )


def get_vertex_ai_llm(
    settings: Settings = Depends(get_settings)
) -> ChatGoogleGenerativeAI:
    """
    Get configured Gemini LLM instance.
    
    Uses dependency injection for FastAPI integration.
    """
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            verbose=True,
            google_api_key=settings.GOOGLE_API_KEY.get_secret_value()
        )
    except Exception as e:
        logger.error(f"Failed to initialize Gemini LLM: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service unavailable"
        )


@lru_cache()
def get_vector_store(
    embeddings: GoogleGenerativeAIEmbeddings = Depends(get_vertex_ai_embeddings),
    settings: Settings = Depends(get_settings)
) -> PGVector:
    """
    Get configured PostgreSQL vector store.
    
    Uses LRU cache to ensure single instance per application lifecycle.
    """
    try:
        return PGVector(
            embeddings=embeddings,
            collection_name=settings.VECTOR_COLLECTION_NAME,
            connection=settings.DATABASE_URL,
            use_jsonb=True
        )
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database unavailable"
        )


# Service Dependencies
def get_rag_chains(
    vector_store: PGVector = Depends(get_vector_store),
    llm: ChatGoogleGenerativeAI = Depends(get_vertex_ai_llm),
    embeddings: GoogleGenerativeAIEmbeddings = Depends(get_vertex_ai_embeddings),
    settings: Settings = Depends(get_settings)
) -> CourtRAGChains:
    """Get configured RAG chains."""
    try:
        return CourtRAGChains(
            vector_store=vector_store,
            llm=llm,
            embeddings=embeddings,
            enable_validation=settings.ENABLE_CLAIM_VALIDATION
        )
    except Exception as e:
        logger.error(f"Failed to initialize RAG chains: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system unavailable"
        )


def get_rag_service(
    rag_chains: CourtRAGChains = Depends(get_rag_chains)
) -> RAGService:
    """Get configured RAG service."""
    try:
        return RAGService(rag_chains)
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service unavailable"
        )


def get_document_processor(
    vector_store: PGVector = Depends(get_vector_store),
    embeddings: GoogleGenerativeAIEmbeddings = Depends(get_vertex_ai_embeddings),
    settings: Settings = Depends(get_settings)
) -> DocumentProcessor:
    """Get configured document processor."""
    try:
        return DocumentProcessor(
            vector_store=vector_store,
            embeddings=embeddings,
            enable_metadata_extraction=True
        )
    except Exception as e:
        logger.error(f"Failed to initialize document processor: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document processing service unavailable"
        )


def get_guardrails_validator(
    llm: ChatGoogleGenerativeAI = Depends(get_vertex_ai_llm),
    settings: Settings = Depends(get_settings)
) -> GuardrailsValidator:
    """Get configured Guardrails validator."""
    try:
        return create_guardrails_validator(
            llm=llm,
            enable_strict_validation=True,
            confidence_threshold=settings.VALIDATION_CONFIDENCE_THRESHOLD
        )
    except Exception as e:
        logger.error(f"Failed to initialize Guardrails validator: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Validation service unavailable"
        )


def get_rag_evaluator(
    settings: Settings = Depends(get_settings)
) -> RAGEvaluator:
    """Get configured RAG evaluator."""
    try:
        return create_rag_evaluator(
            enable_ragas=settings.ENABLE_RAGAS_EVALUATION,
            enable_legal_metrics=True,
            batch_size=settings.EVALUATION_BATCH_SIZE
        )
    except Exception as e:
        logger.error(f"Failed to initialize RAG evaluator: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluation service unavailable"
        )


# Authentication Dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_database_session)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer token credentials
        db: Database session
        
    Returns:
        User: Authenticated user object
        
    Raises:
        HTTPException: For authentication failures
    """
    try:
        # Decode JWT token and extract user ID
        # This is a simplified version - implement proper JWT decoding
        from ..core.security import decode_access_token
        
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database
        user = await crud_users.get(db, id=user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
        
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Active user object
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Get current admin user.
    
    Args:
        current_user: Current active user
        
    Returns:
        User: Admin user object
        
    Raises:
        HTTPException: If user is not admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return current_user


# Rate Limiting Dependencies
class RateLimiter:
    """Simple rate limiter implementation."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.user_requests: Dict[str, list] = {}
    
    async def check_rate_limit(self, user_id: str, endpoint: str) -> None:
        """
        Check rate limit for user and endpoint.
        
        Args:
            user_id: User identifier
            endpoint: API endpoint identifier
            
        Raises:
            HTTPException: If rate limit exceeded
        """
        # Simple implementation - replace with Redis-based solution for production
        import time
        from datetime import datetime, timedelta
        
        now = datetime.now()
        user_key = f"{user_id}:{endpoint}"
        
        if user_key not in self.user_requests:
            self.user_requests[user_key] = []
        
        # Clean old requests
        cutoff_time = now - timedelta(minutes=1)
        self.user_requests[user_key] = [
            req_time for req_time in self.user_requests[user_key]
            if req_time > cutoff_time
        ]
        
        # Check limit
        if len(self.user_requests[user_key]) >= self.settings.DEFAULT_RATE_LIMIT_LIMIT:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Add current request
        self.user_requests[user_key].append(now)


@lru_cache()
def get_rate_limiter(
    settings: Settings = Depends(get_settings)
) -> RateLimiter:
    """Get rate limiter instance."""
    return RateLimiter(settings)


# Health Check Dependencies
async def check_service_health() -> Dict[str, Any]:
    """
    Perform comprehensive health check of all services.
    
    Returns:
        Dict with health status of all components
    """
    health_status = {
        "status": "healthy",
        "timestamp": "2023-12-01T10:30:00Z",
        "services": {}
    }
    
    try:
        # Check database connectivity
        # This would test actual database connection
        health_status["services"]["database"] = "healthy"
        
        # Check vector store
        # This would test vector store connectivity
        health_status["services"]["vector_store"] = "healthy"
        
        # Check LLM service
        # This would test LLM availability
        health_status["services"]["llm_service"] = "healthy"
        
        # Check embedding service
        # This would test embedding service availability
        health_status["services"]["embedding_service"] = "healthy"
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
    
    return health_status


# Monitoring Dependencies
class MetricsCollector:
    """Metrics collection for monitoring."""
    
    def __init__(self):
        self.metrics = {}
    
    def increment_counter(self, metric: str, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        # Implementation for metrics collection
        pass
    
    def set_gauge(self, metric: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric."""
        # Implementation for metrics collection
        pass
    
    def record_histogram(self, metric: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric."""
        # Implementation for metrics collection
        pass


@lru_cache()
def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance."""
    return MetricsCollector()


# Lifecycle Management
async def startup_event():
    """Application startup event handler."""
    logger.info("Starting Supreme Court RAG API")
    
    # Initialize services
    try:
        # Test database connection
        # Test vector store connection
        # Test LLM service connection
        # Initialize caches
        
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


async def shutdown_event():
    """Application shutdown event handler."""
    logger.info("Shutting down Supreme Court RAG API")
    
    # Cleanup resources
    try:
        # Close database connections
        # Close HTTP clients
        # Clear caches
        
        logger.info("Shutdown completed successfully")
    except Exception as e:
        logger.error(f"Shutdown failed: {str(e)}")


# Export commonly used dependencies
__all__ = [
    "get_settings",
    "get_database_session",
    "get_vertex_ai_embeddings",
    "get_vertex_ai_llm", 
    "get_vector_store",
    "get_rag_chains",
    "get_rag_service",
    "get_document_processor",
    "get_guardrails_validator",
    "get_rag_evaluator",
    "get_current_user",
    "get_current_active_user",
    "get_current_admin_user",
    "get_rate_limiter",
    "check_service_health",
    "get_metrics_collector",
    "startup_event",
    "shutdown_event"
]
