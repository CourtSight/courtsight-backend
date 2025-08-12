from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import httpx
import asyncio
from dataclasses import dataclass
import logging

from ..schemas.legal_search import (
    EmbeddingRequest, EmbeddingResponse,
    SimilaritySearchRequest, SimilaritySearchResponse,
    LLMAnalysisRequest, LLMAnalysisResponse,
    ValidationRequest, ValidationResponse
)

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for external services."""
    base_url: str
    timeout: int = 30
    max_retries: int = 3
    api_key: Optional[str] = None


class BaseServiceClient(ABC):
    """Base class for external service clients."""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            headers={"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.request(
                    method=method,
                    url=endpoint,
                    json=data,
                    params=params
                )
                response.raise_for_status()
                return response.json()
                
            except httpx.RequestError as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                raise


class EmbeddingServiceClient(BaseServiceClient):
    """
    Client for Embedding Service.
    Maps to F1.3-F1.4: Generate embeddings for text queries.
    """
    
    async def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embedding vector for text."""
        try:
            response_data = await self._make_request(
                method="POST",
                endpoint="/embeddings",
                data=request.model_dump()
            )
            
            return EmbeddingResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")
    
    async def generate_embeddings_batch(
        self, 
        texts: List[str], 
        model_name: str = "multilingual-e5-large"
    ) -> List[EmbeddingResponse]:
        """Generate embeddings for multiple texts."""
        try:
            response_data = await self._make_request(
                method="POST",
                endpoint="/embeddings/batch",
                data={
                    "texts": texts,
                    "model_name": model_name
                }
            )
            
            return [EmbeddingResponse(**item) for item in response_data["embeddings"]]
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate batch embeddings: {str(e)}")


class VectorSearchServiceClient(BaseServiceClient):
    """
    Client for Vector Search Service.
    Maps to F1.5-F1.6: Similarity search in vector database.
    """
    
    async def similarity_search(self, request: SimilaritySearchRequest) -> SimilaritySearchResponse:
        """Perform similarity search in vector database."""
        try:
            response_data = await self._make_request(
                method="POST",
                endpoint="/search/similarity",
                data=request.model_dump()
            )
            
            return SimilaritySearchResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise RuntimeError(f"Failed to perform similarity search: {str(e)}")
    
    async def add_document_vector(
        self,
        document_id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """Add document vector to the database."""
        try:
            await self._make_request(
                method="POST",
                endpoint="/documents",
                data={
                    "document_id": document_id,
                    "embedding": embedding,
                    "metadata": metadata
                }
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document vector: {e}")
            return False
    
    async def update_document_metadata(
        self,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update document metadata in vector database."""
        try:
            await self._make_request(
                method="PUT",
                endpoint=f"/documents/{document_id}/metadata",
                data={"metadata": metadata}
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document metadata: {e}")
            return False


class LLMServiceClient(BaseServiceClient):
    """
    Client for LLM Analysis Service.
    Maps to F1.8-F1.9: Generate summaries and analysis.
    """
    
    async def analyze_search_results(self, request: LLMAnalysisRequest) -> LLMAnalysisResponse:
        """Generate analysis and summary of search results."""
        try:
            response_data = await self._make_request(
                method="POST",
                endpoint="/analysis/search-results",
                data=request.model_dump()
            )
            
            return LLMAnalysisResponse(**response_data)
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            raise RuntimeError(f"Failed to generate LLM analysis: {str(e)}")
    
    async def extract_citations(self, text: str) -> List[str]:
        """Extract legal citations from text."""
        try:
            response_data = await self._make_request(
                method="POST",
                endpoint="/analysis/extract-citations",
                data={"text": text}
            )
            
            return response_data.get("citations", [])
            
        except Exception as e:
            logger.error(f"Citation extraction failed: {e}")
            return []
    
    async def summarize_document(
        self, 
        document_text: str, 
        max_length: int = 300
    ) -> str:
        """Generate summary for a single document."""
        try:
            response_data = await self._make_request(
                method="POST",
                endpoint="/analysis/summarize",
                data={
                    "text": document_text,
                    "max_length": max_length
                }
            )
            
            return response_data.get("summary", "")
            
        except Exception as e:
            logger.error(f"Document summarization failed: {e}")
            return ""


class ValidationServiceClient(BaseServiceClient):
    """
    Client for Citation Validation Service.
    Maps to F1.10-F1.11: Validate citations and references.
    """
    
    async def validate_citations(self, request: ValidationRequest) -> ValidationResponse:
        """Validate citations against source documents."""
        try:
            response_data = await self._make_request(
                method="POST",
                endpoint="/validation/citations",
                data=request.model_dump()
            )
            
            return ValidationResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Citation validation failed: {e}")
            raise RuntimeError(f"Failed to validate citations: {str(e)}")
    
    async def check_document_authenticity(
        self,
        document_id: str,
        content_hash: str
    ) -> Dict[str, Any]:
        """Check if document is authentic and not tampered."""
        try:
            response_data = await self._make_request(
                method="POST",
                endpoint="/validation/authenticity",
                data={
                    "document_id": document_id,
                    "content_hash": content_hash
                }
            )
            
            return response_data
            
        except Exception as e:
            logger.error(f"Document authenticity check failed: {e}")
            return {"is_authentic": False, "confidence": 0.0}


class ExportImportServiceClient(BaseServiceClient):
    """
    Client for Export/Import Service.
    For bulk operations and data exchange.
    """
    
    async def export_search_results(
        self,
        results: List[Dict[str, Any]],
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """Export search results to specified format."""
        try:
            response_data = await self._make_request(
                method="POST",
                endpoint="/export/search-results",
                data={
                    "results": results,
                    "format": format_type
                }
            )
            
            return response_data
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise RuntimeError(f"Failed to export results: {str(e)}")
    
    async def import_documents_bulk(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Import multiple documents for processing."""
        try:
            response_data = await self._make_request(
                method="POST",
                endpoint="/import/documents",
                data={"documents": documents}
            )
            
            return response_data
            
        except Exception as e:
            logger.error(f"Bulk import failed: {e}")
            raise RuntimeError(f"Failed to import documents: {str(e)}")


# Factory function to create service clients
class ServiceClientFactory:
    """Factory for creating service clients with proper configuration."""
    
    @staticmethod
    def create_embedding_client(config: ServiceConfig) -> EmbeddingServiceClient:
        return EmbeddingServiceClient(config)
    
    @staticmethod
    def create_vector_search_client(config: ServiceConfig) -> VectorSearchServiceClient:
        return VectorSearchServiceClient(config)
    
    @staticmethod
    def create_llm_client(config: ServiceConfig) -> LLMServiceClient:
        return LLMServiceClient(config)
    
    @staticmethod
    def create_validation_client(config: ServiceConfig) -> ValidationServiceClient:
        return ValidationServiceClient(config)
    
    @staticmethod
    def create_export_import_client(config: ServiceConfig) -> ExportImportServiceClient:
        return ExportImportServiceClient(config)
