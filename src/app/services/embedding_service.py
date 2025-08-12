"""
Embedding Service Client for CourtSight Feature 1.
Handles text embedding generation using external embedding APIs.
"""
import httpx
import asyncio
from typing import List, Optional
from pydantic import BaseModel

from app.core.config import settings
from app.core.logger import logging

logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    text: str
    model: str = "all-MiniLM-L6-v2"  # Default embedding model


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    embedding: List[float]
    model: str
    text_length: int


class EmbeddingServiceClient:
    """
    Client for external embedding service.
    Maps to F1.2: Convert text to embeddings for semantic search.
    """
    
    def __init__(self):
        self.base_url = getattr(settings, 'EMBEDDING_SERVICE_URL', 'http://localhost:8001')
        self.api_key = getattr(settings, 'EMBEDDING_API_KEY', None)
        self.timeout = 30.0
        
    async def generate_embedding(
        self, 
        text: str, 
        model: str = "all-MiniLM-L6-v2"
    ) -> List[float]:
        """
        Generate embedding vector for the given text.
        
        Args:
            text: Input text to embed
            model: Embedding model to use
            
        Returns:
            List of float values representing the embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            # Prepare request
            request_data = EmbeddingRequest(text=text, model=model)
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make API call
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    json=request_data.model_dump(),
                    headers=headers
                )
                
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                
                if "embedding" in result:
                    embedding = result["embedding"]
                    
                    # Validate embedding dimension (should be 384 for all-MiniLM-L6-v2)
                    if len(embedding) != 384:
                        logger.warning(
                            f"Unexpected embedding dimension: {len(embedding)}, expected 384"
                        )
                    
                    logger.info(f"Generated embedding for text of length {len(text)}")
                    return embedding
                else:
                    raise ValueError("Invalid response format from embedding service")
                    
        except httpx.TimeoutException:
            logger.error("Embedding service timeout")
            raise Exception("Embedding service is not responding")
        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding service HTTP error: {e.response.status_code}")
            raise Exception(f"Embedding service error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise Exception(f"Failed to generate embedding: {str(e)}")
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str], 
        model: str = "all-MiniLM-L6-v2"
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
            
        Returns:
            List of embedding vectors
        """
        try:
            # Process texts in batches to avoid overwhelming the service
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Generate embeddings for this batch
                batch_tasks = [
                    self.generate_embedding(text, model) 
                    for text in batch
                ]
                
                batch_embeddings = await asyncio.gather(*batch_tasks)
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")
    
    async def health_check(self) -> bool:
        """
        Check if the embedding service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Embedding service health check failed: {e}")
            return False


# Global instance
embedding_service = EmbeddingServiceClient()
