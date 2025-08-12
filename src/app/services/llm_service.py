"""
LLM Service Client for CourtSight Feature 1.
Handles query processing and result enhancement using Large Language Models.
"""
import httpx
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from app.core.config import settings
from app.core.logger import logging

logger = logging.getLogger(__name__)
from app.schemas.legal_search import LegalDocumentRead


class QueryProcessingRequest(BaseModel):
    """Request model for query processing."""
    query: str
    context: Optional[str] = None
    max_tokens: int = 150


class QueryProcessingResponse(BaseModel):
    """Response model for query processing."""
    processed_query: str
    extracted_entities: List[str]
    intent: str


class ResultEnhancementRequest(BaseModel):
    """Request model for result enhancement."""
    query: str
    results: List[Dict[str, Any]]
    max_tokens: int = 500


class ResultEnhancementResponse(BaseModel):
    """Response model for result enhancement."""
    enhanced_results: List[Dict[str, Any]]
    summary: str


class LLMServiceClient:
    """
    Client for external LLM service.
    Maps to F1.1: Natural language query processing and result enhancement.
    """
    
    def __init__(self):
        self.base_url = getattr(settings, 'LLM_SERVICE_URL', 'http://localhost:8002')
        self.api_key = getattr(settings, 'LLM_API_KEY', None)
        self.timeout = 60.0  # LLM calls can take longer
        
    async def process_query(
        self, 
        query: str, 
        context: Optional[str] = None
    ) -> QueryProcessingResponse:
        """
        Process natural language query to extract intent and entities.
        
        Args:
            query: Raw user query
            context: Optional context for query understanding
            
        Returns:
            Processed query with extracted entities and intent
            
        Raises:
            Exception: If query processing fails
        """
        try:
            # Prepare request
            request_data = QueryProcessingRequest(
                query=query,
                context=context,
                max_tokens=150
            )
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make API call
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/process-query",
                    json=request_data.model_dump(),
                    headers=headers
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Processed query: '{query[:50]}...'")
                return QueryProcessingResponse(**result)
                
        except httpx.TimeoutException:
            logger.error("LLM service timeout during query processing")
            # Fallback: return original query
            return QueryProcessingResponse(
                processed_query=query,
                extracted_entities=[],
                intent="search"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM service HTTP error: {e.response.status_code}")
            # Fallback: return original query
            return QueryProcessingResponse(
                processed_query=query,
                extracted_entities=[],
                intent="search"
            )
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            # Fallback: return original query
            return QueryProcessingResponse(
                processed_query=query,
                extracted_entities=[],
                intent="search"
            )
    
    async def enhance_results(
        self, 
        query: str, 
        results: List[LegalDocumentRead]
    ) -> ResultEnhancementResponse:
        """
        Enhance search results with AI-generated insights and summaries.
        
        Args:
            query: Original search query
            results: Search results to enhance
            
        Returns:
            Enhanced results with AI insights
        """
        try:
            # Convert results to dict format for LLM processing
            results_data = []
            for result in results:
                results_data.append({
                    "id": result.id,
                    "title": result.title,
                    "summary": result.summary or "",
                    "case_number": result.case_number,
                    "court_name": result.court_name,
                    "relevance_score": result.relevance_score,
                    "matched_snippets": result.matched_snippets
                })
            
            # Prepare request
            request_data = ResultEnhancementRequest(
                query=query,
                results=results_data,
                max_tokens=500
            )
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make API call
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/enhance-results",
                    json=request_data.model_dump(),
                    headers=headers
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Enhanced {len(results)} search results")
                return ResultEnhancementResponse(**result)
                
        except httpx.TimeoutException:
            logger.error("LLM service timeout during result enhancement")
            # Fallback: return original results
            return ResultEnhancementResponse(
                enhanced_results=results_data,
                summary=f"Found {len(results)} relevant documents for your query."
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM service HTTP error: {e.response.status_code}")
            # Fallback: return original results
            return ResultEnhancementResponse(
                enhanced_results=results_data,
                summary=f"Found {len(results)} relevant documents for your query."
            )
        except Exception as e:
            logger.error(f"Result enhancement failed: {e}")
            # Fallback: return original results
            return ResultEnhancementResponse(
                enhanced_results=results_data,
                summary=f"Found {len(results)} relevant documents for your query."
            )
    
    async def generate_summary(
        self, 
        documents: List[LegalDocumentRead], 
        query: str
    ) -> str:
        """
        Generate a summary of multiple legal documents in relation to a query.
        
        Args:
            documents: Legal documents to summarize
            query: User query for context
            
        Returns:
            Generated summary text
        """
        try:
            # Prepare document content for summarization
            doc_content = []
            for doc in documents:
                content = f"Case: {doc.case_number}\nTitle: {doc.title}\n"
                if doc.summary:
                    content += f"Summary: {doc.summary}\n"
                doc_content.append(content)
            
            request_data = {
                "query": query,
                "documents": doc_content,
                "max_tokens": 300
            }
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make API call
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/generate-summary",
                    json=request_data,
                    headers=headers
                )
                
                response.raise_for_status()
                result = response.json()
                
                summary = result.get("summary", "Summary not available")
                logger.info(f"Generated summary for {len(documents)} documents")
                return summary
                
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Found {len(documents)} relevant legal documents related to your query."
    
    async def health_check(self) -> bool:
        """
        Check if the LLM service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"LLM service health check failed: {e}")
            return False


# Global instance
llm_service = LLMServiceClient()
