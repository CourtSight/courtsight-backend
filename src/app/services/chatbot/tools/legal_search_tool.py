"""
Legal Search Tool for ReAct Agent.
Integrates with existing CourtSight RAG system for document retrieval.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

from ...rag_service import RAGService
from ...rag.chains import create_rag_chains
from ....schemas.search import SearchRequest

logger = logging.getLogger(__name__)


class LegalSearchInput(BaseModel):
    """Input schema for legal search tool."""
    query: str = Field(description="Legal question or search query")
    max_results: int = Field(default=5, description="Maximum number of results to return")
    document_type: Optional[str] = Field(
        default=None, 
        description="Type of document to search (putusan_ma, peraturan, uu)"
    )


class LegalSearchTool(BaseTool):
    """
    Tool for searching legal documents using the existing RAG system.
    
    This tool provides the ReAct agent with access to Supreme Court decisions,
    laws, and regulations through semantic search.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "legal_search"
    description: str = """
    Search for relevant legal documents including Supreme Court decisions, laws, and regulations.
    Use this tool when you need to find specific legal precedents, court decisions, or legal provisions.
    
    Input should be a clear legal query describing what you're looking for.
    """
    args_schema: type = LegalSearchInput
    rag_service: Optional[RAGService] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_rag_service()
    
    def _initialize_rag_service(self) -> None:
        """Initialize RAG service connection."""
        try:
            rag_chains = create_rag_chains()
            self.rag_service = RAGService(rag_chains)
            logger.info("Legal search tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            self.rag_service = None
    
    def _run(self, query: str, max_results: int = 5, document_type: Optional[str] = None) -> str:
        """
        Execute legal document search.
        
        Args:
            query: The legal question or search query
            max_results: Maximum number of results to return
            document_type: Optional filter for document type
            
        Returns:
            Formatted search results as string
        """
        if not self.rag_service:
            return "Error: Legal search service is not available. Please try again later."
        
        try:
            # For now, use a simpler approach - call the retrieval service directly
            from ....services.retrieval import get_retrieval_service
            from ....services.retrieval.base import RetrievalStrategy
            
            retrieval_service = get_retrieval_service()
            
            # Use the correct method signature based on RetrievalService.retrieve
            retrieval_response = retrieval_service.retrieve(
                query=query,
                strategy=RetrievalStrategy.PARENT_CHILD,  # Use ParentChildRetriever as we added dummy documents there
                top_k=max_results,
                include_scores=True
            )
            
            # Format results for the agent
            return self._format_retrieval_results(retrieval_response)
            
        except Exception as e:
            logger.error(f"Legal search failed: {str(e)}")
            return f"No legal documents found for the given query. Error: {str(e)}"
    
    async def _arun(self, query: str, max_results: int = 5, document_type: Optional[str] = None) -> str:
        """Async version of the tool execution."""
        if not self.rag_service:
            return "Error: Legal search service is not available. Please try again later."
        
        try:
            from ....services.retrieval import get_retrieval_service
            from ....services.retrieval.base import RetrievalStrategy
            
            retrieval_service = get_retrieval_service()
            
            # Use the correct method signature based on RetrievalService.retrieve
            retrieval_response = retrieval_service.retrieve(
                query=query,
                strategy=RetrievalStrategy.PARENT_CHILD,  # Use ParentChildRetriever as we added dummy documents there
                top_k=max_results,
                filters={"document_type": document_type} if document_type else None,
                include_scores=True
            )
            
            return self._format_retrieval_results(retrieval_response)

        except Exception as e:
            logger.error(f"Async legal search failed: {str(e)}")
            return f"Error performing legal search: {str(e)}"
    
    def _format_retrieval_results(self, retrieval_response) -> str:
        """
        Format retrieval results into a readable string for the agent.
        
        Args:
            retrieval_response: List of Document objects from retrieval service
            
        Returns:
            Formatted string with search results
        """
        try:
            # The retrieval_response should be a List[Document] from the service
            if not retrieval_response:
                return "No legal documents found for the given query."
            
            documents = retrieval_response
            if not documents:
                return "No legal documents found for the given query."
            
            formatted_results = ["## Legal Search Results:\n"]
            
            for i, doc in enumerate(documents[:5], 1):  # Limit to top 5 results
                title = doc.metadata.get('title', f'Document {i}')
                formatted_results.append(f"### {i}. {title}")
                
                # Add document type if available
                doc_type = doc.metadata.get('kategori', doc.metadata.get('source', 'Unknown'))
                formatted_results.append(f"**Type:** {doc_type}")
                
                # Add relevance score if available
                if hasattr(doc, 'score') and doc.score:
                    formatted_results.append(f"**Relevance:** {doc.score:.3f}")
                
                # Add content excerpt
                content = doc.page_content
                if content:
                    # Truncate long content
                    if len(content) > 400:
                        content = content[:400] + "..."
                    formatted_results.append(f"**Content:** {content}")
                
                # Add source information
                source = doc.metadata.get('nomor_putusan', doc.metadata.get('nomor', doc.metadata.get('source', 'Unknown')))
                if source:
                    formatted_results.append(f"**Source:** {source}")
                
                formatted_results.append("")  # Empty line between results
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Error formatting retrieval results: {str(e)}")
            return f"Found legal documents but error in formatting: {str(e)}"

    def _format_search_results(self, search_response) -> str:
        """
        Format search results into a readable string for the agent.
        
        Args:
            search_response: Response from RAG service
            
        Returns:
            Formatted string with search results
        """
        try:
            if not search_response or not hasattr(search_response, 'results'):
                return "No legal documents found for the given query."
            
            results = search_response.results
            if not results:
                return "No legal documents found for the given query."
            
            formatted_results = ["## Legal Search Results:\n"]
            
            for i, result in enumerate(results[:5], 1):  # Limit to top 5 results
                formatted_results.append(f"### {i}. {getattr(result, 'title', 'Document')}")
                
                # Add document type if available
                doc_type = getattr(result, 'document_type', 'Unknown')
                formatted_results.append(f"**Type:** {doc_type}")
                
                # Add relevance score if available
                score = getattr(result, 'relevance_score', getattr(result, 'score', None))
                if score:
                    formatted_results.append(f"**Relevance:** {score:.2f}")
                
                # Add content excerpt
                content = getattr(result, 'content', getattr(result, 'excerpt', ''))
                if content:
                    # Truncate long content
                    if len(content) > 300:
                        content = content[:300] + "..."
                    formatted_results.append(f"**Content:** {content}")
                
                # Add document ID or source if available
                doc_id = getattr(result, 'document_id', getattr(result, 'id', None))
                if doc_id:
                    formatted_results.append(f"**Source ID:** {doc_id}")
                
                formatted_results.append("")  # Empty line between results
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Error formatting search results: {str(e)}")
            return f"Found legal documents but error in formatting: {str(e)}"