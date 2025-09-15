"""
Parent-Child Retrieval Implementation

This module implements the parent-child chunking strategy for document retrieval.
Uses PostgreSQL-based custom document store for production-ready parent document storage.
"""

import logging
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

from ...utils.parent_doc_store import PostgreSQLParentDocStore
from .base import BaseRetriever,RetrievalRequest,RetrievalStrategy,RetrievalResponse, RetrievedDocument,DocumentResult

logger = logging.getLogger(__name__)


class ParentChildRetriever(BaseRetriever):
    """
    Parent-Child Retrieval Strategy
    
    Uses custom PostgreSQL-based document store for parent document storage
    and PGVector for child chunk embeddings and retrieval.
    """
    
    def __init__(
        self,
        vector_store: PGVector,
        embeddings_model,
        collection_name: str = "parent_documents",
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 20,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 200,
    ):
        """
        Initialize Parent-Child Retrieval Strategy
        
        Args:
            vector_store: PGVector instance for child chunk storage
            embeddings_model: Embeddings model for vector generation
            collection_name: Collection name for parent documents
            child_chunk_size: Size of child chunks for embedding
            child_chunk_overlap: Overlap between child chunks
            parent_chunk_size: Size of parent chunks for storage
            parent_chunk_overlap: Overlap between parent chunks
        """
        self.vector_store = vector_store
        self.embeddings_model = embeddings_model
        self.collection_name = collection_name
        
        # Initialize custom PostgreSQL document store
        self.parent_doc_store = PostgreSQLParentDocStore(
            collection_name=collection_name
        )
        
        # Text splitters for different chunk sizes
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
        )
        
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )
        
        # Initialize retriever
        self._initialize_retriever()
    
    def _initialize_retriever(self) -> None:
        """Initialize the parent document retriever with custom store."""
        try:
            self.retriever = ParentDocumentRetriever(
                vectorstore=self.vector_store,
                docstore=self.parent_doc_store,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
                search_kwargs={"k": 10},  # Retrieve more child chunks initially
            )
            logger.info(f"Parent-child retriever initialized for collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize parent-child retriever: {e}")
            raise

    def retrieve(
        self,
        request,  # We'll accept dict or object for flexibility
        **kwargs
    ) -> RetrievalResponse:
        """
        Retrieve documents using parent-child strategy
        
        Args:
            request: Retrieval request containing query and parameters
            **kwargs: Additional parameters
            
        Returns:
            RetrievalResponse with retrieved parent documents
        """
        try:
            # Handle both dict and object requests
            if hasattr(request, 'query'):
                query = request.query
                top_k = getattr(request, 'top_k', 5)
                min_score = getattr(request, 'min_score', 0.0)
            else:
                query = request.get('query', '')
                top_k = request.get('top_k', 5)
                min_score = request.get('min_score', 0.0)
            
            logger.info(f"Parent-child retrieval for query: {query}")
            
            # Configure search parameters
            search_kwargs = {
                "k": top_k,
                "score_threshold": min_score,
            }
            
            # Update retriever search kwargs
            self.retriever.search_kwargs.update(search_kwargs)
            
            # Perform retrieval - this returns parent documents
            documents = self._async_retrieve(query)

            # Convert to response format
            document_results = [
                RetrievedDocument(
                    id=f"parent_doc_{i}",
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=doc.metadata.get("score", 0.0),
                    source=doc.metadata.get("source", "unknown"),
                )
                for i, doc in enumerate(documents)
            ]
            
            logger.info(f"Retrieved {len(document_results)} parent documents")
            
            return RetrievalResponse(
                query=query,
                strategy_used=RetrievalStrategy.PARENT_CHILD,
                documents=document_results,
                total_found=len(document_results),
                processing_time=0.0,  # We'll add timing later
                metadata={
                    "collection_name": self.collection_name,
                    "child_chunk_size": self.child_splitter._chunk_size,
                    "parent_chunk_size": self.parent_splitter._chunk_size,
                    "search_kwargs": search_kwargs,
                }
            )
            
        except Exception as e:
            logger.error(f"Parent-child retrieval failed: {e}")
            raise
    
    def _async_retrieve(self, query: str) -> List[Document]:
        """
        Sync wrapper for retriever invoke
        
        Args:
            query: Search query
            
        Returns:
            List of retrieved parent documents
        """
        try:
            # Use the new invoke method instead of deprecated get_relevant_documents
            documents = self.retriever.invoke(query)
            return documents
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            raise

    def add_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Add documents to the parent-child retrieval system
        
        Args:
            documents: List of documents to add
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with addition results
        """
        try:
            logger.info(f"Adding {len(documents)} documents to parent-child system")
            
            # Add documents to retriever (handles both parent and child storage)
            self.retriever.add_documents(documents)
            
            # Get statistics from document store
            doc_count =  self._get_document_count()
            
            logger.info(f"Successfully added documents. Total documents: {doc_count}")
            
            return {
                "added_documents": len(documents),
                "total_documents": doc_count,
                "collection_name": self.collection_name,
            }
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def _get_document_count(self) -> int:
        """Get total document count from parent document store."""
        try:
            return  self.parent_doc_store.get_document_count()
        except Exception as e:
            logger.warning(f"Could not get document count: {e}")
            return 0
    
    def clear_collection(self) -> Dict[str, Any]:
        """
        Clear all documents from the collection
        
        Returns:
            Dictionary with clearing results
        """
        try:
            logger.info(f"Clearing parent-child collection: {self.collection_name}")
            
            # Clear parent document store
            self.parent_doc_store.clear_collection()

            # Note: Vector store clearing depends on PGVector implementation
            # You may need to implement vector store clearing separately
            
            return {
                "status": "success",
                "message": f"Cleared collection: {self.collection_name}",
                "collection_name": self.collection_name,
            }
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the strategy configuration
        
        Returns:
            Dictionary with strategy information
        """
        return {
            "strategy_name": "parent_child",
            "collection_name": self.collection_name,
            "child_chunk_size": self.child_splitter._chunk_size,
            "child_chunk_overlap": self.child_splitter._chunk_overlap,
            "parent_chunk_size": self.parent_splitter._chunk_size,
            "parent_chunk_overlap": self.parent_splitter._chunk_overlap,
            "document_store_type": "postgresql_custom",
            "vector_store_type": "pgvector",
        }
        
    def supports_filters(self) -> bool:
        """Parent-child retriever has limited filter support."""
        return False  # Filters would need to be applied post-retrieval

    def get_strategy(self) -> RetrievalStrategy:
        """Get the strategy this retriever implements."""
        return RetrievalStrategy.PARENT_CHILD

from ...utils.parent_doc_store import PostgreSQLParentDocStore
from .base import BaseRetriever,RetrievalRequest,RetrievalStrategy,RetrievalResponse, RetrievedDocument,DocumentResult

logger = logging.getLogger(__name__)


class ParentChildRetriever(BaseRetriever):
    """
    Parent-Child Retrieval Strategy
    
    Uses custom PostgreSQL-based document store for parent document storage
    and PGVector for child chunk embeddings and retrieval.
    """
    
    def __init__(
        self,
        vector_store: PGVector,
        embeddings_model,
        collection_name: str = "parent_documents",
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 20,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 200,
    ):
        """
        Initialize Parent-Child Retrieval Strategy
        
        Args:
            vector_store: PGVector instance for child chunk storage
            embeddings_model: Embeddings model for vector generation
            collection_name: Collection name for parent documents
            child_chunk_size: Size of child chunks for embedding
            child_chunk_overlap: Overlap between child chunks
            parent_chunk_size: Size of parent chunks for storage
            parent_chunk_overlap: Overlap between parent chunks
        """
        self.vector_store = vector_store
        self.embeddings_model = embeddings_model
        self.collection_name = collection_name
        
        # Initialize custom PostgreSQL document store
        self.parent_doc_store = PostgreSQLParentDocStore(
            collection_name=collection_name
        )
        
        # Text splitters for different chunk sizes
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
        )
        
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )
        
        # Initialize retriever
        self._initialize_retriever()
    
    def _initialize_retriever(self) -> None:
        """Initialize the parent document retriever with custom store."""
        try:
            self.retriever = ParentDocumentRetriever(
                vectorstore=self.vector_store,
                docstore=self.parent_doc_store,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
                search_kwargs={"k": 10},  # Retrieve more child chunks initially
            )
            logger.info(f"Parent-child retriever initialized for collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize parent-child retriever: {e}")
            raise

    def retrieve(
        self,
        request,  # We'll accept dict or object for flexibility
        **kwargs
    ) -> RetrievalResponse:
        """
        Retrieve documents using parent-child strategy
        
        Args:
            request: Retrieval request containing query and parameters
            **kwargs: Additional parameters
            
        Returns:
            RetrievalResponse with retrieved parent documents
        """
        try:
            # Handle both dict and object requests
            if hasattr(request, 'query'):
                query = request.query
                top_k = getattr(request, 'top_k', 5)
                min_score = getattr(request, 'min_score', 0.0)
            else:
                query = request.get('query', '')
                top_k = request.get('top_k', 5)
                min_score = request.get('min_score', 0.0)
            
            logger.info(f"Parent-child retrieval for query: {query}")
            
            # Configure search parameters
            search_kwargs = {
                "k": top_k,
                "score_threshold": min_score,
            }
            
            # Update retriever search kwargs
            self.retriever.search_kwargs.update(search_kwargs)
            
            # Perform retrieval - this returns parent documents
            documents = self._async_retrieve(query)

            # Convert to response format
            document_results = [
                RetrievedDocument(
                    id=f"parent_doc_{i}",
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=doc.metadata.get("score", 0.0),
                    source=doc.metadata.get("source", "unknown"),
                )
                for i, doc in enumerate(documents)
            ]
            
            logger.info(f"Retrieved {len(document_results)} parent documents")
            
            return RetrievalResponse(
                query=query,
                strategy_used=RetrievalStrategy.PARENT_CHILD,
                documents=document_results,
                total_found=len(document_results),
                processing_time=0.0,  # We'll add timing later
                metadata={
                    "collection_name": self.collection_name,
                    "child_chunk_size": self.child_splitter._chunk_size,
                    "parent_chunk_size": self.parent_splitter._chunk_size,
                    "search_kwargs": search_kwargs,
                }
            )
            
        except Exception as e:
            logger.error(f"Parent-child retrieval failed: {e}")
            raise
    
    def _async_retrieve(self, query: str) -> List[Document]:
        """
        Async wrapper for retriever get_relevant_documents
        
        Args:
            query: Search query
            
        Returns:
            List of retrieved parent documents
        """
        try:
            # Note: LangChain retrievers are typically sync
            # For true async, consider implementing async versions
            documents = self.retriever.invoke(query)
            return documents
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            raise

    def add_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Add documents to the parent-child retrieval system
        
        Args:
            documents: List of documents to add
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with addition results
        """
        try:
            logger.info(f"Adding {len(documents)} documents to parent-child system")
            
            # Add documents to retriever (handles both parent and child storage)
            self.retriever.add_documents(documents)
            
            # Get statistics from document store
            doc_count =  self._get_document_count()
            
            logger.info(f"Successfully added documents. Total documents: {doc_count}")
            
            return {
                "added_documents": len(documents),
                "total_documents": doc_count,
                "collection_name": self.collection_name,
            }
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def _get_document_count(self) -> int:
        """Get total document count from parent document store."""
        try:
            return  self.parent_doc_store.get_document_count()
        except Exception as e:
            logger.warning(f"Could not get document count: {e}")
            return 0
    
    def clear_collection(self) -> Dict[str, Any]:
        """
        Clear all documents from the collection
        
        Returns:
            Dictionary with clearing results
        """
        try:
            logger.info(f"Clearing parent-child collection: {self.collection_name}")
            
            # Clear parent document store
            self.parent_doc_store.clear_collection()

            # Note: Vector store clearing depends on PGVector implementation
            # You may need to implement vector store clearing separately
            
            return {
                "status": "success",
                "message": f"Cleared collection: {self.collection_name}",
                "collection_name": self.collection_name,
            }
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the strategy configuration
        
        Returns:
            Dictionary with strategy information
        """
        return {
            "strategy_name": "parent_child",
            "collection_name": self.collection_name,
            "child_chunk_size": self.child_splitter._chunk_size,
            "child_chunk_overlap": self.child_splitter._chunk_overlap,
            "parent_chunk_size": self.parent_splitter._chunk_size,
            "parent_chunk_overlap": self.parent_splitter._chunk_overlap,
            "document_store_type": "postgresql_custom",
            "vector_store_type": "pgvector",
        }
        
    def supports_filters(self) -> bool:
        """Parent-child retriever has limited filter support."""
        return False  # Filters would need to be applied post-retrieval

    def get_strategy(self) -> RetrievalStrategy:
        """Get the strategy this retriever implements."""
        return RetrievalStrategy.PARENT_CHILD
