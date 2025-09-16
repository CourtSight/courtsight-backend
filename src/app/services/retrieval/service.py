"""
Main retrieval service orchestrator.
Provides unified interface for all retrieval strategies.
"""

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

from src.app.core.database import get_vector_store

from .base import BaseRetriever, RetrievalRequest, RetrievalStrategy
from .hybrid import HybridRetriever
from .parent_child import ParentChildRetriever
from .vector_search import VectorSearchRetriever
from ...core.config import get_settings
settings = get_settings()

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Unified retrieval service supporting multiple strategies.
    Implements driven architecture pattern with singleton database connections.
    """

    def __init__(self, vector_store: PGVector | None = None, use_redis_store: bool = True):
        # Use singleton vector store if not provided
        self.vector_store = vector_store if vector_store is not None else get_vector_store()
        self.use_redis_store = use_redis_store
        self._retrievers: Dict[RetrievalStrategy, BaseRetriever] = {}
        self._default_strategy = RetrievalStrategy.VECTOR_SEARCH

        # Initialize all retrievers
        self._initialize_retrievers()

    def _initialize_retrievers(self):
        """Initialize all available retrievers."""
        try:
            # Vector search retriever
            self._retrievers[RetrievalStrategy.VECTOR_SEARCH] = VectorSearchRetriever(
                self.vector_store
            )

            # Parent-child retriever
            self._retrievers[RetrievalStrategy.PARENT_CHILD] = ParentChildRetriever(
                vector_store=self.vector_store,
                embeddings_model=self.vector_store.embeddings,
                collection_name=settings.VECTOR_COLLECTION_NAME
            )

            # Hybrid retriever
            self._retrievers[RetrievalStrategy.HYBRID] = HybridRetriever(
                self.vector_store,
                use_redis_store=self.use_redis_store
            )

            logger.info(f"Initialized {len(self._retrievers)} retrievers")

        except Exception as e:
            logger.error(f"Failed to initialize retrievers: {e}")
            raise

    def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = None,
        top_k: int = 5,
        filters: Dict[str, Any] | None = None,
        include_scores: bool = True,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve documents using specified strategy.

        Args:
            query: Search query
            strategy: Retrieval strategy to use
            top_k: Number of documents to retrieve
            filters: Optional filters to apply
            include_scores: Whether to include similarity scores
            **kwargs: Additional strategy-specific parameters

        Returns:
            List of retrieved documents
        """
        # Use default strategy if none specified
        if strategy is None:
            strategy = self._default_strategy

        # Get retriever for strategy
        retriever = self._get_retriever(strategy)
        if not retriever:
            logger.error(f"No retriever available for strategy: {strategy}")
            return []

        # Create retrieval request
        request = RetrievalRequest(
            query=query,
            top_k=top_k,
            filters=filters,
            include_scores=include_scores,
            **kwargs
        )

        # Execute retrieval
        try:
            response = retriever.retrieve(request)
            
            # Handle different response types
            if hasattr(response, 'documents'):
                # RetrievalResponse format (from parent-child)
                documents = []
                for doc_result in response.documents:
                    # Convert RetrievedDocument to Document
                    from langchain_core.documents import Document
                    doc = Document(
                        page_content=doc_result.content,
                        metadata=doc_result.metadata or {}
                    )
                    doc.metadata["service_strategy"] = strategy.value
                    doc.metadata["service_query"] = query
                    documents.append(doc)
            else:
                # List[Document] format (from vector search)
                documents = response
                for doc in documents:
                    doc.metadata["service_strategy"] = strategy.value
                    doc.metadata["service_query"] = query

            logger.info(
                f"Retrieved {len(documents)} documents using {strategy.value} strategy"
            )
            return documents

        except Exception as e:
            logger.error(f"Retrieval failed with {strategy.value}: {e}")
            return []

    async def multi_strategy_retrieve(
        self,
        query: str,
        strategies: List[RetrievalStrategy],
        top_k: int = 5,
        merge_results: bool = True,
        **kwargs
    ) -> Dict[RetrievalStrategy, List[Document]]:
        """
        Retrieve documents using multiple strategies.

        Args:
            query: Search query
            strategies: List of strategies to use
            top_k: Number of documents per strategy
            merge_results: Whether to merge and deduplicate results
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping strategies to their results
        """
        results = {}

        for strategy in strategies:
            try:
                docs = self.retrieve(query, strategy, top_k, **kwargs)
                results[strategy] = docs
            except Exception as e:
                logger.error(f"Strategy {strategy.value} failed: {e}")
                results[strategy] = []

        if merge_results and len(strategies) > 1:
            merged_docs = self._merge_strategy_results(results, top_k)
            results["merged"] = merged_docs

        return results

    def _merge_strategy_results(
        self,
        results: Dict[RetrievalStrategy, List[Document]],
        top_k: int
    ) -> List[Document]:
        """Merge results from multiple strategies."""
        all_docs = []
        seen_content = set()

        # Collect unique documents
        for strategy, docs in results.items():
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)

                    # Add merge metadata
                    doc.metadata["merged_from"] = [strategy.value]
                    all_docs.append(doc)
                else:
                    # Find existing doc and update merge metadata
                    for existing_doc in all_docs:
                        if hash(existing_doc.page_content) == content_hash:
                            if "merged_from" not in existing_doc.metadata:
                                existing_doc.metadata["merged_from"] = []
                            existing_doc.metadata["merged_from"].append(strategy.value)
                            break

        # Sort by highest score if available
        def get_score(doc):
            return (
                doc.metadata.get("hybrid_score", 0) or
                doc.metadata.get("similarity_score", 0) or
                0.5
            )

        all_docs.sort(key=get_score, reverse=True)
        return all_docs[:top_k]

    def _get_retriever(self, strategy: RetrievalStrategy) -> BaseRetriever | None:
        """Get retriever for specified strategy."""
        return self._retrievers.get(strategy)

    def get_available_strategies(self) -> List[RetrievalStrategy]:
        """Get list of available strategies."""
        return list(self._retrievers.keys())

    def set_default_strategy(self, strategy: RetrievalStrategy):
        """Set default retrieval strategy."""
        if strategy in self._retrievers:
            self._default_strategy = strategy
            logger.info(f"Default strategy set to: {strategy.value}")
        else:
            logger.error(f"Strategy not available: {strategy.value}")

    async def add_documents(
        self,
        documents: List[Document],
        strategy: RetrievalStrategy = RetrievalStrategy.PARENT_CHILD
    ):
        """
        Add documents to specified retriever.
        Defaults to parent-child as it includes vector store.
        """
        retriever = self._get_retriever(strategy)
        if retriever:
            retriever.add_documents(documents)
            logger.info(f"Added {len(documents)} documents via {strategy.value}")
        else:
            logger.error(f"Cannot add documents: {strategy.value} not available")

    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        status = {
            "default_strategy": self._default_strategy.value,
            "available_strategies": [s.value for s in self._retrievers.keys()],
            "retrievers": {}
        }

        for strategy, retriever in self._retrievers.items():
            try:
                if hasattr(retriever, 'get_retriever_status'):
                    status["retrievers"][strategy.value] = retriever.get_retriever_status()
                else:
                    status["retrievers"][strategy.value] = {
                        "available": True,
                        "type": type(retriever).__name__
                    }
            except Exception as e:
                status["retrievers"][strategy.value] = {
                    "available": False,
                    "error": str(e)
                }

        return status


# Factory function for easy service creation
@asynccontextmanager
async def create_retrieval_service(
    connection_string: str,
    embedding_model: GoogleGenerativeAIEmbeddings,
    collection_name: str = "putusan_child_chunks",
    use_redis_store: bool = True
):
    """
    Factory function to create retrieval service with proper setup and cleanup.
    """
    vector_store = None
    service = None

    try:
        # Initialize vector store
        vector_store = PGVector(
            embeddings=embedding_model,
            connection=connection_string,
            collection_name=collection_name,
        )

        # Create service
        service = RetrievalService(vector_store, use_redis_store)

        logger.info(f"Retrieval service created with collection: {collection_name}")
        yield service

    except Exception as e:
        logger.error(f"Failed to create retrieval service: {e}")
        raise
    finally:
        # Cleanup if needed
        if vector_store:
            try:
                await vector_store.aclose()
            except:
                pass


# Global singleton instance
_retrieval_service_instance: RetrievalService | None = None


@lru_cache
def get_retrieval_service() -> RetrievalService:
    """
    Get singleton retrieval service instance.
    Uses singleton pattern to avoid repeatedly creating instances and database connections.
    """
    global _retrieval_service_instance

    if _retrieval_service_instance is None:
        logger.info("Creating new RetrievalService singleton instance with database connection manager")

        # Use singleton database connection manager
        vector_store = get_vector_store()

        # Create singleton instance
        _retrieval_service_instance = RetrievalService(
            vector_store=vector_store,
            use_redis_store=True
        )

        logger.info("RetrievalService singleton instance created successfully")
    else:
        logger.debug("Returning existing RetrievalService singleton instance")

    return _retrieval_service_instance


def create_retrieval_service(
    vector_store: PGVector | None = None,
    use_redis_store: bool = True,
    collection_name: str = "putusan_child_chunks"
) -> RetrievalService:
    """
    Factory function to create a new retrieval service instance.
    Use this when you need a fresh instance instead of the singleton.

    Args:
        vector_store: Optional pre-configured vector store (uses singleton if None)
        use_redis_store: Whether to use Redis for document persistence
        collection_name: Collection name for vector store

    Returns:
        New RetrievalService instance
    """
    if vector_store is None:
        # Use singleton database connection manager
        vector_store = get_vector_store(collection_name)

    return RetrievalService(vector_store=vector_store, use_redis_store=use_redis_store)


def reset_retrieval_service() -> None:
    """
    Reset the singleton instance. Useful for testing or configuration changes.
    """
    global _retrieval_service_instance
    _retrieval_service_instance = None
    logger.info("RetrievalService singleton instance reset")


def get_retrieval_service_status() -> Dict[str, Any]:
    """
    Get status of the retrieval service singleton.

    Returns:
        Status information including initialization state
    """
    global _retrieval_service_instance

    status = {
        "singleton_initialized": _retrieval_service_instance is not None,
        "instance_type": type(_retrieval_service_instance).__name__ if _retrieval_service_instance else None
    }

    if _retrieval_service_instance:
        try:
            service_status = _retrieval_service_instance.get_service_status()
            status.update(service_status)
        except Exception as e:
            status["error"] = str(e)

    return status





# Optimized retrieval functions using singleton service
async def search_with_retrieval_service(
    query: str,
    strategy: RetrievalStrategy = RetrievalStrategy.VECTOR_SEARCH,
    top_k: int = 5,
    filters: Dict[str, Any] | None = None,
    include_scores: bool = True
) -> List[Document]:
    """
    Search documents using the retrieval service singleton.
    Provides access to multiple retrieval strategies with unified interface.

    Args:
        query: Search query
        strategy: Retrieval strategy (VECTOR_SEARCH, PARENT_CHILD, HYBRID)
        top_k: Number of documents to retrieve
        filters: Optional filters to apply
        include_scores: Whether to include similarity scores

    Returns:
        List of retrieved documents
    """
    try:
        # Get singleton retrieval service instance
        retrieval_service = get_retrieval_service()

        # Perform retrieval
        documents = retrieval_service.retrieve(
            query=query,
            strategy=strategy,
            top_k=top_k,
            filters=filters,
            include_scores=include_scores
        )

        logger.info(f"Retrieved {len(documents)} documents using {strategy.value}")
        return documents

    except Exception as e:
        logger.error(f"Retrieval service search failed: {e}")
        return []


async def multi_strategy_search(
    query: str,
    strategies: List[RetrievalStrategy] = None,
    top_k: int = 5,
    merge_results: bool = True
) -> Dict[str, List[Document]]:
    """
    Search using multiple strategies for comparison and optimization.

    Args:
        query: Search query
        strategies: List of strategies to use (defaults to all available)
        top_k: Number of documents per strategy
        merge_results: Whether to merge and deduplicate results

    Returns:
        Dictionary mapping strategy names to their results
    """
    if strategies is None:
        strategies = [
            RetrievalStrategy.VECTOR_SEARCH,
            RetrievalStrategy.PARENT_CHILD,
            RetrievalStrategy.HYBRID
        ]

    try:
        retrieval_service = get_retrieval_service()

        results = retrieval_service.multi_strategy_retrieve(
            query=query,
            strategies=strategies,
            top_k=top_k,
            merge_results=merge_results
        )

        # Convert enum keys to string keys for JSON serialization
        string_results = {}
        for key, docs in results.items():
            if isinstance(key, RetrievalStrategy):
                string_results[key.value] = docs
            else:
                string_results[str(key)] = docs

        return string_results

    except Exception as e:
        logger.error(f"Multi-strategy search failed: {e}")
        return {}
