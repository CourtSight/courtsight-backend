"""
Hybrid retriever implementation.
Combines multiple retrieval strategies for optimal results.
"""

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_postgres import PGVector

from .base import BaseRetriever, RetrievalRequest, RetrievalStrategy
from .parent_child import ParentChildRetriever
from .vector_search import VectorSearchRetriever

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """Hybrid retriever that combines multiple strategies."""

    def __init__(
        self,
        vector_store: PGVector,
        use_redis_store: bool = True,
        vector_weight: float = 0.6,
        parent_child_weight: float = 0.4,
        **kwargs
    ):
        super().__init__("hybrid", **kwargs)
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.parent_child_weight = parent_child_weight

        # Initialize sub-retrievers
        self.vector_retriever = VectorSearchRetriever(vector_store, **kwargs)
        self.parent_child_retriever = ParentChildRetriever(
            vector_store=vector_store,
            embeddings_model=vector_store.embeddings,
            collection_name="hybrid_parent_child"
        )

    def retrieve(self, request: RetrievalRequest) -> List[Document]:
        """Retrieve documents using hybrid strategy."""
        try:
            # Execute both strategies in parallel
            # Option 1: Sequential execution (what you requested)
            vector_docs =  self._safe_retrieve(self.vector_retriever, request)
            parent_child_docs =  self._safe_retrieve(self.parent_child_retriever, request)

            # Combine and rank results
            combined_docs = self._combine_results(
                vector_docs,
                parent_child_docs,
                request.top_k
            )

            # Add hybrid metadata
            for doc in combined_docs:
                doc.metadata["retrieval_strategy"] = "hybrid"
                doc.metadata["hybrid_weights"] = {
                    "vector": self.vector_weight,
                    "parent_child": self.parent_child_weight
                }

            logger.info(f"Hybrid retrieval returned {len(combined_docs)} documents")
            return combined_docs

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []

    def _safe_retrieve(self, retriever: BaseRetriever, request: RetrievalRequest) -> List[Document]:
        """Safely execute retrieval with error handling."""
        try:
            return  retriever.retrieve(request)
        except Exception as e:
            logger.error(f"Retriever {retriever.name} failed: {e}")
            return []

    def _combine_results(
        self,
        vector_docs: List[Document],
        parent_child_docs: List[Document],
        top_k: int
    ) -> List[Document]:
        """Combine and rank results from different strategies."""
        combined_docs = []
        seen_content = set()

        # Score and combine documents
        scored_docs = []

        # Process vector search results
        for doc in vector_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)

                # Calculate weighted score
                vector_score = doc.metadata.get("similarity_score", 0.5)
                weighted_score = vector_score * self.vector_weight

                doc.metadata["hybrid_score"] = weighted_score
                doc.metadata["vector_score"] = vector_score
                doc.metadata["source_strategy"] = "vector"

                scored_docs.append((doc, weighted_score))

        # Process parent-child results
        for doc in parent_child_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)

                # Parent-child documents get base score
                parent_child_score = 0.7  # Base score for parent-child
                weighted_score = parent_child_score * self.parent_child_weight

                doc.metadata["hybrid_score"] = weighted_score
                doc.metadata["parent_child_score"] = parent_child_score
                doc.metadata["source_strategy"] = "parent_child"

                scored_docs.append((doc, weighted_score))
            else:
                # Document already exists from vector search, boost its score
                for existing_doc, existing_score in scored_docs:
                    if hash(existing_doc.page_content) == content_hash:
                        boost = 0.1 * self.parent_child_weight
                        new_score = existing_score + boost
                        existing_doc.metadata["hybrid_score"] = new_score
                        existing_doc.metadata["source_strategy"] = "both"

                        # Update the score in the list
                        scored_docs = [(d, new_score if d == existing_doc else s) for d, s in scored_docs]
                        break

        # Sort by hybrid score and return top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        combined_docs = [doc for doc, score in scored_docs[:top_k]]

        return combined_docs

    def supports_filters(self) -> bool:
        """Hybrid retriever supports filters if any sub-retriever does."""
        return self.vector_retriever.supports_filters()

    def get_strategy(self) -> RetrievalStrategy:
        """Get the strategy this retriever implements."""
        return RetrievalStrategy.HYBRID

    async def add_documents(self, documents: List[Document]):
        """Add documents to both sub-retrievers."""
        try:
            # Add to parent-child retriever (which includes vector store)
            await self.parent_child_retriever.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to hybrid retriever")
        except Exception as e:
            logger.error(f"Failed to add documents to hybrid retriever: {e}")
            raise

    def get_retriever_status(self) -> Dict[str, Any]:
        """Get status of all sub-retrievers."""
        status = {
            "vector_retriever": {
                "available": True,
                "collection": self.vector_store.collection_name
            },
            "parent_child_retriever": {
                "available": True,
                "parent_count": self.parent_child_retriever._get_document_count(),
                "doc_store_type": type(self.parent_child_retriever.doc_store).__name__
            },
            "weights": {
                "vector": self.vector_weight,
                "parent_child": self.parent_child_weight
            }
        }
        return status
