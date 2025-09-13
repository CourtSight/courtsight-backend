"""
Parent-child retriever implementation.
Uses ParentDocumentRetriever for hierarchical document retrieval.
"""

import logging
from typing import List

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.storage import RedisStore
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from redis import Redis

from ...core.config import settings
from .base import BaseRetriever, RetrievalRequest, RetrievalStrategy

logger = logging.getLogger(__name__)

class ParentChildRetriever(BaseRetriever):
    """Parent-child hierarchical document retriever."""

    def __init__(
        self,
        vector_store: PGVector,
        use_redis_store: bool = True,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 500,
        **kwargs
    ):
        super().__init__("parent_child", **kwargs)
        self.vector_store = vector_store
        self.use_redis_store = use_redis_store
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size

        # Setup document store for parent documents
        self.doc_store = self._setup_document_store()

        # Setup parent-child retriever
        self.retriever = self._setup_parent_child_retriever()

    def _setup_document_store(self):
        """Setup persistent document store for parent documents."""
        if self.use_redis_store:
            try:
                redis_client = Redis(
                    host=settings.REDIS_CACHE_HOST,
                    port=settings.REDIS_CACHE_PORT,
                )
                # Use collection-specific prefix for parent documents
                doc_store = RedisStore(client=redis_client)
                logger.info("Using Redis store for parent documents")
                return doc_store
            except Exception as e:
                logger.warning(f"Failed to connect to Redis, falling back to InMemoryStore: {e}")
                return InMemoryStore()
        else:
            return InMemoryStore()

    def _setup_parent_child_retriever(self) -> ParentDocumentRetriever:
        """Setup the parent-child retriever with proper chunking."""
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=2000,
            separators=["\n\n", "\n", ".", " ", ""],
            add_start_index=True,
        )

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=400,
            separators=["\n\n", "\n", ".", " ", ""],
            add_start_index=True,
        )

        return ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.doc_store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": 10}  # Search more child chunks
        )

    def retrieve(self, request: RetrievalRequest) -> List[Document]:
        """Retrieve documents using parent-child strategy."""
        try:
            # Execute parent-child retrieval
            documents = self.retriever.invoke(request.query)

            # Limit to requested top_k
            documents = documents[:request.top_k]

            # Add metadata
            for doc in documents:
                doc.metadata["retrieval_strategy"] = "parent_child"
                doc.metadata["chunk_size"] = self.parent_chunk_size

                # Add parent/child relationship info
                if "parent_id" in doc.metadata:
                    doc.metadata["is_parent_doc"] = True
                else:
                    doc.metadata["is_parent_doc"] = False

            logger.info(f"Parent-child retrieval returned {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Parent-child retrieval failed: {e}")
            return []

    def supports_filters(self) -> bool:
        """Parent-child retriever has limited filter support."""
        return False  # Filters would need to be applied post-retrieval

    def get_strategy(self) -> RetrievalStrategy:
        """Get the strategy this retriever implements."""
        return RetrievalStrategy.PARENT_CHILD

    async def add_documents(self, documents: List[Document]):
        """Add documents to the parent-child retriever."""
        try:
            self.retriever.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to parent-child retriever")
        except Exception as e:
            logger.error(f"Failed to add documents to parent-child retriever: {e}")
            raise

    def get_parent_count(self) -> int:
        """Get the number of parent documents stored."""
        try:
            if hasattr(self.doc_store, 'redis_client'):
                # Redis store
                keys = self.doc_store.redis_client.keys("*")
                return len(keys)
            elif hasattr(self.doc_store, 'store'):
                # InMemory store
                return len(self.doc_store.store)
            else:
                # Unknown store type
                return 0
        except Exception as e:
            logger.error(f"Failed to get parent document count: {e}")
            return 0
