"""
Custom PostgreSQL Document Store for Parent Documents.
Implements LangChain BaseStore interface with PostgreSQL backend.
"""

import json
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from ..core.config import get_settings

logger = logging.getLogger(__name__)


class PostgreSQLParentDocStore(BaseStore[str, Document]):
    """
    PostgreSQL-based document store for parent documents.
    Implements LangChain BaseStore interface for persistence.
    """

    def __init__(
        self, 
        database_url: Optional[str] = None, 
        collection_name: str = "default",
        table_name: str = "parent_documents"
    ):
        """
        Initialize PostgreSQL document store.
        
        Args:
            database_url: PostgreSQL connection URL
            collection_name: Collection name for document grouping
            table_name: Table name for storing documents
        """
        self.settings = get_settings()
        self.database_url = database_url or self.settings.DATABASE_URL
        self.collection_name = collection_name
        self.table_name = table_name
        
        # Create engine with connection pooling
        self.engine = self._create_engine()
        
        # Ensure table exists
        self._ensure_table_exists()
        
        logger.info(f"PostgreSQL parent doc store initialized for collection: {collection_name}")

    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine with proper configuration."""
        try:
            # Convert async URL to sync for SQLAlchemy Core
            sync_url = self.database_url.replace("postgresql+asyncpg://", "postgresql://")
            
            engine = create_engine(
                sync_url,
                pool_size=1,  # Limited pool size as per your requirement
                max_overflow=0,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            logger.info("PostgreSQL engine created successfully")
            return engine
            
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL engine: {e}")
            raise

    def _ensure_table_exists(self):
        """Ensure the parent_documents table exists."""
        try:
            with self.engine.connect() as conn:
                # Check if table exists
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = :table_name
                    )
                """), {"table_name": self.table_name})
                
                table_exists = result.scalar()
                
                if not table_exists:
                    logger.warning(f"Table {self.table_name} does not exist. Please run migrations.")
                    
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")

    def mget(self, keys: List[str]) -> List[Optional[Document]]:
        """
        Get multiple documents by keys.
        
        Args:
            keys: List of document IDs
            
        Returns:
            List of documents (None for missing keys)
        """
        try:
            with self.engine.connect() as conn:
                # Build parameterized query
                placeholders = ", ".join([f":key_{i}" for i in range(len(keys))])
                params = {f"key_{i}": key for i, key in enumerate(keys)}
                params["collection_name"] = self.collection_name
                
                query = text(f"""
                    SELECT id, content, metadata 
                    FROM {self.table_name} 
                    WHERE id IN ({placeholders}) 
                    AND collection_name = :collection_name
                """)
                
                result = conn.execute(query, params)
                rows = result.fetchall()
                
                # Create mapping of id to document
                doc_map = {}
                for row in rows:
                    # Handle metadata - could be string or dict depending on DB driver
                    if row.metadata:
                        if isinstance(row.metadata, str):
                            metadata = json.loads(row.metadata)
                        else:
                            metadata = row.metadata
                    else:
                        metadata = {}
                    doc = Document(page_content=row.content, metadata=metadata)
                    doc_map[row.id] = doc
                
                # Return documents in the same order as keys
                documents = [doc_map.get(key) for key in keys]
                
                logger.debug(f"Retrieved {len([d for d in documents if d is not None])}/{len(keys)} documents")
                return documents
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in mget: {e}")
            return [None] * len(keys)
        except Exception as e:
            logger.error(f"Unexpected error in mget: {e}")
            return [None] * len(keys)

    def mset(self, key_value_pairs: List[Tuple[str, Document]]) -> None:
        """
        Set multiple documents.
        
        Args:
            key_value_pairs: List of (key, document) pairs
        """
        try:
            with self.engine.begin() as conn:  # Use transaction
                for key, document in key_value_pairs:
                    # Serialize metadata
                    metadata_json = json.dumps(document.metadata) if document.metadata else None
                    
                    # Upsert document
                    conn.execute(text(f"""
                        INSERT INTO {self.table_name} (id, content, metadata, collection_name)
                        VALUES (:id, :content, :metadata, :collection_name)
                        ON CONFLICT (id) DO UPDATE SET 
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            updated_at = CURRENT_TIMESTAMP
                    """), {
                        "id": key,
                        "content": document.page_content,
                        "metadata": metadata_json,
                        "collection_name": self.collection_name
                    })
                
                logger.info(f"Stored {len(key_value_pairs)} documents in collection: {self.collection_name}")
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in mset: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in mset: {e}")
            raise

    def mdelete(self, keys: List[str]) -> None:
        """
        Delete multiple documents by keys.
        
        Args:
            keys: List of document IDs to delete
        """
        try:
            with self.engine.begin() as conn:  # Use transaction
                if keys:
                    # Build parameterized query
                    placeholders = ", ".join([f":key_{i}" for i in range(len(keys))])
                    params = {f"key_{i}": key for i, key in enumerate(keys)}
                    params["collection_name"] = self.collection_name
                    
                    query = text(f"""
                        DELETE FROM {self.table_name} 
                        WHERE id IN ({placeholders}) 
                        AND collection_name = :collection_name
                    """)
                    
                    result = conn.execute(query, params)
                    deleted_count = result.rowcount
                    
                    logger.info(f"Deleted {deleted_count} documents from collection: {self.collection_name}")
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in mdelete: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in mdelete: {e}")
            raise

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """
        Yield all keys, optionally with prefix filtering.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Yields:
            Document IDs
        """
        try:
            with self.engine.connect() as conn:
                if prefix:
                    query = text(f"""
                        SELECT id FROM {self.table_name} 
                        WHERE id LIKE :prefix 
                        AND collection_name = :collection_name
                        ORDER BY id
                    """)
                    params = {"prefix": f"{prefix}%", "collection_name": self.collection_name}
                else:
                    query = text(f"""
                        SELECT id FROM {self.table_name} 
                        WHERE collection_name = :collection_name
                        ORDER BY id
                    """)
                    params = {"collection_name": self.collection_name}
                
                result = conn.execute(query, params)
                
                for row in result:
                    yield row.id
                    
        except SQLAlchemyError as e:
            logger.error(f"Database error in yield_keys: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in yield_keys: {e}")

    def get_document_count(self) -> int:
        """
        Get the total number of documents in this collection.
        
        Returns:
            Number of documents
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) as count 
                    FROM {self.table_name} 
                    WHERE collection_name = :collection_name
                """), {"collection_name": self.collection_name})
                
                count = result.scalar()
                return count or 0
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_document_count: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error in get_document_count: {e}")
            return 0

    def clear_collection(self) -> None:
        """Clear all documents in this collection."""
        try:
            with self.engine.begin() as conn:
                result = conn.execute(text(f"""
                    DELETE FROM {self.table_name} 
                    WHERE collection_name = :collection_name
                """), {"collection_name": self.collection_name})
                
                deleted_count = result.rowcount
                logger.info(f"Cleared {deleted_count} documents from collection: {self.collection_name}")
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in clear_collection: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in clear_collection: {e}")
            raise

    def close(self):
        """Close database connections."""
        try:
            self.engine.dispose()
            logger.info("PostgreSQL parent doc store connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.close()
        except:
            pass
