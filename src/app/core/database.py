"""
Database Connection Manager with Singleton Pattern
Manages PostgreSQL connections and PGVector instances efficiently to avoid connection pool exhaustion.
"""

import logging
from typing import Optional, Dict, Any
from functools import lru_cache
from contextlib import asynccontextmanager
import asyncio
import threading

from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..core.config import get_settings

logger = logging.getLogger(__name__)


class DatabaseConnectionManager:
    """
    Singleton Database Connection Manager.
    Manages database connections and PGVector instances efficiently.
    """
    
    _instance: Optional['DatabaseConnectionManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'DatabaseConnectionManager':
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the connection manager only once."""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.settings = get_settings()
        self._vector_stores: Dict[str, PGVector] = {}
        self._embeddings_instance: Optional[GoogleGenerativeAIEmbeddings] = None
        self._connection_pool_size = 2  # Based on your limitation
        self._active_connections = 0
        self._connection_lock = threading.Lock()
        self._initialized = True
        
        logger.info("Database Connection Manager initialized")
    
    def _get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Get singleton embeddings instance."""
        if self._embeddings_instance is None:
            self._embeddings_instance = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=self.settings.GOOGLE_API_KEY.get_secret_value()
            )
        return self._embeddings_instance
    
    def get_vector_store(
        self, 
        collection_name: str = "ma_putusan_pc_chunks",
        use_jsonb: bool = True
    ) -> PGVector:
        """
        Get or create PGVector instance with singleton pattern.
        
        Args:
            collection_name: Name of the vector collection
            use_jsonb: Whether to use JSONB for metadata
            
        Returns:
            PGVector instance (reused if exists)
        """
        store_key = f"{collection_name}_{use_jsonb}"
        
        # Return existing instance if available
        if store_key in self._vector_stores:
            logger.debug(f"Reusing existing vector store for collection: {collection_name}")
            return self._vector_stores[store_key]
        
        # Create new instance with connection management
        with self._connection_lock:
            if self._active_connections >= self._connection_pool_size:
                logger.warning(f"Connection pool limit reached ({self._connection_pool_size})")
                # Return existing instance or wait
                if self._vector_stores:
                    existing_key = list(self._vector_stores.keys())[0]
                    logger.info(f"Returning existing vector store: {existing_key}")
                    return self._vector_stores[existing_key]
            
            try:
                logger.info(f"Creating new vector store for collection: {collection_name}")
                
                vector_store = PGVector(
                    embeddings=self._get_embeddings(),
                    connection=self.settings.DATABASE_URL,
                    collection_name=collection_name,
                    use_jsonb=use_jsonb
                )
                
                self._vector_stores[store_key] = vector_store
                self._active_connections += 1
                
                logger.info(f"Vector store created. Active connections: {self._active_connections}")
                return vector_store
                
            except Exception as e:
                logger.error(f"Failed to create vector store: {e}")
                raise
    
    def get_default_vector_store(self) -> PGVector:
        """Get the default vector store instance."""
        return self.get_vector_store(
            collection_name=self.settings.VECTOR_COLLECTION_NAME,
            use_jsonb=True
        )
    
    async def close_connection(self, collection_name: str = None):
        """
        Close specific or all database connections.
        
        Args:
            collection_name: Specific collection to close, or None for all
        """
        with self._connection_lock:
            if collection_name:
                store_key = f"{collection_name}_True"
                if store_key in self._vector_stores:
                    try:
                        # PGVector doesn't have explicit close method, but we can remove reference
                        del self._vector_stores[store_key]
                        self._active_connections = max(0, self._active_connections - 1)
                        logger.info(f"Closed connection for collection: {collection_name}")
                    except Exception as e:
                        logger.error(f"Error closing connection for {collection_name}: {e}")
            else:
                # Close all connections
                try:
                    self._vector_stores.clear()
                    self._active_connections = 0
                    logger.info("Closed all database connections")
                except Exception as e:
                    logger.error(f"Error closing all connections: {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status."""
        return {
            "active_connections": self._active_connections,
            "max_connections": self._connection_pool_size,
            "vector_stores": list(self._vector_stores.keys()),
            "available_slots": self._connection_pool_size - self._active_connections
        }
    
    def optimize_connections(self):
        """Optimize connections by removing unused instances."""
        with self._connection_lock:
            before_count = len(self._vector_stores)
            
            # Keep only the most recently used vector store if we have too many
            if len(self._vector_stores) > 1:
                # Keep only the default collection
                default_key = f"{self.settings.VECTOR_COLLECTION_NAME}_True"
                if default_key in self._vector_stores:
                    default_store = self._vector_stores[default_key]
                    self._vector_stores.clear()
                    self._vector_stores[default_key] = default_store
                    self._active_connections = 1
                else:
                    # Keep only the first one
                    if self._vector_stores:
                        first_key = list(self._vector_stores.keys())[0]
                        first_store = self._vector_stores[first_key]
                        self._vector_stores.clear()
                        self._vector_stores[first_key] = first_store
                        self._active_connections = 1
            
            after_count = len(self._vector_stores)
            logger.info(f"Connection optimization: {before_count} -> {after_count} instances")


# Singleton instance functions
@lru_cache()
def get_db_manager() -> DatabaseConnectionManager:
    """Get singleton database connection manager."""
    return DatabaseConnectionManager()


def get_vector_store(collection_name: str = None) -> PGVector:
    """
    Get vector store instance with singleton connection management.
    
    Args:
        collection_name: Collection name (uses default if None)
        
    Returns:
        PGVector instance
    """
    manager = get_db_manager()
    
    if collection_name:
        return manager.get_vector_store(collection_name)
    else:
        return manager.get_default_vector_store()


async def close_db_connections(collection_name: str = None):
    """Close database connections."""
    manager = get_db_manager()
    await manager.close_connection(collection_name)


def get_db_status() -> Dict[str, Any]:
    """Get database connection status."""
    manager = get_db_manager()
    return manager.get_connection_status()


def optimize_db_connections():
    """Optimize database connections."""
    manager = get_db_manager()
    manager.optimize_connections()


# Context manager for connection lifecycle
@asynccontextmanager
async def database_lifecycle():
    """Context manager for database lifecycle management."""
    manager = get_db_manager()
    try:
        logger.info("Database connections initialized")
        yield manager
    finally:
        await manager.close_connection()
        logger.info("Database connections closed")
