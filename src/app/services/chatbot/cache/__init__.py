"""
Enhanced Redis Caching Service for Sprint 2.
Provides comprehensive caching for conversations, entities, and performance optimization.
"""

import logging
import json
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import hashlib

import redis.asyncio as redis
from redis.asyncio import Redis
from pydantic import BaseModel, Field

from ...core.config import settings

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Configuration for Redis caching."""
    conversation_ttl: int = Field(default=3600, description="TTL for conversation cache (seconds)")
    entity_ttl: int = Field(default=7200, description="TTL for entity cache (seconds)")
    memory_ttl: int = Field(default=1800, description="TTL for memory state cache (seconds)")
    tool_result_ttl: int = Field(default=3600, description="TTL for tool results cache (seconds)")
    performance_ttl: int = Field(default=300, description="TTL for performance metrics cache (seconds)")
    max_memory_size: int = Field(default=100, description="Maximum memory states to cache")


class EnhancedRedisCacheService:
    """
    Enhanced Redis caching service for Sprint 2.
    
    Provides caching for:
    - Conversation history and metadata
    - Entity tracking and relationships
    - Memory states and topic clusters
    - Tool execution results
    - Performance metrics and analytics
    - LLM response caching
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize Redis cache service.
        
        Args:
            config: Cache configuration options
        """
        self.config = config or CacheConfig()
        self.redis_client: Optional[Redis] = None
        self.connection_pool = None
        
        # Cache key prefixes
        self.prefixes = {
            "conversation": "chat:conv:",
            "entity": "chat:entity:",
            "memory": "chat:memory:",
            "tool_result": "chat:tool:",
            "performance": "chat:perf:",
            "llm_response": "chat:llm:",
            "user_session": "chat:user:",
            "workflow": "chat:workflow:"
        }
        
        self._initialize_redis()
    
    def _initialize_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            # Create connection pool
            self.connection_pool = redis.ConnectionPool.from_url(
                settings.redis_cache.REDIS_CACHE_URL,
                max_connections=20,
                decode_responses=True
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self.connection_pool)
            
            logger.info("Redis cache service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache service: {str(e)}")
            self.redis_client = None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Redis connection health.
        
        Returns:
            Health status information
        """
        try:
            if not self.redis_client:
                return {"status": "unhealthy", "error": "Redis client not initialized"}
            
            # Test connection
            await self.redis_client.ping()
            
            # Get basic info
            info = await self.redis_client.info("memory")
            
            return {
                "status": "healthy",
                "memory_usage": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "cache_prefixes": list(self.prefixes.keys())
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}
    
    def _generate_cache_key(self, prefix: str, identifier: str, suffix: str = "") -> str:
        """
        Generate cache key with consistent format.
        
        Args:
            prefix: Cache prefix type
            identifier: Unique identifier
            suffix: Optional suffix
            
        Returns:
            Formatted cache key
        """
        base_key = f"{self.prefixes[prefix]}{identifier}"
        if suffix:
            base_key += f":{suffix}"
        return base_key
    
    def _hash_content(self, content: str) -> str:
        """
        Generate hash for content-based caching.
        
        Args:
            content: Content to hash
            
        Returns:
            Hash string
        """
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def cache_conversation(self, conversation_id: str, messages: List[Dict[str, Any]], 
                                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Cache conversation history.
        
        Args:
            conversation_id: Conversation identifier
            messages: List of conversation messages
            metadata: Optional conversation metadata
            
        Returns:
            Success status
        """
        try:
            if not self.redis_client:
                return False
            
            cache_key = self._generate_cache_key("conversation", conversation_id)
            
            cache_data = {
                "messages": messages,
                "metadata": metadata or {},
                "cached_at": datetime.now().isoformat(),
                "message_count": len(messages)
            }
            
            # Store with TTL
            await self.redis_client.setex(
                cache_key,
                self.config.conversation_ttl,
                json.dumps(cache_data, default=str)
            )
            
            logger.debug(f"Cached conversation {conversation_id} with {len(messages)} messages")
            return True
            
        except Exception as e:
            logger.error(f"Error caching conversation: {str(e)}")
            return False
    
    async def get_cached_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Cached conversation data or None
        """
        try:
            if not self.redis_client:
                return None
            
            cache_key = self._generate_cache_key("conversation", conversation_id)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                logger.debug(f"Retrieved cached conversation {conversation_id}")
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached conversation: {str(e)}")
            return None
    
    async def cache_entity_data(self, entity_name: str, entity_data: Dict[str, Any]) -> bool:
        """
        Cache entity tracking data.
        
        Args:
            entity_name: Entity identifier
            entity_data: Entity information
            
        Returns:
            Success status
        """
        try:
            if not self.redis_client:
                return False
            
            cache_key = self._generate_cache_key("entity", entity_name)
            
            cache_data = {
                "entity_data": entity_data,
                "cached_at": datetime.now().isoformat()
            }
            
            await self.redis_client.setex(
                cache_key,
                self.config.entity_ttl,
                json.dumps(cache_data, default=str)
            )
            
            logger.debug(f"Cached entity data for {entity_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching entity data: {str(e)}")
            return False
    
    async def get_cached_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached entity data.
        
        Args:
            entity_name: Entity identifier
            
        Returns:
            Cached entity data or None
        """
        try:
            if not self.redis_client:
                return None
            
            cache_key = self._generate_cache_key("entity", entity_name)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return data.get("entity_data")
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached entity: {str(e)}")
            return None
    
    async def cache_memory_state(self, user_id: str, memory_state: Dict[str, Any]) -> bool:
        """
        Cache enhanced memory state.
        
        Args:
            user_id: User identifier
            memory_state: Complete memory state
            
        Returns:
            Success status
        """
        try:
            if not self.redis_client:
                return False
            
            cache_key = self._generate_cache_key("memory", user_id)
            
            cache_data = {
                "memory_state": memory_state,
                "cached_at": datetime.now().isoformat(),
                "entities_count": len(memory_state.get("entities", {})),
                "topics_count": len(memory_state.get("topics", {})),
                "messages_count": len(memory_state.get("messages", []))
            }
            
            await self.redis_client.setex(
                cache_key,
                self.config.memory_ttl,
                json.dumps(cache_data, default=str)
            )
            
            logger.debug(f"Cached memory state for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching memory state: {str(e)}")
            return False
    
    async def get_cached_memory_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached memory state.
        
        Args:
            user_id: User identifier
            
        Returns:
            Cached memory state or None
        """
        try:
            if not self.redis_client:
                return None
            
            cache_key = self._generate_cache_key("memory", user_id)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return data.get("memory_state")
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached memory state: {str(e)}")
            return None
    
    async def cache_tool_result(self, tool_name: str, input_hash: str, 
                               result: Any, ttl: Optional[int] = None) -> bool:
        """
        Cache tool execution result.
        
        Args:
            tool_name: Name of the tool
            input_hash: Hash of input parameters
            result: Tool execution result
            ttl: Custom TTL (optional)
            
        Returns:
            Success status
        """
        try:
            if not self.redis_client:
                return False
            
            cache_key = self._generate_cache_key("tool_result", f"{tool_name}:{input_hash}")
            
            cache_data = {
                "result": result,
                "tool_name": tool_name,
                "cached_at": datetime.now().isoformat()
            }
            
            ttl_seconds = ttl or self.config.tool_result_ttl
            
            await self.redis_client.setex(
                cache_key,
                ttl_seconds,
                json.dumps(cache_data, default=str)
            )
            
            logger.debug(f"Cached result for tool {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching tool result: {str(e)}")
            return False
    
    async def get_cached_tool_result(self, tool_name: str, input_hash: str) -> Optional[Any]:
        """
        Retrieve cached tool result.
        
        Args:
            tool_name: Name of the tool
            input_hash: Hash of input parameters
            
        Returns:
            Cached tool result or None
        """
        try:
            if not self.redis_client:
                return None
            
            cache_key = self._generate_cache_key("tool_result", f"{tool_name}:{input_hash}")
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return data.get("result")
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached tool result: {str(e)}")
            return None
    
    async def cache_llm_response(self, prompt_hash: str, response: str, 
                                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Cache LLM response to avoid repeated calls.
        
        Args:
            prompt_hash: Hash of the prompt
            response: LLM response
            metadata: Optional response metadata
            
        Returns:
            Success status
        """
        try:
            if not self.redis_client:
                return False
            
            cache_key = self._generate_cache_key("llm_response", prompt_hash)
            
            cache_data = {
                "response": response,
                "metadata": metadata or {},
                "cached_at": datetime.now().isoformat()
            }
            
            # Use longer TTL for LLM responses
            await self.redis_client.setex(
                cache_key,
                self.config.tool_result_ttl * 2,  # 2 hours
                json.dumps(cache_data, default=str)
            )
            
            logger.debug(f"Cached LLM response for prompt hash {prompt_hash[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error caching LLM response: {str(e)}")
            return False
    
    async def get_cached_llm_response(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached LLM response.
        
        Args:
            prompt_hash: Hash of the prompt
            
        Returns:
            Cached LLM response data or None
        """
        try:
            if not self.redis_client:
                return None
            
            cache_key = self._generate_cache_key("llm_response", prompt_hash)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                logger.debug(f"Retrieved cached LLM response for prompt hash {prompt_hash[:8]}...")
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached LLM response: {str(e)}")
            return None
    
    async def cache_performance_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Cache performance metrics.
        
        Args:
            metrics: Performance metrics data
            
        Returns:
            Success status
        """
        try:
            if not self.redis_client:
                return False
            
            cache_key = self._generate_cache_key("performance", "current")
            
            cache_data = {
                "metrics": metrics,
                "cached_at": datetime.now().isoformat()
            }
            
            await self.redis_client.setex(
                cache_key,
                self.config.performance_ttl,
                json.dumps(cache_data, default=str)
            )
            
            logger.debug("Cached performance metrics")
            return True
            
        except Exception as e:
            logger.error(f"Error caching performance metrics: {str(e)}")
            return False
    
    async def get_cached_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached performance metrics.
        
        Returns:
            Cached performance metrics or None
        """
        try:
            if not self.redis_client:
                return None
            
            cache_key = self._generate_cache_key("performance", "current")
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return data.get("metrics")
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached performance metrics: {str(e)}")
            return None
    
    async def invalidate_conversation(self, conversation_id: str) -> bool:
        """
        Invalidate cached conversation.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Success status
        """
        try:
            if not self.redis_client:
                return False
            
            cache_key = self._generate_cache_key("conversation", conversation_id)
            result = await self.redis_client.delete(cache_key)
            
            logger.debug(f"Invalidated conversation cache for {conversation_id}")
            return result > 0
            
        except Exception as e:
            logger.error(f"Error invalidating conversation cache: {str(e)}")
            return False
    
    async def invalidate_user_data(self, user_id: str) -> bool:
        """
        Invalidate all cached data for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Success status
        """
        try:
            if not self.redis_client:
                return False
            
            # Find all keys related to user
            patterns = [
                self._generate_cache_key("memory", user_id),
                self._generate_cache_key("user_session", user_id)
            ]
            
            deleted_count = 0
            for pattern in patterns:
                result = await self.redis_client.delete(pattern)
                deleted_count += result
            
            logger.debug(f"Invalidated {deleted_count} cache entries for user {user_id}")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error invalidating user cache: {str(e)}")
            return False
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache usage statistics.
        
        Returns:
            Cache statistics
        """
        try:
            if not self.redis_client:
                return {"error": "Redis client not available"}
            
            # Get cache sizes by prefix
            stats = {}
            total_keys = 0
            
            for cache_type, prefix in self.prefixes.items():
                pattern = f"{prefix}*"
                keys = await self.redis_client.keys(pattern)
                count = len(keys) if keys else 0
                stats[cache_type] = count
                total_keys += count
            
            # Get Redis info
            info = await self.redis_client.info("memory")
            
            return {
                "total_keys": total_keys,
                "keys_by_type": stats,
                "memory_usage": info.get("used_memory_human", "unknown"),
                "cache_hit_rate": "N/A",  # Would need separate tracking
                "config": {
                    "conversation_ttl": self.config.conversation_ttl,
                    "entity_ttl": self.config.entity_ttl,
                    "memory_ttl": self.config.memory_ttl,
                    "tool_result_ttl": self.config.tool_result_ttl
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {str(e)}")
            return {"error": str(e)}
    
    async def clear_all_cache(self) -> bool:
        """
        Clear all cached data (use with caution).
        
        Returns:
            Success status
        """
        try:
            if not self.redis_client:
                return False
            
            # Get all keys with our prefixes
            all_keys = []
            for prefix in self.prefixes.values():
                pattern = f"{prefix}*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    all_keys.extend(keys)
            
            if all_keys:
                deleted_count = await self.redis_client.delete(*all_keys)
                logger.info(f"Cleared {deleted_count} cache entries")
                return deleted_count > 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing all cache: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Close Redis connections."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.connection_pool:
                await self.connection_pool.disconnect()
            
            logger.info("Redis cache service closed")
            
        except Exception as e:
            logger.error(f"Error closing Redis cache service: {str(e)}")


# Global instance
_enhanced_redis_cache: Optional[EnhancedRedisCacheService] = None


def get_enhanced_redis_cache() -> EnhancedRedisCacheService:
    """Get or create enhanced Redis cache service instance."""
    global _enhanced_redis_cache
    
    if _enhanced_redis_cache is None:
        _enhanced_redis_cache = EnhancedRedisCacheService()
    
    return _enhanced_redis_cache


async def cleanup_redis_cache() -> None:
    """Cleanup Redis cache service on shutdown."""
    global _enhanced_redis_cache
    
    if _enhanced_redis_cache:
        await _enhanced_redis_cache.close()
        _enhanced_redis_cache = None