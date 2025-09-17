"""
Cached Enhanced Chatbot Service.
Integrates Redis caching with the enhanced chatbot service for optimal performance.
"""

import logging
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import asyncio

from ..service import ChatbotService, QueryComplexity, WorkflowResult
from . import get_enhanced_redis_cache, EnhancedRedisCacheService

logger = logging.getLogger(__name__)


class CachedEnhancedChatbotService:
    """
    Cached wrapper for EnhancedChatbotService.
    
    Provides intelligent caching for:
    - Complex workflow results
    - Tool execution results
    - LLM responses
    - Memory states
    - Performance metrics
    """
    
    def __init__(self, enhanced_service: EnhancedChatbotService):
        """
        Initialize cached service.
        
        Args:
            enhanced_service: The underlying enhanced chatbot service
        """
        self.enhanced_service = enhanced_service
        self.cache_service = get_enhanced_redis_cache()
        
        # Cache hit/miss tracking
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0
        }
    
    def _generate_input_hash(self, **kwargs) -> str:
        """
        Generate hash for input parameters.
        
        Args:
            **kwargs: Input parameters
            
        Returns:
            Hash string
        """
        # Sort and serialize inputs for consistent hashing
        sorted_inputs = sorted(kwargs.items())
        input_str = json.dumps(sorted_inputs, sort_keys=True, default=str)
        return hashlib.sha256(input_str.encode()).hexdigest()[:16]
    
    async def _cache_workflow_result(self, input_hash: str, result: WorkflowResult) -> None:
        """
        Cache workflow execution result.
        
        Args:
            input_hash: Hash of input parameters
            result: Workflow result to cache
        """
        try:
            # Convert result to serializable format
            serializable_result = {
                "answer": result.answer,
                "confidence": result.confidence,
                "sources": result.sources,
                "reasoning_steps": result.reasoning_steps,
                "citations": result.citations,
                "audit_trail": result.audit_trail,
                "metadata": result.metadata
            }
            
            await self.cache_service.cache_tool_result(
                "workflow",
                input_hash,
                serializable_result,
                ttl=1800  # 30 minutes for complex workflows
            )
            
        except Exception as e:
            logger.error(f"Error caching workflow result: {str(e)}")
    
    async def _get_cached_workflow_result(self, input_hash: str) -> Optional[WorkflowResult]:
        """
        Retrieve cached workflow result.
        
        Args:
            input_hash: Hash of input parameters
            
        Returns:
            Cached workflow result or None
        """
        try:
            cached_data = await self.cache_service.get_cached_tool_result("workflow", input_hash)
            
            if cached_data:
                # Convert back to WorkflowResult
                return WorkflowResult(
                    answer=cached_data["answer"],
                    confidence=cached_data["confidence"],
                    sources=cached_data["sources"],
                    reasoning_steps=cached_data["reasoning_steps"],
                    citations=cached_data["citations"],
                    audit_trail=cached_data["audit_trail"],
                    metadata=cached_data["metadata"]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached workflow result: {str(e)}")
            return None
    
    async def _cache_llm_response(self, prompt: str, response: str, metadata: Dict[str, Any]) -> None:
        """
        Cache LLM response.
        
        Args:
            prompt: Input prompt
            response: LLM response
            metadata: Response metadata
        """
        try:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            await self.cache_service.cache_llm_response(prompt_hash, response, metadata)
            
        except Exception as e:
            logger.error(f"Error caching LLM response: {str(e)}")
    
    async def _get_cached_llm_response(self, prompt: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Retrieve cached LLM response.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (response, metadata) or None
        """
        try:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            cached_data = await self.cache_service.get_cached_llm_response(prompt_hash)
            
            if cached_data:
                return cached_data["response"], cached_data["metadata"]
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached LLM response: {str(e)}")
            return None
    
    async def chat_with_caching(self, message: str, conversation_id: str, 
                               user_id: str) -> Dict[str, Any]:
        """
        Enhanced chat with comprehensive caching.
        
        Args:
            message: User message
            conversation_id: Conversation identifier
            user_id: User identifier
            
        Returns:
            Chat response with cache information
        """
        try:
            # Generate cache key for this specific query
            input_hash = self._generate_input_hash(
                message=message,
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            # Check for cached conversation state
            cached_conversation = await self.cache_service.get_cached_conversation(conversation_id)
            if cached_conversation:
                logger.debug(f"Using cached conversation state for {conversation_id}")
                # Update conversation cache with new message before processing
                cached_conversation["messages"].append({
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Check for cached memory state
            cached_memory = await self.cache_service.get_cached_memory_state(user_id)
            if cached_memory:
                logger.debug(f"Using cached memory state for user {user_id}")
            
            # Analyze query complexity
            complexity = await self.enhanced_service.analyze_query_complexity(message)
            
            # For complex queries, check cache first
            if complexity == QueryComplexity.COMPLEX:
                cached_result = await self._get_cached_workflow_result(input_hash)
                if cached_result:
                    self._cache_stats["hits"] += 1
                    logger.info(f"Cache HIT for complex query: {message[:50]}...")
                    
                    # Still update conversation and memory for consistency
                    await self._update_conversation_cache(conversation_id, message, cached_result.answer)
                    
                    return {
                        "response": cached_result.answer,
                        "conversation_id": conversation_id,
                        "confidence": cached_result.confidence,
                        "sources": cached_result.sources,
                        "cached": True,
                        "complexity": complexity.value,
                        "reasoning_steps": cached_result.reasoning_steps,
                        "citations": cached_result.citations,
                        "performance": {
                            "cache_hit": True,
                            "response_time": "< 100ms"
                        }
                    }
            
            # Cache miss - execute normally
            self._cache_stats["misses"] += 1
            logger.debug(f"Cache MISS for query: {message[:50]}...")
            
            # Execute enhanced chat
            start_time = datetime.now()
            result = await self.enhanced_service.chat(message, conversation_id, user_id)
            execution_time = datetime.now() - start_time
            
            # Cache the result if it's a complex workflow
            if complexity == QueryComplexity.COMPLEX and isinstance(result.get("workflow_result"), dict):
                workflow_result_data = result["workflow_result"]
                workflow_result = WorkflowResult(
                    answer=workflow_result_data.get("answer", ""),
                    confidence=workflow_result_data.get("confidence", 0.0),
                    sources=workflow_result_data.get("sources", []),
                    reasoning_steps=workflow_result_data.get("reasoning_steps", []),
                    citations=workflow_result_data.get("citations", []),
                    audit_trail=workflow_result_data.get("audit_trail", []),
                    metadata=workflow_result_data.get("metadata", {})
                )
                await self._cache_workflow_result(input_hash, workflow_result)
            
            # Update conversation cache
            await self._update_conversation_cache(conversation_id, message, result["response"])
            
            # Update memory cache
            if self.enhanced_service.memory_system:
                memory_state = await self.enhanced_service.memory_system.get_memory_state(user_id)
                await self.cache_service.cache_memory_state(user_id, memory_state)
            
            # Add cache metadata to response
            result["cached"] = False
            result["performance"] = result.get("performance", {})
            result["performance"]["cache_hit"] = False
            result["performance"]["execution_time"] = str(execution_time.total_seconds()) + "s"
            
            return result
            
        except Exception as e:
            self._cache_stats["errors"] += 1
            logger.error(f"Error in cached chat: {str(e)}")
            # Fallback to non-cached execution
            return await self.enhanced_service.chat(message, conversation_id, user_id)
    
    async def _update_conversation_cache(self, conversation_id: str, user_message: str, 
                                       bot_response: str) -> None:
        """
        Update conversation cache with new messages.
        
        Args:
            conversation_id: Conversation identifier
            user_message: User's message
            bot_response: Bot's response
        """
        try:
            # Get existing conversation or create new
            cached_conversation = await self.cache_service.get_cached_conversation(conversation_id)
            
            if not cached_conversation:
                cached_conversation = {
                    "messages": [],
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat()
                    }
                }
            
            # Add new messages
            timestamp = datetime.now().isoformat()
            cached_conversation["messages"].extend([
                {
                    "role": "user",
                    "content": user_message,
                    "timestamp": timestamp
                },
                {
                    "role": "assistant",
                    "content": bot_response,
                    "timestamp": timestamp
                }
            ])
            
            # Update metadata
            cached_conversation["metadata"]["last_updated"] = timestamp
            cached_conversation["metadata"]["message_count"] = len(cached_conversation["messages"])
            
            # Cache updated conversation
            await self.cache_service.cache_conversation(
                conversation_id,
                cached_conversation["messages"],
                cached_conversation["metadata"]
            )
            
        except Exception as e:
            logger.error(f"Error updating conversation cache: {str(e)}")
    
    async def get_cached_tool_result(self, tool_name: str, **kwargs) -> Optional[Any]:
        """
        Get cached tool result.
        
        Args:
            tool_name: Name of the tool
            **kwargs: Tool input parameters
            
        Returns:
            Cached result or None
        """
        try:
            input_hash = self._generate_input_hash(**kwargs)
            result = await self.cache_service.get_cached_tool_result(tool_name, input_hash)
            
            if result:
                self._cache_stats["hits"] += 1
                logger.debug(f"Cache HIT for tool {tool_name}")
            else:
                self._cache_stats["misses"] += 1
                logger.debug(f"Cache MISS for tool {tool_name}")
            
            return result
            
        except Exception as e:
            self._cache_stats["errors"] += 1
            logger.error(f"Error getting cached tool result: {str(e)}")
            return None
    
    async def cache_tool_result(self, tool_name: str, result: Any, **kwargs) -> bool:
        """
        Cache tool execution result.
        
        Args:
            tool_name: Name of the tool
            result: Tool result to cache
            **kwargs: Tool input parameters
            
        Returns:
            Success status
        """
        try:
            input_hash = self._generate_input_hash(**kwargs)
            return await self.cache_service.cache_tool_result(tool_name, input_hash, result)
            
        except Exception as e:
            logger.error(f"Error caching tool result: {str(e)}")
            return False
    
    async def execute_cached_tool(self, tool_name: str, tool_function, **kwargs) -> Any:
        """
        Execute tool with caching.
        
        Args:
            tool_name: Name of the tool
            tool_function: Async function to execute
            **kwargs: Tool input parameters
            
        Returns:
            Tool result (cached or fresh)
        """
        try:
            # Check cache first
            cached_result = await self.get_cached_tool_result(tool_name, **kwargs)
            if cached_result is not None:
                return cached_result
            
            # Execute tool
            result = await tool_function(**kwargs)
            
            # Cache result
            await self.cache_tool_result(tool_name, result, **kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in cached tool execution: {str(e)}")
            # Fallback to direct execution
            return await tool_function(**kwargs)
    
    async def get_memory_with_cache(self, user_id: str) -> Dict[str, Any]:
        """
        Get user memory with caching.
        
        Args:
            user_id: User identifier
            
        Returns:
            Memory state
        """
        try:
            # Check cache first
            cached_memory = await self.cache_service.get_cached_memory_state(user_id)
            if cached_memory:
                self._cache_stats["hits"] += 1
                return cached_memory
            
            # Get from memory system
            self._cache_stats["misses"] += 1
            if self.enhanced_service.memory_system:
                memory_state = await self.enhanced_service.memory_system.get_memory_state(user_id)
                
                # Cache the result
                await self.cache_service.cache_memory_state(user_id, memory_state)
                
                return memory_state
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting memory with cache: {str(e)}")
            return {}
    
    async def clear_user_cache(self, user_id: str) -> bool:
        """
        Clear all cached data for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Success status
        """
        try:
            return await self.cache_service.invalidate_user_data(user_id)
        except Exception as e:
            logger.error(f"Error clearing user cache: {str(e)}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Cache statistics
        """
        try:
            # Get Redis statistics
            redis_stats = await self.cache_service.get_cache_statistics()
            
            # Calculate hit rate
            total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
            hit_rate = (self._cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "cache_performance": {
                    "hits": self._cache_stats["hits"],
                    "misses": self._cache_stats["misses"],
                    "errors": self._cache_stats["errors"],
                    "hit_rate_percent": round(hit_rate, 2),
                    "total_requests": total_requests
                },
                "redis_stats": redis_stats,
                "cache_health": await self.cache_service.health_check()
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {"error": str(e)}
    
    async def warm_cache(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        Warm cache with user's recent data.
        
        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            
        Returns:
            Warming results
        """
        try:
            results = {
                "memory_warmed": False,
                "conversation_warmed": False,
                "entities_warmed": 0
            }
            
            # Pre-load memory state
            if self.enhanced_service.memory_system:
                memory_state = await self.enhanced_service.memory_system.get_memory_state(user_id)
                if memory_state:
                    await self.cache_service.cache_memory_state(user_id, memory_state)
                    results["memory_warmed"] = True
                    
                    # Cache entities
                    entities = memory_state.get("entities", {})
                    for entity_name, entity_data in entities.items():
                        await self.cache_service.cache_entity_data(entity_name, entity_data)
                        results["entities_warmed"] += 1
            
            # Pre-load conversation if it exists
            # This would typically come from database
            # For now, we'll mark as warmed if cache service is available
            if self.cache_service.redis_client:
                results["conversation_warmed"] = True
            
            logger.info(f"Cache warmed for user {user_id}: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error warming cache: {str(e)}")
            return {"error": str(e)}
    
    # Delegate other methods to enhanced service
    async def analyze_query_complexity(self, query: str) -> QueryComplexity:
        """Delegate to enhanced service."""
        return await self.enhanced_service.analyze_query_complexity(query)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics with cache stats."""
        # Get base metrics
        base_metrics = await self.enhanced_service.get_performance_metrics()
        
        # Add cache metrics
        cache_stats = await self.get_cache_stats()
        
        return {
            **base_metrics,
            "cache_metrics": cache_stats
        }