"""
Enhanced Memory System for Sprint 2.
Provides conversation memory with entity tracking and topic clustering.
"""

import logging
import json
import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum

from langchain_core.memory import BaseMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

from ...llm_service import get_llm_service

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities that can be tracked."""
    PERSON = "person"
    ORGANIZATION = "organization" 
    LEGAL_CASE = "legal_case"
    LEGAL_PRINCIPLE = "legal_principle"
    STATUTE = "statute"
    REGULATION = "regulation"
    COURT = "court"
    DATE = "date"
    LOCATION = "location"
    OTHER = "other"


class TopicCategory(Enum):
    """Categories for topic clustering."""
    CIVIL_LAW = "civil_law"
    CRIMINAL_LAW = "criminal_law"
    ADMINISTRATIVE_LAW = "administrative_law"
    CONSTITUTIONAL_LAW = "constitutional_law"
    COMMERCIAL_LAW = "commercial_law"
    LABOR_LAW = "labor_law"
    TAX_LAW = "tax_law"
    FAMILY_LAW = "family_law"
    PROCEDURAL_LAW = "procedural_law"
    OTHER = "other"


@dataclass
class Entity:
    """Represents an entity mentioned in conversation."""
    name: str
    entity_type: EntityType
    mentions: List[str]
    first_mentioned: datetime
    last_mentioned: datetime
    confidence: float
    context: str
    related_entities: Set[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "name": self.name,
            "entity_type": self.entity_type.value,
            "mentions": self.mentions,
            "first_mentioned": self.first_mentioned.isoformat(),
            "last_mentioned": self.last_mentioned.isoformat(),
            "confidence": self.confidence,
            "context": self.context,
            "related_entities": list(self.related_entities)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create entity from dictionary."""
        return cls(
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            mentions=data["mentions"],
            first_mentioned=datetime.fromisoformat(data["first_mentioned"]),
            last_mentioned=datetime.fromisoformat(data["last_mentioned"]),
            confidence=data["confidence"],
            context=data["context"],
            related_entities=set(data["related_entities"])
        )


@dataclass
class Topic:
    """Represents a topic cluster in conversation."""
    topic_id: str
    category: TopicCategory
    keywords: Set[str]
    messages: List[int]  # Message indices
    relevance_score: float
    created: datetime
    last_updated: datetime
    summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert topic to dictionary."""
        return {
            "topic_id": self.topic_id,
            "category": self.category.value,
            "keywords": list(self.keywords),
            "messages": self.messages,
            "relevance_score": self.relevance_score,
            "created": self.created.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "summary": self.summary
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Topic':
        """Create topic from dictionary."""
        return cls(
            topic_id=data["topic_id"],
            category=TopicCategory(data["category"]),
            keywords=set(data["keywords"]),
            messages=data["messages"],
            relevance_score=data["relevance_score"],
            created=datetime.fromisoformat(data["created"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            summary=data["summary"]
        )


@dataclass 
class ConversationMessage:
    """Enhanced message with metadata."""
    content: str
    message_type: str  # "human" or "ai"
    timestamp: datetime
    entities: Set[str]
    topics: Set[str]
    sentiment: str
    importance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "content": self.content,
            "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat(),
            "entities": list(self.entities),
            "topics": list(self.topics),
            "sentiment": self.sentiment,
            "importance_score": self.importance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """Create message from dictionary."""
        return cls(
            content=data["content"],
            message_type=data["message_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            entities=set(data["entities"]),
            topics=set(data["topics"]),
            sentiment=data["sentiment"],
            importance_score=data["importance_score"]
        )


class MemorySystem:
    """
    Enhanced memory system with entity tracking and topic clustering.
    
    Features:
    - Entity recognition and tracking
    - Topic clustering and categorization  
    - Conversation summarization
    - Contextual retrieval
    - Memory compression
    - Importance-based retention
    """
    
    def __init__(self, 
                 max_messages: int = 100,
                 max_entities: int = 50,
                 max_topics: int = 20,
                 entity_confidence_threshold: float = 0.7,
                 topic_relevance_threshold: float = 0.6):
        """
        Initialize enhanced memory system.
        
        Args:
            max_messages: Maximum messages to keep in memory
            max_entities: Maximum entities to track
            max_topics: Maximum topics to maintain
            entity_confidence_threshold: Minimum confidence for entity tracking
            topic_relevance_threshold: Minimum relevance for topic clustering
        """
        self.max_messages = max_messages
        self.max_entities = max_entities
        self.max_topics = max_topics
        self.entity_confidence_threshold = entity_confidence_threshold
        self.topic_relevance_threshold = topic_relevance_threshold
        
        # Memory storage
        self.messages: deque = deque(maxlen=max_messages)
        self.entities: Dict[str, Entity] = {}
        self.topics: Dict[str, Topic] = {}
        
        # Indices for efficient retrieval
        self.entity_to_messages: Dict[str, Set[int]] = defaultdict(set)
        self.topic_to_messages: Dict[str, Set[int]] = defaultdict(set)
        self.message_index = 0
        
        # Services
        self.llm_service = None
        self._initialize_services()
    
    def _initialize_services(self) -> None:
        """Initialize LLM service."""
        try:
            self.llm_service = get_llm_service()
            logger.info("Enhanced memory system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory system services: {str(e)}")
    
    async def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to memory with entity and topic extraction.
        
        Args:
            message: Message to add to memory
        """
        try:
            # Convert to conversation message
            conv_message = await self._process_message(message)
            
            # Add to messages
            self.messages.append(conv_message)
            current_index = self.message_index
            self.message_index += 1
            
            # Update entity-message mappings
            for entity_name in conv_message.entities:
                self.entity_to_messages[entity_name].add(current_index)
            
            # Update topic-message mappings
            for topic_id in conv_message.topics:
                self.topic_to_messages[topic_id].add(current_index)
            
            # Manage memory limits
            await self._manage_memory_limits()
            
            logger.debug(f"Added message {current_index} with {len(conv_message.entities)} entities and {len(conv_message.topics)} topics")
            
        except Exception as e:
            logger.error(f"Error adding message to memory: {str(e)}")
    
    async def _process_message(self, message: BaseMessage) -> ConversationMessage:
        """
        Process message to extract entities, topics, and metadata.
        
        Args:
            message: Raw message
            
        Returns:
            Processed conversation message
        """
        content = message.content
        message_type = "human" if isinstance(message, HumanMessage) else "ai"
        timestamp = datetime.now()
        
        # Extract entities and topics in parallel
        entities_task = self._extract_entities(content)
        topics_task = self._extract_topics(content)
        sentiment_task = self._analyze_sentiment(content)
        importance_task = self._calculate_importance(content, message_type)
        
        # Wait for all tasks
        entities, topics, sentiment, importance = await asyncio.gather(
            entities_task, topics_task, sentiment_task, importance_task
        )
        
        return ConversationMessage(
            content=content,
            message_type=message_type,
            timestamp=timestamp,
            entities=entities,
            topics=topics,
            sentiment=sentiment,
            importance_score=importance
        )
    
    async def _extract_entities(self, content: str) -> Set[str]:
        """
        Extract entities from message content.
        
        Args:
            content: Message content
            
        Returns:
            Set of entity names
        """
        if not self.llm_service:
            return set()
        
        try:
            extraction_prompt = f"""
            Ekstrak entitas hukum dari teks berikut:
            
            Teks: {content}
            
            Identifikasi dan kategorikan entitas berikut:
            - Nama orang (hakim, pengacara, pihak berperkara)
            - Organisasi/institusi (pengadilan, firma hukum, perusahaan)
            - Kasus hukum (nomor perkara, nama kasus)
            - Prinsip hukum (asas, doktrin)
            - Peraturan (UU, PP, Perpres)
            - Pengadilan (MA, PT, PN)
            - Tanggal penting
            - Lokasi
            
            Berikan dalam format JSON:
            {{
                "entities": [
                    {{
                        "name": "nama entitas",
                        "type": "person|organization|legal_case|legal_principle|statute|regulation|court|date|location|other",
                        "confidence": 0.0-1.0,
                        "context": "konteks singkat"
                    }}
                ]
            }}
            """
            
            result = await self.llm_service.llm.ainvoke(extraction_prompt)
            
            try:
                data = json.loads(result.content)
                extracted_entities = set()
                
                for entity_data in data.get("entities", []):
                    name = entity_data.get("name", "").strip()
                    entity_type = EntityType(entity_data.get("type", "other"))
                    confidence = float(entity_data.get("confidence", 0.5))
                    context = entity_data.get("context", "")
                    
                    if name and confidence >= self.entity_confidence_threshold:
                        # Update or create entity
                        await self._update_entity(name, entity_type, content, confidence, context)
                        extracted_entities.add(name)
                
                return extracted_entities
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Error parsing entity extraction result: {str(e)}")
                return set()
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return set()
    
    async def _extract_topics(self, content: str) -> Set[str]:
        """
        Extract and cluster topics from message content.
        
        Args:
            content: Message content
            
        Returns:
            Set of topic IDs
        """
        if not self.llm_service:
            return set()
        
        try:
            topic_prompt = f"""
            Analisis topik hukum dari teks berikut:
            
            Teks: {content}
            
            Tentukan kategori hukum dan kata kunci utama:
            
            Kategori hukum:
            - civil_law (hukum perdata)
            - criminal_law (hukum pidana)  
            - administrative_law (hukum administrasi)
            - constitutional_law (hukum tata negara)
            - commercial_law (hukum dagang)
            - labor_law (hukum ketenagakerjaan)
            - tax_law (hukum pajak)
            - family_law (hukum keluarga)
            - procedural_law (hukum acara)
            - other
            
            Berikan dalam format JSON:
            {{
                "topics": [
                    {{
                        "category": "kategori dari daftar di atas",
                        "keywords": ["kata kunci 1", "kata kunci 2"],
                        "relevance": 0.0-1.0,
                        "summary": "ringkasan singkat topik"
                    }}
                ]
            }}
            """
            
            result = await self.llm_service.llm.ainvoke(topic_prompt)
            
            try:
                data = json.loads(result.content)
                extracted_topics = set()
                
                for topic_data in data.get("topics", []):
                    category = TopicCategory(topic_data.get("category", "other"))
                    keywords = set(topic_data.get("keywords", []))
                    relevance = float(topic_data.get("relevance", 0.5))
                    summary = topic_data.get("summary", "")
                    
                    if relevance >= self.topic_relevance_threshold:
                        # Update or create topic
                        topic_id = await self._update_topic(category, keywords, relevance, summary)
                        extracted_topics.add(topic_id)
                
                return extracted_topics
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Error parsing topic extraction result: {str(e)}")
                return set()
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {str(e)}")
            return set()
    
    async def _analyze_sentiment(self, content: str) -> str:
        """
        Analyze sentiment of message content.
        
        Args:
            content: Message content
            
        Returns:
            Sentiment classification
        """
        # Simple keyword-based sentiment analysis for now
        positive_keywords = ["setuju", "baik", "benar", "bagus", "terima kasih", "suka"]
        negative_keywords = ["tidak", "salah", "buruk", "jelek", "tolak", "benci"]
        
        content_lower = content.lower()
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in content_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in content_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    async def _calculate_importance(self, content: str, message_type: str) -> float:
        """
        Calculate importance score for message.
        
        Args:
            content: Message content
            message_type: Type of message (human/ai)
            
        Returns:
            Importance score (0.0-1.0)
        """
        score = 0.5  # Base score
        
        # Length factor
        if len(content) > 200:
            score += 0.1
        elif len(content) < 50:
            score -= 0.1
        
        # Legal keywords boost
        legal_keywords = [
            "putusan", "pengadilan", "mahkamah", "peraturan", "undang-undang",
            "pasal", "ayat", "preseden", "yurisprudensi", "hukum", "legal"
        ]
        
        content_lower = content.lower()
        legal_score = sum(0.05 for keyword in legal_keywords if keyword in content_lower)
        score += min(legal_score, 0.3)  # Cap at 0.3
        
        # Question boost (higher importance for questions)
        if "?" in content or any(word in content_lower for word in ["apa", "bagaimana", "mengapa", "kapan", "dimana"]):
            score += 0.1
        
        # Human messages slightly more important
        if message_type == "human":
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    async def _update_entity(self, name: str, entity_type: EntityType, content: str, 
                            confidence: float, context: str) -> None:
        """
        Update or create entity in memory.
        
        Args:
            name: Entity name
            entity_type: Type of entity
            content: Content where entity was mentioned
            confidence: Confidence score
            context: Context of mention
        """
        now = datetime.now()
        
        if name in self.entities:
            # Update existing entity
            entity = self.entities[name]
            entity.mentions.append(content[:100] + "..." if len(content) > 100 else content)
            entity.last_mentioned = now
            entity.confidence = max(entity.confidence, confidence)
            if context and context not in entity.context:
                entity.context += f"; {context}"
        else:
            # Create new entity
            entity = Entity(
                name=name,
                entity_type=entity_type,
                mentions=[content[:100] + "..." if len(content) > 100 else content],
                first_mentioned=now,
                last_mentioned=now,
                confidence=confidence,
                context=context,
                related_entities=set()
            )
            self.entities[name] = entity
        
        # Find related entities (entities mentioned together)
        for other_name, other_entity in self.entities.items():
            if other_name != name and other_name in content:
                entity.related_entities.add(other_name)
                other_entity.related_entities.add(name)
    
    async def _update_topic(self, category: TopicCategory, keywords: Set[str], 
                           relevance: float, summary: str) -> str:
        """
        Update or create topic in memory.
        
        Args:
            category: Topic category
            keywords: Keywords for topic
            relevance: Relevance score
            summary: Topic summary
            
        Returns:
            Topic ID
        """
        # Create topic ID from category and top keywords
        top_keywords = sorted(keywords)[:3]
        topic_id = f"{category.value}_{'-'.join(top_keywords)}"
        
        now = datetime.now()
        
        if topic_id in self.topics:
            # Update existing topic
            topic = self.topics[topic_id]
            topic.keywords.update(keywords)
            topic.relevance_score = max(topic.relevance_score, relevance)
            topic.last_updated = now
            topic.messages.append(self.message_index)
            if summary and summary not in topic.summary:
                topic.summary += f"; {summary}"
        else:
            # Create new topic
            topic = Topic(
                topic_id=topic_id,
                category=category,
                keywords=keywords,
                messages=[self.message_index],
                relevance_score=relevance,
                created=now,
                last_updated=now,
                summary=summary
            )
            self.topics[topic_id] = topic
        
        return topic_id
    
    async def _manage_memory_limits(self) -> None:
        """Manage memory limits by removing old/irrelevant items."""
        try:
            # Clean old entities
            if len(self.entities) > self.max_entities:
                # Sort by last mentioned and confidence
                sorted_entities = sorted(
                    self.entities.items(),
                    key=lambda x: (x[1].last_mentioned, x[1].confidence),
                    reverse=True
                )
                
                # Keep top entities
                entities_to_keep = dict(sorted_entities[:self.max_entities])
                entities_to_remove = set(self.entities.keys()) - set(entities_to_keep.keys())
                
                for entity_name in entities_to_remove:
                    del self.entities[entity_name]
                    if entity_name in self.entity_to_messages:
                        del self.entity_to_messages[entity_name]
            
            # Clean old topics  
            if len(self.topics) > self.max_topics:
                # Sort by last updated and relevance
                sorted_topics = sorted(
                    self.topics.items(),
                    key=lambda x: (x[1].last_updated, x[1].relevance_score),
                    reverse=True
                )
                
                # Keep top topics
                topics_to_keep = dict(sorted_topics[:self.max_topics])
                topics_to_remove = set(self.topics.keys()) - set(topics_to_keep.keys())
                
                for topic_id in topics_to_remove:
                    del self.topics[topic_id]
                    if topic_id in self.topic_to_messages:
                        del self.topic_to_messages[topic_id]
            
        except Exception as e:
            logger.error(f"Error managing memory limits: {str(e)}")
    
    def get_relevant_context(self, query: str, max_messages: int = 10) -> List[ConversationMessage]:
        """
        Get relevant context for a query.
        
        Args:
            query: Query to find relevant context for
            max_messages: Maximum messages to return
            
        Returns:
            List of relevant messages
        """
        try:
            # Simple keyword matching for now
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            scored_messages = []
            
            for i, message in enumerate(self.messages):
                score = 0.0
                
                # Content similarity
                message_words = set(message.content.lower().split())
                overlap = len(query_words.intersection(message_words))
                score += overlap / max(len(query_words), 1) * 0.4
                
                # Entity matching
                for entity_name in message.entities:
                    if entity_name.lower() in query_lower:
                        score += 0.3
                
                # Topic matching
                for topic_id in message.topics:
                    if topic_id in self.topics:
                        topic = self.topics[topic_id]
                        topic_overlap = len(query_words.intersection(topic.keywords))
                        score += topic_overlap / max(len(topic.keywords), 1) * 0.2
                
                # Importance boost
                score += message.importance_score * 0.1
                
                if score > 0:
                    scored_messages.append((score, message))
            
            # Sort by score and return top messages
            scored_messages.sort(key=lambda x: x[0], reverse=True)
            return [msg for score, msg in scored_messages[:max_messages]]
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return list(self.messages)[-max_messages:]
    
    def get_entity_history(self, entity_name: str) -> Optional[Entity]:
        """
        Get history for a specific entity.
        
        Args:
            entity_name: Name of entity
            
        Returns:
            Entity object or None
        """
        return self.entities.get(entity_name)
    
    def get_topic_summary(self, category: TopicCategory) -> Dict[str, Any]:
        """
        Get summary for a topic category.
        
        Args:
            category: Topic category
            
        Returns:
            Topic summary
        """
        category_topics = [topic for topic in self.topics.values() if topic.category == category]
        
        if not category_topics:
            return {"category": category.value, "topic_count": 0, "messages": 0}
        
        total_messages = sum(len(topic.messages) for topic in category_topics)
        avg_relevance = sum(topic.relevance_score for topic in category_topics) / len(category_topics)
        
        return {
            "category": category.value,
            "topic_count": len(category_topics),
            "messages": total_messages,
            "avg_relevance": avg_relevance,
            "keywords": list(set().union(*[topic.keywords for topic in category_topics]))
        }
    
    def export_memory(self) -> Dict[str, Any]:
        """
        Export memory state for persistence.
        
        Returns:
            Memory state dictionary
        """
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "entities": {name: entity.to_dict() for name, entity in self.entities.items()},
            "topics": {topic_id: topic.to_dict() for topic_id, topic in self.topics.items()},
            "message_index": self.message_index,
            "config": {
                "max_messages": self.max_messages,
                "max_entities": self.max_entities,
                "max_topics": self.max_topics,
                "entity_confidence_threshold": self.entity_confidence_threshold,
                "topic_relevance_threshold": self.topic_relevance_threshold
            }
        }
    
    def import_memory(self, memory_state: Dict[str, Any]) -> None:
        """
        Import memory state from persistence.
        
        Args:
            memory_state: Memory state dictionary
        """
        try:
            # Import messages
            self.messages.clear()
            for msg_data in memory_state.get("messages", []):
                self.messages.append(ConversationMessage.from_dict(msg_data))
            
            # Import entities
            self.entities.clear()
            for name, entity_data in memory_state.get("entities", {}).items():
                self.entities[name] = Entity.from_dict(entity_data)
            
            # Import topics
            self.topics.clear()
            for topic_id, topic_data in memory_state.get("topics", {}).items():
                self.topics[topic_id] = Topic.from_dict(topic_data)
            
            # Import config
            self.message_index = memory_state.get("message_index", 0)
            
            # Rebuild indices
            self._rebuild_indices()
            
            logger.info("Memory state imported successfully")
            
        except Exception as e:
            logger.error(f"Error importing memory state: {str(e)}")
    
    def _rebuild_indices(self) -> None:
        """Rebuild indices after import."""
        self.entity_to_messages.clear()
        self.topic_to_messages.clear()
        
        for i, message in enumerate(self.messages):
            for entity_name in message.entities:
                self.entity_to_messages[entity_name].add(i)
            
            for topic_id in message.topics:
                self.topic_to_messages[topic_id].add(i)
    
    def clear_memory(self) -> None:
        """Clear all memory."""
        self.messages.clear()
        self.entities.clear()
        self.topics.clear()
        self.entity_to_messages.clear()
        self.topic_to_messages.clear()
        self.message_index = 0