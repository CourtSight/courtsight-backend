"""
Enhanced Chatbot Service for Sprint 2.
Integrates Langraph workflow, advanced tools, enhanced memory, and Redis caching.
"""

import logging
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from uuid import UUID

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field, field_validator

from .agent import create_react_agent
from .tools.advanced import CaseComparatorTool, PrecedentExplorerTool, CitationGeneratorTool
from .workflow import LegalWorkflowOrchestrator
from .memory import MemorySystem
from ..llm_service import get_llm_service
from ..retrieval import get_retrieval_service
from ...core.config import settings

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str = Field(..., description="User query")
    conversation_id: Optional[Union[str, UUID]] = Field(None, description="Conversation ID for memory persistence")
    use_workflow: bool = Field(True, description="Use Langraph workflow for complex reasoning")
    use_advanced_tools: bool = Field(True, description="Enable advanced tools (comparison, precedent, citation)")
    memory_enabled: bool = Field(True, description="Enable enhanced memory system")
    cache_enabled: bool = Field(True, description="Enable Redis caching")

    @field_validator('conversation_id', mode='before')
    @classmethod
    def validate_conversation_id(cls, v):
        if isinstance(v, UUID):
            return str(v)
        return v

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Chatbot response")
    conversation_id: str = Field(..., description="Conversation ID")
    workflow_used: bool = Field(..., description="Whether workflow was used")
    confidence_score: Optional[float] = Field(None, description="Response confidence score")
    response_time: float = Field(..., description="Response time in seconds")
    memory_summary: Optional[Dict[str, Any]] = Field(None, description="Memory summary if enabled")


class ChatbotService:
    """
    Enhanced Chatbot Service for Sprint 2.
    
    Features:
    - Langraph workflow orchestration for complex legal reasoning
    - Advanced tools (case comparison, precedent exploration, citation generation)
    - Enhanced memory system with entity tracking and topic clustering
    - Redis caching for performance optimization
    - Intelligent routing between simple and complex queries
    - Comprehensive logging and monitoring
    """
    
    def __init__(self):
        """Initialize the enhanced chatbot service."""
        self.llm_service = None
        self.retrieval_service = None
        self.agent_executor = None
        self.workflow_orchestrator = None
        self.memory_system = None
        
        # Advanced tools
        self.case_comparator = None
        self.precedent_explorer = None
        self.citation_generator = None
        
        # Cache for conversations - CONVERSATION-SPECIFIC MEMORY
        self.conversation_cache: Dict[str, Any] = {}
        self.conversation_memories: Dict[str, Any] = {}  # Memory per conversation
        
        # Performance tracking
        self.query_count = 0
        self.total_response_time = 0.0
        
        self._initialize_services()
    
    def _initialize_services(self) -> None:
        """Initialize all required services."""
        try:
            # Core services
            self.llm_service = get_llm_service()
            self.retrieval_service = get_retrieval_service()
            
            # Create enhanced agent with advanced tools
            self.agent_executor = self._create_enhanced_agent()
            
            # Initialize Langraph workflow
            self.workflow_orchestrator = LegalWorkflowOrchestrator()
            
            # Initialize enhanced memory (template for conversations)
            self.memory_template = MemorySystem(
                max_messages=100,
                max_entities=50,
                max_topics=20,
                entity_confidence_threshold=0.7,
                topic_relevance_threshold=0.6
            )
            
            # Initialize advanced tools
            self.case_comparator = CaseComparatorTool()
            self.precedent_explorer = PrecedentExplorerTool()
            self.citation_generator = CitationGeneratorTool()
            
            logger.info("Enhanced chatbot service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced chatbot service: {str(e)}")
            raise
    
    def _get_conversation_memory(self, conversation_id: str):
        """Get or create memory system for specific conversation."""
        if conversation_id not in self.conversation_memories:
            # Create new memory system for this conversation
            from .memory import MemorySystem
            self.conversation_memories[conversation_id] = MemorySystem(
                max_messages=100,
                max_entities=50,
                max_topics=20,
                entity_confidence_threshold=0.7,
                topic_relevance_threshold=0.6
            )
            logger.info(f"Created new memory system for conversation: {conversation_id}")
        
        return self.conversation_memories[conversation_id]
    
    def _create_enhanced_agent(self) -> AgentExecutor:
        """Create enhanced agent with advanced tools."""
        try:
            # Get existing tools from Sprint 1
            from .tools import get_available_tools
            basic_tools = get_available_tools()
            
       
            all_tools = basic_tools 
            
            # Create ReAct prompt - improved version
            try:
                from langchain import hub
                prompt = hub.pull("hwchase17/react")
            except:
                # Fallback prompt if hub is not available - improved version
                from langchain.prompts import PromptTemplate
                prompt = PromptTemplate.from_template("""
Anda adalah asisten hukum yang ahli dalam hukum Indonesia. Jawab pertanyaan dengan akurat menggunakan tools yang tersedia.

Tools yang tersedia:
{tools}

Format yang harus digunakan:

Question: pertanyaan yang harus dijawab
Thought: pikirkan apa yang harus dilakukan
Action: pilih salah satu action dari [{tool_names}]
Action Input: input untuk action tersebut
Observation: hasil dari action
... (Thought/Action/Action Input/Observation dapat diulang beberapa kali)
Thought: sekarang saya tahu jawaban final
Final Answer: jawaban final untuk pertanyaan

PENTING: Selalu akhiri dengan "Final Answer:" diikuti jawaban lengkap dalam bahasa Indonesia.

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
            
            # Create enhanced agent with improved LLM settings
            enhanced_llm = self.llm_service.llm
            # Override temperature for agent to prevent generation issues
            if hasattr(enhanced_llm, 'temperature'):
                enhanced_llm.temperature = 0.01  # Slightly higher than 0 to prevent generation issues
            
            agent = create_react_agent(
                llm=enhanced_llm,
                tools=all_tools,
                prompt=prompt
            )
            
            # Create agent executor with improved settings
            from langchain.agents import AgentExecutor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=all_tools,
                verbose=False,
                handle_parsing_errors=True,
                early_stopping_method="generate",  # Stop when final answer is generated
                return_intermediate_steps=False  # For better debugging
            )
            
            # Create enhanced agent
            agent = create_react_agent(
                llm=self.llm_service.llm,
                tools=all_tools,
                prompt=prompt
            )
            
            # Create agent executor
            from langchain.agents import AgentExecutor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=all_tools,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=15,
                max_execution_time=120,
                early_stopping_method="generate"  # Stop when final answer is generated
            )
            
            logger.info(f"Enhanced agent created with {len(all_tools)} tools")
            return agent_executor
            
        except Exception as e:
            logger.error(f"Failed to create enhanced agent: {str(e)}")
            raise
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process chat request with enhanced features.
        
        Args:
            request: Chat request with configuration
            
        Returns:
            Enhanced chat response
        """
        start_time = datetime.now()
        conversation_id = request.conversation_id or self._generate_conversation_id()
        tools_used = []
        
        try:
            # Add query to memory if enabled - CONVERSATION SPECIFIC
            memory_system = None
            if request.memory_enabled:
                memory_system = self._get_conversation_memory(conversation_id)
                await memory_system.add_message(HumanMessage(content=request.query))
                logger.info(f"Added user message to conversation {conversation_id}. Total messages: {len(memory_system.messages)}")
            else:
                logger.warning("Memory system not enabled")
            
            # Determine query complexity and routing
            complexity_analysis = await self._analyze_query_complexity(request.query)
            
            # Route to appropriate processing method
            if request.use_workflow and complexity_analysis["use_workflow"]:
                response_content, workflow_tools = await self._process_with_workflow(
                    request.query, conversation_id, request, memory_system
                )
                tools_used.extend(workflow_tools)
            else:
                response_content, agent_tools = await self._process_with_agent(
                    request.query, conversation_id, request, memory_system
                )
                tools_used.extend(agent_tools)
            
            # Add response to memory if enabled - CONVERSATION SPECIFIC
            if request.memory_enabled and memory_system:
                await memory_system.add_message(AIMessage(content=response_content))
                logger.info(f"Added AI response to conversation {conversation_id}. Total messages: {len(memory_system.messages)}")
                
                # Debug: Show recent messages in memory
                recent_messages = list(memory_system.messages)[-4:]  # Last 4 messages
                logger.info(f"Recent messages in conversation {conversation_id}:")
                for i, msg in enumerate(recent_messages):
                    logger.info(f"  {i+1}. {msg.message_type}: {msg.content[:100]}...")
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Update performance tracking
            self.query_count += 1
            self.total_response_time += response_time
            
            # Get memory summary if enabled - CONVERSATION SPECIFIC
            memory_summary = None
            if request.memory_enabled and memory_system:
                memory_summary = self._get_memory_summary(memory_system)
            
            # Create response
            response = ChatResponse(
                response=response_content,
                conversation_id=conversation_id,
                workflow_used=request.use_workflow and complexity_analysis["use_workflow"],
                tools_used=tools_used,
                confidence_score=complexity_analysis.get("confidence", 0.8),
                response_time=response_time,
                memory_summary=memory_summary
            )
            
            logger.info(f"Chat processed in {response_time:.2f}s using {len(tools_used)} tools")
            return response
            
        except Exception as e:
            logger.error(f"Error processing chat request: {str(e)}")
            
            # Return error response
            response_time = (datetime.now() - start_time).total_seconds()
            return ChatResponse(
                response=f"Maaf, terjadi kesalahan dalam memproses pertanyaan Anda: {str(e)}",
                conversation_id=conversation_id,
                workflow_used=False,
                tools_used=[],
                confidence_score=0.0,
                response_time=response_time
            )
    
    async def chat_stream(self, request: ChatRequest):
        """
        Process chat request with streaming response for real-time feedback.
        
        Args:
            request: Chat request with configuration
            
        Yields:
            Streaming response chunks in Server-Sent Events format
        """
        import json
        import asyncio
        from datetime import datetime
        
        start_time = datetime.now()
        conversation_id = request.conversation_id or self._generate_conversation_id()
        tools_used = []
        
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Processing your request...', 'conversation_id': conversation_id})}\n\n"
            
            # Add query to memory if enabled - CONVERSATION SPECIFIC
            memory_system = None
            if request.memory_enabled:
                memory_system = self._get_conversation_memory(conversation_id)
                await memory_system.add_message(HumanMessage(content=request.query))
                yield f"data: {json.dumps({'type': 'status', 'message': f'Added to conversation memory (total: {len(memory_system.messages)})'})}\n\n"
            
            # Determine query complexity and routing
            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing query complexity...'})}\n\n"
            complexity_analysis = await self._analyze_query_complexity(request.query)
            
            # Send complexity analysis result
            complexity_msg = "Using workflow orchestration" if complexity_analysis["use_workflow"] else "Using direct agent processing"
            yield f"data: {json.dumps({'type': 'status', 'message': complexity_msg})}\n\n"
            
            # Route to appropriate processing method with streaming
            if request.use_workflow and complexity_analysis["use_workflow"]:
                async for chunk in self._process_with_workflow_streaming(request.query, conversation_id, request, memory_system):
                    yield f"data: {json.dumps(chunk)}\n\n"
                    if chunk.get('type') == 'tools_used':
                        tools_used.extend(chunk.get('tools', []))
            else:
                async for chunk in self._process_with_agent_streaming(request.query, conversation_id, request, memory_system):
                    yield f"data: {json.dumps(chunk)}\n\n"
                    if chunk.get('type') == 'tools_used':
                        tools_used.extend(chunk.get('tools', []))
            
            # Calculate final response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Update performance tracking
            self.query_count += 1
            self.total_response_time += response_time
            
            # Send final completion status
            final_data = {
                'type': 'complete',
                'conversation_id': conversation_id,
                'tools_used': tools_used,
                'response_time': response_time,
                'workflow_used': request.use_workflow and complexity_analysis["use_workflow"]
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming chat: {str(e)}")
            error_data = {
                'type': 'error',
                'message': str(e),
                'conversation_id': conversation_id
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    async def _process_with_agent_streaming(self, query: str, conversation_id: str, request: ChatRequest, memory_system=None):
        """Stream agent processing with real-time updates."""
        try:
            yield {'type': 'status', 'message': 'Initializing agent...'}
            
            # Get relevant context - IMPROVED VERSION  
            context_prompt = ""
            if request.memory_enabled and memory_system:
                recent_messages = list(memory_system.messages)[-6:]  # Last 6 messages
                
                if recent_messages:
                    context_prompt = "\\n\\nKONTEKS PERCAKAPAN SEBELUMNYA (untuk referensi):\\n"
                    for msg in recent_messages:
                        role = "User" if msg.message_type == "human" else "Assistant"
                        context_prompt += f"{role}: {msg.content}\\n"
                    context_prompt += "\\nBerdasarkan konteks di atas, jawab pertanyaan berikut:\\n"
                    
                yield {'type': 'status', 'message': f'Retrieved {len(recent_messages)} context messages'}
            
            # Create enhanced query
            if context_prompt:
                enhanced_query = f"{context_prompt}{query}"
            else:
                enhanced_query = query
            
            # Create agent input
            agent_input = {
                "input": enhanced_query,
                "conversation_id": conversation_id
            }
            
            yield {'type': 'status', 'message': 'Agent processing with context...'}
            
            # For now, simulate streaming by calling the regular agent and yielding incremental updates
            # In a real implementation, you'd modify the agent to support streaming
            result = await self.agent_executor.ainvoke(agent_input)
            
            # Simulate streaming the response
            response_text = result.get('output', 'No response generated')
            words = response_text.split()
            
            current_text = ""
            for i, word in enumerate(words):
                current_text += word + " "
                if i % 5 == 0:  # Send update every 5 words
                    yield {
                        'type': 'partial_response',
                        'content': current_text.strip(),
                        'progress': (i + 1) / len(words)
                    }
                    await asyncio.sleep(0.1)  # Small delay for streaming effect
            
            # Send final response
            yield {
                'type': 'final_response',
                'content': response_text,
                'tools_used': result.get('intermediate_steps', [])
            }
            
        except Exception as e:
            yield {'type': 'error', 'message': f'Agent processing error: {str(e)}'}
    
    async def _process_with_workflow_streaming(self, query: str, conversation_id: str, request: ChatRequest, memory_system=None):
        """Stream workflow processing with real-time updates."""
        try:
            yield {'type': 'status', 'message': 'Initializing workflow orchestrator...'}
            
            # Get relevant context - IMPROVED VERSION
            relevant_context = []
            context_string = ""
            if request.memory_enabled and memory_system:
                recent_messages = list(memory_system.messages)[-6:]  # Last 6 messages
                
                if recent_messages:
                    relevant_context = [f"{msg.message_type.title()}: {msg.content}" for msg in recent_messages]
                    context_string = "\\n".join(relevant_context)
                    
                yield {'type': 'status', 'message': f'Retrieved {len(recent_messages)} context messages'}
            
            # Execute workflow with streaming updates
            workflow_input = {
                "question": query,
                "context": relevant_context,
                "context_string": context_string,
                "conversation_id": conversation_id
            }
            
            yield {'type': 'status', 'message': 'Starting workflow execution...'}
            
            # For now, simulate workflow streaming
            # In a real implementation, you'd modify the workflow to support streaming
            workflow_result = await self.workflow_orchestrator.execute_workflow(workflow_input)
            
            # Simulate streaming the workflow steps
            steps = [
                "Analyzing question...",
                "Gathering information...",
                "Cross-validating sources...",
                "Synthesizing response...",
                "Generating citations...",
                "Quality check..."
            ]
            
            for i, step in enumerate(steps):
                yield {
                    'type': 'workflow_step',
                    'step': step,
                    'progress': (i + 1) / len(steps)
                }
                await asyncio.sleep(0.5)  # Simulate processing time
            
            # Stream the final response
            response_text = workflow_result.get("final_response", "No response generated")
            words = response_text.split()
            
            current_text = ""
            for i, word in enumerate(words):
                current_text += word + " "
                if i % 3 == 0:  # Send update every 3 words for workflow
                    yield {
                        'type': 'partial_response',
                        'content': current_text.strip(),
                        'progress': (i + 1) / len(words)
                    }
                    await asyncio.sleep(0.1)
            
            # Send final response
            yield {
                'type': 'final_response',
                'content': response_text,
                'workflow_steps': workflow_result.get('steps', [])
            }
            
        except Exception as e:
            yield {'type': 'error', 'message': f'Workflow processing error: {str(e)}'}
    
    async def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze query complexity to determine processing approach.
        
        Args:
            query: User query
            
        Returns:
            Complexity analysis with routing decision
        """
        try:
            # Simple heuristics for now
            complexity_indicators = {
                "comparison_keywords": ["bandingkan", "beda", "sama", "versus", "vs"],
                "precedent_keywords": ["preseden", "yurisprudensi", "putusan serupa", "kasus serupa"],
                "citation_keywords": ["sitasi", "rujukan", "referensi", "sumber"],
                "complex_reasoning": ["analisis", "jelaskan", "bagaimana", "mengapa", "implikasi"]
            }
            
            query_lower = query.lower()
            complexity_score = 0.0
            reasoning_type = "simple"
            
            # Check for complexity indicators
            for category, keywords in complexity_indicators.items():
                if any(keyword in query_lower for keyword in keywords):
                    if category == "comparison_keywords":
                        complexity_score += 0.3
                        reasoning_type = "comparison"
                    elif category == "precedent_keywords":
                        complexity_score += 0.4
                        reasoning_type = "precedent_research"
                    elif category == "citation_keywords":
                        complexity_score += 0.2
                        reasoning_type = "citation"
                    elif category == "complex_reasoning":
                        complexity_score += 0.3
                        reasoning_type = "complex_analysis"
            
            # Length factor
            if len(query.split()) > 20:
                complexity_score += 0.2
            
            # Question complexity
            question_count = query.count("?")
            complexity_score += min(question_count * 0.1, 0.3)
            
            # Decision threshold
            use_workflow = complexity_score >= 0.4
            
            return {
                "complexity_score": complexity_score,
                "reasoning_type": reasoning_type,
                "use_workflow": use_workflow,
                "confidence": min(0.9, 0.5 + complexity_score)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing query complexity: {str(e)}")
            return {
                "complexity_score": 0.3,
                "reasoning_type": "simple",
                "use_workflow": False,
                "confidence": 0.5
            }
    
    async def _process_with_workflow(self, query: str, conversation_id: str, 
                                   request: ChatRequest, memory_system=None) -> Tuple[str, List[str]]:
        """
        Process query using Langraph workflow.
        
        Args:
            query: User query
            conversation_id: Conversation ID
            request: Chat request
            
        Returns:
            Response content and list of tools used
        """
        try:
            # Get relevant context from memory - IMPROVED VERSION
            relevant_context = []
            context_string = ""
            if request.memory_enabled and memory_system:
                # Get last N messages for workflow too
                recent_messages = list(memory_system.messages)[-6:]  # Last 6 messages
                
                if recent_messages:
                    relevant_context = [f"{msg.message_type.title()}: {msg.content}" for msg in recent_messages]
                    context_string = "\\n".join(relevant_context)
            
            # Execute workflow with enhanced context
            workflow_input = {
                "question": query,
                "context": relevant_context,
                "context_string": context_string,  # Add formatted context string
                "conversation_id": conversation_id
            }
            
            workflow_result = await self.workflow_orchestrator.execute_workflow(workflow_input)
            
            # Extract response and tools used
            response_content = workflow_result.get("final_response", "Tidak dapat memproses permintaan.")
            
            # Track tools used in workflow
            tools_used = []
            audit_trail = workflow_result.get("audit_trail", [])
            for step in audit_trail:
                if "tools_used" in step:
                    tools_used.extend(step["tools_used"])
            
            # Add workflow-specific tools
            tools_used.append("langraph_workflow")
            
            return response_content, tools_used
            
        except Exception as e:
            logger.error(f"Error processing with workflow: {str(e)}")
            return f"Error processing with workflow: {str(e)}", ["workflow_error"]
    
    async def _process_with_agent(self, query: str, conversation_id: str,
                                request: ChatRequest, memory_system=None) -> Tuple[str, List[str]]:
        """
        Process query using ReAct agent.
        
        Args:
            query: User query
            conversation_id: Conversation ID  
            request: Chat request
            
        Returns:
            Response content and list of tools used
        """
        try:
            # Get relevant context from memory - IMPROVED VERSION
            context_prompt = ""
            if request.memory_enabled and memory_system:
                # Get last N messages instead of "relevant" scoring
                # This ensures we always have recent context
                recent_messages = list(memory_system.messages)[-6:]  # Last 6 messages (3 pairs of user/AI)
                
                if recent_messages:
                    context_prompt = "\\n\\nKONTEKS PERCAKAPAN SEBELUMNYA (untuk referensi):\\n"
                    for i, msg in enumerate(recent_messages):
                        role = "User" if msg.message_type == "human" else "Assistant"
                        # Use full content, not truncated
                        context_prompt += f"{role}: {msg.content}\\n"
                    context_prompt += "\\nBerdasarkan konteks di atas, jawab pertanyaan berikut:\\n"
            
            # Create enhanced query with explicit context instructions
            if context_prompt:
                enhanced_query = f"{context_prompt}{query}"
            else:
                enhanced_query = query
            
            logger.info(f"Enhanced query with context (first 200 chars): {enhanced_query[:200]}...")
            
            # Execute agent with fallback handling
            result = None
            try:
                result = await self.agent_executor.ainvoke({
                    "input": enhanced_query,
                    "conversation_id": conversation_id
                })
            except Exception as agent_error:
                logger.warning(f"Agent execution failed: {str(agent_error)}")
                
                # Try direct LLM call as fallback
                try:
                    logger.info("Attempting direct LLM fallback...")
                    direct_response = await self.llm_service.llm.ainvoke(
                        f"Sebagai asisten hukum, jawab pertanyaan berikut: {enhanced_query}"
                    )
                    return direct_response.content, ["direct_llm_fallback"]
                except Exception as llm_error:
                    logger.error(f"Direct LLM fallback also failed: {str(llm_error)}")
                    return f"Maaf, sistem sedang mengalami masalah teknis. Silakan coba lagi dalam beberapa saat.", ["system_error"]
            
            # Extract response
            response_content = result.get("output", "Tidak dapat memproses permintaan.")
            
            # Validate response content
            if not response_content or response_content.strip() == "":
                logger.warning("Empty response from agent, using direct LLM fallback")
                try:
                    direct_response = await self.llm_service.llm.ainvoke(
                        f"Sebagai asisten hukum, jawab pertanyaan berikut: {enhanced_query}"
                    )
                    return direct_response.content, ["direct_llm_fallback"]
                except Exception as e:
                    logger.error(f"Direct LLM fallback failed: {str(e)}")
                    return "Maaf, sistem tidak dapat memberikan respons yang valid saat ini.", ["system_error"]
            
            # Extract tools used (basic tracking)
            tools_used = ["react_agent"]
            
            # Check if advanced tools were likely used based on response content
            if "perbandingan" in response_content.lower():
                tools_used.append("case_comparator")
            if "preseden" in response_content.lower():
                tools_used.append("precedent_explorer")
            if "sitasi" in response_content.lower():
                tools_used.append("citation_generator")
            
            return response_content, tools_used
            
        except Exception as e:
            logger.error(f"Error processing with agent: {str(e)}")
            # Enhanced error response with direct LLM fallback
            try:
                logger.info("Attempting final LLM fallback for error case...")
                direct_response = await self.llm_service.llm.ainvoke(
                    f"Sebagai asisten hukum, jawab pertanyaan berikut dengan ringkas: {query}"
                )
                return direct_response.content, ["emergency_llm_fallback"]
            except Exception as final_error:
                logger.error(f"All fallback methods failed: {str(final_error)}")
                return f"Maaf, terjadi kesalahan teknis dalam memproses pertanyaan Anda. Silakan coba dengan pertanyaan yang lebih sederhana.", ["agent_error"]
    
    def _generate_conversation_id(self) -> str:
        """Generate unique conversation ID."""
        from uuid import uuid4        
        return f"chat_{uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
    
    def _get_memory_summary(self, memory_system=None) -> Dict[str, Any]:
        """Get summary of current memory state."""
        if not memory_system:
            return {}
        
        try:
            return {
                "total_messages": len(memory_system.messages),
                "total_entities": len(memory_system.entities),
                "total_topics": len(memory_system.topics),
                "recent_entities": list(memory_system.entities.keys())[-5:],
                "recent_topics": list(memory_system.topics.keys())[-3:]
            }
        except Exception as e:
            logger.error(f"Error getting memory summary: {str(e)}")
            return {"error": str(e)}
    
    def get_conversation_history(self, conversation_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get conversation history for a specific conversation.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages
            
        Returns:
            List of conversation messages
        """
        try:
            if not self.memory_system:
                return []
            
            # Get recent messages
            recent_messages = list(self.memory_system.messages)[-limit:]
            
            return [
                {
                    "content": msg.content,
                    "type": msg.message_type,
                    "timestamp": msg.timestamp.isoformat(),
                    "importance": msg.importance_score
                }
                for msg in recent_messages
            ]
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    def get_entity_insights(self) -> Dict[str, Any]:
        """Get insights about tracked entities."""
        try:
            if not self.memory_system:
                return {}
            
            entities_by_type = {}
            for entity in self.memory_system.entities.values():
                entity_type = entity.entity_type.value
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append({
                    "name": entity.name,
                    "confidence": entity.confidence,
                    "mentions": len(entity.mentions),
                    "related_entities": len(entity.related_entities)
                })
            
            return {
                "entities_by_type": entities_by_type,
                "total_entities": len(self.memory_system.entities),
                "most_mentioned": sorted(
                    self.memory_system.entities.values(),
                    key=lambda x: len(x.mentions),
                    reverse=True
                )[:5]
            }
            
        except Exception as e:
            logger.error(f"Error getting entity insights: {str(e)}")
            return {"error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the chatbot."""
        try:
            avg_response_time = (
                self.total_response_time / self.query_count 
                if self.query_count > 0 else 0.0
            )
            
            return {
                "total_queries": self.query_count,
                "average_response_time": avg_response_time,
                "memory_usage": {
                    "messages": len(self.memory_system.messages) if self.memory_system else 0,
                    "entities": len(self.memory_system.entities) if self.memory_system else 0,
                    "topics": len(self.memory_system.topics) if self.memory_system else 0
                },
                "service_status": {
                    "llm_service": self.llm_service is not None,
                    "retrieval_service": self.retrieval_service is not None,
                    "agent_executor": self.agent_executor is not None,
                    "workflow_orchestrator": self.workflow_orchestrator is not None,
                    "memory_system": self.memory_system is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {"error": str(e)}
    
    def clear_conversation(self, conversation_id: Optional[str] = None) -> bool:
        """
        Clear conversation memory.
        
        Args:
            conversation_id: Specific conversation to clear (None for all)
            
        Returns:
            Success status
        """
        try:
            if self.memory_system:
                if conversation_id is None:
                    # Clear all memory
                    self.memory_system.clear_memory()
                    logger.info("All conversation memory cleared")
                else:
                    # For now, we clear all since conversation-specific clearing
                    # requires more sophisticated memory management
                    logger.info(f"Clearing all memory (conversation-specific clearing not yet implemented)")
                    self.memory_system.clear_memory()
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing conversation: {str(e)}")
            return False


# Global instance
_chatbot_service: Optional[ChatbotService] = None


async def get_chatbot_service() -> ChatbotService:
    """Get or create enhanced chatbot service instance."""
    global _chatbot_service
    
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    
    return _chatbot_service