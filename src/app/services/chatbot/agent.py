"""
Legal Chatbot Agent implementation for Sprint 1.
Basic ReAct Agent with conversation memory and tool integration.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from ..llm_service import get_llm_service
from .tools import LegalSearchTool, BasicSummarizerTool, ValidatorTool

logger = logging.getLogger(__name__)


class LegalChatbotAgentV1:
    """
    Sprint 1 - Basic ReAct Agent implementation.
    
    Features:
    - ReAct reasoning pattern
    - 3 core tools (LegalSearch, Summarizer, Validator)
    - Conversation memory
    - Basic error handling
    - Execution monitoring
    """
    
    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id
        self.llm_service = None
        self.tools: List[BaseTool] = []
        self.memory: Optional[ConversationBufferWindowMemory] = None
        self.agent: Optional[AgentExecutor] = None
        
        # Initialize components
        self._initialize_llm_service()
        self._initialize_tools()
        self._initialize_memory()
        self._create_agent()
    
    def _initialize_llm_service(self) -> None:
        """Initialize LLM service connection."""
        try:
            self.llm_service = get_llm_service()
            logger.info("LLM service initialized for chatbot agent")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            raise RuntimeError(f"Cannot initialize chatbot agent: {str(e)}")
    
    def _initialize_tools(self) -> None:
        """Initialize the 3 core tools for Sprint 1."""
        try:
            self.tools = [
                LegalSearchTool(),
                BasicSummarizerTool(),
                ValidatorTool()
            ]
            logger.info(f"Initialized {len(self.tools)} tools for chatbot agent")
        except Exception as e:
            logger.error(f"Failed to initialize tools: {str(e)}")
            raise RuntimeError(f"Cannot initialize chatbot tools: {str(e)}")
    
    def _initialize_memory(self) -> None:
        """Initialize conversation memory."""
        try:
            self.memory = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 exchanges
                memory_key="chat_history",
                return_messages=True,
                input_key="input",
                output_key="output"
            )
            logger.info("Conversation memory initialized")
        except Exception as e:
            logger.error(f"Failed to initialize memory: {str(e)}")
            # Continue without memory if initialization fails
            self.memory = None
    
    def _create_agent(self) -> None:
        """Create the ReAct agent with tools and memory."""
        try:
            # Create the prompt template for ReAct agent
            prompt = self._create_agent_prompt()
            
            # Create the ReAct agent
            react_agent = create_react_agent(
                llm=self.llm_service.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            # Create agent executor with memory and configuration
            self.agent = AgentExecutor(
                agent=react_agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,  # Enable for debugging
                max_iterations=5,  # Limit iterations to prevent infinite loops
                max_execution_time=30,  # 30 second timeout
                early_stopping_method="force",  # Changed from "generate"
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            
            logger.info("ReAct agent created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create agent: {str(e)}")
            raise RuntimeError(f"Cannot create chatbot agent: {str(e)}")
    
    def _create_agent_prompt(self) -> PromptTemplate:
        """Create the prompt template for the ReAct agent."""
        
        template = """Anda adalah asisten hukum AI CourtSight yang membantu dengan pertanyaan hukum Indonesia.
Anda memiliki akses ke tools yang dapat membantu mencari dokumen hukum, meringkas teks, dan memvalidasi klaim.

TOOLS TERSEDIA:
{tools}

Tool Names: {tool_names}

INSTRUKSI:
1. Analisis pertanyaan pengguna dengan cermat
2. Gunakan tools yang tersedia untuk mencari informasi yang relevan
3. Berikan jawaban yang akurat dengan sitasi yang jelas
4. Gunakan bahasa Indonesia yang formal namun mudah dipahami
5. Jika tidak yakin, gunakan validator tool untuk memverifikasi informasi

FORMAT REASONING:
Gunakan format berikut untuk reasoning Anda:

Thought: [Analisis pertanyaan dan rencana tindakan]
Action: [Tool yang akan digunakan]
Action Input: [Input untuk tool]
Observation: [Hasil dari tool]
... (ulangi jika perlu menggunakan tool lain)
Thought: [Analisis final berdasarkan semua informasi]
Final Answer: [Jawaban akhir dengan sitasi]

CONTOH:
Thought: Pengguna bertanya tentang dasar hukum sengketa tanah. Saya perlu mencari putusan dan peraturan terkait.
Action: legal_search
Action Input: dasar hukum sengketa tanah putusan mahkamah agung
Observation: [Hasil pencarian]
Thought: Sekarang saya akan memvalidasi informasi yang ditemukan.
Action: validator
Action Input: [klaim yang perlu divalidasi]
Observation: [Hasil validasi]
Final Answer: [Jawaban lengkap dengan sitasi]

CHAT HISTORY:
{chat_history}

PERTANYAAN: {input}
{agent_scratchpad}"""
        
        return PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad", "chat_history"],
            template=template
        )
    
    def process_question(self, question: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a user question and return the agent's response.
        
        Args:
            question: User's question
            user_id: Optional user ID for context
            
        Returns:
            Dictionary containing response and metadata
        """
        if not self.agent:
            return {
                "answer": "Error: Chatbot agent is not properly initialized.",
                "success": False,
                "error": "Agent initialization failed"
            }
        
        start_time = time.time()
        
        try:
            # Prepare input for the agent
            agent_input = {
                "input": question
            }
            
            # Execute the agent
            result = self.agent.invoke(agent_input)
            
            processing_time = time.time() - start_time
            
            # Extract information from the result
            answer = result.get("output", "No response generated")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Format the response
            response = {
                "answer": answer,
                "conversation_id": self.conversation_id,
                "processing_time": processing_time,
                "reasoning_steps": self._format_reasoning_steps(intermediate_steps),
                "tool_calls": self._extract_tool_calls(intermediate_steps),
                "success": True,
                "timestamp": time.time()
            }
            
            logger.info(f"Question processed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing question: {str(e)}")
            
            return {
                "answer": f"Maaf, terjadi kesalahan dalam memproses pertanyaan Anda: {str(e)}",
                "conversation_id": self.conversation_id,
                "processing_time": processing_time,
                "reasoning_steps": [],
                "tool_calls": [],
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _format_reasoning_steps(self, intermediate_steps: List) -> List[Dict[str, Any]]:
        """Format intermediate steps into reasoning steps."""
        reasoning_steps = []
        
        for i, (action, observation) in enumerate(intermediate_steps):
            step = {
                "step_number": i + 1,
                "action": str(action),
                "tool_used": getattr(action, 'tool', None),
                "tool_input": getattr(action, 'tool_input', None),
                "result": str(observation),
                "timestamp": time.time()
            }
            reasoning_steps.append(step)
        
        return reasoning_steps
    
    def _extract_tool_calls(self, intermediate_steps: List) -> List[Dict[str, Any]]:
        """Extract tool call information from intermediate steps."""
        tool_calls = []
        
        for action, observation in intermediate_steps:
            if hasattr(action, 'tool'):
                tool_call = {
                    "tool_name": action.tool,
                    "input_data": getattr(action, 'tool_input', {}),
                    "output_data": str(observation),
                    "execution_time": 0.0,  # Could be measured if needed
                    "success": True
                }
                tool_calls.append(tool_call)
        
        return tool_calls
    
    def get_conversation_memory(self) -> List[Dict[str, Any]]:
        """Get the current conversation memory."""
        if not self.memory:
            return []
        
        try:
            messages = self.memory.chat_memory.messages
            formatted_messages = []
            
            for message in messages:
                formatted_messages.append({
                    "role": message.type,
                    "content": message.content,
                    "timestamp": getattr(message, 'timestamp', time.time())
                })
            
            return formatted_messages
            
        except Exception as e:
            logger.error(f"Error retrieving conversation memory: {str(e)}")
            return []
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the agent components."""
        status = {
            "llm_service": bool(self.llm_service),
            "tools_count": len(self.tools),
            "memory_initialized": bool(self.memory),
            "agent_initialized": bool(self.agent),
            "overall_status": "healthy"
        }
        
        if not all([status["llm_service"], status["agent_initialized"]]):
            status["overall_status"] = "unhealthy"
        
        return status