"""
Langraph Workflow Implementation for Sprint 2.
State machine workflow for complex legal reasoning with audit trail.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langchain_core.tools import BaseTool

from ..llm_service import get_llm_service
from ...core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LegalReasoningState(TypedDict):
    """State structure for the legal reasoning workflow."""
    
    # Input
    question: str
    user_id: Optional[int]
    conversation_id: Optional[str]
    
    # Analysis phase
    question_analysis: Dict[str, Any]
    complexity_score: float
    required_tools: List[str]
    
    # Information gathering phase
    gathered_information: Annotated[List[Dict[str, Any]], "add_info"]
    tool_results: Dict[str, Any]
    
    # Cross-validation phase
    cross_validation_results: Dict[str, Any]
    confidence_score: float
    
    # Synthesis phase
    synthesized_answer: str
    key_points: List[str]
    
    # Citation phase
    citations: List[Dict[str, Any]]
    formatted_citations: str
    
    # Quality check phase
    quality_score: float
    quality_issues: List[str]
    
    # Workflow metadata
    reasoning_path: Annotated[List[str], "add_step"]
    execution_time: float
    step_timings: Dict[str, float]
    error_log: List[str]


class LegalWorkflowOrchestrator:
    """
    Sprint 2 - Langraph workflow orchestrator for complex legal reasoning.
    
    Features:
    - 6-step state machine workflow
    - Audit trail and execution monitoring
    - Dynamic tool selection based on question complexity
    - Quality assurance and confidence scoring
    - Error handling and recovery
    """
    
    def __init__(self):
        self.llm_service = None
        self.available_tools: Dict[str, BaseTool] = {}
        self.workflow: Optional[StateGraph] = None
        
        # Initialize components
        self._initialize_llm_service()
        self._initialize_workflow()
    
    def _initialize_llm_service(self) -> None:
        """Initialize LLM service connection."""
        try:
            self.llm_service = get_llm_service()
            logger.info("LLM service initialized for workflow orchestrator")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            raise RuntimeError(f"Cannot initialize workflow orchestrator: {str(e)}")
    
    def _initialize_workflow(self) -> None:
        """Create the Langraph state machine workflow."""
        try:
            # Create state graph
            workflow = StateGraph(LegalReasoningState)
            
            # Add workflow nodes
            workflow.add_node("analyze_question", self.analyze_question)
            workflow.add_node("gather_information", self.gather_information)
            workflow.add_node("cross_validate", self.cross_validate)
            workflow.add_node("synthesize_answer", self.synthesize_answer)
            workflow.add_node("generate_citations", self.generate_citations)
            workflow.add_node("quality_check", self.quality_check)
            
            # Define workflow transitions
            workflow.set_entry_point("analyze_question")
            
            # Conditional edge from analysis
            workflow.add_conditional_edges(
                "analyze_question",
                self._route_after_analysis,
                {
                    "simple": "synthesize_answer",     # Simple questions go directly to synthesis
                    "complex": "gather_information",   # Complex questions need information gathering
                    "error": END                       # Error handling
                }
            )
            
            # Sequential edges for complex workflow
            workflow.add_edge("gather_information", "cross_validate")
            workflow.add_edge("cross_validate", "synthesize_answer")
            workflow.add_edge("synthesize_answer", "generate_citations")
            workflow.add_edge("generate_citations", "quality_check")
            workflow.add_edge("quality_check", END)
            
            # Compile workflow
            self.workflow = workflow.compile()
            
            logger.info("Langraph workflow initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize workflow: {str(e)}")
            raise RuntimeError(f"Cannot create workflow: {str(e)}")
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool for use in the workflow."""
        self.available_tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    async def execute_workflow(self, workflow_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow with input parameters (service interface).
        
        Args:
            workflow_input: Dictionary containing:
                - question: The legal question to process
                - context: Optional context information  
                - conversation_id: Optional conversation ID
                
        Returns:
            Workflow execution results formatted for service consumption
        """
        try:
            question = workflow_input.get("question", "")
            context = workflow_input.get("context", [])
            conversation_id = workflow_input.get("conversation_id")
            
            if not question:
                return {
                    "final_response": "Pertanyaan tidak ditemukan",
                    "success": False,
                    "error": "No question provided"
                }
            
            # Add context to question if provided
            enhanced_question = question
            if context:
                context_str = " ".join(context[:3])  # Limit context
                enhanced_question = f"{question}\\n\\nKonteks: {context_str}"
            
            # Execute the workflow
            result = await self.process_question(
                question=enhanced_question,
                conversation_id=conversation_id
            )
            
            # Adapt response format for service consumption
            if result.get("success", False):
                return {
                    "final_response": result.get("answer", ""),
                    "confidence_score": result.get("confidence_score", 0.0),
                    "quality_score": result.get("quality_score", 0.0),
                    "citations": result.get("citations", ""),
                    "key_points": result.get("key_points", []),
                    "reasoning_path": result.get("reasoning_path", []),
                    "execution_time": result.get("execution_time", 0.0),
                    "tool_calls": result.get("tool_calls", []),
                    "audit_trail": [
                        {
                            "step": "workflow_execution",
                            "tools_used": [tool.get("name", "") for tool in result.get("tool_calls", [])],
                            "execution_time": result.get("execution_time", 0.0),
                            "quality_score": result.get("quality_score", 0.0)
                        }
                    ],
                    "success": True
                }
            else:
                return {
                    "final_response": result.get("answer", "Workflow execution failed"),
                    "error": result.get("error", "Unknown workflow error"),
                    "success": False
                }
                
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "final_response": f"Maaf, terjadi kesalahan dalam workflow: {str(e)}",
                "error": str(e),
                "success": False
            }

    async def process_question(self, question: str, user_id: Optional[int] = None, 
                             conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a question through the complete workflow.
        
        Args:
            question: User's legal question
            user_id: Optional user ID for context
            conversation_id: Optional conversation ID for context
            
        Returns:
            Complete workflow results with audit trail
        """
        if not self.workflow:
            return {
                "error": "Workflow not initialized",
                "success": False
            }
        
        start_time = time.time()
        
        try:
            # Initialize state
            initial_state: LegalReasoningState = {
                "question": question,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "question_analysis": {},
                "complexity_score": 0.0,
                "required_tools": [],
                "gathered_information": [],
                "tool_results": {},
                "cross_validation_results": {},
                "confidence_score": 0.0,
                "synthesized_answer": "",
                "key_points": [],
                "citations": [],
                "formatted_citations": "",
                "quality_score": 0.0,
                "quality_issues": [],
                "reasoning_path": [],
                "execution_time": 0.0,
                "step_timings": {},
                "error_log": []
            }
            
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Calculate total execution time
            execution_time = time.time() - start_time
            final_state["execution_time"] = execution_time
            
            # Format response
            response = {
                "answer": final_state["synthesized_answer"],
                "citations": final_state["formatted_citations"],
                "confidence_score": final_state["confidence_score"],
                "quality_score": final_state["quality_score"],
                "key_points": final_state["key_points"],
                "reasoning_path": final_state["reasoning_path"],
                "execution_time": execution_time,
                "step_timings": final_state["step_timings"],
                "tool_calls": self._extract_tool_calls_from_state(final_state),
                "complexity_analysis": final_state["question_analysis"],
                "quality_issues": final_state["quality_issues"],
                "conversation_id": conversation_id,
                "success": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Workflow completed successfully in {execution_time:.2f}s")
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Workflow execution failed: {str(e)}")
            
            return {
                "answer": f"Maaf, terjadi kesalahan dalam memproses pertanyaan Anda: {str(e)}",
                "error": str(e),
                "execution_time": execution_time,
                "success": False,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # ======= WORKFLOW STEP IMPLEMENTATIONS =======
    
    async def analyze_question(self, state: LegalReasoningState) -> LegalReasoningState:
        """
        Step 1: Analyze question complexity and determine required tools.
        """
        step_start = time.time()
        
        try:
            state["reasoning_path"].append("Starting question analysis...")
            
            # Analyze question using LLM
            analysis_prompt = f"""
            Analisis pertanyaan hukum berikut dan tentukan:
            1. Tingkat kompleksitas (1-10)
            2. Area hukum yang terlibat
            3. Tools yang dibutuhkan untuk menjawab
            4. Estimasi langkah penelusuran
            
            Pertanyaan: {state['question']}
            
            Berikan analisis dalam format JSON.
            """
            
            # Call LLM for analysis
            analysis_result = await self.llm_service.llm.ainvoke(analysis_prompt)
            
            # Parse analysis (simplified for now)
            complexity_score = self._extract_complexity_score(analysis_result.content)
            required_tools = self._determine_required_tools(state["question"], complexity_score)
            
            # Update state
            state["question_analysis"] = {
                "complexity_score": complexity_score,
                "legal_areas": self._identify_legal_areas(state["question"]),
                "question_type": self._classify_question_type(state["question"]),
                "estimated_steps": len(required_tools) + 2,
                "analysis_result": analysis_result.content
            }
            state["complexity_score"] = complexity_score
            state["required_tools"] = required_tools
            
            step_time = time.time() - step_start
            state["step_timings"]["analyze_question"] = step_time
            state["reasoning_path"].append(f"Question analysis completed in {step_time:.2f}s - Complexity: {complexity_score}")
            
        except Exception as e:
            state["error_log"].append(f"Question analysis failed: {str(e)}")
            state["reasoning_path"].append(f"Question analysis failed: {str(e)}")
            logger.error(f"Question analysis failed: {str(e)}")
        
        return state
    
    async def gather_information(self, state: LegalReasoningState) -> LegalReasoningState:
        """
        Step 2: Gather information using required tools.
        """
        step_start = time.time()
        
        try:
            state["reasoning_path"].append("Starting information gathering...")
            
            # Execute tools based on required_tools
            for tool_name in state["required_tools"]:
                if tool_name in self.available_tools:
                    tool = self.available_tools[tool_name]
                    
                    # Execute tool
                    tool_start = time.time()
                    tool_result = await tool.ainvoke(state["question"])
                    tool_time = time.time() - tool_start
                    
                    # Store result
                    tool_info = {
                        "tool_name": tool_name,
                        "result": tool_result,
                        "execution_time": tool_time,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    state["gathered_information"].append(tool_info)
                    state["tool_results"][tool_name] = tool_result
                    
                    state["reasoning_path"].append(f"Executed {tool_name} in {tool_time:.2f}s")
                
                else:
                    state["error_log"].append(f"Tool {tool_name} not available")
            
            step_time = time.time() - step_start
            state["step_timings"]["gather_information"] = step_time
            state["reasoning_path"].append(f"Information gathering completed in {step_time:.2f}s")
            
        except Exception as e:
            state["error_log"].append(f"Information gathering failed: {str(e)}")
            state["reasoning_path"].append(f"Information gathering failed: {str(e)}")
            logger.error(f"Information gathering failed: {str(e)}")
        
        return state
    
    async def cross_validate(self, state: LegalReasoningState) -> LegalReasoningState:
        """
        Step 3: Cross-validate information from multiple sources.
        """
        step_start = time.time()
        
        try:
            state["reasoning_path"].append("Starting cross-validation...")
            
            # Analyze consistency across tool results
            validation_results = {}
            confidence_scores = []
            
            for info in state["gathered_information"]:
                tool_name = info["tool_name"]
                result = info["result"]
                
                # Simple validation logic (can be enhanced)
                validation_results[tool_name] = {
                    "result_length": len(str(result)),
                    "contains_legal_terms": self._contains_legal_terms(str(result)),
                    "confidence": self._calculate_result_confidence(str(result))
                }
                confidence_scores.append(validation_results[tool_name]["confidence"])
            
            # Calculate overall confidence
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            state["cross_validation_results"] = validation_results
            state["confidence_score"] = overall_confidence
            
            step_time = time.time() - step_start
            state["step_timings"]["cross_validate"] = step_time
            state["reasoning_path"].append(f"Cross-validation completed in {step_time:.2f}s - Confidence: {overall_confidence:.2f}")
            
        except Exception as e:
            state["error_log"].append(f"Cross-validation failed: {str(e)}")
            state["reasoning_path"].append(f"Cross-validation failed: {str(e)}")
            logger.error(f"Cross-validation failed: {str(e)}")
        
        return state
    
    async def synthesize_answer(self, state: LegalReasoningState) -> LegalReasoningState:
        """
        Step 4: Synthesize information into a coherent answer.
        """
        step_start = time.time()
        
        try:
            state["reasoning_path"].append("Starting answer synthesis...")
            
            # Prepare synthesis prompt
            all_information = "\\n\\n".join([
                f"**{info['tool_name']}**: {info['result']}"
                for info in state["gathered_information"]
            ])
            
            synthesis_prompt = f"""
            Berdasarkan informasi yang dikumpulkan, berikan jawaban yang komprehensif untuk pertanyaan:
            
            **Pertanyaan:** {state['question']}
            
            **Informasi yang dikumpulkan:**
            {all_information}
            
            **Tingkat kepercayaan:** {state['confidence_score']:.2f}
            
            Berikan jawaban yang:
            1. Menjawab pertanyaan secara langsung
            2. Menggunakan informasi yang valid
            3. Menyebutkan dasar hukum yang relevan
            4. Memberikan penjelasan yang mudah dipahami
            5. Menyertakan batasan atau catatan jika diperlukan
            """
            
            # Generate synthesized answer
            synthesis_result = await self.llm_service.llm.ainvoke(synthesis_prompt)
            
            # Extract key points
            key_points = self._extract_key_points(synthesis_result.content)
            
            state["synthesized_answer"] = synthesis_result.content
            state["key_points"] = key_points
            
            step_time = time.time() - step_start
            state["step_timings"]["synthesize_answer"] = step_time
            state["reasoning_path"].append(f"Answer synthesis completed in {step_time:.2f}s")
            
        except Exception as e:
            state["error_log"].append(f"Answer synthesis failed: {str(e)}")
            state["reasoning_path"].append(f"Answer synthesis failed: {str(e)}")
            logger.error(f"Answer synthesis failed: {str(e)}")
        
        return state
    
    async def generate_citations(self, state: LegalReasoningState) -> LegalReasoningState:
        """
        Step 5: Generate proper legal citations.
        """
        step_start = time.time()
        
        try:
            state["reasoning_path"].append("Starting citation generation...")
            
            # Extract citations from tool results
            citations = []
            
            for info in state["gathered_information"]:
                if "legal_search" in info["tool_name"]:
                    # Extract citation info from legal search results
                    citations.extend(self._extract_citations_from_text(str(info["result"])))
            
            # Format citations
            formatted_citations = self._format_citations(citations)
            
            state["citations"] = citations
            state["formatted_citations"] = formatted_citations
            
            step_time = time.time() - step_start
            state["step_timings"]["generate_citations"] = step_time
            state["reasoning_path"].append(f"Citation generation completed in {step_time:.2f}s")
            
        except Exception as e:
            state["error_log"].append(f"Citation generation failed: {str(e)}")
            state["reasoning_path"].append(f"Citation generation failed: {str(e)}")
            logger.error(f"Citation generation failed: {str(e)}")
        
        return state
    
    async def quality_check(self, state: LegalReasoningState) -> LegalReasoningState:
        """
        Step 6: Perform final quality check.
        """
        step_start = time.time()
        
        try:
            state["reasoning_path"].append("Starting quality check...")
            
            quality_issues = []
            quality_score = 1.0
            
            # Check answer completeness
            if len(state["synthesized_answer"]) < 100:
                quality_issues.append("Answer too short")
                quality_score -= 0.2
            
            # Check confidence score
            if state["confidence_score"] < 0.7:
                quality_issues.append("Low confidence score")
                quality_score -= 0.1
            
            # Check citation availability
            if not state["citations"]:
                quality_issues.append("No citations found")
                quality_score -= 0.1
            
            # Check for errors
            if state["error_log"]:
                quality_issues.append(f"{len(state['error_log'])} errors occurred")
                quality_score -= 0.1 * len(state["error_log"])
            
            # Ensure quality score is not negative
            quality_score = max(0.0, quality_score)
            
            state["quality_score"] = quality_score
            state["quality_issues"] = quality_issues
            
            step_time = time.time() - step_start
            state["step_timings"]["quality_check"] = step_time
            state["reasoning_path"].append(f"Quality check completed in {step_time:.2f}s - Score: {quality_score:.2f}")
            
        except Exception as e:
            state["error_log"].append(f"Quality check failed: {str(e)}")
            state["reasoning_path"].append(f"Quality check failed: {str(e)}")
            logger.error(f"Quality check failed: {str(e)}")
        
        return state
    
    # ======= HELPER METHODS =======
    
    def _route_after_analysis(self, state: LegalReasoningState) -> str:
        """Route workflow after question analysis."""
        if state.get("error_log"):
            return "error"
        elif state.get("complexity_score", 0) < 3:
            return "simple"
        else:
            return "complex"
    
    def _extract_complexity_score(self, analysis_text: str) -> float:
        """Extract complexity score from LLM analysis."""
        # Simple extraction logic - can be enhanced with regex or JSON parsing
        try:
            if "sangat kompleks" in analysis_text.lower():
                return 8.0
            elif "kompleks" in analysis_text.lower():
                return 6.0
            elif "sedang" in analysis_text.lower():
                return 4.0
            elif "sederhana" in analysis_text.lower():
                return 2.0
            else:
                return 5.0  # Default
        except:
            return 5.0
    
    def _determine_required_tools(self, question: str, complexity_score: float) -> List[str]:
        """Determine which tools are needed based on question and complexity."""
        required_tools = ["legal_search"]  # Always include legal search
        
        if complexity_score > 5:
            required_tools.append("validator")
        
        if any(keyword in question.lower() for keyword in ["ringkas", "rangkum", "summary"]):
            required_tools.append("summarizer")
        
        return required_tools
    
    def _identify_legal_areas(self, question: str) -> List[str]:
        """Identify legal areas from the question."""
        legal_areas = []
        
        area_keywords = {
            "pidana": ["pidana", "pencurian", "korupsi", "pembunuhan"],
            "perdata": ["perdata", "kontrak", "wanprestasi", "ganti rugi"],
            "tata_usaha_negara": ["administrasi", "perizinan", "tun"],
            "tata_negara": ["konstitusi", "pemilu", "kekuasaan"]
        }
        
        question_lower = question.lower()
        for area, keywords in area_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                legal_areas.append(area)
        
        return legal_areas if legal_areas else ["umum"]
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of legal question."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["apa", "what", "definisi"]):
            return "definition"
        elif any(word in question_lower for word in ["bagaimana", "cara", "prosedur"]):
            return "procedure"
        elif any(word in question_lower for word in ["sanksi", "hukuman", "denda"]):
            return "penalty"
        elif any(word in question_lower for word in ["syarat", "ketentuan", "persyaratan"]):
            return "requirements"
        else:
            return "general"
    
    def _contains_legal_terms(self, text: str) -> bool:
        """Check if text contains legal terms."""
        legal_terms = ["pasal", "undang-undang", "putusan", "pengadilan", "mahkamah", "hukum"]
        text_lower = text.lower()
        return any(term in text_lower for term in legal_terms)
    
    def _calculate_result_confidence(self, result_text: str) -> float:
        """Calculate confidence score for a result."""
        # Simple confidence calculation
        score = 0.5  # Base score
        
        if self._contains_legal_terms(result_text):
            score += 0.2
        
        if len(result_text) > 100:
            score += 0.2
        
        if "pasal" in result_text.lower():
            score += 0.1
        
        return min(1.0, score)
    
    def _extract_key_points(self, answer_text: str) -> List[str]:
        """Extract key points from synthesized answer."""
        # Simple extraction - split by numbered points or bullets
        key_points = []
        
        lines = answer_text.split("\\n")
        for line in lines:
            line = line.strip()
            if line and (line.startswith(("1.", "2.", "3.", "•", "-", "*")) or "penting" in line.lower()):
                key_points.append(line)
        
        return key_points[:5]  # Limit to 5 key points
    
    def _extract_citations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract citation information from text."""
        citations = []
        
        # Look for common citation patterns
        if "pasal" in text.lower():
            # Extract pasal references
            import re
            pasal_matches = re.findall(r"pasal\\s+(\\d+)", text, re.IGNORECASE)
            for match in pasal_matches:
                citations.append({
                    "type": "pasal",
                    "reference": f"Pasal {match}",
                    "source": "KUHP"  # Default, should be extracted properly
                })
        
        return citations
    
    def _format_citations(self, citations: List[Dict[str, Any]]) -> str:
        """Format citations into readable text."""
        if not citations:
            return ""
        
        formatted = "\\n**Sumber Hukum:**\\n"
        for i, citation in enumerate(citations, 1):
            formatted += f"{i}. {citation.get('reference', 'Unknown')} - {citation.get('source', 'Unknown')}\\n"
        
        return formatted
    
    def _extract_tool_calls_from_state(self, state: LegalReasoningState) -> List[Dict[str, Any]]:
        """Extract tool call information from workflow state."""
        tool_calls = []
        
        for info in state.get("gathered_information", []):
            tool_calls.append({
                "tool_name": info["tool_name"],
                "execution_time": info["execution_time"],
                "timestamp": info["timestamp"],
                "success": True
            })
        
        return tool_calls