"""
Basic Summarizer Tool for ReAct Agent.
Provides text summarization capabilities using the existing LLM service.
"""

import logging
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

from ...llm_service import get_llm_service

logger = logging.getLogger(__name__)


class SummarizerInput(BaseModel):
    """Input schema for summarizer tool."""
    text: str = Field(description="Text to be summarized")
    max_length: int = Field(
        default=200, 
        description="Maximum length of summary in words"
    )
    focus: Optional[str] = Field(
        default=None,
        description="Specific aspect to focus on in the summary (e.g., 'legal implications', 'key facts')"
    )


class BasicSummarizerTool(BaseTool):
    """
    Tool for summarizing legal documents and text.
    
    This tool helps the ReAct agent create concise summaries of long legal texts,
    court decisions, or other documents to provide clear and digestible information.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "summarizer"
    description: str = """
    Summarize long legal texts or documents into concise, clear summaries.
    Use this tool when you need to condense lengthy court decisions, legal documents, 
    or other text into key points and main conclusions.
    
    Input should be the text you want to summarize.
    """
    args_schema: type = SummarizerInput
    llm_service: Optional[object] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_llm_service()
    
    def _initialize_llm_service(self) -> None:
        """Initialize LLM service connection."""
        try:
            self.llm_service = get_llm_service()
            logger.info("Summarizer tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            self.llm_service = None
    
    def _run(self, text: str, max_length: int = 200, focus: Optional[str] = None) -> str:
        """
        Execute text summarization.
        
        Args:
            text: The text to be summarized
            max_length: Maximum length of summary in words
            focus: Optional specific aspect to focus on
            
        Returns:
            Summarized text
        """
        if not self.llm_service:
            return "Error: Summarization service is not available. Please try again later."
        
        if not text or len(text.strip()) < 50:
            return "Error: Text is too short to summarize. Please provide longer text."
        
        try:
            # Create summarization prompt
            prompt = self._create_summarization_prompt(text, max_length, focus)
            
            # Use the existing LLM service
            response = self.llm_service.llm.invoke(prompt)
            
            # Extract the summary from the response
            if hasattr(response, 'content'):
                summary = response.content
            else:
                summary = str(response)
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return f"Error performing summarization: {str(e)}"
    
    async def _arun(self, text: str, max_length: int = 200, focus: Optional[str] = None) -> str:
        """Async version of the tool execution."""
        if not self.llm_service:
            return "Error: Summarization service is not available. Please try again later."
        
        if not text or len(text.strip()) < 50:
            return "Error: Text is too short to summarize. Please provide longer text."
        
        try:
            prompt = self._create_summarization_prompt(text, max_length, focus)
            
            # Use async invoke if available
            if hasattr(self.llm_service.llm, 'ainvoke'):
                response = await self.llm_service.llm.ainvoke(prompt)
            else:
                response = self.llm_service.llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                summary = response.content
            else:
                summary = str(response)
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Async summarization failed: {str(e)}")
            return f"Error performing summarization: {str(e)}"
    
    def _create_summarization_prompt(self, text: str, max_length: int, focus: Optional[str] = None) -> str:
        """
        Create a structured prompt for summarization.
        
        Args:
            text: Text to summarize
            max_length: Maximum word count for summary
            focus: Optional focus area
            
        Returns:
            Formatted prompt for the LLM
        """
        focus_instruction = ""
        if focus:
            focus_instruction = f"\nFokus khusus pada: {focus}"
        
        prompt = f"""Anda adalah asisten hukum yang ahli dalam merangkum dokumen hukum.
Tugas Anda adalah membuat ringkasan yang jelas dan terstruktur dari teks berikut.

INSTRUKSI RINGKASAN:
1. Buat ringkasan maksimal {max_length} kata
2. Fokus pada poin-poin utama dan kesimpulan penting
3. Gunakan bahasa yang jelas dan mudah dipahami
4. Pertahankan akurasi informasi hukum
5. Struktur dengan bullet points atau paragraf pendek{focus_instruction}

TEKS YANG AKAN DIRINGKAS:
{text}

RINGKASAN:"""
        
        return prompt
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def _validate_summary_length(self, summary: str, max_length: int) -> str:
        """Validate and trim summary if needed."""
        word_count = self._count_words(summary)
        
        if word_count <= max_length:
            return summary
        
        # If summary is too long, try to trim it intelligently
        words = summary.split()
        trimmed = " ".join(words[:max_length])
        
        # Try to end at a sentence boundary
        sentences = trimmed.split('.')
        if len(sentences) > 1:
            return '.'.join(sentences[:-1]) + '.'
        
        return trimmed + "..."