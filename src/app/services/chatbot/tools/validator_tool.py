"""
Validator Tool for ReAct Agent.
Provides fact-checking and claim validation capabilities.
"""

import logging
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

from ...llm_service import get_llm_service

logger = logging.getLogger(__name__)


class ValidatorInput(BaseModel):
    """Input schema for validator tool."""
    claim: str = Field(description="The claim or statement to validate")
    source_context: Optional[str] = Field(
        default=None,
        description="Source context or reference material to validate against"
    )


class ValidationResult(BaseModel):
    """Result schema for validation."""
    is_valid: bool = Field(description="Whether the claim appears to be valid")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Explanation of the validation result")
    issues_found: list[str] = Field(description="List of potential issues or concerns")


class ValidatorTool(BaseTool):
    """
    Tool for validating legal claims and statements.
    
    This tool helps the ReAct agent verify the accuracy of legal statements,
    check for factual consistency, and identify potential issues or concerns.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "validator"
    description: str = """
    Validate legal claims, statements, or facts for accuracy and consistency.
    Use this tool when you need to fact-check legal statements, verify claims
    against source material, or assess the validity of legal arguments.
    
    Input should be the claim you want to validate, optionally with source context.
    """
    args_schema: type = ValidatorInput
    llm_service: Optional[object] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_llm_service()
    
    def _initialize_llm_service(self) -> None:
        """Initialize LLM service connection."""
        try:
            self.llm_service = get_llm_service()
            logger.info("Validator tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            self.llm_service = None
    
    def _run(self, claim: str, source_context: Optional[str] = None) -> str:
        """
        Execute claim validation.
        
        Args:
            claim: The claim or statement to validate
            source_context: Optional source context for validation
            
        Returns:
            Validation result as formatted string
        """
        if not self.llm_service:
            return "Error: Validation service is not available. Please try again later."
        
        if not claim or len(claim.strip()) < 10:
            return "Error: Claim is too short or empty. Please provide a clear claim to validate."
        
        try:
            # Create validation prompt
            prompt = self._create_validation_prompt(claim, source_context)
            
            # Use the existing LLM service
            response = self.llm_service.llm.invoke(prompt)
            
            # Extract the validation result from the response
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return f"Error performing validation: {str(e)}"
    
    async def _arun(self, claim: str, source_context: Optional[str] = None) -> str:
        """Async version of the tool execution."""
        if not self.llm_service:
            return "Error: Validation service is not available. Please try again later."
        
        if not claim or len(claim.strip()) < 10:
            return "Error: Claim is too short or empty. Please provide a clear claim to validate."
        
        try:
            prompt = self._create_validation_prompt(claim, source_context)
            
            # Use async invoke if available
            if hasattr(self.llm_service.llm, 'ainvoke'):
                response = await self.llm_service.llm.ainvoke(prompt)
            else:
                response = self.llm_service.llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Async validation failed: {str(e)}")
            return f"Error performing validation: {str(e)}"
    
    def _create_validation_prompt(self, claim: str, source_context: Optional[str] = None) -> str:
        """
        Create a structured prompt for claim validation.
        
        Args:
            claim: The claim to validate
            source_context: Optional source context
            
        Returns:
            Formatted prompt for the LLM
        """
        context_section = ""
        if source_context:
            context_section = f"""
KONTEKS SUMBER:
{source_context}
"""
        
        prompt = f"""Anda adalah ahli hukum yang bertugas memvalidasi klaim dan pernyataan hukum.
Analisis klaim berikut dengan teliti dan berikan penilaian yang objektif.

INSTRUKSI VALIDASI:
1. Evaluasi keakuratan faktual dari klaim
2. Periksa konsistensi dengan prinsip hukum yang berlaku
3. Identifikasi potensi masalah atau keraguan
4. Berikan tingkat kepercayaan (0-100%)
5. Jelaskan alasan di balik penilaian Anda{context_section}

KLAIM YANG AKAN DIVALIDASI:
{claim}

HASIL VALIDASI:
Berikan analisis dalam format berikut:
- Status: [VALID/TIDAK VALID/MERAGUKAN]
- Tingkat Kepercayaan: [0-100%]
- Penjelasan: [Alasan dan analisis detail]
- Isu/Masalah: [Daftar potensi masalah jika ada]
- Rekomendasi: [Saran tindak lanjut jika diperlukan]"""
        
        return prompt
    
    def _parse_validation_result(self, result_text: str) -> ValidationResult:
        """
        Parse the LLM response into a structured ValidationResult.
        
        Args:
            result_text: Raw text response from LLM
            
        Returns:
            Structured ValidationResult object
        """
        try:
            # This is a simplified parser - in production, you might want
            # to use more sophisticated parsing or structured output
            
            lines = result_text.split('\n')
            
            is_valid = False
            confidence = 0.5
            reasoning = result_text
            issues_found = []
            
            for line in lines:
                line = line.strip()
                if 'Status:' in line:
                    status = line.split('Status:')[1].strip().lower()
                    is_valid = 'valid' in status and 'tidak' not in status
                elif 'Tingkat Kepercayaan:' in line or 'Confidence:' in line:
                    try:
                        # Extract percentage
                        conf_text = line.split(':')[1].strip()
                        conf_number = ''.join(filter(str.isdigit, conf_text))
                        if conf_number:
                            confidence = float(conf_number) / 100.0
                    except:
                        pass
                elif 'Isu/Masalah:' in line or 'Issues:' in line:
                    issues_text = line.split(':')[1].strip()
                    if issues_text:
                        issues_found.append(issues_text)
            
            return ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                reasoning=reasoning,
                issues_found=issues_found
            )
            
        except Exception as e:
            logger.error(f"Error parsing validation result: {str(e)}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                reasoning=f"Error parsing validation result: {str(e)}",
                issues_found=["Parsing error occurred"]
            )