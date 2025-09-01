"""
Guardrails integration for claim validation in Supreme Court RAG system.
Implements PRD requirement for factual accuracy verification using
Guardrails AI as a service layer with LangChain integration.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

try:
    import guardrails as gd
    from guardrails.validators import (
        ValidLength,
        ValidRange,
        OnTopic,
        ProvenanceLLM
    )
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    logging.warning("Guardrails AI not available. Install with: pip install guardrails-ai")

logger = logging.getLogger(__name__)


class Claim(BaseModel):
    """Individual factual claim to be validated."""
    text: str = Field(..., description="The claim text")
    claim_id: str = Field(..., description="Unique identifier for the claim")
    source_context: str = Field(..., description="Context from which claim was extracted")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Terdakwa dijatuhi hukuman 5 tahun penjara",
                "claim_id": "claim_001",
                "source_context": "Dalam putusan kasasi No. 123/K/Pid/2023..."
            }
        }


class ValidationEvidence(BaseModel):
    """Evidence supporting or refuting a claim."""
    evidence_text: str = Field(..., description="Text evidence")
    source_document: str = Field(..., description="Source document identifier")
    chunk_id: str = Field(..., description="Specific chunk containing evidence")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "evidence_text": "Pengadilan menjatuhkan pidana penjara selama 5 tahun",
                "source_document": "putusan_123_k_pid_2023",
                "chunk_id": "chunk_45",
                "relevance_score": 0.92
            }
        }


class ClaimValidationResult(BaseModel):
    """Result of validating a single claim."""
    claim: Claim = Field(..., description="The claim that was validated")
    status: str = Field(..., description="Supported/Partially Supported/Unsupported/Uncertain")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in validation")
    evidence: List[ValidationEvidence] = Field(default_factory=list, description="Supporting evidence")
    contradictions: List[ValidationEvidence] = Field(default_factory=list, description="Contradictory evidence")
    reasoning: str = Field(..., description="Explanation of validation decision")
    timestamp: datetime = Field(default_factory=datetime.now, description="Validation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "claim": {
                    "text": "Terdakwa dijatuhi hukuman 5 tahun penjara",
                    "claim_id": "claim_001",
                    "source_context": "Dalam putusan kasasi..."
                },
                "status": "Supported",
                "confidence": 0.95,
                "evidence": [
                    {
                        "evidence_text": "Pengadilan menjatuhkan pidana penjara selama 5 tahun",
                        "source_document": "putusan_123",
                        "chunk_id": "chunk_45",
                        "relevance_score": 0.92
                    }
                ],
                "reasoning": "Klaim didukung langsung oleh teks putusan"
            }
        }


class BatchValidationResult(BaseModel):
    """Result of validating multiple claims."""
    claims: List[ClaimValidationResult] = Field(..., description="Individual claim results")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    filtered_text: str = Field(..., description="Original text with unsupported claims removed")
    summary: Dict[str, int] = Field(..., description="Summary of validation results")
    processing_time: float = Field(..., description="Total processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "claims": [],
                "overall_confidence": 0.85,
                "filtered_text": "Berdasarkan putusan yang tersedia...",
                "summary": {
                    "total_claims": 5,
                    "supported": 3,
                    "partially_supported": 1,
                    "unsupported": 1,
                    "uncertain": 0
                },
                "processing_time": 2.3
            }
        }


class GuardrailsValidator:
    """
    Guardrails AI integration for claim validation.
    
    Implements PRD requirement for factual accuracy verification
    using Guardrails as a service layer within LangChain workflows.
    """
    
    def __init__(
        self,
        llm,  # Will be VertexAILLM
        enable_strict_validation: bool = True,
        confidence_threshold: float = 0.7
    ):
        self.llm = llm
        self.enable_strict_validation = enable_strict_validation
        self.confidence_threshold = confidence_threshold
        
        if not GUARDRAILS_AVAILABLE:
            logger.warning("Guardrails not available, using fallback validation")
            self.use_guardrails = False
        else:
            self.use_guardrails = True
            self._initialize_guardrails()
    
    def _initialize_guardrails(self) -> None:
        """Initialize Guardrails validators and guards."""
        if not self.use_guardrails:
            return
        
        # Define validation schema for legal claims
        self.claim_extraction_guard = gd.Guard.from_pydantic(
            output_class=List[Claim],
            prompt="""
            Extract all factual claims from the following legal text that can be verified:
            
            Text: ${text}
            
            For each claim, provide:
            1. The exact claim text
            2. A unique identifier
            3. The source context
            
            Focus on specific factual assertions about:
            - Legal decisions and rulings
            - Sentence lengths and penalties
            - Case details and procedures
            - Legal precedents and citations
            
            Avoid extracting:
            - Legal opinions or interpretations
            - Procedural descriptions
            - General legal principles
            
            ${gr.complete_json_suffix_v2}
            """,
            validators=[
                ValidLength(min=10, max=500, on_fail="reask"),
                OnTopic(
                    valid_topics=["legal decisions", "court rulings", "legal facts"],
                    on_fail="filter"
                )
            ]
        )
        
        # Define validation guard for claim verification
        self.claim_validation_guard = gd.Guard.from_pydantic(
            output_class=ClaimValidationResult,
            prompt="""
            Validate the following claim against the provided evidence:
            
            CLAIM: ${claim_text}
            
            EVIDENCE DOCUMENTS:
            ${evidence_context}
            
            Determine if the claim is:
            - "Supported": Directly supported by evidence
            - "Partially Supported": Partially supported with some uncertainty
            - "Unsupported": Contradicted by evidence or not mentioned
            - "Uncertain": Insufficient evidence to determine
            
            Provide:
            1. Validation status
            2. Confidence score (0.0-1.0)
            3. Supporting evidence with source references
            4. Clear reasoning for the decision
            
            ${gr.complete_json_suffix_v2}
            """,
            validators=[
                ValidRange(min=0, max=1, on_fail="reask"),  # For confidence score
                ProvenanceLLM(
                    sources=["evidence_context"],
                    on_fail="reask"
                )
            ]
        )
    
    async def extract_claims(self, text: str) -> List[Claim]:
        """
        Extract verifiable claims from text using Guardrails.
        
        Args:
            text: Input text to extract claims from
            
        Returns:
            List of extracted claims
        """
        if not self.use_guardrails:
            return await self._fallback_extract_claims(text)
        
        try:
            # Use Guardrails to extract claims
            result = self.claim_extraction_guard(
                llm_api=self.llm,
                prompt_params={"text": text},
                num_reasks=2
            )
            
            if result.validation_passed:
                return result.validated_output
            else:
                logger.warning(f"Claim extraction validation failed: {result.error}")
                return await self._fallback_extract_claims(text)
                
        except Exception as e:
            logger.error(f"Guardrails claim extraction failed: {str(e)}")
            return await self._fallback_extract_claims(text)
    
    async def validate_claim(
        self,
        claim: Claim,
        evidence_context: str
    ) -> ClaimValidationResult:
        """
        Validate a single claim against evidence using Guardrails.
        
        Args:
            claim: Claim to validate
            evidence_context: Relevant evidence text
            
        Returns:
            Validation result for the claim
        """
        if not self.use_guardrails:
            return await self._fallback_validate_claim(claim, evidence_context)
        
        try:
            # Use Guardrails to validate claim
            result = self.claim_validation_guard(
                llm_api=self.llm,
                prompt_params={
                    "claim_text": claim.text,
                    "evidence_context": evidence_context
                },
                num_reasks=2
            )
            
            if result.validation_passed:
                return result.validated_output
            else:
                logger.warning(f"Claim validation failed: {result.error}")
                return await self._fallback_validate_claim(claim, evidence_context)
                
        except Exception as e:
            logger.error(f"Guardrails validation failed: {str(e)}")
            return await self._fallback_validate_claim(claim, evidence_context)
    
    async def validate_batch(
        self,
        text: str,
        evidence_documents: List[Dict[str, Any]]
    ) -> BatchValidationResult:
        """
        Validate all claims in a text against evidence documents.
        
        Implements the complete validation workflow as specified in PRD.
        
        Args:
            text: Text containing claims to validate
            evidence_documents: List of evidence documents with content and metadata
            
        Returns:
            Complete batch validation result
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Extract claims from text
            claims = await self.extract_claims(text)
            
            if not claims:
                return BatchValidationResult(
                    claims=[],
                    overall_confidence=0.0,
                    filtered_text=text,
                    summary={"total_claims": 0, "supported": 0, "partially_supported": 0, 
                           "unsupported": 0, "uncertain": 0},
                    processing_time=0.0
                )
            
            # Step 2: Prepare evidence context
            evidence_context = self._format_evidence_context(evidence_documents)
            
            # Step 3: Validate each claim
            claim_results = []
            for claim in claims:
                validation_result = await self.validate_claim(claim, evidence_context)
                claim_results.append(validation_result)
            
            # Step 4: Calculate overall metrics
            overall_confidence = self._calculate_overall_confidence(claim_results)
            summary = self._calculate_summary(claim_results)
            
            # Step 5: Filter text based on validation results
            filtered_text = self._filter_unsupported_claims(text, claim_results)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return BatchValidationResult(
                claims=claim_results,
                overall_confidence=overall_confidence,
                filtered_text=filtered_text,
                summary=summary,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Batch validation failed: {str(e)}")
            raise
    
    def _format_evidence_context(self, evidence_documents: List[Dict[str, Any]]) -> str:
        """Format evidence documents into context string."""
        context_parts = []
        
        for i, doc in enumerate(evidence_documents):
            context_parts.append(
                f"[DOKUMEN {i+1}]\n"
                f"Sumber: {doc.get('source', 'Unknown')}\n"
                f"Konten: {doc.get('content', '')}\n"
                f"{'='*50}\n"
            )
        
        return '\n'.join(context_parts)
    
    def _calculate_overall_confidence(self, claim_results: List[ClaimValidationResult]) -> float:
        """Calculate overall confidence score from individual claim validations."""
        if not claim_results:
            return 0.0
        
        total_confidence = sum(result.confidence for result in claim_results)
        return total_confidence / len(claim_results)
    
    def _calculate_summary(self, claim_results: List[ClaimValidationResult]) -> Dict[str, int]:
        """Calculate summary statistics from claim validation results."""
        summary = {
            "total_claims": len(claim_results),
            "supported": 0,
            "partially_supported": 0,
            "unsupported": 0,
            "uncertain": 0
        }
        
        for result in claim_results:
            status = result.status.lower().replace(" ", "_")
            if status in summary:
                summary[status] += 1
        
        return summary
    
    def _filter_unsupported_claims(
        self,
        original_text: str,
        claim_results: List[ClaimValidationResult]
    ) -> str:
        """Remove unsupported claims from the original text."""
        filtered_text = original_text
        
        # Remove claims marked as "Unsupported"
        for result in claim_results:
            if result.status == "Unsupported":
                # Simple approach: remove the claim text
                # More sophisticated approach would preserve context
                filtered_text = filtered_text.replace(result.claim.text, "")
        
        # Clean up formatting
        import re
        filtered_text = re.sub(r'\s+', ' ', filtered_text)
        filtered_text = re.sub(r'[.]{2,}', '.', filtered_text)
        
        return filtered_text.strip()
    
    # Fallback methods when Guardrails is not available
    async def _fallback_extract_claims(self, text: str) -> List[Claim]:
        """Fallback claim extraction using LLM without Guardrails."""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Ekstrak klaim-klaim faktual yang dapat diverifikasi dari teks hukum berikut:
            
            {text}
            
            Berikan dalam format JSON dengan struktur:
            [
                {{
                    "text": "teks klaim",
                    "claim_id": "claim_001",
                    "source_context": "konteks sumber"
                }}
            ]
            """
        )
        
        # Use LLM to extract claims
        chain = prompt | self.llm
        result = await chain.ainvoke({"text": text})
        
        # Parse JSON response (simplified)
        try:
            import json
            claims_data = json.loads(result.content)
            return [Claim(**claim) for claim in claims_data]
        except:
            return []
    
    async def _fallback_validate_claim(
        self,
        claim: Claim,
        evidence_context: str
    ) -> ClaimValidationResult:
        """Fallback claim validation using LLM without Guardrails."""
        prompt = PromptTemplate(
            input_variables=["claim_text", "evidence_context"],
            template="""
            Validasi klaim berikut terhadap bukti yang tersedia:
            
            KLAIM: {claim_text}
            
            BUKTI: {evidence_context}
            
            Tentukan status validasi dan berikan alasan.
            """
        )
        
        chain = prompt | self.llm
        result = await chain.ainvoke({
            "claim_text": claim.text,
            "evidence_context": evidence_context
        })
        
        # Simple fallback validation
        return ClaimValidationResult(
            claim=claim,
            status="Uncertain",
            confidence=0.5,
            evidence=[],
            contradictions=[],
            reasoning=result.content
        )


def create_guardrails_validator(
    llm,  # Will be VertexAILLM
    enable_strict_validation: bool = True,
    confidence_threshold: float = 0.7
) -> GuardrailsValidator:
    """
    Factory function to create Guardrails validator.
    
    Args:
        llm: LangChain LLM for validation
        enable_strict_validation: Whether to use strict validation rules
        confidence_threshold: Minimum confidence threshold for acceptance
        
    Returns:
        Configured GuardrailsValidator instance
    """
    return GuardrailsValidator(
        llm=llm,
        enable_strict_validation=enable_strict_validation,
        confidence_threshold=confidence_threshold
    )
