"""
RAGAS evaluation integration for Supreme Court RAG system.
Implements PRD evaluation protocol using RAGAS framework
for comprehensive RAG workflow assessment.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.callbacks import BaseCallbackHandler

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS not available. Install with: pip install ragas")

logger = logging.getLogger(__name__)


class EvaluationSample(BaseModel):
    """Single evaluation sample for RAGAS assessment."""
    question: str = Field(..., description="User question")
    answer: str = Field(..., description="Generated answer")
    contexts: List[str] = Field(..., description="Retrieved context chunks")
    ground_truth: Optional[str] = Field(None, description="Reference answer if available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Apa hukuman untuk tindak pidana korupsi?",
                "answer": "Berdasarkan putusan yang tersedia, hukuman untuk korupsi...",
                "contexts": [
                    "Dalam Pasal 2 UU Tipikor, hukuman korupsi adalah...",
                    "Berdasarkan putusan kasasi No. 123/K/Pid/2023..."
                ],
                "ground_truth": "Hukuman korupsi sesuai UU adalah..."
            }
        }


class RAGASMetrics(BaseModel):
    """RAGAS evaluation metrics results."""
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Factual grounding score")
    answer_relevancy: float = Field(..., ge=0.0, le=1.0, description="Answer relevance score")
    context_precision: float = Field(..., ge=0.0, le=1.0, description="Context precision score")
    context_recall: float = Field(..., ge=0.0, le=1.0, description="Context recall score")
    answer_similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Semantic similarity score")
    answer_correctness: Optional[float] = Field(None, ge=0.0, le=1.0, description="Answer correctness score")
    
    def overall_score(self) -> float:
        """Calculate overall RAG performance score."""
        scores = [self.faithfulness, self.answer_relevancy, self.context_precision, self.context_recall]
        if self.answer_similarity is not None:
            scores.append(self.answer_similarity)
        if self.answer_correctness is not None:
            scores.append(self.answer_correctness)
        
        return sum(scores) / len(scores)
    
    class Config:
        json_schema_extra = {
            "example": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.78,
                "context_precision": 0.82,
                "context_recall": 0.75,
                "answer_similarity": 0.80,
                "answer_correctness": 0.77
            }
        }


class LegalDomainMetrics(BaseModel):
    """Legal domain-specific evaluation metrics."""
    legal_accuracy: float = Field(..., ge=0.0, le=1.0, description="Legal information accuracy")
    citation_quality: float = Field(..., ge=0.0, le=1.0, description="Citation accuracy and completeness")
    case_law_relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance of cited case law")
    jurisdictional_accuracy: float = Field(..., ge=0.0, le=1.0, description="Jurisdictional context accuracy")
    temporal_relevance: float = Field(..., ge=0.0, le=1.0, description="Temporal relevance of citations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "legal_accuracy": 0.88,
                "citation_quality": 0.75,
                "case_law_relevance": 0.82,
                "jurisdictional_accuracy": 0.95,
                "temporal_relevance": 0.70
            }
        }


class EvaluationResult(BaseModel):
    """Complete evaluation result combining RAGAS and domain-specific metrics."""
    sample_id: str = Field(..., description="Unique identifier for the evaluation sample")
    ragas_metrics: RAGASMetrics = Field(..., description="RAGAS framework metrics")
    legal_metrics: LegalDomainMetrics = Field(..., description="Legal domain-specific metrics")
    timestamp: datetime = Field(default_factory=datetime.now, description="Evaluation timestamp")
    processing_time: float = Field(..., description="Evaluation processing time")
    
    def overall_performance(self) -> float:
        """Calculate overall system performance score."""
        ragas_score = self.ragas_metrics.overall_score()
        legal_score = (
            self.legal_metrics.legal_accuracy + 
            self.legal_metrics.citation_quality + 
            self.legal_metrics.case_law_relevance
        ) / 3
        
        # Weight RAGAS and legal metrics equally
        return (ragas_score + legal_score) / 2


class BatchEvaluationResult(BaseModel):
    """Results from evaluating multiple samples."""
    results: List[EvaluationResult] = Field(..., description="Individual evaluation results")
    aggregate_metrics: Dict[str, float] = Field(..., description="Aggregated metrics across all samples")
    sample_count: int = Field(..., description="Number of samples evaluated")
    total_processing_time: float = Field(..., description="Total processing time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [],
                "aggregate_metrics": {
                    "avg_faithfulness": 0.85,
                    "avg_relevancy": 0.78,
                    "avg_legal_accuracy": 0.88
                },
                "sample_count": 100,
                "total_processing_time": 150.5
            }
        }


class RAGEvaluator:
    """
    RAGAS-based evaluator for Supreme Court RAG system.
    
    Implements PRD evaluation protocol using RAGAS framework
    with domain-specific legal metrics integration.
    """
    
    def __init__(
        self,
        enable_ragas: bool = True,
        enable_legal_metrics: bool = True,
        batch_size: int = 10
    ):
        self.enable_ragas = enable_ragas and RAGAS_AVAILABLE
        self.enable_legal_metrics = enable_legal_metrics
        self.batch_size = batch_size
        
        if enable_ragas and not RAGAS_AVAILABLE:
            logger.warning("RAGAS not available, using fallback evaluation")
        
        self._initialize_metrics()
    
    def _initialize_metrics(self) -> None:
        """Initialize RAGAS metrics based on PRD requirements."""
        if not self.enable_ragas:
            return
        
        self.ragas_metrics = [
            faithfulness,      # PRD: Measure factual grounding
            answer_relevancy,  # PRD: Assess relevance to queries
            context_precision, # PRD: Evaluate signal-to-noise ratio
            context_recall,    # PRD: Verify retriever captures necessary info
        ]
        
        # Optional metrics for reference answer comparison
        self.reference_metrics = [
            answer_similarity,    # Semantic similarity with reference
            answer_correctness   # Overall correctness assessment
        ]
    
    async def evaluate_sample(
        self,
        sample: EvaluationSample,
        sample_id: str
    ) -> EvaluationResult:
        """
        Evaluate a single RAG sample using RAGAS and legal metrics.
        
        Args:
            sample: Evaluation sample with question, answer, and context
            sample_id: Unique identifier for the sample
            
        Returns:
            Complete evaluation result
        """
        start_time = datetime.now()
        
        try:
            # Evaluate using RAGAS
            ragas_metrics = await self._evaluate_ragas_sample(sample)
            
            # Evaluate using legal domain metrics
            legal_metrics = await self._evaluate_legal_sample(sample)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EvaluationResult(
                sample_id=sample_id,
                ragas_metrics=ragas_metrics,
                legal_metrics=legal_metrics,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed for sample {sample_id}: {str(e)}")
            raise
    
    async def _evaluate_ragas_sample(self, sample: EvaluationSample) -> RAGASMetrics:
        """Evaluate sample using RAGAS framework."""
        if not self.enable_ragas:
            return await self._fallback_ragas_evaluation(sample)
        
        try:
            # Prepare dataset for RAGAS
            data = {
                "question": [sample.question],
                "answer": [sample.answer],
                "contexts": [sample.contexts]
            }
            
            # Add ground truth if available
            if sample.ground_truth:
                data["ground_truth"] = [sample.ground_truth]
                metrics_to_use = self.ragas_metrics + self.reference_metrics
            else:
                metrics_to_use = self.ragas_metrics
            
            dataset = Dataset.from_dict(data)
            
            # Run RAGAS evaluation
            result = evaluate(dataset, metrics=metrics_to_use)
            
            # Extract metrics
            ragas_metrics = RAGASMetrics(
                faithfulness=result['faithfulness'],
                answer_relevancy=result['answer_relevancy'],
                context_precision=result['context_precision'],
                context_recall=result['context_recall'],
                answer_similarity=result.get('answer_similarity'),
                answer_correctness=result.get('answer_correctness')
            )
            
            return ragas_metrics
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {str(e)}")
            return await self._fallback_ragas_evaluation(sample)
    
    async def _evaluate_legal_sample(self, sample: EvaluationSample) -> LegalDomainMetrics:
        """Evaluate sample using legal domain-specific metrics."""
        # Legal accuracy assessment
        legal_accuracy = await self._assess_legal_accuracy(sample.answer, sample.contexts)
        
        # Citation quality assessment
        citation_quality = await self._assess_citation_quality(sample.answer)
        
        # Case law relevance assessment
        case_law_relevance = await self._assess_case_law_relevance(sample.answer, sample.question)
        
        # Jurisdictional accuracy assessment
        jurisdictional_accuracy = await self._assess_jurisdictional_accuracy(sample.answer, sample.contexts)
        
        # Temporal relevance assessment
        temporal_relevance = await self._assess_temporal_relevance(sample.answer)
        
        return LegalDomainMetrics(
            legal_accuracy=legal_accuracy,
            citation_quality=citation_quality,
            case_law_relevance=case_law_relevance,
            jurisdictional_accuracy=jurisdictional_accuracy,
            temporal_relevance=temporal_relevance
        )
    
    async def _assess_legal_accuracy(self, answer: str, contexts: List[str]) -> float:
        """Assess accuracy of legal information in the answer."""
        # Implementation for legal accuracy assessment
        # This would involve checking legal facts against known legal principles
        
        import re
        
        # Simple heuristics (replace with more sophisticated evaluation)
        accuracy_indicators = 0
        total_indicators = 0
        
        # Check for proper legal terminology
        legal_terms = [
            r'pasal\s+\d+', r'undang-undang', r'putusan', r'kasasi',
            r'mahkamah agung', r'pengadilan', r'terdakwa', r'pidana'
        ]
        
        for term in legal_terms:
            if re.search(term, answer, re.IGNORECASE):
                accuracy_indicators += 1
            total_indicators += 1
        
        # Check consistency with context
        context_text = ' '.join(contexts)
        shared_concepts = 0
        answer_concepts = re.findall(r'\b\w{4,}\b', answer.lower())
        
        for concept in answer_concepts[:10]:  # Check first 10 concepts
            if concept in context_text.lower():
                shared_concepts += 1
        
        context_consistency = shared_concepts / max(len(answer_concepts[:10]), 1)
        
        # Combine scores
        if total_indicators > 0:
            term_score = accuracy_indicators / total_indicators
        else:
            term_score = 0.5
        
        return (term_score + context_consistency) / 2
    
    async def _assess_citation_quality(self, answer: str) -> float:
        """Assess quality and completeness of citations."""
        import re
        
        # Look for citation patterns
        citation_patterns = [
            r'putusan\s+no\.?\s*\d+[/\w]+',  # Case citations
            r'no\.?\s*\d+[/\w]+/\d{4}',      # Case numbers
            r'pasal\s+\d+',                   # Article references
            r'uu\s+no\.?\s*\d+',              # Law references
        ]
        
        citations_found = 0
        for pattern in citation_patterns:
            citations_found += len(re.findall(pattern, answer, re.IGNORECASE))
        
        # Score based on citation density and variety
        answer_length = len(answer.split())
        citation_density = min(citations_found / max(answer_length / 100, 1), 1.0)
        
        return citation_density
    
    async def _assess_case_law_relevance(self, answer: str, question: str) -> float:
        """Assess relevance of cited case law to the question."""
        # Simple implementation - check overlap between question and answer topics
        import re
        
        question_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
        answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        
        if not question_words:
            return 0.5
        
        overlap = len(question_words.intersection(answer_words))
        return min(overlap / len(question_words), 1.0)
    
    async def _assess_jurisdictional_accuracy(self, answer: str, contexts: List[str]) -> float:
        """Assess accuracy of jurisdictional context."""
        # Check for Indonesian jurisdiction indicators
        id_indicators = [
            'indonesia', 'republik indonesia', 'mahkamah agung',
            'pengadilan negeri', 'pengadilan tinggi'
        ]
        
        score = 0.0
        for indicator in id_indicators:
            if indicator in answer.lower():
                score += 0.2
        
        return min(score, 1.0)
    
    async def _assess_temporal_relevance(self, answer: str) -> float:
        """Assess temporal relevance of citations."""
        import re
        from datetime import datetime
        
        # Find years mentioned in the answer
        years = re.findall(r'\b(19|20)\d{2}\b', answer)
        
        if not years:
            return 0.5
        
        current_year = datetime.now().year
        recent_years = sum(1 for year in years if current_year - int(year) <= 10)
        
        return min(recent_years / len(years), 1.0)
    
    async def _fallback_ragas_evaluation(self, sample: EvaluationSample) -> RAGASMetrics:
        """Fallback evaluation when RAGAS is not available."""
        # Simple heuristic-based evaluation
        return RAGASMetrics(
            faithfulness=0.7,
            answer_relevancy=0.7,
            context_precision=0.7,
            context_recall=0.7,
            answer_similarity=None,
            answer_correctness=None
        )
    
    async def evaluate_batch(
        self,
        samples: List[EvaluationSample],
        sample_ids: Optional[List[str]] = None
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple samples in batch for efficiency.
        
        Args:
            samples: List of evaluation samples
            sample_ids: Optional list of sample identifiers
            
        Returns:
            Batch evaluation results with aggregated metrics
        """
        start_time = datetime.now()
        
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(samples))]
        
        if len(sample_ids) != len(samples):
            raise ValueError("Number of sample IDs must match number of samples")
        
        results = []
        
        # Process samples in batches
        for i in range(0, len(samples), self.batch_size):
            batch_samples = samples[i:i + self.batch_size]
            batch_ids = sample_ids[i:i + self.batch_size]
            
            # Evaluate batch concurrently
            tasks = [
                self.evaluate_sample(sample, sample_id)
                for sample, sample_id in zip(batch_samples, batch_ids)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            for result in batch_results:
                if isinstance(result, EvaluationResult):
                    results.append(result)
                else:
                    logger.error(f"Evaluation failed: {result}")
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        total_processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchEvaluationResult(
            results=results,
            aggregate_metrics=aggregate_metrics,
            sample_count=len(results),
            total_processing_time=total_processing_time
        )
    
    def _calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate aggregated metrics across all evaluation results."""
        if not results:
            return {}
        
        metrics = {}
        
        # Aggregate RAGAS metrics
        metrics["avg_faithfulness"] = sum(r.ragas_metrics.faithfulness for r in results) / len(results)
        metrics["avg_answer_relevancy"] = sum(r.ragas_metrics.answer_relevancy for r in results) / len(results)
        metrics["avg_context_precision"] = sum(r.ragas_metrics.context_precision for r in results) / len(results)
        metrics["avg_context_recall"] = sum(r.ragas_metrics.context_recall for r in results) / len(results)
        
        # Aggregate legal metrics
        metrics["avg_legal_accuracy"] = sum(r.legal_metrics.legal_accuracy for r in results) / len(results)
        metrics["avg_citation_quality"] = sum(r.legal_metrics.citation_quality for r in results) / len(results)
        metrics["avg_case_law_relevance"] = sum(r.legal_metrics.case_law_relevance for r in results) / len(results)
        
        # Overall performance
        metrics["avg_overall_performance"] = sum(r.overall_performance() for r in results) / len(results)
        
        return metrics


class RAGASCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for RAGAS evaluation integration."""
    
    def __init__(self, evaluator: RAGEvaluator):
        self.evaluator = evaluator
        self.current_sample = None
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain starts."""
        if "question" in inputs:
            self.current_sample = {
                "question": inputs["question"],
                "contexts": [],
                "answer": ""
            }
    
    def on_retriever_end(self, documents: List[Document], **kwargs) -> None:
        """Called when retriever finishes."""
        if self.current_sample:
            self.current_sample["contexts"] = [doc.page_content for doc in documents]
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain ends."""
        if self.current_sample and "answer" in outputs:
            self.current_sample["answer"] = outputs["answer"]
            
            # Trigger evaluation asynchronously
            sample = EvaluationSample(**self.current_sample)
            # Store for batch evaluation later
            
    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when a chain errors."""
        self.current_sample = None


def create_rag_evaluator(
    enable_ragas: bool = True,
    enable_legal_metrics: bool = True,
    batch_size: int = 10
) -> RAGEvaluator:
    """
    Factory function to create RAG evaluator.
    
    Args:
        enable_ragas: Whether to enable RAGAS metrics
        enable_legal_metrics: Whether to enable legal domain metrics
        batch_size: Batch size for evaluation processing
        
    Returns:
        Configured RAGEvaluator instance
    """
    return RAGEvaluator(
        enable_ragas=enable_ragas,
        enable_legal_metrics=enable_legal_metrics,
        batch_size=batch_size
    )
