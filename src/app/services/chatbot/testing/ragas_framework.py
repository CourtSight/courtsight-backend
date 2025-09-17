"""
RAGAS Testing Framework for RAG Quality Validation.
Provides comprehensive evaluation of RAG responses using multiple metrics.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json

from pydantic import BaseModel, Field

try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
        answer_similarity,
        answer_correctness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS not available. Install with: pip install ragas")

from ....core.config import settings

logger = logging.getLogger(__name__)


class RAGEvaluationMetrics(BaseModel):
    """RAGAS evaluation metrics."""
    answer_relevancy: Optional[float] = Field(default=None, description="How relevant the answer is to the question")
    context_precision: Optional[float] = Field(default=None, description="How precise the retrieved context is")
    context_recall: Optional[float] = Field(default=None, description="How much of the relevant context was retrieved")
    faithfulness: Optional[float] = Field(default=None, description="How faithful the answer is to the context")
    answer_similarity: Optional[float] = Field(default=None, description="Semantic similarity to reference answer")
    answer_correctness: Optional[float] = Field(default=None, description="Overall correctness of the answer")


class RAGTestCase(BaseModel):
    """Individual RAG test case."""
    question: str = Field(description="Test question")
    contexts: List[str] = Field(description="Retrieved contexts")
    answer: str = Field(description="Generated answer")
    ground_truth: Optional[str] = Field(default=None, description="Reference/expected answer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RAGEvaluationResult(BaseModel):
    """Complete RAG evaluation result."""
    test_case: RAGTestCase
    metrics: RAGEvaluationMetrics
    evaluation_time: datetime
    evaluation_duration: float
    errors: List[str] = Field(default_factory=list)


class RAGASTestingFramework:
    """
    RAGAS-based testing framework for RAG quality validation.
    
    Provides comprehensive evaluation of:
    - Answer relevancy to questions
    - Context precision and recall
    - Faithfulness to source material
    - Answer similarity and correctness
    """
    
    def __init__(self):
        """Initialize RAGAS testing framework."""
        self.available = RAGAS_AVAILABLE
        self.evaluation_history: List[RAGEvaluationResult] = []
        
        if not self.available:
            logger.warning("RAGAS not available. Install with: pip install ragas")
    
    async def evaluate_single_response(self, 
                                     question: str,
                                     contexts: List[str],
                                     answer: str,
                                     ground_truth: Optional[str] = None,
                                     metadata: Optional[Dict[str, Any]] = None) -> RAGEvaluationResult:
        """
        Evaluate a single RAG response.
        
        Args:
            question: The question asked
            contexts: Retrieved contexts used for answer generation
            answer: Generated answer
            ground_truth: Optional reference answer
            metadata: Optional additional metadata
            
        Returns:
            Evaluation result with metrics
        """
        start_time = datetime.now()
        
        test_case = RAGTestCase(
            question=question,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth,
            metadata=metadata or {}
        )
        
        if not self.available:
            logger.warning("RAGAS not available for evaluation")
            return RAGEvaluationResult(
                test_case=test_case,
                metrics=RAGEvaluationMetrics(),
                evaluation_time=start_time,
                evaluation_duration=0.0,
                errors=["RAGAS not installed"]
            )
        
        try:
            # Prepare data for RAGAS
            data = {
                "question": [question],
                "contexts": [contexts],
                "answer": [answer]
            }
            
            # Add ground truth if available
            if ground_truth:
                data["ground_truth"] = [ground_truth]
            
            # Create dataset
            dataset = Dataset.from_dict(data)
            
            # Define metrics to evaluate
            metrics_to_eval = [
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness
            ]
            
            # Add metrics that require ground truth
            if ground_truth:
                metrics_to_eval.extend([
                    answer_similarity,
                    answer_correctness
                ])
            
            # Run evaluation
            logger.debug(f"Evaluating RAG response with {len(metrics_to_eval)} metrics")
            
            # Execute evaluation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            eval_result = await loop.run_in_executor(
                None,
                lambda: evaluate(dataset, metrics=metrics_to_eval)
            )
            
            # Extract metrics
            metrics = RAGEvaluationMetrics(
                answer_relevancy=eval_result.get("answer_relevancy", [None])[0],
                context_precision=eval_result.get("context_precision", [None])[0],
                context_recall=eval_result.get("context_recall", [None])[0],
                faithfulness=eval_result.get("faithfulness", [None])[0],
                answer_similarity=eval_result.get("answer_similarity", [None])[0] if ground_truth else None,
                answer_correctness=eval_result.get("answer_correctness", [None])[0] if ground_truth else None
            )
            
            evaluation_duration = (datetime.now() - start_time).total_seconds()
            
            result = RAGEvaluationResult(
                test_case=test_case,
                metrics=metrics,
                evaluation_time=start_time,
                evaluation_duration=evaluation_duration,
                errors=[]
            )
            
            # Store in history
            self.evaluation_history.append(result)
            
            logger.info(f"RAG evaluation completed in {evaluation_duration:.2f}s")
            logger.info(f"Metrics: relevancy={metrics.answer_relevancy:.3f}, "
                       f"precision={metrics.context_precision:.3f}, "
                       f"recall={metrics.context_recall:.3f}, "
                       f"faithfulness={metrics.faithfulness:.3f}")
            
            return result
            
        except Exception as e:
            evaluation_duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"RAGAS evaluation failed: {str(e)}"
            logger.error(error_msg)
            
            result = RAGEvaluationResult(
                test_case=test_case,
                metrics=RAGEvaluationMetrics(),
                evaluation_time=start_time,
                evaluation_duration=evaluation_duration,
                errors=[error_msg]
            )
            
            self.evaluation_history.append(result)
            return result
    
    async def evaluate_batch_responses(self, test_cases: List[RAGTestCase]) -> List[RAGEvaluationResult]:
        """
        Evaluate multiple RAG responses in batch.
        
        Args:
            test_cases: List of test cases to evaluate
            
        Returns:
            List of evaluation results
        """
        if not self.available:
            logger.warning("RAGAS not available for batch evaluation")
            return [
                RAGEvaluationResult(
                    test_case=case,
                    metrics=RAGEvaluationMetrics(),
                    evaluation_time=datetime.now(),
                    evaluation_duration=0.0,
                    errors=["RAGAS not installed"]
                )
                for case in test_cases
            ]
        
        logger.info(f"Starting batch evaluation of {len(test_cases)} test cases")
        
        results = []
        for i, test_case in enumerate(test_cases):
            logger.debug(f"Evaluating test case {i+1}/{len(test_cases)}")
            
            result = await self.evaluate_single_response(
                question=test_case.question,
                contexts=test_case.contexts,
                answer=test_case.answer,
                ground_truth=test_case.ground_truth,
                metadata=test_case.metadata
            )
            
            results.append(result)
            
            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.1)
        
        logger.info(f"Batch evaluation completed: {len(results)} results")
        return results
    
    async def evaluate_chatbot_response(self, 
                                      question: str,
                                      chatbot_response: Dict[str, Any],
                                      ground_truth: Optional[str] = None) -> RAGEvaluationResult:
        """
        Evaluate a chatbot response using RAGAS.
        
        Args:
            question: User question
            chatbot_response: Complete chatbot response with sources
            ground_truth: Optional reference answer
            
        Returns:
            Evaluation result
        """
        try:
            # Extract answer and contexts from chatbot response
            answer = chatbot_response.get("response", "")
            sources = chatbot_response.get("sources", [])
            
            # Convert sources to context strings
            contexts = []
            for source in sources:
                if isinstance(source, dict):
                    # Extract text content from source
                    content = source.get("content", source.get("text", str(source)))
                    contexts.append(content)
                else:
                    contexts.append(str(source))
            
            # Prepare metadata
            metadata = {
                "conversation_id": chatbot_response.get("conversation_id"),
                "confidence": chatbot_response.get("confidence"),
                "complexity": chatbot_response.get("complexity"),
                "workflow_used": "workflow_result" in chatbot_response,
                "agent_used": "agent_result" in chatbot_response,
                "sources_count": len(sources),
                "response_length": len(answer)
            }
            
            return await self.evaluate_single_response(
                question=question,
                contexts=contexts,
                answer=answer,
                ground_truth=ground_truth,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error evaluating chatbot response: {str(e)}")
            
            # Return error result
            test_case = RAGTestCase(
                question=question,
                contexts=[],
                answer=chatbot_response.get("response", ""),
                ground_truth=ground_truth,
                metadata={"evaluation_error": str(e)}
            )
            
            return RAGEvaluationResult(
                test_case=test_case,
                metrics=RAGEvaluationMetrics(),
                evaluation_time=datetime.now(),
                evaluation_duration=0.0,
                errors=[f"Evaluation error: {str(e)}"]
            )
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive evaluation statistics.
        
        Returns:
            Statistics across all evaluations
        """
        if not self.evaluation_history:
            return {
                "total_evaluations": 0,
                "average_metrics": {},
                "error_rate": 0.0
            }
        
        # Calculate average metrics
        metrics_sums = {
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": [],
            "faithfulness": [],
            "answer_similarity": [],
            "answer_correctness": []
        }
        
        successful_evaluations = 0
        total_errors = 0
        
        for result in self.evaluation_history:
            if not result.errors:
                successful_evaluations += 1
                
                metrics = result.metrics
                if metrics.answer_relevancy is not None:
                    metrics_sums["answer_relevancy"].append(metrics.answer_relevancy)
                if metrics.context_precision is not None:
                    metrics_sums["context_precision"].append(metrics.context_precision)
                if metrics.context_recall is not None:
                    metrics_sums["context_recall"].append(metrics.context_recall)
                if metrics.faithfulness is not None:
                    metrics_sums["faithfulness"].append(metrics.faithfulness)
                if metrics.answer_similarity is not None:
                    metrics_sums["answer_similarity"].append(metrics.answer_similarity)
                if metrics.answer_correctness is not None:
                    metrics_sums["answer_correctness"].append(metrics.answer_correctness)
            else:
                total_errors += len(result.errors)
        
        # Calculate averages
        average_metrics = {}
        for metric_name, values in metrics_sums.items():
            if values:
                average_metrics[metric_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        # Calculate error rate
        error_rate = total_errors / len(self.evaluation_history) if self.evaluation_history else 0.0
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "successful_evaluations": successful_evaluations,
            "average_metrics": average_metrics,
            "error_rate": error_rate,
            "ragas_available": self.available,
            "evaluation_period": {
                "first": self.evaluation_history[0].evaluation_time.isoformat() if self.evaluation_history else None,
                "last": self.evaluation_history[-1].evaluation_time.isoformat() if self.evaluation_history else None
            }
        }
    
    def get_recent_evaluations(self, limit: int = 10) -> List[RAGEvaluationResult]:
        """
        Get recent evaluation results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            Recent evaluation results
        """
        return self.evaluation_history[-limit:] if self.evaluation_history else []
    
    async def create_test_suite(self, test_questions: List[str], 
                               chatbot_service) -> List[RAGTestCase]:
        """
        Create a test suite by generating responses for questions.
        
        Args:
            test_questions: List of test questions
            chatbot_service: Chatbot service to generate responses
            
        Returns:
            List of test cases
        """
        test_cases = []
        
        for i, question in enumerate(test_questions):
            try:
                logger.debug(f"Generating response for test question {i+1}/{len(test_questions)}")
                
                # Generate response using chatbot
                response = await chatbot_service.chat(
                    message=question,
                    conversation_id=f"test_conversation_{i}",
                    user_id="test_user"
                )
                
                # Extract contexts from sources
                sources = response.get("sources", [])
                contexts = []
                for source in sources:
                    if isinstance(source, dict):
                        content = source.get("content", source.get("text", str(source)))
                        contexts.append(content)
                    else:
                        contexts.append(str(source))
                
                test_case = RAGTestCase(
                    question=question,
                    contexts=contexts,
                    answer=response.get("response", ""),
                    metadata={
                        "test_id": i,
                        "generated_at": datetime.now().isoformat(),
                        "confidence": response.get("confidence"),
                        "sources_count": len(sources)
                    }
                )
                
                test_cases.append(test_case)
                
                # Small delay between requests
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error generating test case for question '{question}': {str(e)}")
                
                # Create empty test case for failed generation
                test_case = RAGTestCase(
                    question=question,
                    contexts=[],
                    answer="",
                    metadata={
                        "test_id": i,
                        "generation_error": str(e),
                        "generated_at": datetime.now().isoformat()
                    }
                )
                test_cases.append(test_case)
        
        logger.info(f"Created test suite with {len(test_cases)} test cases")
        return test_cases
    
    def export_evaluation_results(self, format: str = "json") -> str:
        """
        Export evaluation results to specified format.
        
        Args:
            format: Export format ("json", "csv")
            
        Returns:
            Serialized evaluation data
        """
        if format.lower() == "json":
            # Convert to JSON-serializable format
            export_data = []
            for result in self.evaluation_history:
                export_data.append({
                    "question": result.test_case.question,
                    "answer": result.test_case.answer,
                    "contexts_count": len(result.test_case.contexts),
                    "ground_truth": result.test_case.ground_truth,
                    "metrics": result.metrics.dict(),
                    "evaluation_time": result.evaluation_time.isoformat(),
                    "evaluation_duration": result.evaluation_duration,
                    "errors": result.errors,
                    "metadata": result.test_case.metadata
                })
            
            return json.dumps(export_data, indent=2)
        
        elif format.lower() == "csv":
            # Create CSV format
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "question", "answer", "contexts_count", "ground_truth",
                "answer_relevancy", "context_precision", "context_recall",
                "faithfulness", "answer_similarity", "answer_correctness",
                "evaluation_time", "evaluation_duration", "errors"
            ])
            
            # Write data
            for result in self.evaluation_history:
                writer.writerow([
                    result.test_case.question,
                    result.test_case.answer,
                    len(result.test_case.contexts),
                    result.test_case.ground_truth or "",
                    result.metrics.answer_relevancy or "",
                    result.metrics.context_precision or "",
                    result.metrics.context_recall or "",
                    result.metrics.faithfulness or "",
                    result.metrics.answer_similarity or "",
                    result.metrics.answer_correctness or "",
                    result.evaluation_time.isoformat(),
                    result.evaluation_duration,
                    "; ".join(result.errors)
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global instance
_ragas_framework: Optional[RAGASTestingFramework] = None


def get_ragas_framework() -> RAGASTestingFramework:
    """Get or create RAGAS testing framework instance."""
    global _ragas_framework
    
    if _ragas_framework is None:
        _ragas_framework = RAGASTestingFramework()
    
    return _ragas_framework