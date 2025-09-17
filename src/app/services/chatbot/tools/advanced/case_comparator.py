"""
Case Comparator Tool for Sprint 2.
Compares multiple court decisions for similarities and differences.
"""

import logging
import asyncio
import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

try:
    from ....retrieval import RetrievalService, get_retrieval_service
    from ....retrieval.base import RetrievalStrategy  
    from ....llm_service import LLMService, get_llm_service
except ImportError:
    # Fallback for different import paths
    from src.app.services.retrieval import RetrievalService, get_retrieval_service
    from src.app.services.retrieval.base import RetrievalStrategy
    from src.app.services.llm_service import LLMService, get_llm_service

logger = logging.getLogger(__name__)


class CaseComparatorInput(BaseModel):
    """Input schema for case comparator tool."""
    case_identifiers: List[str] = Field(
        description="List of case IDs, case numbers, or case names to compare (2-3 cases max)"
    )
    comparison_aspects: Optional[List[str]] = Field(
        default=["legal_basis", "judicial_reasoning", "verdict", "precedent_value"],
        description="Aspects to compare: legal_basis, judicial_reasoning, verdict, precedent_value"
    )


class CaseComparatorTool(BaseTool):
    """
    Tool for comparing multiple court decisions with deep analysis.
    
    This tool provides comprehensive comparison of court cases including:
    - Legal basis comparison
    - Judicial reasoning analysis
    - Verdict differences
    - Precedent value assessment
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "case_comparator"
    description: str = """
    Compare multiple court decisions to find similarities and differences.
    Use this tool when you need to analyze multiple court cases on the same legal issue.
    
    Input should be a list of case identifiers (2-3 cases maximum) and optional comparison aspects.
    """
    args_schema: type = CaseComparatorInput
    retrieval_service: Optional[Any] = None
    llm_service: Optional[Any] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_services()
    
    def _initialize_services(self) -> None:
        """Initialize retrieval and LLM services."""
        try:
            self.retrieval_service = get_retrieval_service()
            self.llm_service = get_llm_service()
            logger.info("Case comparator tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize case comparator services: {str(e)}")
            self.retrieval_service = None
            self.llm_service = None
    
    def _run(self, case_identifiers: List[str], comparison_aspects: Optional[List[str]] = None) -> str:
        """
        Execute case comparison synchronously.
        
        Args:
            case_identifiers: List of case IDs/numbers/names to compare
            comparison_aspects: Aspects to compare
            
        Returns:
            Formatted comparison analysis as string
        """
        if not self.retrieval_service or not self.llm_service:
            return "Error: Case comparator services are not available. Please try again later."
        
        try:
            # Run async method in sync context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._arun(case_identifiers, comparison_aspects))
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Case comparison failed: {str(e)}")
            return f"Error performing case comparison: {str(e)}"
    
    async def _arun(self, case_identifiers: List[str], comparison_aspects: Optional[List[str]] = None) -> str:
        """
        Execute case comparison asynchronously.
        
        Args:
            case_identifiers: List of case IDs/numbers/names to compare
            comparison_aspects: Aspects to compare
            
        Returns:
            Formatted comparison analysis as string
        """
        if not self.retrieval_service or not self.llm_service:
            return "Error: Case comparator services are not available. Please try again later."
        
        try:
            # Validate input
            if len(case_identifiers) < 2:
                return "Error: At least 2 cases are required for comparison."
            
            if len(case_identifiers) > 3:
                return "Error: Maximum 3 cases can be compared at once."
            
            # Set default comparison aspects
            if not comparison_aspects:
                comparison_aspects = ["legal_basis", "judicial_reasoning", "verdict", "precedent_value"]
            
            # Fetch case details
            cases = await self._fetch_cases(case_identifiers)
            
            if len(cases) < 2:
                return f"Error: Could not retrieve sufficient cases. Found {len(cases)} out of {len(case_identifiers)} requested."
            
            # Perform comparison analysis
            comparison_result = await self._analyze_cases(cases, comparison_aspects)
            
            # Format and return results
            return self._format_comparison_result(comparison_result)
            
        except Exception as e:
            logger.error(f"Async case comparison failed: {str(e)}")
            return f"Error performing case comparison: {str(e)}"
    
    async def _fetch_cases(self, case_identifiers: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch case details from retrieval service.
        
        Args:
            case_identifiers: List of case identifiers
            
        Returns:
            List of case documents with metadata
        """
        cases = []
        
        # Create tasks for parallel fetching
        fetch_tasks = []
        for identifier in case_identifiers:
            task = self._fetch_single_case(identifier)
            fetch_tasks.append(task)
        
        # Execute tasks in parallel
        case_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(case_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch case {case_identifiers[i]}: {str(result)}")
                continue
            
            if result:
                cases.append(result)
        
        return cases
    
    async def _fetch_single_case(self, case_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single case from retrieval service.
        
        Args:
            case_identifier: Case ID, number, or name
            
        Returns:
            Case document if found, None otherwise
        """
        try:
            # Search for the case using various strategies
            search_query = f"putusan {case_identifier}"
            
            # Try with different retrieval strategies
            documents = self.retrieval_service.retrieve(
                query=search_query,
                strategy=RetrievalStrategy.PARENT_CHILD,
                top_k=3,
                include_scores=True
            )
            
            if not documents:
                # Try with vector search
                documents = self.retrieval_service.retrieve(
                    query=search_query,
                    strategy=RetrievalStrategy.VECTOR_SEARCH,
                    top_k=3,
                    include_scores=True
                )
            
            # Find the best matching document
            best_match = None
            best_score = 0.0
            
            for doc in documents:
                # Check if document metadata contains the case identifier
                metadata = doc.metadata
                doc_id = metadata.get('nomor_putusan', metadata.get('nomor', ''))
                doc_title = metadata.get('title', '')
                
                if (case_identifier.lower() in doc_id.lower() or 
                    case_identifier.lower() in doc_title.lower()):
                    score = getattr(doc, 'score', 1.0)
                    if score > best_score:
                        best_score = score
                        best_match = doc
            
            if best_match:
                return {
                    "identifier": case_identifier,
                    "content": best_match.page_content,
                    "metadata": best_match.metadata,
                    "score": best_score
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching case {case_identifier}: {str(e)}")
            return None
    
    async def _analyze_cases(self, cases: List[Dict[str, Any]], comparison_aspects: List[str]) -> Dict[str, Any]:
        """
        Perform comparative analysis of cases.
        
        Args:
            cases: List of case documents
            comparison_aspects: Aspects to compare
            
        Returns:
            Comparison analysis results
        """
        try:
            # Prepare cases for analysis
            case_summaries = []
            for i, case in enumerate(cases):
                summary = {
                    "case_number": i + 1,
                    "identifier": case["identifier"],
                    "title": case["metadata"].get("title", f"Case {i+1}"),
                    "content_excerpt": case["content"][:1000] + "..." if len(case["content"]) > 1000 else case["content"],
                    "metadata": case["metadata"]
                }
                case_summaries.append(summary)
            
            # Create comparison prompt
            comparison_prompt = self._create_comparison_prompt(case_summaries, comparison_aspects)
            
            # Get LLM analysis
            analysis_result = await self.llm_service.llm.ainvoke(comparison_prompt)
            
            # Parse and structure the analysis
            analysis = {
                "cases": case_summaries,
                "comparison_aspects": comparison_aspects,
                "analysis": analysis_result.content,
                "similarities": self._extract_similarities(analysis_result.content),
                "differences": self._extract_differences(analysis_result.content),
                "precedent_analysis": self._extract_precedent_analysis(analysis_result.content)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing cases: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "cases": cases,
                "comparison_aspects": comparison_aspects
            }
    
    def _create_comparison_prompt(self, case_summaries: List[Dict[str, Any]], comparison_aspects: List[str]) -> str:
        """Create prompt for LLM case comparison."""
        
        cases_text = ""
        for case in case_summaries:
            cases_text += f"""
**Case {case['case_number']}: {case['title']}**
- Identifier: {case['identifier']}
- Content: {case['content_excerpt']}
- Metadata: {json.dumps(case['metadata'], indent=2)}

"""
        
        aspects_text = ", ".join(comparison_aspects)
        
        prompt = f"""
Lakukan analisis perbandingan mendalam terhadap putusan-putusan pengadilan berikut:

{cases_text}

**Aspek yang harus dibandingkan:**
{aspects_text}

**Instruksi Analisis:**
1. **Persamaan (Similarities)**: Identifikasi kesamaan dalam dasar hukum, pertimbangan hakim, dan putusan
2. **Perbedaan (Differences)**: Analisis perbedaan pendekatan, interpretasi hukum, dan hasil putusan
3. **Nilai Preseden (Precedent Value)**: Evaluasi kekuatan setiap putusan sebagai preseden hukum
4. **Konsistensi Yuridis**: Analisis konsistensi dengan putusan-putusan sebelumnya

**Format Output:**
Berikan analisis dalam format yang terstruktur dengan sub-bagian untuk setiap aspek perbandingan.
Gunakan bahasa hukum yang tepat namun mudah dipahami.
Sertakan kutipan spesifik dari putusan jika relevan.

**Analisis Perbandingan:**
"""
        
        return prompt
    
    def _extract_similarities(self, analysis_text: str) -> List[str]:
        """Extract similarities from analysis text."""
        similarities = []
        
        # Look for similarity indicators
        lines = analysis_text.split('\\n')
        in_similarity_section = False
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['persamaan', 'similarities', 'kesamaan']):
                in_similarity_section = True
                continue
            elif any(keyword in line.lower() for keyword in ['perbedaan', 'differences', 'analisis']):
                in_similarity_section = False
            
            if in_similarity_section and line and (line.startswith(('-', '•', '*')) or line[0].isdigit()):
                similarities.append(line)
        
        return similarities[:5]  # Limit to 5 similarities
    
    def _extract_differences(self, analysis_text: str) -> List[str]:
        """Extract differences from analysis text."""
        differences = []
        
        # Look for difference indicators
        lines = analysis_text.split('\\n')
        in_difference_section = False
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['perbedaan', 'differences']):
                in_difference_section = True
                continue
            elif any(keyword in line.lower() for keyword in ['preseden', 'precedent', 'konsistensi']):
                in_difference_section = False
            
            if in_difference_section and line and (line.startswith(('-', '•', '*')) or line[0].isdigit()):
                differences.append(line)
        
        return differences[:5]  # Limit to 5 differences
    
    def _extract_precedent_analysis(self, analysis_text: str) -> str:
        """Extract precedent value analysis from text."""
        lines = analysis_text.split('\\n')
        precedent_lines = []
        in_precedent_section = False
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['preseden', 'precedent']):
                in_precedent_section = True
                precedent_lines.append(line)
                continue
            elif in_precedent_section and any(keyword in line.lower() for keyword in ['konsistensi', 'kesimpulan']):
                break
            
            if in_precedent_section and line:
                precedent_lines.append(line)
        
        return '\\n'.join(precedent_lines[:10])  # Limit length
    
    def _format_comparison_result(self, comparison_result: Dict[str, Any]) -> str:
        """Format comparison results into readable text."""
        
        if "error" in comparison_result:
            return f"Error in case comparison: {comparison_result['error']}"
        
        try:
            cases = comparison_result.get("cases", [])
            analysis = comparison_result.get("analysis", "")
            similarities = comparison_result.get("similarities", [])
            differences = comparison_result.get("differences", [])
            precedent_analysis = comparison_result.get("precedent_analysis", "")
            
            formatted_result = f"""## Analisis Perbandingan Putusan

### Kasus yang Dibandingkan:
"""
            
            for case in cases:
                formatted_result += f"**{case['case_number']}.** {case['title']} ({case['identifier']})\\n"
            
            formatted_result += f"""
### Analisis Lengkap:
{analysis}

### Persamaan Utama:
"""
            
            for i, similarity in enumerate(similarities, 1):
                formatted_result += f"{i}. {similarity}\\n"
            
            formatted_result += f"""
### Perbedaan Utama:
"""
            
            for i, difference in enumerate(differences, 1):
                formatted_result += f"{i}. {difference}\\n"
            
            if precedent_analysis:
                formatted_result += f"""
### Nilai Preseden:
{precedent_analysis}
"""
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error formatting comparison result: {str(e)}")
            return f"Comparison completed but formatting failed: {str(e)}"