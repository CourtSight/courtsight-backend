"""
Precedent Explorer Tool for Sprint 2.
Finds legal precedents using semantic similarity search.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
import numpy as np

try:
    from ...retrieval import get_retrieval_service
    from ...retrieval.base import RetrievalStrategy
    from ...llm_service import LLMService, get_llm_service
except ImportError:
    # Fallback for different import paths
    from src.app.services.retrieval import get_retrieval_service
    from src.app.services.retrieval.base import RetrievalStrategy
    from src.app.services.llm_service import LLMService, get_llm_service

logger = logging.getLogger(__name__)


class PrecedentExplorerInput(BaseModel):
    """Input schema for precedent explorer tool."""
    legal_issue: str = Field(
        description="Legal issue or question to find precedents for"
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity threshold (0.0-1.0) for considering a case as precedent"
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of precedent cases to return"
    )
    precedent_strength_filter: Optional[str] = Field(
        default=None,
        description="Filter by precedent strength: 'high', 'medium', 'low', or None for all"
    )


class PrecedentExplorerTool(BaseTool):
    """
    Tool for finding legal precedents using semantic similarity.
    
    This tool provides comprehensive precedent discovery including:
    - Semantic similarity search on case database
    - Timeline analysis of precedents
    - Precedent strength assessment
    - Related cases discovery
    - Evolution of legal interpretation
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "precedent_explorer"
    description: str = """
    Find legal precedents using semantic similarity search on court decisions.
    Use this tool when you need to discover relevant precedent cases for a legal issue.
    
    Input should be a clear description of the legal issue and optional similarity threshold.
    """
    args_schema: type = PrecedentExplorerInput
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
            logger.info("Precedent explorer tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize precedent explorer services: {str(e)}")
            self.retrieval_service = None
            self.llm_service = None
    
    def _run(self, legal_issue: str, similarity_threshold: float = 0.7, 
             max_results: int = 5, precedent_strength_filter: Optional[str] = None) -> str:
        """
        Execute precedent search synchronously.
        
        Args:
            legal_issue: Legal issue to find precedents for
            similarity_threshold: Minimum similarity threshold
            max_results: Maximum number of results
            precedent_strength_filter: Filter by precedent strength
            
        Returns:
            Formatted precedent analysis as string
        """
        if not self.retrieval_service or not self.llm_service:
            return "Error: Precedent explorer services are not available. Please try again later."
        
        try:
            # Run async method in sync context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._arun(legal_issue, similarity_threshold, max_results, precedent_strength_filter)
                )
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Precedent exploration failed: {str(e)}")
            return f"Error performing precedent search: {str(e)}"
    
    async def _arun(self, legal_issue: str, similarity_threshold: float = 0.7,
                    max_results: int = 5, precedent_strength_filter: Optional[str] = None) -> str:
        """
        Execute precedent search asynchronously.
        
        Args:
            legal_issue: Legal issue to find precedents for
            similarity_threshold: Minimum similarity threshold
            max_results: Maximum number of results
            precedent_strength_filter: Filter by precedent strength
            
        Returns:
            Formatted precedent analysis as string
        """
        if not self.retrieval_service or not self.llm_service:
            return "Error: Precedent explorer services are not available. Please try again later."
        
        try:
            # Validate inputs
            if not legal_issue.strip():
                return "Error: Legal issue cannot be empty."
            
            if not 0.0 <= similarity_threshold <= 1.0:
                return "Error: Similarity threshold must be between 0.0 and 1.0."
            
            if max_results < 1 or max_results > 20:
                return "Error: Max results must be between 1 and 20."
            
            # Enhance the legal issue query for better search
            enhanced_query = await self._enhance_legal_issue_query(legal_issue)
            
            # Find similar cases using multiple strategies
            similar_cases = await self._find_similar_cases(enhanced_query, max_results * 2)  # Get more for filtering
            
            if not similar_cases:
                return f"No precedent cases found for the legal issue: {legal_issue}"
            
            # Analyze precedent strength
            precedent_analysis = await self._analyze_precedent_strength(similar_cases, legal_issue)
            
            # Filter by precedent strength if specified
            if precedent_strength_filter:
                precedent_analysis = self._filter_by_strength(precedent_analysis, precedent_strength_filter)
            
            # Limit results
            precedent_analysis = precedent_analysis[:max_results]
            
            # Generate timeline analysis
            timeline_analysis = self._generate_timeline_analysis(precedent_analysis)
            
            # Format and return results
            return self._format_precedent_report(
                legal_issue, precedent_analysis, timeline_analysis, similarity_threshold
            )
            
        except Exception as e:
            logger.error(f"Async precedent exploration failed: {str(e)}")
            return f"Error performing precedent search: {str(e)}"
    
    async def _enhance_legal_issue_query(self, legal_issue: str) -> str:
        """
        Enhance the legal issue query for better search results.
        
        Args:
            legal_issue: Original legal issue
            
        Returns:
            Enhanced query string
        """
        try:
            enhancement_prompt = f"""
            Tingkatkan query pencarian berikut untuk menemukan preseden hukum yang relevan:
            
            Isu Hukum: {legal_issue}
            
            Berikan query yang diperbaiki dengan:
            1. Kata kunci hukum yang tepat
            2. Istilah teknis yang relevan
            3. Sinonim yang umum digunakan dalam putusan pengadilan
            
            Format: berikan hanya query yang diperbaiki tanpa penjelasan tambahan.
            """
            
            result = await self.llm_service.llm.ainvoke(enhancement_prompt)
            enhanced_query = result.content.strip()
            
            # If enhancement fails, use original
            if not enhanced_query or len(enhanced_query) < len(legal_issue) // 2:
                return legal_issue
            
            logger.info(f"Enhanced query from '{legal_issue}' to '{enhanced_query}'")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {str(e)}")
            return legal_issue
    
    async def _find_similar_cases(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Find similar cases using multiple retrieval strategies.
        
        Args:
            query: Enhanced query string
            max_results: Maximum number of results
            
        Returns:
            List of similar cases with metadata
        """
        similar_cases = []
        
        try:
            # Strategy 1: Parent-Child retrieval
            documents_pc = self.retrieval_service.retrieve(
                query=query,
                strategy=RetrievalStrategy.PARENT_CHILD,
                top_k=max_results,
                include_scores=True
            )
            
            # Strategy 2: Vector search
            documents_vs = self.retrieval_service.retrieve(
                query=query,
                strategy=RetrievalStrategy.VECTOR_SEARCH,
                top_k=max_results,
                include_scores=True
            )
            
            # # Strategy 3: Hybrid search (if available)
            # try:
            #     documents_hybrid = self.retrieval_service.retrieve(
            #         query=query,
            #         strategy=RetrievalStrategy.HYBRID,
            #         top_k=max_results,
            #         include_scores=True
            #     )
            # except:
            #     documents_hybrid = []
            
            # Combine and deduplicate results
            all_documents = documents_pc + documents_vs 
            seen_cases = set()
            
            for doc in all_documents:
                # Create unique identifier for the case
                case_id = doc.metadata.get('nomor_putusan', doc.metadata.get('nomor', doc.page_content[:100]))
                
                if case_id not in seen_cases:
                    seen_cases.add(case_id)
                    
                    case_info = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, 'score', 0.8),  # Default score if not available
                        "case_id": case_id,
                        "title": doc.metadata.get('title', f'Putusan {case_id}'),
                        "source": doc.metadata.get('source', 'Unknown')
                    }
                    similar_cases.append(case_info)
            
            # Sort by relevance score
            similar_cases.sort(key=lambda x: x['score'], reverse=True)
            
            return similar_cases[:max_results]
            
        except Exception as e:
            logger.error(f"Error finding similar cases: {str(e)}")
            return []
    
    async def _analyze_precedent_strength(self, cases: List[Dict[str, Any]], legal_issue: str) -> List[Dict[str, Any]]:
        """
        Analyze the precedent strength of each case.
        
        Args:
            cases: List of similar cases
            legal_issue: Original legal issue
            
        Returns:
            Cases with precedent strength analysis
        """
        analyzed_cases = []
        
        for case in cases:
            try:
                # Analyze precedent strength using LLM
                strength_analysis = await self._assess_single_case_strength(case, legal_issue)
                
                # Add analysis to case
                case_with_analysis = case.copy()
                case_with_analysis.update({
                    "precedent_strength": strength_analysis["strength"],
                    "strength_score": strength_analysis["score"],
                    "strength_reasoning": strength_analysis["reasoning"],
                    "legal_principle": strength_analysis["legal_principle"],
                    "applicability": strength_analysis["applicability"]
                })
                
                analyzed_cases.append(case_with_analysis)
                
            except Exception as e:
                logger.error(f"Error analyzing precedent strength for case {case.get('case_id', 'unknown')}: {str(e)}")
                # Add case with default analysis
                case_with_analysis = case.copy()
                case_with_analysis.update({
                    "precedent_strength": "medium",
                    "strength_score": 0.5,
                    "strength_reasoning": "Analysis failed",
                    "legal_principle": "Unknown",
                    "applicability": "Uncertain"
                })
                analyzed_cases.append(case_with_analysis)
        
        # Sort by strength score
        analyzed_cases.sort(key=lambda x: x["strength_score"], reverse=True)
        
        return analyzed_cases
    
    async def _assess_single_case_strength(self, case: Dict[str, Any], legal_issue: str) -> Dict[str, Any]:
        """
        Assess precedent strength for a single case.
        
        Args:
            case: Case information
            legal_issue: Legal issue being researched
            
        Returns:
            Precedent strength analysis
        """
        try:
            assessment_prompt = f"""
            Analisis kekuatan preseden dari putusan pengadilan berikut untuk isu hukum yang diberikan:
            
            **Isu Hukum:** {legal_issue}
            
            **Putusan:** {case['title']}
            **Konten:** {case['content'][:1500]}...
            **Metadata:** {json.dumps(case['metadata'], indent=2)}
            
            Berikan analisis dalam format JSON dengan struktur berikut:
            {{
                "strength": "high|medium|low",
                "score": 0.0-1.0,
                "reasoning": "alasan mengapa precedent ini kuat/lemah",
                "legal_principle": "prinsip hukum utama yang ditetapkan",
                "applicability": "seberapa applicable untuk isu yang diberikan"
            }}
            
            Pertimbangkan faktor:
            1. Tingkat pengadilan (MA, PT, PN)
            2. Relevansi dengan isu hukum
            3. Kejelasan prinsip hukum
            4. Konsistensi dengan putusan lain
            5. Kekuatan pertimbangan hukum
            """
            
            result = await self.llm_service.llm.ainvoke(assessment_prompt)
            
            # Try to parse JSON response
            try:
                analysis = json.loads(result.content)
                
                # Validate required fields
                required_fields = ["strength", "score", "reasoning", "legal_principle", "applicability"]
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = "Unknown" if field != "score" else 0.5
                
                # Validate strength value
                if analysis["strength"] not in ["high", "medium", "low"]:
                    analysis["strength"] = "medium"
                
                # Validate score range
                analysis["score"] = max(0.0, min(1.0, float(analysis["score"])))
                
                return analysis
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                
                # Fallback analysis based on simple heuristics
                return self._simple_strength_assessment(case, legal_issue)
            
        except Exception as e:
            logger.error(f"Precedent strength assessment failed: {str(e)}")
            return self._simple_strength_assessment(case, legal_issue)
    
    def _simple_strength_assessment(self, case: Dict[str, Any], legal_issue: str) -> Dict[str, Any]:
        """
        Simple fallback precedent strength assessment.
        
        Args:
            case: Case information
            legal_issue: Legal issue
            
        Returns:
            Basic strength analysis
        """
        score = 0.5  # Base score
        strength = "medium"
        
        # Check court level
        content = case.get('content', '').lower()
        metadata = case.get('metadata', {})
        
        if 'mahkamah agung' in content or 'ma' in metadata.get('source', '').lower():
            score += 0.3
            strength = "high"
        elif 'pengadilan tinggi' in content:
            score += 0.1
        
        # Check relevance by keyword matching
        legal_issue_words = set(legal_issue.lower().split())
        content_words = set(content.lower().split())
        
        overlap = len(legal_issue_words.intersection(content_words))
        relevance_score = overlap / max(len(legal_issue_words), 1)
        score += relevance_score * 0.2
        
        # Adjust strength based on final score
        if score >= 0.8:
            strength = "high"
        elif score >= 0.6:
            strength = "medium"
        else:
            strength = "low"
        
        return {
            "strength": strength,
            "score": min(1.0, score),
            "reasoning": f"Automated assessment based on court level and keyword relevance",
            "legal_principle": "To be determined from detailed analysis",
            "applicability": f"Estimated {int(relevance_score * 100)}% relevant to the legal issue"
        }
    
    def _filter_by_strength(self, cases: List[Dict[str, Any]], strength_filter: str) -> List[Dict[str, Any]]:
        """
        Filter cases by precedent strength.
        
        Args:
            cases: List of analyzed cases
            strength_filter: 'high', 'medium', or 'low'
            
        Returns:
            Filtered list of cases
        """
        if strength_filter.lower() not in ['high', 'medium', 'low']:
            return cases
        
        return [case for case in cases if case.get('precedent_strength') == strength_filter.lower()]
    
    def _generate_timeline_analysis(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate timeline analysis of precedent evolution.
        
        Args:
            cases: List of analyzed cases
            
        Returns:
            Timeline analysis
        """
        try:
            # Extract years from cases
            case_years = []
            for case in cases:
                # Try to extract year from metadata or content
                year = self._extract_year_from_case(case)
                if year:
                    case_years.append((year, case))
            
            # Sort by year
            case_years.sort(key=lambda x: x[0])
            
            # Analyze evolution
            timeline_analysis = {
                "earliest_case": case_years[0] if case_years else None,
                "latest_case": case_years[-1] if case_years else None,
                "case_count_by_decade": self._count_cases_by_decade(case_years),
                "evolution_trend": self._analyze_evolution_trend(case_years),
                "time_span": len(set(year for year, _ in case_years)) if case_years else 0
            }
            
            return timeline_analysis
            
        except Exception as e:
            logger.error(f"Timeline analysis failed: {str(e)}")
            return {"error": "Timeline analysis failed", "case_count": len(cases)}
    
    def _extract_year_from_case(self, case: Dict[str, Any]) -> Optional[int]:
        """Extract year from case metadata or content."""
        try:
            # Try metadata first
            metadata = case.get('metadata', {})
            
            # Look for year in various metadata fields
            for field in ['tahun', 'year', 'tanggal', 'date']:
                if field in metadata:
                    value = str(metadata[field])
                    year = self._extract_year_from_string(value)
                    if year:
                        return year
            
            # Try case ID/number
            case_id = case.get('case_id', '')
            year = self._extract_year_from_string(case_id)
            if year:
                return year
            
            # Try content (first 500 characters)
            content = case.get('content', '')[:500]
            year = self._extract_year_from_string(content)
            return year
            
        except Exception as e:
            logger.error(f"Year extraction failed: {str(e)}")
            return None
    
    def _extract_year_from_string(self, text: str) -> Optional[int]:
        """Extract 4-digit year from string."""
        import re
        
        # Look for 4-digit years (1900-2030)
        year_pattern = r'\\b(19\\d{2}|20[0-3]\\d)\\b'
        matches = re.findall(year_pattern, text)
        
        if matches:
            # Return the most recent reasonable year
            years = [int(year) for year in matches]
            reasonable_years = [year for year in years if 1950 <= year <= 2030]
            return max(reasonable_years) if reasonable_years else max(years)
        
        return None
    
    def _count_cases_by_decade(self, case_years: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, int]:
        """Count cases by decade."""
        decade_counts = {}
        
        for year, _ in case_years:
            decade = (year // 10) * 10
            decade_key = f"{decade}s"
            decade_counts[decade_key] = decade_counts.get(decade_key, 0) + 1
        
        return decade_counts
    
    def _analyze_evolution_trend(self, case_years: List[Tuple[int, Dict[str, Any]]]) -> str:
        """Analyze the evolution trend of legal interpretation."""
        if len(case_years) < 2:
            return "Insufficient data for trend analysis"
        
        earliest_year = case_years[0][0]
        latest_year = case_years[-1][0]
        span = latest_year - earliest_year
        
        if span < 5:
            return "Recent cases, no significant evolution trend"
        elif span < 15:
            return "Medium-term evolution spanning multiple years"
        else:
            return "Long-term evolution showing development of legal interpretation"
    
    def _format_precedent_report(self, legal_issue: str, precedent_analysis: List[Dict[str, Any]], 
                                 timeline_analysis: Dict[str, Any], similarity_threshold: float) -> str:
        """
        Format precedent analysis into readable report.
        
        Args:
            legal_issue: Original legal issue
            precedent_analysis: Analyzed precedent cases
            timeline_analysis: Timeline analysis
            similarity_threshold: Used similarity threshold
            
        Returns:
            Formatted precedent report
        """
        try:
            report = f"""## Analisis Preseden Hukum

### Isu Hukum:
{legal_issue}

### Parameter Pencarian:
- Threshold Similarity: {similarity_threshold}
- Jumlah Preseden Ditemukan: {len(precedent_analysis)}

### Preseden yang Ditemukan:

"""
            
            # Add each precedent case
            for i, case in enumerate(precedent_analysis, 1):
                strength_icon = "🔴" if case['precedent_strength'] == 'high' else "🟡" if case['precedent_strength'] == 'medium' else "🟢"
                
                report += f"""#### {i}. {case['title']} {strength_icon}

**Kekuatan Preseden:** {case['precedent_strength'].title()} ({case['strength_score']:.2f})
**Prinsip Hukum:** {case['legal_principle']}
**Aplikabilitas:** {case['applicability']}
**Alasan:** {case['strength_reasoning']}

**Ringkasan Putusan:**
{case['content'][:300]}...

---
"""
            
            # Add timeline analysis
            if not timeline_analysis.get('error'):
                report += f"""
### Analisis Timeline:

**Rentang Waktu:** {timeline_analysis.get('time_span', 0)} tahun
**Tren Evolusi:** {timeline_analysis.get('evolution_trend', 'Unknown')}

"""
                
                if timeline_analysis.get('case_count_by_decade'):
                    report += "**Distribusi per Dekade:**\\n"
                    for decade, count in timeline_analysis['case_count_by_decade'].items():
                        report += f"- {decade}: {count} kasus\\n"
                
                if timeline_analysis.get('earliest_case'):
                    earliest = timeline_analysis['earliest_case']
                    report += f"**Preseden Paling Awal:** {earliest[0]} - {earliest[1]['title']}\\n"
                
                if timeline_analysis.get('latest_case'):
                    latest = timeline_analysis['latest_case']
                    report += f"**Preseden Paling Baru:** {latest[0]} - {latest[1]['title']}\\n"
            
            # Add summary
            high_strength = len([c for c in precedent_analysis if c['precedent_strength'] == 'high'])
            medium_strength = len([c for c in precedent_analysis if c['precedent_strength'] == 'medium'])
            low_strength = len([c for c in precedent_analysis if c['precedent_strength'] == 'low'])
            
            report += f"""
### Ringkasan:
- **Preseden Kuat:** {high_strength} kasus
- **Preseden Sedang:** {medium_strength} kasus  
- **Preseden Lemah:** {low_strength} kasus

**Rekomendasi:** {"Preseden yang ditemukan memberikan dasar hukum yang kuat" if high_strength > 0 else "Preseden yang tersedia memberikan dasar hukum yang terbatas, pertimbangkan penelitian lebih lanjut"}
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error formatting precedent report: {str(e)}")
            return f"Precedent analysis completed but formatting failed: {str(e)}"