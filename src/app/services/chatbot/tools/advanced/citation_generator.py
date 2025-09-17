"""
Citation Generator Tool for Sprint 2.
Generates proper legal citations in multiple formats.
"""

import logging
import json
import re
from typing import Any, Dict, Optional, List, Union, ClassVar
from datetime import datetime

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

try:
    from ...llm_service import LLMService, get_llm_service
except ImportError:
    # Fallback for different import paths
    from src.app.services.llm_service import LLMService, get_llm_service

logger = logging.getLogger(__name__)


class CitationGeneratorInput(BaseModel):
    """Input schema for citation generator tool."""
    case_reference: str = Field(
        description="Case information or document reference to generate citation for"
    )
    citation_format: str = Field(
        default="indonesian_legal",
        description="Citation format: 'indonesian_legal', 'apa', 'bluebook', 'oscola', or 'custom'"
    )
    include_page_numbers: bool = Field(
        default=False,
        description="Include page numbers in citation if available"
    )
    include_pinpoint: bool = Field(
        default=False,
        description="Include pinpoint citations for specific paragraphs or sections"
    )


class CitationGeneratorTool(BaseTool):
    """
    Tool for generating legal citations in multiple formats.
    
    This tool provides comprehensive citation generation including:
    - Multiple citation formats (Indonesian legal, APA, Bluebook, OSCOLA)
    - Automatic metadata extraction from case information
    - Validation of citation elements
    - Parallel citation support
    - Custom format handling
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "citation_generator"
    description: str = """
    Generate properly formatted legal citations for court cases and legal documents.
    Use this tool when you need to create citations for legal references.
    
    Input should include case information and desired citation format.
    """
    args_schema: type = CitationGeneratorInput
    llm_service: Optional[Any] = None
    
    # Citation format templates
    CITATION_FORMATS: ClassVar[Dict[str, Dict[str, str]]] = {
        "indonesian_legal": {
            "court_decision": "{title}, Putusan {court} No. {case_number} tanggal {date}{page_ref}",
            "regulation": "{title}, {type} No. {number} Tahun {year}{page_ref}",
            "book": "{author}, {title}, {publisher}, {place}, {year}{page_ref}",
            "journal": "{author}, \"{article_title}\", {journal_name}, Vol. {volume}, No. {issue}, {year}{page_ref}"
        },
        "apa": {
            "court_decision": "{court}. ({year}). {title} [Case No. {case_number}]. {court_location}",
            "regulation": "{issuing_body}. ({year}). {title} [{type} No. {number}]. {location}",
            "book": "{author}. ({year}). {title}. {publisher}",
            "journal": "{author}. ({year}). {article_title}. {journal_name}, {volume}({issue}), {pages}"
        },
        "bluebook": {
            "court_decision": "{title}, {case_number} ({court} {date})",
            "regulation": "{title}, {type} No. {number} ({year})",
            "book": "{author}, {title} ({year})",
            "journal": "{author}, {article_title}, {volume} {journal_name} {pages} ({year})"
        },
        "oscola": {
            "court_decision": "{title} [{year}] {court} {case_number}",
            "regulation": "{title} [{year}] {type} {number}",
            "book": "{author}, {title} ({publisher} {year})",
            "journal": "{author}, '{article_title}' ({year}) {volume} {journal_name} {pages}"
        }
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_services()
    
    def _initialize_services(self) -> None:
        """Initialize LLM service."""
        try:
            self.llm_service = get_llm_service()
            logger.info("Citation generator tool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize citation generator service: {str(e)}")
            self.llm_service = None
    
    def _run(self, case_reference: str, citation_format: str = "indonesian_legal",
             include_page_numbers: bool = False, include_pinpoint: bool = False) -> str:
        """
        Execute citation generation synchronously.
        
        Args:
            case_reference: Case information to cite
            citation_format: Desired citation format
            include_page_numbers: Include page numbers
            include_pinpoint: Include pinpoint citations
            
        Returns:
            Generated citation as string
        """
        if not self.llm_service:
            return "Error: Citation generator service is not available. Please try again later."
        
        try:
            # Run async method in sync context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._arun(case_reference, citation_format, include_page_numbers, include_pinpoint)
                )
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Citation generation failed: {str(e)}")
            return f"Error generating citation: {str(e)}"
    
    async def _arun(self, case_reference: str, citation_format: str = "indonesian_legal",
                    include_page_numbers: bool = False, include_pinpoint: bool = False) -> str:
        """
        Execute citation generation asynchronously.
        
        Args:
            case_reference: Case information to cite
            citation_format: Desired citation format
            include_page_numbers: Include page numbers
            include_pinpoint: Include pinpoint citations
            
        Returns:
            Generated citation as string
        """
        if not self.llm_service:
            return "Error: Citation generator service is not available. Please try again later."
        
        try:
            # Validate inputs
            if not case_reference.strip():
                return "Error: Case reference cannot be empty."
            
            valid_formats = list(self.CITATION_FORMATS.keys()) + ["custom"]
            if citation_format not in valid_formats:
                return f"Error: Invalid citation format. Valid formats: {', '.join(valid_formats)}"
            
            # Extract metadata from case reference
            metadata = await self._extract_citation_metadata(case_reference)
            
            if not metadata:
                return "Error: Could not extract citation metadata from the provided case reference."
            
            # Determine document type
            doc_type = self._determine_document_type(metadata)
            
            # Generate citation based on format
            if citation_format == "custom":
                citation = await self._generate_custom_citation(metadata, case_reference)
            else:
                citation = self._generate_standard_citation(
                    metadata, citation_format, doc_type, include_page_numbers, include_pinpoint
                )
            
            # Validate and clean citation
            cleaned_citation = self._clean_citation(citation)
            
            # Add additional information if requested
            result = self._format_citation_output(
                cleaned_citation, metadata, citation_format, include_page_numbers, include_pinpoint
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Async citation generation failed: {str(e)}")
            return f"Error generating citation: {str(e)}"
    
    async def _extract_citation_metadata(self, case_reference: str) -> Optional[Dict[str, Any]]:
        """
        Extract citation metadata from case reference using LLM.
        
        Args:
            case_reference: Raw case reference text
            
        Returns:
            Extracted metadata dictionary
        """
        try:
            extraction_prompt = f"""
            Ekstrak metadata sitasi dari referensi kasus/dokumen hukum berikut:
            
            **Referensi:** {case_reference}
            
            Berikan metadata dalam format JSON dengan struktur berikut:
            {{
                "title": "judul putusan/dokumen",
                "case_number": "nomor putusan/perkara",
                "court": "nama pengadilan",
                "court_level": "tingkat pengadilan (PN/PT/MA)",
                "date": "tanggal putusan",
                "year": "tahun putusan",
                "judges": ["nama hakim"],
                "parties": {{
                    "plaintiff": "nama penggugat",
                    "defendant": "nama tergugat"
                }},
                "legal_area": "bidang hukum",
                "document_type": "court_decision|regulation|book|journal|other",
                "author": "penulis (jika applicable)",
                "publisher": "penerbit (jika applicable)",
                "journal_name": "nama jurnal (jika applicable)",
                "volume": "volume (jika applicable)",
                "issue": "issue (jika applicable)",
                "pages": "halaman",
                "pinpoint": "referensi spesifik (paragraf/pasal)",
                "url": "URL (jika ada)",
                "accessed_date": "tanggal akses (jika applicable)"
            }}
            
            Jika informasi tidak tersedia, gunakan null atau string kosong.
            Pastikan format JSON valid.
            """
            
            result = await self.llm_service.llm.ainvoke(extraction_prompt)
            
            # Try to parse JSON response
            try:
                metadata = json.loads(result.content)
                
                # Clean and validate metadata
                metadata = self._validate_metadata(metadata)
                
                logger.info(f"Successfully extracted metadata for: {metadata.get('title', 'Unknown')}")
                return metadata
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                
                # Fallback: try to extract basic info with regex
                return self._extract_basic_metadata_regex(case_reference)
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
            return self._extract_basic_metadata_regex(case_reference)
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean extracted metadata.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Cleaned metadata dictionary
        """
        # Required fields with defaults
        required_fields = {
            "title": "Unknown Title",
            "document_type": "other",
            "year": datetime.now().year
        }
        
        for field, default in required_fields.items():
            if field not in metadata or not metadata[field]:
                metadata[field] = default
        
        # Clean string fields
        string_fields = ["title", "case_number", "court", "author", "publisher"]
        for field in string_fields:
            if field in metadata and metadata[field]:
                metadata[field] = str(metadata[field]).strip()
        
        # Validate year
        if "year" in metadata:
            try:
                year = int(metadata["year"])
                if not (1900 <= year <= datetime.now().year + 1):
                    metadata["year"] = datetime.now().year
            except (ValueError, TypeError):
                metadata["year"] = datetime.now().year
        
        # Validate document type
        valid_types = ["court_decision", "regulation", "book", "journal", "other"]
        if metadata.get("document_type") not in valid_types:
            metadata["document_type"] = "other"
        
        return metadata
    
    def _extract_basic_metadata_regex(self, case_reference: str) -> Dict[str, Any]:
        """
        Fallback metadata extraction using regex patterns.
        
        Args:
            case_reference: Raw case reference text
            
        Returns:
            Basic metadata dictionary
        """
        metadata = {
            "title": "",
            "case_number": "",
            "court": "",
            "date": "",
            "year": datetime.now().year,
            "document_type": "other"
        }
        
        try:
            # Extract case number patterns
            case_number_patterns = [
                r'No\\.?\\s*(\\d+/[^\\s]+/\\d{4})',
                r'Nomor\\s*(\\d+/[^\\s]+/\\d{4})',
                r'Perkara\\s*No\\.?\\s*(\\d+/[^\\s]+/\\d{4})'
            ]
            
            for pattern in case_number_patterns:
                match = re.search(pattern, case_reference, re.IGNORECASE)
                if match:
                    metadata["case_number"] = match.group(1)
                    break
            
            # Extract year from case number or text
            year_patterns = [r'\\b(19\\d{2}|20\\d{2})\\b']
            for pattern in year_patterns:
                matches = re.findall(pattern, case_reference)
                if matches:
                    metadata["year"] = int(matches[-1])  # Use the last year found
                    break
            
            # Extract court information
            court_patterns = [
                r'(Mahkamah Agung|MA)',
                r'(Pengadilan Tinggi|PT)',
                r'(Pengadilan Negeri|PN)',
                r'(Pengadilan\\s+\\w+)'
            ]
            
            for pattern in court_patterns:
                match = re.search(pattern, case_reference, re.IGNORECASE)
                if match:
                    metadata["court"] = match.group(1)
                    break
            
            # Determine document type
            if any(keyword in case_reference.lower() for keyword in ['putusan', 'pengadilan', 'mahkamah']):
                metadata["document_type"] = "court_decision"
            elif any(keyword in case_reference.lower() for keyword in ['undang-undang', 'peraturan', 'pp', 'uu']):
                metadata["document_type"] = "regulation"
            
            # Use first part as title if no specific title found
            if not metadata["title"]:
                # Take first 100 characters as title
                metadata["title"] = case_reference[:100].strip()
                if len(case_reference) > 100:
                    metadata["title"] += "..."
            
            logger.info("Used regex fallback for metadata extraction")
            return metadata
            
        except Exception as e:
            logger.error(f"Regex metadata extraction failed: {str(e)}")
            
            # Absolute fallback
            return {
                "title": case_reference[:50] + "..." if len(case_reference) > 50 else case_reference,
                "document_type": "other",
                "year": datetime.now().year,
                "case_number": "",
                "court": "",
                "date": ""
            }
    
    def _determine_document_type(self, metadata: Dict[str, Any]) -> str:
        """
        Determine document type from metadata.
        
        Args:
            metadata: Extracted metadata
            
        Returns:
            Document type string
        """
        # If already determined, use it
        if metadata.get("document_type") and metadata["document_type"] != "other":
            return metadata["document_type"]
        
        # Check for specific indicators
        title = metadata.get("title", "").lower()
        case_number = metadata.get("case_number", "").lower()
        court = metadata.get("court", "").lower()
        
        if any(keyword in title for keyword in ["putusan", "pengadilan", "mahkamah"]) or court:
            return "court_decision"
        elif any(keyword in title for keyword in ["undang-undang", "peraturan", "pp", "uu"]):
            return "regulation"
        elif metadata.get("journal_name") or metadata.get("volume"):
            return "journal"
        elif metadata.get("publisher") or metadata.get("author"):
            return "book"
        else:
            return "court_decision"  # Default for legal documents
    
    def _generate_standard_citation(self, metadata: Dict[str, Any], citation_format: str, 
                                    doc_type: str, include_page_numbers: bool, 
                                    include_pinpoint: bool) -> str:
        """
        Generate citation using standard format templates.
        
        Args:
            metadata: Citation metadata
            citation_format: Citation format name
            doc_type: Document type
            include_page_numbers: Include page numbers
            include_pinpoint: Include pinpoint citations
            
        Returns:
            Formatted citation string
        """
        try:
            # Get format template
            format_templates = self.CITATION_FORMATS.get(citation_format, {})
            template = format_templates.get(doc_type, format_templates.get("court_decision", ""))
            
            if not template:
                return f"No template available for {doc_type} in {citation_format} format"
            
            # Prepare citation data
            citation_data = self._prepare_citation_data(metadata, include_page_numbers, include_pinpoint)
            
            # Format citation
            try:
                citation = template.format(**citation_data)
            except KeyError as e:
                # Handle missing fields gracefully
                logger.warning(f"Missing field in citation template: {str(e)}")
                citation = self._format_with_fallback(template, citation_data)
            
            return citation
            
        except Exception as e:
            logger.error(f"Standard citation generation failed: {str(e)}")
            return f"Error generating {citation_format} citation: {str(e)}"
    
    def _prepare_citation_data(self, metadata: Dict[str, Any], include_page_numbers: bool, 
                               include_pinpoint: bool) -> Dict[str, str]:
        """
        Prepare citation data for template formatting.
        
        Args:
            metadata: Raw metadata
            include_page_numbers: Include page numbers
            include_pinpoint: Include pinpoint citations
            
        Returns:
            Prepared citation data dictionary
        """
        citation_data = {}
        
        # Basic fields
        citation_data["title"] = metadata.get("title", "Unknown Title")
        citation_data["case_number"] = metadata.get("case_number", "")
        citation_data["court"] = metadata.get("court", "")
        citation_data["date"] = metadata.get("date", "")
        citation_data["year"] = str(metadata.get("year", ""))
        citation_data["author"] = metadata.get("author", "")
        citation_data["publisher"] = metadata.get("publisher", "")
        citation_data["journal_name"] = metadata.get("journal_name", "")
        citation_data["volume"] = metadata.get("volume", "")
        citation_data["issue"] = metadata.get("issue", "")
        
        # Handle page references
        page_ref = ""
        if include_page_numbers and metadata.get("pages"):
            page_ref = f", hlm. {metadata['pages']}"
        
        if include_pinpoint and metadata.get("pinpoint"):
            pinpoint_ref = f", {metadata['pinpoint']}"
            page_ref += pinpoint_ref
        
        citation_data["page_ref"] = page_ref
        citation_data["pages"] = metadata.get("pages", "")
        
        # Additional fields
        citation_data["type"] = metadata.get("type", "")
        citation_data["number"] = metadata.get("number", "")
        citation_data["place"] = metadata.get("place", "")
        citation_data["court_location"] = metadata.get("court_location", "")
        citation_data["issuing_body"] = metadata.get("issuing_body", "")
        citation_data["location"] = metadata.get("location", "")
        citation_data["article_title"] = metadata.get("article_title", "")
        
        # Clean empty fields
        for key, value in citation_data.items():
            if not value:
                citation_data[key] = ""
        
        return citation_data
    
    def _format_with_fallback(self, template: str, citation_data: Dict[str, str]) -> str:
        """
        Format template with fallback for missing fields.
        
        Args:
            template: Citation template
            citation_data: Citation data
            
        Returns:
            Formatted citation with fallbacks
        """
        # Replace missing fields with empty strings
        import string
        
        class SafeFormatter(string.Formatter):
            def get_value(self, key, args, kwargs):
                if isinstance(key, str):
                    try:
                        return kwargs[key]
                    except KeyError:
                        return ""
                else:
                    return super().get_value(key, args, kwargs)
        
        formatter = SafeFormatter()
        return formatter.format(template, **citation_data)
    
    async def _generate_custom_citation(self, metadata: Dict[str, Any], case_reference: str) -> str:
        """
        Generate custom citation using LLM.
        
        Args:
            metadata: Citation metadata
            case_reference: Original case reference
            
        Returns:
            Custom formatted citation
        """
        try:
            custom_prompt = f"""
            Buat sitasi yang tepat dan profesional untuk dokumen hukum berikut:
            
            **Referensi Asli:** {case_reference}
            
            **Metadata yang Diekstrak:**
            {json.dumps(metadata, indent=2, ensure_ascii=False)}
            
            Buat sitasi yang:
            1. Mengikuti standar sitasi hukum Indonesia
            2. Mencakup semua informasi penting yang tersedia
            3. Menggunakan format yang konsisten dan profesional
            4. Sesuai dengan jenis dokumen (putusan pengadilan, peraturan, dll.)
            
            Berikan hanya sitasi final tanpa penjelasan tambahan.
            """
            
            result = await self.llm_service.llm.ainvoke(custom_prompt)
            citation = result.content.strip()
            
            # Basic validation
            if len(citation) < 10:
                return f"Custom citation generation failed. Fallback: {metadata.get('title', case_reference[:100])}"
            
            return citation
            
        except Exception as e:
            logger.error(f"Custom citation generation failed: {str(e)}")
            return f"Custom citation error. Basic info: {metadata.get('title', case_reference[:100])}"
    
    def _clean_citation(self, citation: str) -> str:
        """
        Clean and validate citation format.
        
        Args:
            citation: Raw citation string
            
        Returns:
            Cleaned citation string
        """
        # Remove extra whitespace
        citation = re.sub(r'\\s+', ' ', citation.strip())
        
        # Remove empty parentheses and brackets
        citation = re.sub(r'\\([\\s]*\\)', '', citation)
        citation = re.sub(r'\\[[\\s]*\\]', '', citation)
        
        # Remove trailing commas and periods before punctuation
        citation = re.sub(r',\\s*([,.;])', r'\\1', citation)
        
        # Fix spacing around punctuation
        citation = re.sub(r'\\s*,\\s*', ', ', citation)
        citation = re.sub(r'\\s*\\.\\s*', '. ', citation)
        
        # Remove multiple consecutive commas or periods
        citation = re.sub(r'[,]{2,}', ',', citation)
        citation = re.sub(r'[.]{2,}', '.', citation)
        
        return citation.strip()
    
    def _format_citation_output(self, citation: str, metadata: Dict[str, Any], 
                                citation_format: str, include_page_numbers: bool, 
                                include_pinpoint: bool) -> str:
        """
        Format final citation output with additional information.
        
        Args:
            citation: Generated citation
            metadata: Citation metadata
            citation_format: Used citation format
            include_page_numbers: Whether page numbers were included
            include_pinpoint: Whether pinpoint citations were included
            
        Returns:
            Formatted output string
        """
        try:
            output = f"""## Sitasi yang Dihasilkan

### Format: {citation_format.replace('_', ' ').title()}

**Sitasi:**
{citation}

### Detail Metadata:
"""
            
            # Add key metadata
            if metadata.get("title"):
                output += f"- **Judul:** {metadata['title']}\\n"
            if metadata.get("case_number"):
                output += f"- **Nomor Perkara:** {metadata['case_number']}\\n"
            if metadata.get("court"):
                output += f"- **Pengadilan:** {metadata['court']}\\n"
            if metadata.get("date"):
                output += f"- **Tanggal:** {metadata['date']}\\n"
            if metadata.get("year"):
                output += f"- **Tahun:** {metadata['year']}\\n"
            if metadata.get("author"):
                output += f"- **Penulis:** {metadata['author']}\\n"
            if metadata.get("publisher"):
                output += f"- **Penerbit:** {metadata['publisher']}\\n"
            
            # Add options used
            output += f"""
### Opsi yang Digunakan:
- **Format Sitasi:** {citation_format}
- **Termasuk Nomor Halaman:** {'Ya' if include_page_numbers else 'Tidak'}
- **Termasuk Pinpoint:** {'Ya' if include_pinpoint else 'Tidak'}
- **Jenis Dokumen:** {metadata.get('document_type', 'Tidak diketahui').replace('_', ' ').title()}
"""
            
            # Add alternative formats suggestion
            other_formats = [f for f in self.CITATION_FORMATS.keys() if f != citation_format]
            if other_formats:
                output += f"""
### Format Alternatif Tersedia:
{', '.join(other_formats)}

*Gunakan tool ini lagi dengan parameter citation_format yang berbeda untuk mendapatkan format lain.*
"""
            
            return output
            
        except Exception as e:
            logger.error(f"Error formatting citation output: {str(e)}")
            return f"**Generated Citation:** {citation}\\n\\n*Error formatting additional details: {str(e)}*"