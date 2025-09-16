"""
LangChain Expression Language (LCEL) implementation for Supreme Court RAG system.
This module implements the core RAG pipeline using LCEL for optimal composability
and monitoring as specified in the PRD requirements.
"""

import json
import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

from src.app.core.database import get_vector_store

# Pydantic models for structured outputs
from src.app.schemas.search import SearchResult as SchemaSearchResult
from src.app.schemas.search import ValidationStatus
from src.app.services.llm_service import get_llm_service
from src.app.services.retrieval import RetrievalService, RetrievalStrategy, get_retrieval_service

logger = logging.getLogger(__name__)

def clean_json_response(response: str) -> str:
    """
    Comprehensive JSON cleaning function to handle LLM markdown responses.
    Removes markdown code blocks and extracts pure JSON content.
    """
    if not isinstance(response, str):
        return str(response)

    response = response.strip()

    # Remove markdown code blocks
    if response.startswith('```json'):
        response = response[7:]
    elif response.startswith('```'):
        response = response[3:]

    if response.endswith('```'):
        response = response[:-3]

    response = response.strip()

    # Try to find JSON object if embedded in text
    if '{' in response and '}' in response:
        start = response.find('{')

        # Find matching closing brace
        brace_count = 0
        end = start
        for i, char in enumerate(response[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        response = response[start:end]

    return response

# Use the schema SearchResult instead of defining our own
SearchResult = SchemaSearchResult

class CourtRAGChains:
    """
    LangChain-first implementation of the Supreme Court RAG system.
    Uses LCEL for complete pipeline orchestration as per PRD specifications.
    """

    def __init__(
        self,
        vector_store: PGVector,
        llm: ChatGoogleGenerativeAI,
        embeddings: GoogleGenerativeAIEmbeddings,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.PARENT_CHILD,
        retrieval_service: RetrievalService | None = None
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.embeddings = embeddings
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_service = retrieval_service

        # Core LCEL chains
        self.search_chain = self._build_search_chain()

    def _build_search_chain(self) -> RunnableSequence:
        """
        Build the core search and generation chain using LCEL.
        Implements the RAG flow: Query -> Retrieve -> Generate -> Format
        """

        # Indonesian legal system prompt template as per existing workdir logic
        system_prompt = ChatPromptTemplate.from_template("""
        Anda adalah mesin pencari cerdas berbasis AI yang khusus dirancang untuk tugas pencarian dan analisis hukum di Indonesia, dengan fokus pada putusan Mahkamah Agung dan dokumen hukum terkait.

        INSTRUKSI PENTING UNTUK MESIN PENCARI HUKUM:
        - **Fokus pada Relevansi**: Prioritaskan hasil yang paling relevan dengan pertanyaan pengguna berdasarkan konteks dokumen yang disediakan.
        - **Akurasi Hukum**: Berikan informasi yang akurat, faktual, dan didasarkan sepenuhnya pada konteks. Jangan tambahkan asumsi, interpretasi pribadi, atau informasi di luar dokumen sumber.
        - **Ekstraksi Poin Kunci**: Identifikasi poin-poin hukum utama, preseden, dan argumen hukum dari dokumen, termasuk referensi ke pasal undang-undang atau yurisprudensi terkait.
        - **Sitasi Lengkap**: Sertakan sitasi yang lengkap dan dapat diverifikasi, termasuk nomor putusan, bagian dokumen, link_pdf, source, dan halaman jika tersedia. Gunakan format standar hukum Indonesia.
        - **Bahasa Formal**: Gunakan bahasa hukum formal namun jelas dan mudah dipahami, hindari jargon berlebihan.
        - **Penanganan Ketidakcukupan**: Jika konteks tidak cukup untuk menjawab sepenuhnya, nyatakan dengan jelas dan sarankan pencarian tambahan. Jangan berhalusinasi atau mengisi kekosongan dengan informasi eksternal.
        - **Optimasi untuk Pencarian**: Strukturkan respons untuk memfasilitasi navigasi cepat, seperti highlighting bagian relevan dan saran query terkait.

        KONTEKS DOKUMEN YANG TERSEDIA:
        {context}

        PERTANYAAN PENGGUNA:
        {question}

        TUGAS ANDA:
        Analisis konteks di atas untuk memberikan hasil pencarian yang optimal. Jika tidak ada dokumen relevan, kembalikan respons kosong yang sesuai.

        Berikan respons DALAM FORMAT JSON PRESISI dengan struktur berikut (pastikan semua field terisi berdasarkan konteks):
        {{
            "summary": "Ringkasan singkat dan faktual dari hasil pencarian, maksimal 200 kata, fokus pada jawaban langsung untuk query",
            "key_points": ["Poin hukum kunci 1 dengan sitasi", "Poin hukum kunci 2 dengan sitasi", ...],
            "source_documents": [
            {{
                "title": "Judul dokumen lengkap",
                "case_number": "Nomor perkara resmi (contoh: 123/PID/2023)",
                "excerpt": "Kutipan relevan langsung dari dokumen, maksimal 300 kata, dengan highlighting bagian penting",
                "source": "Sumber dokumen dari metadata retrieved",
                "link_pdf": "URL atau path ke dokumen sumber dari metadata retrieved",
                "validation_status": "Supported/Partially Supported/Unsupported/Uncertain",
                "relevance_score": 0.95,
                "legal_areas": ["Area hukum 1", "Area hukum 2"]
            }}
            ],
            "confidence_score": 0.0-1.0 (skor kepercayaan keseluruhan berdasarkan relevansi dan validasi),
            "legal_areas": ["Area hukum utama 1", "Area hukum utama 2"],
            "metadata": {{
                "total_documents_found": int,
                "search_timestamp": datetime,
                "search_strategy": "query strategy used (e.g., semantic similarity + legal context matching)",
                "disclaimer": "Hasil berdasarkan dokumen tersedia; konsultasikan ahli hukum untuk nasihat profesional"
            }}
        }}

        CATATAN TAMBAHAN:
        - Jika query tidak relevan dengan hukum Indonesia, kembalikan summary: "Query tidak terkait dengan domain hukum Indonesia."
        - Pastikan respons dapat di-parse sebagai JSON valid tanpa tambahan teks luar.
        - Optimalkan untuk efisiensi: Hindari redundansi dan fokus pada informasi actionable.
        """)

        # Context formatting function
        def format_context(docs: List[Document]) -> str:
            """Format retrieved documents for LLM context."""
            context_parts = []
            for i, doc in enumerate(docs):
                metadata = doc.metadata
                metadata.get('chunk_id', f'chunk_{i}')
                case_number = metadata.get('case_number', 'Unknown')
                # Use link from metadata if available
                link_pdf = metadata.get('link_pdf', 'Link tidak tersedia')
                source = metadata.get('source', 'Sumber tidak tersedia')

                context_parts.append(
                    f"[DOKUMEN {i+1}]\n"
                    f"Nomor Perkara: {case_number}\n"
                    f"Konten: {doc.page_content}\n"
                    f"Link PDF: {link_pdf}\n"
                    f"Sumber: {source}\n"
                    f"{'='*50}\n"
                )
            return '\n'.join(context_parts)

        # Custom JSON parsing function with error handling
        def safe_json_parse(response: str) -> Dict[str, Any]:
            """Safely parse JSON with comprehensive cleaning and fallbacks"""
            try:
                # Clean the response first
                cleaned = clean_json_response(response)

                # Parse JSON
                parsed = json.loads(cleaned)

                # Validate required fields exist
                required_fields = ['summary', 'key_points', 'source_documents', 'confidence_score', 'legal_areas']
                for field in required_fields:
                    if field not in parsed:
                        logger.warning(f"Missing required field: {field}")
                        if field == 'summary':
                            parsed[field] = "Ringkasan tidak tersedia"
                        elif field == 'key_points':
                            parsed[field] = []
                        elif field == 'source_documents':
                            parsed[field] = []
                        elif field == 'confidence_score':
                            parsed[field] = 0.5
                        elif field == 'legal_areas':
                            parsed[field] = []

                # Ensure optional fields have defaults
                if 'related_queries' not in parsed:
                    parsed['related_queries'] = []
                if 'metadata' not in parsed:
                    parsed['metadata'] = None

                return parsed

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
                logger.error(f"Failed response: {response[:500]}...")

                # Return fallback structure
                return {
                    "summary": f"JSON parsing failed: {str(e)}. Raw response: {response[:200]}...",
                    "key_points": ["Response could not be parsed as JSON"],
                    "source_documents": [],
                    "confidence_score": 0.0,
                    "legal_areas": [],
                    "related_queries": [],
                    "metadata": {
                        "error": str(e),
                        "raw_response_preview": response[:200]
                    }
                }
            except Exception as e:
                logger.error(f"Unexpected error in JSON parsing: {e}")
                return {
                    "summary": f"Unexpected parsing error: {str(e)}",
                    "key_points": ["Parsing error occurred"],
                    "source_documents": [],
                    "confidence_score": 0.0,
                    "legal_areas": [],
                    "related_queries": [],
                    "metadata": {"error": str(e)}
                }

        # Retrieval function that handles async properly
        def retrieve_documents(query_dict: Dict[str, str]) -> List[Document]:
            """Synchronous wrapper for async retrieval."""
            try:
                query = query_dict.get("query", "")

                # If we have a retrieval service, use it
                if self.retrieval_service:
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # We're in an async context, can't use run_until_complete
                            # Fall back to vector store retrieval
                            return self._fallback_retrieval(query)
                        else:
                            return loop.run_until_complete(
                                self.retrieval_service.retrieve(
                                    query=query,
                                    strategy=self.retrieval_strategy,
                                    top_k=5
                                )
                            )
                    except RuntimeError:
                        # Loop is already running, use fallback
                        return self._fallback_retrieval(query)
                else:
                    # Use fallback retrieval
                    return self._fallback_retrieval(query)

            except Exception as e:
                logger.error(f"Document retrieval failed: {e}")
                return self._fallback_retrieval(query_dict.get("query", ""))

        search_chain = (
            {
                "context": RunnableLambda(lambda x: {"query": x["query"]})
                | RunnableLambda(retrieve_documents)
                | RunnableLambda(format_context),
                "question": RunnableLambda(lambda x: x["query"])
            }
            | system_prompt
            | self.llm
            | RunnableLambda(lambda x: x.content if hasattr(x, 'content') else str(x))
            | RunnableLambda(safe_json_parse)
        )

        return search_chain

    def _fallback_retrieval(self, query: str) -> List[Document]:
        """Fallback retrieval using vector store directly."""
        try:
            # Build search kwargs
            search_kwargs = {"k": 5}

            # Apply filters if they exist
            if hasattr(self, '_current_filters') and self._current_filters:
                search_kwargs["filter"] = self._current_filters

            # Use vector store as fallback
            docs = self.retrieval_service.retrieve(query=query, strategy=RetrievalStrategy.VECTOR_SEARCH, top_k=5)
            # retr
            logger.info(f"Fallback retrieval returned {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Fallback retrieval failed: {e}")
            return []


    def invoke(self, query: str, filters: Dict[str, Any] | None = None, strategy: str | None = None) -> SearchResult:
        """
        Main entry point for the RAG system.

        Args:
            query: User's natural language search query
            filters: Optional filters for jurisdiction, date range, case type

        Returns:
            SearchResult with validated content and citations
        """
        try:
            # Apply filters to retriever if provided
            if filters:
                self._apply_filters(filters)

            # Invoke the complete pipeline - now returns parsed JSON dict
            parsed_response = self.search_chain.invoke({"query": query})

            if parsed_response is None:
                # Return a default SearchResult
                return SearchResult(
                    summary="No results found for the query.",
                    key_points=["No relevant documents found."],
                    source_documents=[],
                    validation_status=ValidationStatus.UNCERTAIN,
                    confidence_score=0.0,
                    legal_areas=[]
                )

            # Extract validation status from source documents if available
            validation_status = ValidationStatus.UNCERTAIN
            if parsed_response.get('source_documents'):
                # Get validation status from first document or use default logic
                first_doc = parsed_response['source_documents'][0]
                if isinstance(first_doc, dict) and 'validation_status' in first_doc:
                    status_str = first_doc['validation_status']
                    if status_str == 'Supported':
                        validation_status = ValidationStatus.SUPPORTED
                    elif status_str == 'Partially Supported':
                        validation_status = ValidationStatus.PARTIALLY_SUPPORTED
                    elif status_str == 'Unsupported':
                        validation_status = ValidationStatus.UNSUPPORTED

            # Create SearchResult from parsed data
            return SearchResult(
                summary=parsed_response.get('summary', 'No summary available'),
                key_points=parsed_response.get('key_points', []),
                source_documents=parsed_response.get('source_documents', []),
                validation_status=validation_status,
                confidence_score=parsed_response.get('confidence_score', 0.5),
                legal_areas=parsed_response.get('legal_areas', []),
                related_queries=parsed_response.get('related_queries', []),
                metadata=parsed_response.get('metadata')
            )
        except Exception as e:
            logger.error(f"Error in invoke method: {str(e)}")
            # Return a default SearchResult on error
            return SearchResult(
                summary=f"Error processing query: {str(e)}",
                key_points=["An error occurred during search."],
                source_documents=[],
                validation_status=ValidationStatus.UNCERTAIN,
                confidence_score=0.0,
                legal_areas=[],
                related_queries=[],
                metadata=None
            )

    async def ainvoke(self, query: str, filters: Dict[str, Any] | None = None) -> SearchResult:
        """
        Main entry point for the RAG system.

        Args:
            query: User's natural language search query
            filters: Optional filters for jurisdiction, date range, case type

        Returns:
            SearchResult with validated content and citations
        """
        try:
            # Apply filters to retriever if provided
            if filters:
                self._apply_filters(filters)

            # Run the synchronous invoke in a thread pool to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self._invoke_sync(query, filters))

            return result

        except Exception as e:
            # Return a default SearchResult on error
            return SearchResult(
                summary=f"Error processing query: {str(e)}",
                key_points=["An error occurred during search."],
                source_documents=[],
                validation_status=ValidationStatus.UNCERTAIN,
                confidence_score=0.0,
                legal_areas=[],
                related_queries=[],
                metadata=None
            )

    def _invoke_sync(self, query: str, filters: Dict[str, Any] | None = None) -> SearchResult:
        """Synchronous implementation of the search logic."""
        # Apply filters to retriever if provided
        if filters:
            self._apply_filters(filters)

        # Invoke the complete pipeline - now returns parsed JSON dict
        parsed_response = self.search_chain.invoke({"query": query})

        if parsed_response is None:
            # Return a default SearchResult
            return SearchResult(
                summary="No results found for the query.",
                key_points=["No relevant documents found."],
                source_documents=[],
                validation_status=ValidationStatus.UNCERTAIN,
                confidence_score=0.0,
                legal_areas=[],
                related_queries=[],
                metadata=None
            )

        # Extract validation status from source documents if available
        validation_status = ValidationStatus.UNCERTAIN
        if parsed_response.get('source_documents'):
            # Get validation status from first document or use default logic
            first_doc = parsed_response['source_documents'][0]
            if isinstance(first_doc, dict) and 'validation_status' in first_doc:
                status_str = first_doc['validation_status']
                if status_str == 'Supported':
                    validation_status = ValidationStatus.SUPPORTED
                elif status_str == 'Partially Supported':
                    validation_status = ValidationStatus.PARTIALLY_SUPPORTED
                elif status_str == 'Unsupported':
                    validation_status = ValidationStatus.UNSUPPORTED

        # Create SearchResult from parsed data
        return SearchResult(
            summary=parsed_response.get('summary', 'No summary available'),
            key_points=parsed_response.get('key_points', []),
            source_documents=parsed_response.get('source_documents', []),
            validation_status=validation_status,
            confidence_score=parsed_response.get('confidence_score', 0.5),
            legal_areas=parsed_response.get('legal_areas', []),
            related_queries=parsed_response.get('related_queries', []),
            metadata=parsed_response.get('metadata')
        )

    def _apply_filters(self, filters: Dict[str, Any]) -> None:
        """Apply search filters to the retriever."""
        # Extract supported filter types from PRD requirements
        jurisdiction_filter = filters.get('jurisdiction')
        date_range_filter = filters.get('date_range')
        case_type_filter = filters.get('case_type')

        # Build metadata filter dict for vector store
        metadata_filters = {}

        if jurisdiction_filter:
            # Map jurisdiction to metadata field
            metadata_filters['jurisdiction'] = jurisdiction_filter

        if date_range_filter:
            # Handle date range filtering
            if isinstance(date_range_filter, dict):
                if 'start_date' in date_range_filter:
                    metadata_filters['date_gte'] = date_range_filter['start_date']
                if 'end_date' in date_range_filter:
                    metadata_filters['date_lte'] = date_range_filter['end_date']

        if case_type_filter:
            # Filter by case type (e.g., civil, criminal, administrative)
            metadata_filters['case_type'] = case_type_filter

        # Apply filters to vector store retriever if it supports metadata filtering
        if hasattr(self.vector_store, 'as_retriever'):
            # Note: Filters will be applied during retrieval in _fallback_retrieval
            # Store filters for later use
            self._current_filters = metadata_filters
        else:
            # Log warning if filtering not supported
            logger.warning(f"Vector store doesn't support metadata filtering. Filters ignored: {filters}")

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store for indexing."""
        # Add documents directly to vector store
        self.vector_store.add_documents(documents)

    async def ainvoke(self, query: str, filters: Dict[str, Any] | None = None) -> SearchResult:
        """Async version of invoke for better performance."""
        # Apply filters to retriever if provided
        if filters:
            self._apply_filters(filters)

        # Run the synchronous invoke in a thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.invoke, query, filters)

        return result

    def retrieve_documents(self, query: str, top_k: int=5) -> List[Document]:
        """Retrieve documents from the RAG system."""
        try:
            return self.retrieval_service.retrieve(
                query=query,
                strategy=RetrievalStrategy.PARENT_CHILD,
                top_k=top_k
            )
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {str(e)}")
            return []


def create_rag_chains(
    database_url: str = None,  # Made optional since we use singleton
    collection_name: str = "putusan_child_chunks",
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.PARENT_CHILD,
) -> CourtRAGChains:
    """
    Factory function to create configured RAG chains with singleton database connections.

    Args:
        database_url: PostgreSQL connection string (optional, uses singleton if None)
        collection_name: Vector store collection name
        retrieval_strategy: Strategy for document retrieval

    Returns:
        Configured CourtRAGChains instance
    """
    llm_service = get_llm_service()
    retrieval_service = get_retrieval_service()

    # Use singleton database connection manager
    if collection_name:
        vector_store = get_vector_store(collection_name)
    else:
        vector_store = get_vector_store()

    llm = llm_service.llm
    embeddings = llm_service.embeddings

    return CourtRAGChains(
        vector_store=vector_store,
        llm=llm,
        embeddings=embeddings,
        retrieval_strategy=retrieval_strategy,
        retrieval_service=retrieval_service
    )
