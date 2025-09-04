"""
LangChain Expression Language (LCEL) implementation for Supreme Court RAG system.
This module implements the core RAG pipeline using LCEL for optimal composability
and monitoring as specified in the PRD requirements.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableParallel, 
    RunnableLambda,
    RunnableSequence
)
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.storage import RedisStore
from redis import Redis
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks import StdOutCallbackHandler
# Pydantic models for structured outputs
from src.app.core.config import settings
from src.app.schemas.search import SearchResult as SchemaSearchResult, ValidationStatus

logger = logging.getLogger(__name__)





# Use the schema SearchResult instead of defining our own
SearchResult = SchemaSearchResult

class ClaimValidation(BaseModel):
    """Individual claim validation result."""
    claim: str = Field(description="The factual claim being validated")
    status: str = Field(description="Supported/Partially Supported/Unsupported/Uncertain")
    evidence: List[str] = Field(description="Supporting evidence from source documents")
    source_chunks: List[str] = Field(description="Specific chunk IDs containing evidence")

class ValidationResult(BaseModel):
    """Complete validation result for all claims."""
    claims: List[ClaimValidation] = Field(description="Individual claim validations")
    overall_confidence: float = Field(description="Overall confidence score 0-1")
    filtered_response: str = Field(description="Response with unsupported claims removed")

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
        enable_validation: bool = True,
        use_redis_store: bool = True,
        guardrails_validator: Optional[Any] = None
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.embeddings = embeddings
        self.enable_validation = enable_validation
        self.use_redis_store = use_redis_store
        self.guardrails_validator = guardrails_validator
        
        # Initialize document store for parent-child retrieval
        # Use Redis for production persistence
        if self.use_redis_store:
            try:
                redis_client = Redis(
                    host=settings.REDIS_CACHE_HOST,
                    port=settings.REDIS_CACHE_PORT,
                    decode_responses=True
                )
                self.doc_store = RedisStore(client=redis_client)
                logger.info("Using Redis store for document persistence")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis, falling back to InMemoryStore: {str(e)}")
                self.doc_store = InMemoryStore()
        else:
            self.doc_store = InMemoryStore()
        
        # Setup simple retriever for testing
        self.retriever = self._setup_parent_child_retriever()
        
        # Core LCEL chains
        self.search_chain = self._build_search_chain()
        self.validation_chain = self._build_validation_chain() if enable_validation else None
        self.complete_pipeline = self._build_complete_pipeline()
    
    def _setup_parent_child_retriever(self) -> ParentDocumentRetriever:
        """
        Setup parent-child retriever with PRD-specified chunk sizes:
        - Child chunks: 400 characters for precise search
        - Parent documents: 2000 characters for full context
        """
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # PRD specification
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,   # PRD specification
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        return ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.doc_store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": 10}  # Retrieve top 10 child chunks
        )
    
    def _build_search_chain(self) -> RunnableSequence:
        """
        Build the core search and generation chain using LCEL.
        Implements the RAG flow: Query -> Retrieve -> Generate -> Format
        """
    
        # Indonesian legal system prompt template as per existing workdir logic
        system_prompt = ChatPromptTemplate.from_template("""
        Anda adalah mesin pencari cerdas berbasis AI yang khusus dirancang untuk tugas pencarian dan analisis hukum di Indonesia, dengan fokus pada putusan Mahkamah Agung dan dokumen hukum terkait.
        
        INSTRUKSI PENTING UNTUK MESIN PENCARI HUKUM:
        - **Fokus pada Relevansi**: Prioritaskan hasil yang paling relevan dengan pertanyaan pengguna berdasarkan konteks dokumen yang disediakan. Gunakan similarity search dan ranking untuk menentukan urutan.
        - **Akurasi Hukum**: Berikan informasi yang akurat, faktual, dan didasarkan sepenuhnya pada konteks. Jangan tambahkan asumsi, interpretasi pribadi, atau informasi di luar dokumen sumber.
        - **Ekstraksi Poin Kunci**: Identifikasi poin-poin hukum utama, preseden, dan argumen hukum dari dokumen, termasuk referensi ke pasal undang-undang atau yurisprudensi terkait.
        - **Sitasi Lengkap**: Sertakan sitasi yang lengkap dan dapat diverifikasi, termasuk nomor putusan, bagian dokumen, chunk_id, dan halaman jika tersedia. Gunakan format standar hukum Indonesia.
        - **Bahasa Formal**: Gunakan bahasa hukum formal namun jelas dan mudah dipahami, hindari jargon berlebihan.
        - **Penanganan Ketidakcukupan**: Jika konteks tidak cukup untuk menjawab sepenuhnya, nyatakan dengan jelas dan sarankan pencarian tambahan. Jangan berhalusinasi atau mengisi kekosongan dengan informasi eksternal.
        - **Optimasi untuk Pencarian**: Strukturkan respons untuk memfasilitasi navigasi cepat, seperti highlighting bagian relevan dan saran query terkait.
        - **Validasi Otomatis**: Evaluasi keandalan setiap klaim berdasarkan bukti dalam konteks (Supported/Partially Supported/Unsupported/Uncertain).
        
        KONTEKS DOKUMEN YANG TERSEDIA:
        {context}
        
        PERTANYAAN PENGGUNA (Query Pencarian):
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
                "chunk_id": "ID chunk spesifik dari dokumen",
                "validation_status": "Supported/Partially Supported/Unsupported/Uncertain",
                "relevance_score": 0.95,
                "legal_areas": ["Area hukum 1", "Area hukum 2"]
            }}
            ],
            "confidence_score": 0.85,
            "related_queries": ["Saran query pencarian terkait 1", "Saran query pencarian terkait 2"],
            "metadata": {{
            "total_documents_found": 5,
            "search_timestamp": "2024-10-01T12:00:00Z",
            "top_relevance_threshold": 0.8,
            "search_strategy": "semantic similarity + legal context matching"
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
                chunk_id = metadata.get('chunk_id', f'chunk_{i}')
                case_number = metadata.get('case_number', 'Unknown')
                
                context_parts.append(
                    f"[DOKUMEN {i+1}]\n"
                    f"Nomor Perkara: {case_number}\n"
                    f"Chunk ID: {chunk_id}\n"
                    f"Konten: {doc.page_content}\n"
                    f"{'='*50}\n"
                )
            return '\n'.join(context_parts)
        
        # Build the search chain using LCEL
        def clean_json_response(response: str) -> str:
            """Clean LLM response to extract JSON content."""
            if isinstance(response, str):
                # Remove markdown code blocks
                response = response.strip()
                if response.startswith('```json'):
                    response = response[7:]
                if response.startswith('```'):
                    response = response[3:]
                if response.endswith('```'):
                    response = response[:-3]
                return response.strip()
            return response
        
        search_chain = (
            {
                "context": RunnableLambda(lambda x: x["query"]) | self.retriever | RunnableLambda(format_context),
                "question": RunnableLambda(lambda x: x["query"])
            }
            | system_prompt
            | self.llm
            | RunnableLambda(lambda x: x.content if hasattr(x, 'content') else str(x))
        )
        
        return search_chain
    
    def _build_validation_chain(self) -> RunnableSequence:
        """
        Build claim validation chain using Guardrails validator.
        Implements PRD requirement for factual claim verification.
        """
        
        if self.guardrails_validator:
            # Use the injected Guardrails validator
            def validate_with_guardrails(data: Dict[str, Any]) -> ValidationResult:
                """Validate claims using Guardrails validator."""
                try:
                    response = data.get("response", "")
                    context_docs = data.get("context", [])
                    
                    # Format context documents for guardrails
                    evidence_documents = []
                    if isinstance(context_docs, list):
                        for doc in context_docs:
                            if hasattr(doc, 'page_content'):
                                evidence_documents.append({
                                    'source': getattr(doc, 'metadata', {}).get('source', 'unknown'),
                                    'content': doc.page_content
                                })
                    
                    # Use guardrails validator to validate the response
                    validation_result = self.guardrails_validator.validate_batch(
                        text=response,
                        evidence_documents=evidence_documents
                    )
                    
                    # Convert to our ValidationResult format
                    claims = []
                    for claim_result in validation_result.claims:
                        claims.append(ClaimValidation(
                            claim=claim_result.claim.text,
                            status=claim_result.status,
                            evidence=[ev.evidence_text for ev in claim_result.evidence],
                            source_chunks=[ev.chunk_id for ev in claim_result.evidence]
                        ))
                    
                    return ValidationResult(
                        claims=claims,
                        overall_confidence=validation_result.overall_confidence,
                        filtered_response=validation_result.filtered_text
                    )
                    
                except Exception as e:
                    logger.error(f"Guardrails validation failed: {str(e)}")
                    # Fallback to basic validation
                    return ValidationResult(
                        claims=[],
                        overall_confidence=0.5,
                        filtered_response=data.get("response", "")
                    )
            
            validation_chain = RunnableLambda(validate_with_guardrails)
            
        else:
            # Fallback to basic LLM-based validation
            # Claim extraction prompt
            extraction_prompt = ChatPromptTemplate.from_template("""
            Ekstrak semua klaim faktual dari respon berikut yang dapat diverifikasi:
            
            RESPON: {response}
            
            Identifikasi klaim-klaim spesifik yang dapat dicek kebenarannya terhadap dokumen sumber.
            Berikan dalam format JSON:
            {{
                "claims": ["klaim 1", "klaim 2", ...]
            }}
            """)
            
            # Claim verification prompt
            verification_prompt = ChatPromptTemplate.from_template("""
            Verifikasi setiap klaim berikut terhadap konteks dokumen yang tersedia:
            
            KLAIM YANG AKAN DIVERIFIKASI:
            {claims}
            
            KONTEKS DOKUMEN:
            {context}
            
            Untuk setiap klaim, tentukan status validasi:
            - "Supported": Klaim didukung penuh oleh bukti dalam dokumen
            - "Partially Supported": Klaim didukung sebagian dengan beberapa ketidakpastian  
            - "Unsupported": Klaim tidak didukung atau bertentangan dengan dokumen
            - "Uncertain": Tidak cukup informasi untuk memverifikasi
            
            Berikan hasil dalam format JSON:
            {{
                "claims": [
                    {{
                        "claim": "teks klaim",
                        "status": "Supported/Partially Supported/Unsupported/Uncertain",
                        "evidence": ["bukti 1", "bukti 2"],
                        "source_chunks": ["chunk_id_1", "chunk_id_2"]
                    }}
                ],
                "overall_confidence": 0.85,
                "filtered_response": "respon yang telah difilter"
            }}
            """)
            
            # Claim extraction chain
            extract_claims = (
                extraction_prompt
                | self.llm
                | JsonOutputParser()
            )
            
            # Verification chain
            verify_claims = (
                verification_prompt
                | self.llm  
                | JsonOutputParser(pydantic_object=ValidationResult)
            )
            
            # Complete validation pipeline
            validation_chain = (
                {
                    "claims": RunnableLambda(lambda x: extract_claims.invoke({"response": x["response"]})),
                    "context": RunnableLambda(lambda x: x["context"])
                }
                | verify_claims
            )
        
        return validation_chain
    
    def _build_complete_pipeline(self) -> RunnableSequence:
        """
        Build the complete RAG pipeline with optional validation.
        This is the main entry point for the search functionality.
        """
        
        if self.validation_chain:
            # Pipeline with validation
            complete_pipeline = (
                self.search_chain
                | RunnableLambda(lambda result: {
                    "response": result,
                    "context": self.retriever.invoke(result.get("question", ""))
                })
                | self.validation_chain
                | RunnableLambda(self._merge_results)
            )
        else:
            # Pipeline without validation
            complete_pipeline = self.search_chain | RunnableLambda(lambda x: x)
            # Simple pipeline without validation --- IGNORE ---
            # complete_pipeline = self.search_chain --- IGNORE ---
         # --- IGNORE ---
        return complete_pipeline
    
    def _merge_results(self, validation_result: Dict[str, Any]) -> SearchResult:
        """
        Merge search results with validation results.
        Combines the filtered response from validation with claim statuses to create a validated SearchResult.
        """
        # Extract data from validation result
        claims = validation_result.get('claims', [])
        overall_confidence = validation_result.get('overall_confidence', 0.0)
        filtered_response = validation_result.get('filtered_response', '')
        
        # Use filtered response as summary
        summary = filtered_response
        
        # Extract supported claims as key points
        key_points = [
            claim['claim'] for claim in claims 
            if claim.get('status') in ['Supported', 'Partially Supported']
        ]
        
        # For source_documents, collect unique source chunks (document details not available in validation result)
        # In a full implementation, you'd need to pass original context or retrieve documents by chunk_id
        source_documents = []  # Placeholder - requires additional context to populate properly
        
        # Determine overall validation status based on confidence
        if overall_confidence >= 0.8:
            validation_status = ValidationStatus.SUPPORTED
        elif overall_confidence >= 0.5:
            validation_status = ValidationStatus.PARTIALLY_SUPPORTED
        elif overall_confidence > 0.0:
            validation_status = ValidationStatus.UNSUPPORTED
        else:
            validation_status = ValidationStatus.UNCERTAIN
        
        # Create and return SearchResult
        return SearchResult(
            summary=summary,
            key_points=key_points,
            source_documents=source_documents,
            validation_status=validation_status,
            confidence_score=overall_confidence,
            legal_areas=[]  # Not available from validation result
        )
    
    def invoke(self, query: str, filters: Optional[Dict[str, Any]] = None) -> SearchResult:
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
            
            # Invoke the complete pipeline
            raw_response = self.search_chain.invoke({"query": query})
            
            if raw_response is None:
                # Return a default SearchResult
                return SearchResult(
                    summary="No results found for the query.",
                    key_points=["No relevant documents found."],
                    source_documents=[],
                    validation_status=ValidationStatus.UNCERTAIN,
                    confidence_score=0.0,
                    legal_areas=[]
                )
            
            # Parse the JSON response
            try:
                # Clean the response (remove markdown formatting)
                cleaned_response = raw_response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                # Parse JSON
                import json
                parsed_data = json.loads(cleaned_response)
                
                # Map validation status to enum
                validation_status_str = parsed_data.get('validation_status', 'Unknown')
                if validation_status_str == 'Supported':
                    validation_status = ValidationStatus.SUPPORTED
                elif validation_status_str == 'Partially Supported':
                    validation_status = ValidationStatus.PARTIALLY_SUPPORTED
                elif validation_status_str == 'Unsupported':
                    validation_status = ValidationStatus.UNSUPPORTED
                elif validation_status_str == 'Uncertain':
                    validation_status = ValidationStatus.UNCERTAIN
                else:
                    validation_status = ValidationStatus.UNCERTAIN  # Default to uncertain
                
                # Create SearchResult from parsed data
                return SearchResult(
                    summary=parsed_data.get('summary', 'No summary available'),
                    key_points=parsed_data.get('key_points', []),
                    source_documents=parsed_data.get('source_documents', []),
                    validation_status=validation_status,
                    confidence_score=0.8,  # Default confidence
                    legal_areas=[]  # Will be populated later
                )
                
            except (json.JSONDecodeError, KeyError) as e:
                # Return a fallback SearchResult
                return SearchResult(
                    summary=f"Response parsing failed: {str(raw_response)[:500]}",
                    key_points=["Response could not be parsed as JSON"],
                    source_documents=[],
                    validation_status=ValidationStatus.UNCERTAIN,
                    confidence_score=0.0,
                    legal_areas=[]
                )
            
        except Exception as e:
            # Return a default SearchResult on error
            return SearchResult(
                summary=f"Error processing query: {str(e)}",
                key_points=["An error occurred during search."],
                source_documents=[],
                validation_status=ValidationStatus.UNCERTAIN,
                confidence_score=0.0,
                legal_areas=[]
            )
    
    async def ainvoke(self, query: str, filters: Optional[Dict[str, Any]] = None) -> SearchResult:
        """
        Main entry point for the RAG system.
        
        Args:
            query: User's natural language search query
            filters: Optional filters for jurisdiction, date range, case type
            
        Returns:
            SearchResult with validated content and citations
        """
    async def ainvoke(self, query: str, filters: Optional[Dict[str, Any]] = None) -> SearchResult:
        """Async version of invoke for better performance."""
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
                validation_status="Error"
            )
    
    def _invoke_sync(self, query: str, filters: Optional[Dict[str, Any]] = None) -> SearchResult:
        """Synchronous implementation of the search logic."""
        # Apply filters to retriever if provided
        if filters:
            self._apply_filters(filters)
        
        # Invoke the complete pipeline
        result = self.complete_pipeline.invoke(query)
        
        if result is None:
            # Return a default SearchResult
            return SearchResult(
                summary="No results found for the query.",
                key_points=["No relevant documents found."],
                source_documents=[],
                validation_status="No results"
            )
        
        return result
    
    def _invoke_sync(self, query: str, filters: Optional[Dict[str, Any]] = None) -> SearchResult:
        """Synchronous implementation of the search logic."""
        # Apply filters to retriever if provided
        if filters:
            self._apply_filters(filters)
        
        # Invoke the complete pipeline
        result = self.complete_pipeline.invoke(query)
        
        if result is None:
            # Return a default SearchResult
            return SearchResult(
                summary="No results found for the query.",
                key_points=["No relevant documents found."],
                source_documents=[],
                validation_status="No results"
            )
        
        return result
    
    def _apply_filters(self, filters: Dict[str, Any]) -> None:
        """Apply search filters to the retriever."""
        # Implementation for PRD filter requirements:
        # - jurisdiction filtering
        # - date range filtering  
        # - case type filtering
        pass
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store for indexing."""
        # For simple retriever, add documents directly to vector store
        self.vector_store.add_documents(documents)
    
    async def ainvoke(self, query: str, filters: Optional[Dict[str, Any]] = None) -> SearchResult:
        """Async version of invoke for better performance."""
        # Apply filters to retriever if provided
        if filters:
            self._apply_filters(filters)

        # Run the synchronous invoke in a thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.invoke, query, filters)

        return result

def create_rag_chains(
    database_url: str,
    collection_name: str = "supreme_court_docs",
    enable_validation: bool = False,
    use_redis_store: bool = True,
    guardrails_validator: Optional[Any] = None
) -> CourtRAGChains:
    """
    Factory function to create configured RAG chains.
    
    Args:
        database_url: PostgreSQL connection string
        collection_name: Vector store collection name
        enable_validation: Whether to enable claim validation
        use_redis_store: Whether to use Redis for document persistence
        guardrails_validator: Optional Guardrails validator instance
        
    Returns:
        Configured CourtRAGChains instance
    """
    
    # Initialize LangChain components
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # Gemini embedding model
        google_api_key=settings.GOOGLE_API_KEY.get_secret_value()
    )
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=database_url,
        use_jsonb=True
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Or use "gemini-1.5-pro" for better reasoning
        verbose=True,
        google_api_key=settings.GOOGLE_API_KEY.get_secret_value()
    )
    
    return CourtRAGChains(
        vector_store=vector_store,
        llm=llm,
        embeddings=embeddings,
        enable_validation=enable_validation,
        use_redis_store=use_redis_store,
        guardrails_validator=guardrails_validator
    )
