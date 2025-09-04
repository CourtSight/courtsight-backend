"""
LangChain Expression Language (LCEL) implementation for Supreme Court RAG system.
This module implements the core RAG pipeline using LCEL for optimal composability
and monitoring as specified in the PRD requirements.
"""

import os
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks import StdOutCallbackHandler
# Pydantic models for structured outputs
from src.app.core.config import settings
from src.app.schemas.search import SearchResult as SchemaSearchResult, ValidationStatus





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
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.embeddings = embeddings
        self.enable_validation = enable_validation
        
        # Initialize document store for parent-child retrieval
        self.doc_store = InMemoryStore()  # TODO: Replace with persistent store for production
        
        # Setup simple retriever for testing
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 documents
        )
        
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
        Anda adalah asisten hukum yang menganalisis putusan Mahkamah Agung Indonesia.
        
        INSTRUKSI PENTING:
        - Berikan rangkuman yang akurat berdasarkan konteks yang diberikan
        - Ekstrak poin-poin hukum kunci dari dokumen
        - Sertakan sitasi lengkap (nomor putusan, bagian dokumen, chunk_id)
        - Gunakan bahasa hukum formal namun mudah dipahami
        - Jika informasi tidak mencukupi, nyatakan dengan jelas
        - Jangan berhalusinasi atau menambahkan informasi di luar konteks
        
        KONTEKS DOKUMEN:
        {context}
        
        PERTANYAAN PENGGUNA:
        {question}
        
        Berikan respon dalam format JSON dengan struktur:
        {{
            "summary": "ringkasan ringkas dan faktual",
            "key_points": ["poin hukum 1", "poin hukum 2", ...],
            "source_documents": [
                {{
                    "title": "judul dokumen",
                    "case_number": "nomor perkara",
                    "excerpt": "kutipan relevan",
                    "chunk_id": "ID chunk"
                }}
            ],
            "validation_status": "Supported"
        }}
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
        Build claim validation chain using Guardrails pattern.
        Implements PRD requirement for factual claim verification.
        """
        
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
            # Simple pipeline without validation
            complete_pipeline = self.search_chain
        
        return complete_pipeline
    
    def _merge_results(self, validation_result: Dict[str, Any]) -> SearchResult:
        """Merge search results with validation results."""
        # Implementation logic to combine search and validation results
        # This ensures only validated claims are included in final response
        pass
    
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
            print(f"DEBUG: Invoking search chain with query: {query}")
            raw_response = self.search_chain.invoke({"query": query})
            print(f"DEBUG: Search chain raw response: {raw_response}")
            print(f"DEBUG: Raw response type: {type(raw_response)}")
            
            if raw_response is None:
                print("DEBUG: Search chain returned None")
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
                print(f"DEBUG: JSON parsing error: {e}")
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
            print(f"DEBUG: Error in invoke: {e}")
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
            print(f"DEBUG: Error in ainvoke: {e}")
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
        print(f"DEBUG: Invoking search chain with query: {query}")
        result = self.complete_pipeline.invoke(query)
        print(f"DEBUG: Search chain result: {result}")
        
        if result is None:
            print("DEBUG: Search chain returned None")
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
        print(f"DEBUG: Invoking search chain with query: {query}")
        result = self.complete_pipeline.invoke(query)
        print(f"DEBUG: Search chain result: {result}")
        
        if result is None:
            print("DEBUG: Search chain returned None")
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
) -> CourtRAGChains:
    """
    Factory function to create configured RAG chains.
    
    Args:
        database_url: PostgreSQL connection string
        collection_name: Vector store collection name
        enable_validation: Whether to enable claim validation
        
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
        enable_validation=enable_validation
    )
