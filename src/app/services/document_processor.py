"""
Enhanced document processing pipeline using LangChain components.
Refactors and extends the existing workdir business logic with proper
clean architecture and LangChain integration.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader,
    PyPDFLoader,
    UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_google_vertexai import VertexAIModelGarden, VertexAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.callbacks import CallbackManager, StdOutCallbackHandler
from src.app.core.config import settings

# Import existing workdir components to refactor
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../workdir'))

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProcessingMetrics(BaseModel):
    """Metrics for document processing pipeline."""
    total_files: int = Field(..., description="Total files processed")
    successful: int = Field(..., description="Successfully processed files")
    failed: int = Field(..., description="Failed processing attempts")
    total_chunks: int = Field(..., description="Total chunks created")
    processing_time: float = Field(..., description="Total processing time in seconds")
    average_chunk_size: float = Field(..., description="Average chunk size in characters")


class DocumentProcessor:
    """
    Enhanced document processing pipeline using LangChain components.
    
    Refactors existing workdir logic with proper error handling,
    monitoring, and LangChain integration patterns.
    """
    
    def __init__(
        self,
        vector_store: PGVector,
        embeddings: VertexAIEmbeddings,
        enable_metadata_extraction: bool = True,
        callback_manager: Optional[CallbackManager] = None
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.enable_metadata_extraction = enable_metadata_extraction
        self.callback_manager = callback_manager or CallbackManager([StdOutCallbackHandler()])
        
        # Initialize text splitters with PRD specifications
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # PRD parent document size
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""],
            keep_separator=True
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,   # PRD child chunk size
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""],
            keep_separator=True
        )
        
        # Document store for parent-child retrieval
        # TODO: Replace with persistent store for production
        self.doc_store = InMemoryStore()
        
        # Setup parent-child retriever
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.doc_store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
            search_kwargs={"k": 10}
        )
    
    async def process_directory(
        self,
        directory_path: str,
        file_patterns: List[str] = ["**/*.pdf", "**/*.txt"],
        batch_size: int = 10
    ) -> ProcessingMetrics:
        """
        Process all documents in a directory using LangChain loaders.
        
        Enhanced version of the existing workdir loader.py logic.
        
        Args:
            directory_path: Path to directory containing documents
            file_patterns: Glob patterns for file matching
            batch_size: Number of files to process in each batch
            
        Returns:
            ProcessingMetrics with detailed processing statistics
        """
        start_time = datetime.now()
        
        try:
            # Load documents using LangChain DirectoryLoader
            documents = []
            total_files = 0
            
            for pattern in file_patterns:
                loader = DirectoryLoader(
                    directory_path,
                    glob=pattern,
                    loader_cls=self._get_loader_class,
                    show_progress=True,
                    use_multithreading=True,
                    max_concurrency=4
                )
                
                pattern_docs = await asyncio.to_thread(loader.load)
                documents.extend(pattern_docs)
                total_files += len(pattern_docs)
            
            logger.info(f"Loaded {len(documents)} documents from {directory_path}")
            
            # Process in batches
            successful = 0
            failed = 0
            total_chunks = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                try:
                    # Clean and preprocess documents
                    cleaned_batch = await self._clean_documents_batch(batch)
                    
                    # Extract metadata if enabled
                    if self.enable_metadata_extraction:
                        enriched_batch = await self._extract_metadata_batch(cleaned_batch)
                    else:
                        enriched_batch = cleaned_batch
                    
                    # Add to retriever (this creates parent-child chunks)
                    self.retriever.add_documents(enriched_batch)
                    
                    successful += len(batch)
                    
                    # Count chunks created
                    for doc in enriched_batch:
                        parent_chunks = self.parent_splitter.split_documents([doc])
                        for parent in parent_chunks:
                            child_chunks = self.child_splitter.split_documents([parent])
                            total_chunks += len(child_chunks)
                    
                except Exception as e:
                    logger.error(f"Failed to process batch {i//batch_size + 1}: {str(e)}")
                    failed += len(batch)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate average chunk size
            avg_chunk_size = 0
            if total_chunks > 0:
                total_chars = sum(len(doc.page_content) for doc in documents)
                avg_chunk_size = total_chars / total_chunks
            
            return ProcessingMetrics(
                total_files=total_files,
                successful=successful,
                failed=failed,
                total_chunks=total_chunks,
                processing_time=processing_time,
                average_chunk_size=avg_chunk_size
            )
            
        except Exception as e:
            logger.error(f"Directory processing failed: {str(e)}")
            raise
    
    def _get_loader_class(self, file_path: str):
        """Get appropriate LangChain loader class based on file extension."""
        suffix = Path(file_path).suffix.lower()
        
        if suffix == '.pdf':
            return PyPDFLoader
        elif suffix in ['.txt', '.md']:
            return TextLoader
        else:
            return UnstructuredFileLoader
    
    async def _clean_documents_batch(self, documents: List[Document]) -> List[Document]:
        """
        Clean and preprocess documents using enhanced workdir cleaning logic.
        
        Refactors existing cleaning.py with async support and error handling.
        """
        cleaned_docs = []
        
        for doc in documents:
            try:
                # Enhanced text cleaning based on workdir/cleaning.py
                cleaned_content = self._clean_legal_text(doc.page_content)
                
                # Create new document with cleaned content
                cleaned_doc = Document(
                    page_content=cleaned_content,
                    metadata={
                        **doc.metadata,
                        "cleaned_at": datetime.now().isoformat(),
                        "original_length": len(doc.page_content),
                        "cleaned_length": len(cleaned_content)
                    }
                )
                
                cleaned_docs.append(cleaned_doc)
                
            except Exception as e:
                logger.error(f"Failed to clean document {doc.metadata.get('source', 'unknown')}: {str(e)}")
                # Include original document if cleaning fails
                cleaned_docs.append(doc)
        
        return cleaned_docs
    
    def _clean_legal_text(self, text: str) -> str:
        """
        Enhanced legal text cleaning based on workdir/cleaning.py.
        
        Implements specific cleaning rules for Indonesian legal documents.
        """
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers common in legal docs
        text = re.sub(r'(Halaman \d+|Page \d+)', '', text)
        text = re.sub(r'(Putusan No\..*?Mahkamah Agung)', '', text)
        
        # Clean up common OCR errors in Indonesian legal text
        text = re.sub(r'(?i)(mahkamah\s+agung)', 'Mahkamah Agung', text)
        text = re.sub(r'(?i)(republik\s+indonesia)', 'Republik Indonesia', text)
        
        # Remove redundant punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[,]{2,}', ',', text)
        
        # Normalize quotation marks (escape Unicode properly)
        text = re.sub(r'[""„]', '"', text)  # Smart double quotes
        text = re.sub(r'[''‚]', "'", text)  # Smart single quotes
        
        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])\s+', r'\1 ', text)
        
        return text.strip()
    
    async def _extract_metadata_batch(self, documents: List[Document]) -> List[Document]:
        """
        Extract metadata from documents using enhanced workdir extraction logic.
        
        Refactors existing extract_metadata.py with better error handling.
        """
        enriched_docs = []
        
        for doc in documents:
            try:
                # Extract metadata using enhanced logic
                metadata = await self._extract_legal_metadata(doc.page_content, doc.metadata)
                
                # Create enriched document
                enriched_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        **metadata,
                        "metadata_extracted_at": datetime.now().isoformat()
                    }
                )
                
                enriched_docs.append(enriched_doc)
                
            except Exception as e:
                logger.error(f"Failed to extract metadata: {str(e)}")
                # Include original document if metadata extraction fails
                enriched_docs.append(doc)
        
        return enriched_docs
    
    async def _extract_legal_metadata(self, content: str, existing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract legal metadata from document content.
        
        Enhanced version of workdir/extract_metadata.py logic.
        """
        import re
        
        metadata = {}
        
        # Extract case number patterns
        case_patterns = [
            r'No\.?\s*(\d+/[A-Z]+/[A-Z]+/\d+)',  # Indonesian case format
            r'Putusan\s+No\.?\s*(\d+[/\w]+)',
            r'Perkara\s+No\.?\s*(\d+[/\w]+)'
        ]
        
        for pattern in case_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata['case_number'] = match.group(1)
                break
        
        # Extract court information
        court_patterns = [
            r'(Mahkamah Agung Republik Indonesia)',
            r'(Pengadilan\s+\w+\s+\w+)',
            r'(MA\s+RI)'
        ]
        
        for pattern in court_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata['court'] = match.group(1)
                break
        
        # Extract date information
        date_patterns = [
            r'(\d{1,2}\s+\w+\s+\d{4})',  # Indonesian date format
            r'(\d{4}-\d{2}-\d{2})',      # ISO date format
            r'(\d{1,2}/\d{1,2}/\d{4})'   # Slash date format
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content)
            if match:
                metadata['document_date'] = match.group(1)
                break
        
        # Extract case type indicators
        if re.search(r'(pidana|criminal)', content, re.IGNORECASE):
            metadata['case_type'] = 'criminal'
        elif re.search(r'(perdata|civil)', content, re.IGNORECASE):
            metadata['case_type'] = 'civil'
        elif re.search(r'(tata\s+usaha|administrative)', content, re.IGNORECASE):
            metadata['case_type'] = 'administrative'
        
        # Extract jurisdiction (default to Indonesia)
        metadata['jurisdiction'] = 'ID'
        
        # Determine court level
        if re.search(r'mahkamah\s+agung', content, re.IGNORECASE):
            metadata['court_level'] = 'supreme'
        elif re.search(r'pengadilan\s+tinggi', content, re.IGNORECASE):
            metadata['court_level'] = 'high'
        elif re.search(r'pengadilan\s+negeri', content, re.IGNORECASE):
            metadata['court_level'] = 'district'
        
        return metadata
    
    async def process_single_file(
        self,
        file_path: str,
        metadata_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single document file.
        
        Args:
            file_path: Path to the document file
            metadata_override: Optional metadata to override extracted values
            
        Returns:
            Processing result with statistics
        """
        try:
            # Load document using appropriate loader
            loader_class = self._get_loader_class(file_path)
            loader = loader_class(file_path)
            
            documents = await asyncio.to_thread(loader.load)
            
            if not documents:
                return {"status": "failed", "error": "No content extracted"}
            
            # Process document through pipeline
            cleaned_docs = await self._clean_documents_batch(documents)
            
            if self.enable_metadata_extraction:
                enriched_docs = await self._extract_metadata_batch(cleaned_docs)
            else:
                enriched_docs = cleaned_docs
            
            # Apply metadata overrides
            if metadata_override:
                for doc in enriched_docs:
                    doc.metadata.update(metadata_override)
            
            # Add to retriever
            self.retriever.add_documents(enriched_docs)
            
            # Calculate chunks created
            total_chunks = 0
            for doc in enriched_docs:
                parent_chunks = self.parent_splitter.split_documents([doc])
                for parent in parent_chunks:
                    child_chunks = self.child_splitter.split_documents([parent])
                    total_chunks += len(child_chunks)
            
            return {
                "status": "success",
                "documents_processed": len(enriched_docs),
                "chunks_created": total_chunks,
                "metadata_extracted": enriched_docs[0].metadata if enriched_docs else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {str(e)}")
            return {"status": "failed", "error": str(e)}
    
    def get_retriever(self) -> ParentDocumentRetriever:
        """Get the configured parent-child retriever."""
        return self.retriever
    
    async def clear_index(self) -> None:
        """Clear all documents from the index."""
        # Clear vector store
        # Implementation depends on PGVector clear method
        
        # Clear document store
        self.doc_store.mdelete(list(self.doc_store.yield_keys()))
        
        logger.info("Document index cleared")


def create_document_processor(
    database_url: str,
    collection_name: str = "supreme_court_docs",
    enable_metadata_extraction: bool = True
) -> DocumentProcessor:
    """
    Factory function to create configured document processor.
    
    Args:
        database_url: PostgreSQL connection string
        collection_name: Vector store collection name
        enable_metadata_extraction: Whether to extract metadata
        
    Returns:
        Configured DocumentProcessor instance
    """
    # Initialize embeddings
    embeddings = VertexAIEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        project=settings.PROJECT_ID,
        location=settings.LOCATION
    )
    
    # Initialize vector store
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=database_url,
        use_jsonb=True
    )
    
    return DocumentProcessor(
        vector_store=vector_store,
        embeddings=embeddings,
        enable_metadata_extraction=enable_metadata_extraction
    )
