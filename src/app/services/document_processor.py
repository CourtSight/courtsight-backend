"""
Document Processing Pipeline Service for CourtSight
Implements bulk document ingestion from Supreme Court website (putusan3.mahkamahagung.go.id)
"""

import asyncio
import hashlib
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from ..models.legal_document import LegalDocument
from ..crud.crud_legal_documents import CRUDLegalDocument
from ..schemas.legal_search import LegalDocumentCreate
from .embedding_service import embedding_service
from ..core.logger import logger


class ProcessingStatus(Enum):
    """Document processing status enumeration."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    VECTORIZING = "vectorizing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DocumentMetadata:
    """Metadata structure for scraped documents."""
    case_number: str
    title: str
    court_name: str
    jurisdiction: str
    decision_date: date
    case_type: str
    legal_area: str
    document_url: str
    content_hash: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of document processing operation."""
    case_number: str
    status: ProcessingStatus
    document_id: Optional[int] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


class DocumentProcessor:
    """
    Service for processing Supreme Court documents from putusan3.mahkamahagung.go.id
    
    This service handles:
    - Document discovery and metadata extraction
    - Content downloading and text extraction
    - Deduplication based on content hash
    - Text vectorization for semantic search
    - Database storage with proper error handling
    """
    
    def __init__(self):
        """Initialize the document processor."""
        self.crud_legal_documents = CRUDLegalDocument()
        self.session_timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.max_concurrent_downloads = 5
        self.max_retries = 3
        logger.info("DocumentProcessor initialized")
    
    async def process_daily_documents(
        self,
        db: AsyncSession,
        target_date: Optional[date] = None,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Process documents for a specific date from Supreme Court website.
        
        Args:
            db: Database session
            target_date: Date to process (defaults to today)
            force_reprocess: Whether to reprocess existing documents
            
        Returns:
            Processing summary with statistics
        """
        if target_date is None:
            target_date = date.today()
            
        logger.info(f"Starting daily document processing for {target_date}")
        start_time = datetime.now()
        
        try:
            # Step 1: Discover documents for the target date
            discovered_docs = await self._discover_documents_for_date(target_date)
            logger.info(f"Discovered {len(discovered_docs)} documents for {target_date}")
            
            if not discovered_docs:
                return self._create_processing_summary(
                    target_date=target_date,
                    total_discovered=0,
                    total_processed=0,
                    successful=0,
                    failed=0,
                    skipped=0,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Step 2: Filter out existing documents (unless force reprocess)
            documents_to_process = await self._filter_existing_documents(
                db, discovered_docs, force_reprocess
            )
            
            logger.info(f"Processing {len(documents_to_process)} documents (after deduplication)")
            
            # Step 3: Process documents in batches
            results = await self._process_documents_batch(db, documents_to_process)
            
            # Step 4: Generate processing summary
            summary = self._analyze_processing_results(
                target_date=target_date,
                total_discovered=len(discovered_docs),
                results=results,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            logger.info(f"Daily processing completed: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Daily document processing failed: {str(e)}", exc_info=True)
            raise
    
    async def process_document_range(
        self,
        db: AsyncSession,
        start_date: date,
        end_date: date,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Process documents for a date range (bulk historical processing).
        
        Args:
            db: Database session
            start_date: Start date for processing
            end_date: End date for processing
            batch_size: Number of documents to process per batch
            
        Returns:
            Processing summary with statistics
        """
        logger.info(f"Starting bulk processing from {start_date} to {end_date}")
        start_time = datetime.now()
        
        all_results = []
        current_date = start_date
        
        try:
            while current_date <= end_date:
                logger.info(f"Processing date: {current_date}")
                
                # Process each date
                daily_summary = await self.process_daily_documents(
                    db=db,
                    target_date=current_date,
                    force_reprocess=False
                )
                
                all_results.append(daily_summary)
                current_date = date.fromordinal(current_date.toordinal() + 1)
                
                # Add small delay to be respectful to the source website
                await asyncio.sleep(1)
            
            # Aggregate all results
            total_summary = self._aggregate_bulk_results(
                all_results,
                start_date,
                end_date,
                (datetime.now() - start_time).total_seconds()
            )
            
            logger.info(f"Bulk processing completed: {total_summary}")
            return total_summary
            
        except Exception as e:
            logger.error(f"Bulk document processing failed: {str(e)}", exc_info=True)
            raise
    
    async def _discover_documents_for_date(self, target_date: date) -> List[DocumentMetadata]:
        """
        Discover documents available for a specific date from the Supreme Court website.
        
        This method should be implemented based on your team's existing scraping logic.
        It should return a list of DocumentMetadata objects.
        """
        # TODO: Integrate your team's existing scraping logic here
        # This is a placeholder that your team can replace with their implementation
        
        async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
            # Example implementation - replace with your team's logic
            documents = []
            
            try:
                # Your team's scraping logic would go here
                # For example:
                # 1. Navigate to putusan3.mahkamahagung.go.id
                # 2. Search for documents by date
                # 3. Extract metadata from search results
                # 4. Return list of DocumentMetadata objects
                
                logger.info(f"Document discovery for {target_date} - integrate team's scraping logic here")
                
                # Placeholder return - replace with actual implementation
                return documents
                
            except Exception as e:
                logger.error(f"Document discovery failed for {target_date}: {str(e)}")
                return []
    
    async def _filter_existing_documents(
        self,
        db: AsyncSession,
        discovered_docs: List[DocumentMetadata],
        force_reprocess: bool
    ) -> List[DocumentMetadata]:
        """Filter out documents that already exist in the database."""
        if force_reprocess:
            return discovered_docs
        
        # Get existing case numbers from database
        case_numbers = [doc.case_number for doc in discovered_docs]
        
        query = select(LegalDocument.case_number).where(
            LegalDocument.case_number.in_(case_numbers)
        )
        result = await db.execute(query)
        existing_case_numbers = {row[0] for row in result.fetchall()}
        
        # Filter out existing documents
        filtered_docs = [
            doc for doc in discovered_docs 
            if doc.case_number not in existing_case_numbers
        ]
        
        logger.info(f"Filtered {len(discovered_docs) - len(filtered_docs)} existing documents")
        return filtered_docs
    
    async def _process_documents_batch(
        self,
        db: AsyncSession,
        documents: List[DocumentMetadata]
    ) -> List[ProcessingResult]:
        """Process a batch of documents with controlled concurrency."""
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        
        async def process_single_document(doc_metadata: DocumentMetadata) -> ProcessingResult:
            async with semaphore:
                return await self._process_single_document(db, doc_metadata)
        
        # Process documents concurrently with limited concurrency
        tasks = [process_single_document(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    case_number=documents[i].case_number,
                    status=ProcessingStatus.FAILED,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_document(
        self,
        db: AsyncSession,
        doc_metadata: DocumentMetadata
    ) -> ProcessingResult:
        """Process a single document through the complete pipeline."""
        start_time = datetime.now()
        
        try:
            logger.debug(f"Processing document: {doc_metadata.case_number}")
            
            # Step 1: Download document content
            document_content = await self._download_document_content(doc_metadata.document_url)
            if not document_content:
                return ProcessingResult(
                    case_number=doc_metadata.case_number,
                    status=ProcessingStatus.FAILED,
                    error_message="Failed to download document content"
                )
            
            # Step 2: Extract text and create content hash
            extracted_text = await self._extract_text_content(document_content)
            content_hash = self._generate_content_hash(extracted_text)
            
            # Step 3: Check for duplicates by content hash
            if await self._is_duplicate_content(db, content_hash):
                return ProcessingResult(
                    case_number=doc_metadata.case_number,
                    status=ProcessingStatus.SKIPPED,
                    error_message="Duplicate content detected"
                )
            
            # Step 4: Generate embeddings
            embeddings = await self._generate_embeddings(extracted_text)
            
            # Step 5: Save to database
            document_create = LegalDocumentCreate(
                case_number=doc_metadata.case_number,
                court_name=doc_metadata.court_name,
                jurisdiction=doc_metadata.jurisdiction,
                title=doc_metadata.title,
                full_text=extracted_text,
                summary=await self._generate_summary(extracted_text),
                decision_date=doc_metadata.decision_date,
                case_type=doc_metadata.case_type,
                legal_area=doc_metadata.legal_area,
                content_hash=content_hash,
                embedding=embeddings,
                processing_status="completed",
                processed_at=datetime.now()
            )
            
            # Save document
            created_document = await self.crud_legal_documents.create(db, obj_in=document_create)
            await db.commit()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                case_number=doc_metadata.case_number,
                status=ProcessingStatus.COMPLETED,
                document_id=created_document.id,
                processing_time=processing_time
            )
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to process document {doc_metadata.case_number}: {str(e)}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessingResult(
                case_number=doc_metadata.case_number,
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                processing_time=processing_time
            )
    
    async def _download_document_content(self, document_url: str) -> Optional[bytes]:
        """Download document content from URL with retry logic."""
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                    async with session.get(document_url) as response:
                        if response.status == 200:
                            content = await response.read()
                            logger.debug(f"Downloaded {len(content)} bytes from {document_url}")
                            return content
                        else:
                            logger.warning(f"HTTP {response.status} for {document_url}")
                            
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed for {document_url}: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    async def _extract_text_content(self, document_content: bytes) -> str:
        """
        Extract text content from downloaded document.
        
        This method should handle different document formats (PDF, DOC, HTML, etc.)
        based on your team's existing text extraction logic.
        """
        # TODO: Integrate your team's text extraction logic here
        # This should handle PDF, DOC, HTML and other formats
        
        try:
            # Placeholder implementation - replace with your team's logic
            # For example, if documents are HTML:
            # return await self._extract_html_text(document_content)
            # For PDF:
            # return await self._extract_pdf_text(document_content)
            
            logger.debug("Text extraction - integrate team's extraction logic here")
            
            # Temporary fallback - assume content is already text
            try:
                return document_content.decode('utf-8')
            except:
                return document_content.decode('latin-1', errors='ignore')
                
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            return ""
    
    async def _generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the document text."""
        try:
            # Use the existing embedding service
            embeddings = await embedding_service.get_embeddings(text)
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return []
    
    async def _generate_summary(self, text: str, max_length: int = 500) -> str:
        """Generate a summary of the document text."""
        try:
            # Simple extractive summary - take first sentences up to max_length
            sentences = text.split('. ')
            summary = ""
            for sentence in sentences:
                if len(summary + sentence) < max_length:
                    summary += sentence + ". "
                else:
                    break
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate a hash of the document content for deduplication."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def _is_duplicate_content(self, db: AsyncSession, content_hash: str) -> bool:
        """Check if document with this content hash already exists."""
        query = select(LegalDocument.id).where(
            LegalDocument.content_hash == content_hash
        )
        result = await db.execute(query)
        return result.scalar_one_or_none() is not None
    
    def _create_processing_summary(
        self,
        target_date: date,
        total_discovered: int,
        total_processed: int,
        successful: int,
        failed: int,
        skipped: int,
        processing_time: float
    ) -> Dict[str, Any]:
        """Create a processing summary dictionary."""
        return {
            "target_date": target_date.isoformat(),
            "total_discovered": total_discovered,
            "total_processed": total_processed,
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "processing_time_seconds": processing_time,
            "success_rate": (successful / total_processed * 100) if total_processed > 0 else 0.0
        }
    
    def _analyze_processing_results(
        self,
        target_date: date,
        total_discovered: int,
        results: List[ProcessingResult],
        processing_time: float
    ) -> Dict[str, Any]:
        """Analyze processing results and create summary."""
        successful = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
        skipped = sum(1 for r in results if r.status == ProcessingStatus.SKIPPED)
        
        return self._create_processing_summary(
            target_date=target_date,
            total_discovered=total_discovered,
            total_processed=len(results),
            successful=successful,
            failed=failed,
            skipped=skipped,
            processing_time=processing_time
        )
    
    def _aggregate_bulk_results(
        self,
        daily_results: List[Dict[str, Any]],
        start_date: date,
        end_date: date,
        total_processing_time: float
    ) -> Dict[str, Any]:
        """Aggregate results from bulk processing."""
        return {
            "date_range": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days_processed": len(daily_results)
            },
            "totals": {
                "discovered": sum(r["total_discovered"] for r in daily_results),
                "processed": sum(r["total_processed"] for r in daily_results),
                "successful": sum(r["successful"] for r in daily_results),
                "failed": sum(r["failed"] for r in daily_results),
                "skipped": sum(r["skipped"] for r in daily_results)
            },
            "processing_time_seconds": total_processing_time,
            "daily_summaries": daily_results
        }


# Global instance
document_processor = DocumentProcessor()
