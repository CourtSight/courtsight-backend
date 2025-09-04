#!/usr/bin/env python3
"""
Test script for ETL process - loading test documents into vector database.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from app.services.document_processor import create_document_processor
from app.services.rag_service import create_rag_service
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_etl_process():
    """Test ETL process for loading test documents."""
    try:
        logger.info("🚀 Starting ETL test process...")

        # Initialize services
        doc_processor = create_document_processor()
        rag_service = create_rag_service()

        # Path to test documents
        test_doc_path = Path(__file__).parent / "test_doc"
        logger.info(f"📁 Processing documents from: {test_doc_path}")

        if not test_doc_path.exists():
            logger.error(f"❌ Test documents directory not found: {test_doc_path}")
            return

        # List available documents
        pdf_files = list(test_doc_path.glob("*.pdf"))
        logger.info(f"📄 Found {len(pdf_files)} PDF documents:")
        for pdf_file in pdf_files:
            logger.info(f"  - {pdf_file.name}")

        if not pdf_files:
            logger.warning("⚠️  No PDF files found in test_doc directory")
            return

        # Process documents from directory
        logger.info("🔄 Processing documents...")
        metrics = await doc_processor.process_directory(
            directory_path=str(test_doc_path),
            file_patterns=["*.pdf"],
            batch_size=5
        )

        logger.info("📊 Processing Metrics:")
        logger.info(f"  - Total files processed: {metrics.total_files}")
        logger.info(f"  - Documents created: {metrics.documents_created}")
        logger.info(f"  - Chunks created: {metrics.chunks_created}")
        logger.info(f"  - Processing time: {metrics.processing_time:.2f}s")
        logger.info(f"  - Success rate: {metrics.success_rate:.1%}")

        if metrics.errors:
            logger.warning("⚠️  Processing errors:")
            for error in metrics.errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")

        # Test search to verify documents are in the database
        logger.info("🔍 Testing search functionality...")
        test_query = "korupsi"  # Corruption in Indonesian

        search_result = await rag_service.search_documents(
            query=test_query,
            limit=3,
            user_id=1  # Mock user ID
        )

        logger.info(f"✅ Search test completed!")
        logger.info(f"  - Query: '{test_query}'")
        logger.info(f"  - Results found: {len(search_result.results) if search_result.results else 0}")

        if search_result.results:
            logger.info("📋 Sample result:")
            result = search_result.results[0]
            logger.info(f"  - Summary: {result.summary[:100]}...")
            logger.info(f"  - Source: {result.source_documents[0].title if result.source_documents else 'N/A'}")

        logger.info("🎉 ETL test process completed successfully!")

    except Exception as e:
        logger.error(f"❌ ETL test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_etl_process())
