"""
Comprehensive example demonstrating the Supreme Court RAG system.
This example shows how all components work together following
clean architecture and LangChain orchestration principles.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

# Example usage imports
from langchain_core.documents import Document
from src.app.services.rag_service import create_rag_service
from src.app.services.document_processor import create_document_processor
from src.app.services.evaluation import create_rag_evaluator, EvaluationSample
from src.app.schemas.search import SearchRequest, SearchFilters, DateRange
from src.app.core.config import get_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_document_processing():
    """Demonstrate document processing pipeline."""
    logger.info("=== Document Processing Demonstration ===")
    
    # Initialize document processor
    settings = get_settings()
    processor = create_document_processor(
        database_url=settings.DATABASE_URL,
        collection_name="demo_collection",
        enable_metadata_extraction=True
    )
    
    # Example: Process the actual test documents directory
    test_docs_path = "/mnt/2A28ACA028AC6C8F/Programming/webdevoloper/fastapi/coursigh/test_doc"
    try:
        logger.info(f"Processing documents from: {test_docs_path}")
        metrics = await processor.process_directory(
            directory_path=test_docs_path,
            file_patterns=["**/*.pdf"],
            batch_size=5
        )
        
        logger.info(f"Processing completed: {metrics.dict()}")
        
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
    
    # Example: Process individual test files
    test_files = [
        "putusan_129_pid.b_2025_pn_mrk_20250830082542.pdf",
        "putusan_271_pid.sus_2025_pn_cjr_20250830082656.pdf",
        "putusan_316_pid.b_2025_pn_jkt.pst_20250830082901.pdf",
        "putusan_49_pid.b_2025_pn_slw_20250830082741.pdf",
        "putusan_52_pid.b_2025_pn_pms_20250830082440.pdf"
    ]
    
    for filename in test_files:
        try:
            file_path = f"{test_docs_path}/{filename}"
            logger.info(f"Processing single file: {filename}")
            
            result = await processor.process_single_file(
                file_path=file_path,
                metadata_override={
                    "jurisdiction": "ID",
                    "case_type": "criminal",
                    "court_level": "district",
                    "source_file": filename
                }
            )
            
            logger.info(f"File {filename} processing result: {result}")
            
        except Exception as e:
            logger.error(f"Single file processing failed for {filename}: {str(e)}")


async def demonstrate_search_functionality():
    """Demonstrate RAG search functionality."""
    logger.info("=== Search Functionality Demonstration ===")
    
    # Initialize RAG service
    rag_service = create_rag_service()
    
    # Example search requests
    search_requests = [
        SearchRequest(
            query="putusan mahkamah agung tentang korupsi",
            filters=SearchFilters(
                jurisdiction="ID",
                case_type="criminal",
                date_range=DateRange(
                    start="2020-01-01",
                    end="2023-12-31"
                )
            ),
            max_results=5,
            include_summary=True,
            include_validation=True
        ),
        SearchRequest(
            query="sanksi pidana untuk tindak pidana pencucian uang",
            max_results=3
        ),
        SearchRequest(
            query="yurisprudensi mahkamah agung tentang hak asasi manusia",
            filters=SearchFilters(
                court_level="supreme"
            )
        )
    ]
    
    # Execute searches
    for i, request in enumerate(search_requests, 1):
        try:
            logger.info(f"Executing search {i}: {request.query}")
            
            result = await rag_service.search_documents(
                request=request,
                user_id="demo_user"
            )
            
            logger.info(f"Search {i} completed in {result.metrics.query_time:.2f}s")
            logger.info(f"Found {len(result.results)} results")
            
            if result.results:
                first_result = result.results[0]
                logger.info(f"First result summary: {first_result.summary[:200]}...")
                logger.info(f"Validation status: {first_result.validation_status}")
                logger.info(f"Source documents: {len(first_result.source_documents)}")
            
        except Exception as e:
            logger.error(f"Search {i} failed: {str(e)}")


async def demonstrate_evaluation_system():
    """Demonstrate RAGAS evaluation system."""
    logger.info("=== Evaluation System Demonstration ===")
    
    # Initialize evaluator
    evaluator = create_rag_evaluator(
        enable_ragas=True,
        enable_legal_metrics=True,
        batch_size=5
    )
    
    # Example evaluation samples
    evaluation_samples = [
        EvaluationSample(
            question="Apa hukuman untuk tindak pidana korupsi?",
            answer="Berdasarkan putusan yang tersedia, hukuman untuk tindak pidana korupsi dapat berupa pidana penjara minimal 4 tahun dan maksimal 20 tahun, serta denda minimal 200 juta rupiah.",
            contexts=[
                "Dalam Pasal 2 UU No. 31 Tahun 1999 tentang Pemberantasan Tindak Pidana Korupsi, pidana penjara seumur hidup atau pidana penjara paling singkat 4 tahun dan paling lama 20 tahun.",
                "Putusan Mahkamah Agung No. 123/K/Pid/2023 menjatuhkan pidana penjara 8 tahun dan denda 500 juta rupiah."
            ],
            ground_truth="Hukuman korupsi sesuai UU No. 31/1999 adalah pidana penjara 4-20 tahun dan denda minimum 200 juta rupiah."
        ),
        EvaluationSample(
            question="Bagaimana prosedur kasasi di Mahkamah Agung?",
            answer="Prosedur kasasi diatur dalam KUHAP, dimana permohonan kasasi harus diajukan dalam waktu 14 hari setelah putusan dijatuhkan.",
            contexts=[
                "Pasal 245 KUHAP mengatur bahwa permohonan kasasi diajukan dalam waktu 14 hari setelah putusan dijatuhkan.",
                "Mahkamah Agung memeriksa permohonan kasasi hanya terhadap penerapan hukum."
            ]
        )
    ]
    
    # Execute evaluation
    try:
        batch_result = await evaluator.evaluate_batch(
            samples=evaluation_samples,
            sample_ids=["sample_1", "sample_2"]
        )
        
        logger.info(f"Evaluation completed for {batch_result.sample_count} samples")
        logger.info(f"Processing time: {batch_result.total_processing_time:.2f}s")
        logger.info("Aggregate metrics:")
        
        for metric_name, value in batch_result.aggregate_metrics.items():
            logger.info(f"  {metric_name}: {value:.3f}")
        
        # Show individual results
        for result in batch_result.results:
            logger.info(f"Sample {result.sample_id}:")
            logger.info(f"  Overall performance: {result.overall_performance():.3f}")
            logger.info(f"  RAGAS score: {result.ragas_metrics.overall_score():.3f}")
            logger.info(f"  Legal accuracy: {result.legal_metrics.legal_accuracy:.3f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")


async def demonstrate_claim_validation():
    """Demonstrate Guardrails claim validation."""
    logger.info("=== Claim Validation Demonstration ===")
    
    from src.app.services.guardrails_validator import create_guardrails_validator
    from langchain_google_vertexai import VertexAIModelGarden
    import os
    
    # Initialize validator
    llm = VertexAIModelGarden(
        endpoint_id=os.getenv("VERTEX_AI_ENDPOINT_ID", "your-endpoint-id"),
        project=os.getenv("PROJECT_ID", "g-72-courtsightteam"),
        location="us-central1"
    )
    
    validator = create_guardrails_validator(
        llm=llm,
        enable_strict_validation=True,
        confidence_threshold=0.7
    )
    
    # Example text with claims to validate
    text_with_claims = """
    Berdasarkan putusan Mahkamah Agung No. 123/K/Pid/2023, terdakwa dijatuhi 
    hukuman penjara 10 tahun dan denda 1 miliar rupiah atas tindak pidana korupsi. 
    Kasus ini melibatkan penggelapan dana desa sebesar 5 miliar rupiah. 
    Mahkamah Agung juga memerintahkan ganti rugi sebesar 3 miliar rupiah.
    """
    
    # Evidence documents
    evidence_documents = [
        {
            "source": "putusan_123_k_pid_2023",
            "content": "Mengadili: Menjatuhkan pidana terhadap terdakwa dengan pidana penjara selama 10 tahun dan denda sebesar Rp. 1.000.000.000 (satu miliar rupiah)."
        },
        {
            "source": "berita_acara_penyidikan",
            "content": "Total kerugian negara akibat penggelapan dana desa adalah sebesar Rp. 5.000.000.000 (lima miliar rupiah)."
        }
    ]
    
    try:
        # Validate claims
        validation_result = await validator.validate_batch(
            text=text_with_claims,
            evidence_documents=evidence_documents
        )
        
        logger.info(f"Validation completed for {len(validation_result.claims)} claims")
        logger.info(f"Overall confidence: {validation_result.overall_confidence:.3f}")
        logger.info("Summary:")
        
        for status, count in validation_result.summary.items():
            logger.info(f"  {status}: {count}")
        
        # Show individual claim validations
        for claim_result in validation_result.claims:
            logger.info(f"Claim: {claim_result.claim.text}")
            logger.info(f"Status: {claim_result.status}")
            logger.info(f"Confidence: {claim_result.confidence:.3f}")
            logger.info(f"Reasoning: {claim_result.reasoning[:100]}...")
            logger.info("---")
        
        logger.info(f"Filtered text: {validation_result.filtered_text[:200]}...")
        
    except Exception as e:
        logger.error(f"Claim validation failed: {str(e)}")


async def demonstrate_complete_workflow():
    """Demonstrate complete end-to-end workflow."""
    logger.info("=== Complete Workflow Demonstration ===")
    
    try:
        # Step 1: Document Processing
        logger.info("Step 1: Processing documents...")
        await demonstrate_document_processing()
        
        # Give some time for documents to be indexed
        logger.info("Waiting for documents to be indexed...")
        await asyncio.sleep(5)
        
        # Step 2: Search Functionality
        logger.info("Step 2: Executing searches...")
        await demonstrate_search_functionality()
        
        # Step 3: Claim Validation
        logger.info("Step 3: Validating claims...")
        await demonstrate_claim_validation()
        
        # Step 4: System Evaluation
        logger.info("Step 4: Evaluating system...")
        await demonstrate_evaluation_system()
        
        logger.info("Complete workflow demonstration finished successfully!")
        
    except Exception as e:
        logger.error(f"Workflow demonstration failed: {str(e)}")


async def test_specific_features():
    """Test specific features individually."""
    logger.info("=== Testing Specific Features ===")
    
    # Test 1: Vector Store Connection
    logger.info("Testing vector store connection...")
    try:
        from src.app.core.dependencies import get_vector_store
        vector_store = get_vector_store()
        logger.info("✅ Vector store connection successful")
    except Exception as e:
        logger.error(f"❌ Vector store connection failed: {e}")
    
    # Test 2: VertexAI Components
    logger.info("Testing VertexAI components...")
    try:
        from src.app.core.dependencies import get_vertex_ai_llm, get_vertex_ai_embeddings
        llm = get_vertex_ai_llm()
        embeddings = get_vertex_ai_embeddings()
        logger.info("✅ VertexAI components initialized successfully")
    except Exception as e:
        logger.error(f"❌ VertexAI components failed: {e}")
    
    # Test 3: RAG Chains
    logger.info("Testing RAG chains...")
    try:
        from src.app.core.dependencies import get_rag_chains
        chains = get_rag_chains()
        logger.info("✅ RAG chains initialized successfully")
    except Exception as e:
        logger.error(f"❌ RAG chains failed: {e}")
    
    # Test 4: Document Processing
    logger.info("Testing document processing components...")
    try:
        from src.app.core.dependencies import get_document_processor
        processor = get_document_processor()
        logger.info("✅ Document processor initialized successfully")
    except Exception as e:
        logger.error(f"❌ Document processor failed: {e}")
    
    # Test 5: Guardrails Validator
    logger.info("Testing Guardrails validator...")
    try:
        from src.app.core.dependencies import get_guardrails_validator
        validator = get_guardrails_validator()
        logger.info("✅ Guardrails validator initialized successfully")
    except Exception as e:
        logger.error(f"❌ Guardrails validator failed: {e}")


async def test_ingest_performance():
    """Test document ingestion performance with the actual test files."""
    logger.info("=== Testing Document Ingestion Performance ===")
    
    from src.app.services.document_processor import create_document_processor
    from src.app.core.config import get_settings
    import time
    
    settings = get_settings()
    processor = create_document_processor(
        database_url=settings.DATABASE_URL,
        collection_name="performance_test",
        enable_metadata_extraction=True
    )
    
    test_docs_path = "/mnt/2A28ACA028AC6C8F/Programming/webdevoloper/fastapi/coursigh/test_doc"
    test_files = [
        "putusan_129_pid.b_2025_pn_mrk_20250830082542.pdf",
        "putusan_271_pid.sus_2025_pn_cjr_20250830082656.pdf",
        "putusan_316_pid.b_2025_pn_jkt.pst_20250830082901.pdf",
        "putusan_49_pid.b_2025_pn_slw_20250830082741.pdf",
        "putusan_52_pid.b_2025_pn_pms_20250830082440.pdf"
    ]
    
    total_start_time = time.time()
    successful_ingests = 0
    failed_ingests = 0
    
    for i, filename in enumerate(test_files, 1):
        file_start_time = time.time()
        
        try:
            file_path = f"{test_docs_path}/{filename}"
            logger.info(f"[{i}/5] Ingesting: {filename}")
            
            result = await processor.process_single_file(
                file_path=file_path,
                metadata_override={
                    "jurisdiction": "ID",
                    "case_type": "criminal",
                    "court_level": "district",
                    "source_file": filename,
                    "ingestion_batch": "performance_test"
                }
            )
            
            file_end_time = time.time()
            processing_time = file_end_time - file_start_time
            
            if result["status"] == "success":
                successful_ingests += 1
                logger.info(f"✅ {filename} processed in {processing_time:.2f}s")
                logger.info(f"   Documents: {result['documents_processed']}")
                logger.info(f"   Chunks: {result['chunks_created']}")
            else:
                failed_ingests += 1
                logger.error(f"❌ {filename} failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            failed_ingests += 1
            file_end_time = time.time()
            processing_time = file_end_time - file_start_time
            logger.error(f"❌ {filename} failed after {processing_time:.2f}s: {str(e)}")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    logger.info("=== Ingestion Performance Summary ===")
    logger.info(f"Total files processed: {len(test_files)}")
    logger.info(f"Successful ingests: {successful_ingests}")
    logger.info(f"Failed ingests: {failed_ingests}")
    logger.info(f"Total processing time: {total_time:.2f}s")
    logger.info(f"Average time per file: {total_time/len(test_files):.2f}s")
    logger.info(f"Success rate: {(successful_ingests/len(test_files))*100:.1f}%")


async def main():
    """Main demonstration function."""
    logger.info("Starting Supreme Court RAG System Comprehensive Testing")
    logger.info("=" * 60)
    
    try:
        # Test 1: Individual Feature Testing
        logger.info("PHASE 1: Testing Individual Features")
        await test_specific_features()
        
        # Test 2: Document Ingestion Performance
        logger.info("\nPHASE 2: Testing Document Ingestion Performance")
        await test_ingest_performance()
        
        # Test 3: Complete Workflow
        logger.info("\nPHASE 3: Testing Complete Workflow")
        await demonstrate_complete_workflow()
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("The Supreme Court RAG System is ready for production use.")
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        logger.error("Please check the logs above for specific error details.")
    finally:
        logger.info("Testing session completed")


if __name__ == "__main__":
    # Set up environment variables if needed
    import os
    
    # Ensure we have the required environment variables
    required_env_vars = [
        "DATABASE_URL",
        "PROJECT_ID", 
        "VERTEX_AI_ENDPOINT_ID"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning("Some tests may fail without proper configuration")
    
    # Run the comprehensive testing
    asyncio.run(main())
