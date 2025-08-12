# Document Processing Pipeline Integration Guide

## Overview

This document explains how to integrate your team's existing document scraping workflow with the CourtSight backend server. The Document Processing Pipeline provides a comprehensive service layer for bulk document ingestion from `putusan3.mahkamahagung.go.id`.

## 🏗️ Architecture Overview

The document processing system consists of three main layers:

### 1. Service Layer
- **DocumentProcessor** (`src/app/services/document_processor.py`)
- **DocumentScheduler** (`src/app/services/document_scheduler.py`)
- **EmbeddingService** (`src/app/services/embedding_service.py`)

### 2. API Layer
- **Document Processing API** (`src/app/api/v1/document_processing.py`)
- REST endpoints for managing processing tasks

### 3. Database Layer
- Legal documents storage with vector embeddings
- Processing task tracking and status management

## 🔧 Integration Methods

### Method 1: Direct API Integration

Your team can integrate with the backend server using REST API endpoints:

#### Start Daily Processing
```bash
curl -X POST "http://localhost:8000/api/v1/document-processing/start-daily" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "target_date": "2024-01-15",
    "force_reprocess": false,
    "max_documents": 1000
  }'
```

#### Bulk Processing for Date Range
```bash
curl -X POST "http://localhost:8000/api/v1/document-processing/start-bulk" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "batch_size": 100,
    "force_reprocess": false
  }'
```

#### Team Workflow Integration
```bash
curl -X POST "http://localhost:8000/api/v1/document-processing/integrate-team-workflow" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "documents": [
      {
        "case_number": "001/Pid.Sus/2024/PN.JKT",
        "title": "Putusan Perkara Korupsi",
        "court_name": "Pengadilan Negeri Jakarta Pusat",
        "document_url": "https://putusan3.mahkamahagung.go.id/direktori/putusan/...",
        "metadata": {
          "jurisdiction": "Indonesia",
          "case_type": "Pidana Khusus",
          "legal_area": "Hukum Pidana"
        }
      }
    ],
    "source": "team_workflow",
    "processing_date": "2024-01-15"
  }'
```

### Method 2: Direct Service Integration

For deeper integration, your team can directly use the service classes:

```python
from app.services.document_processor import document_processor
from app.core.db.database import async_get_db

async def integrate_team_documents():
    async for db in async_get_db():
        # Process documents discovered by your team's scraping logic
        result = await document_processor.process_daily_documents(
            db=db,
            target_date=date.today() - timedelta(days=1)
        )
        
        print(f"Processing completed: {result}")
        break
```

### Method 3: Custom Document Metadata Integration

Your team can provide custom document metadata for processing:

```python
from app.services.document_processor import DocumentMetadata, document_processor
from datetime import date

# Create document metadata from your scraping results
documents = [
    DocumentMetadata(
        case_number="001/Pid.Sus/2024/PN.JKT",
        title="Putusan Perkara Korupsi",
        court_name="Pengadilan Negeri Jakarta Pusat",
        jurisdiction="Indonesia",
        decision_date=date(2024, 1, 15),
        case_type="Pidana Khusus",
        legal_area="Hukum Pidana",
        document_url="https://putusan3.mahkamahagung.go.id/..."
    )
]

# Process using existing pipeline
async for db in async_get_db():
    results = await document_processor._process_documents_batch(db, documents)
    break
```

## 🔌 Key Integration Points

### 1. Document Discovery (`_discover_documents_for_date`)

**Location**: `src/app/services/document_processor.py:200`

**What to implement**: Replace the placeholder with your team's scraping logic

```python
async def _discover_documents_for_date(self, target_date: date) -> List[DocumentMetadata]:
    """
    TODO: Integrate your team's existing scraping logic here
    
    Your implementation should:
    1. Navigate to putusan3.mahkamahagung.go.id
    2. Search for documents by date
    3. Extract metadata from search results
    4. Return list of DocumentMetadata objects
    """
    
    # Your team's scraping logic here
    documents = []
    
    # Example integration:
    async with aiohttp.ClientSession() as session:
        # Your team's discovery logic
        scraped_data = await your_team_scraper.discover_documents(target_date)
        
        for item in scraped_data:
            documents.append(DocumentMetadata(
                case_number=item["case_number"],
                title=item["title"],
                court_name=item["court_name"],
                jurisdiction=item["jurisdiction"],
                decision_date=item["decision_date"],
                case_type=item["case_type"],
                legal_area=item["legal_area"],
                document_url=item["document_url"]
            ))
    
    return documents
```

### 2. Content Extraction (`_extract_text_content`)

**Location**: `src/app/services/document_processor.py:400`

**What to implement**: Replace with your team's text extraction logic

```python
async def _extract_text_content(self, document_content: bytes) -> str:
    """
    TODO: Integrate your team's text extraction logic here
    
    This should handle different document formats:
    - PDF files
    - DOC/DOCX files  
    - HTML pages
    - Other formats
    """
    
    # Your team's text extraction logic here
    try:
        # Example for different formats:
        if document_content.startswith(b'%PDF'):
            return await your_team_extractor.extract_pdf_text(document_content)
        elif b'html' in document_content.lower():
            return await your_team_extractor.extract_html_text(document_content)
        else:
            return await your_team_extractor.extract_generic_text(document_content)
            
    except Exception as e:
        logger.error(f"Text extraction failed: {str(e)}")
        return ""
```

## 📋 API Endpoints Reference

### Document Processing Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/document-processing/start-daily` | Start daily processing |
| `POST` | `/api/v1/document-processing/start-bulk` | Start bulk processing |
| `GET` | `/api/v1/document-processing/status` | Get processing status |
| `GET` | `/api/v1/document-processing/tasks` | List processing tasks |
| `GET` | `/api/v1/document-processing/tasks/{task_id}` | Get task details |
| `DELETE` | `/api/v1/document-processing/tasks/{task_id}` | Cancel task |
| `GET` | `/api/v1/document-processing/statistics` | Get processing statistics |
| `POST` | `/api/v1/document-processing/schedule-daily` | Schedule daily automation |
| `POST` | `/api/v1/document-processing/integrate-team-workflow` | Team integration endpoint |

### Response Models

#### ProcessingSummaryResponse
```json
{
  "target_date": "2024-01-15",
  "total_discovered": 150,
  "total_processed": 145,
  "successful": 140,
  "failed": 3,
  "skipped": 2,
  "processing_time_seconds": 1200.5,
  "success_rate": 96.5
}
```

#### ProcessingStatusResponse
```json
{
  "is_running": true,
  "current_task": {
    "task_id": "task_123",
    "task_type": "daily_processing",
    "status": "running",
    "progress": 65
  },
  "queue_size": 3,
  "last_completed": {
    "task_id": "task_122",
    "completed_at": "2024-01-15T10:30:00Z",
    "result": {
      "successful": 98,
      "failed": 2
    }
  }
}
```

## 🔄 Automated Scheduling

### Setup Daily Automation

```bash
curl -X POST "http://localhost:8000/api/v1/document-processing/schedule-daily" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d "time_str=02:00&timezone=Asia/Jakarta&enabled=true"
```

This will automatically run document processing every day at 2:00 AM Jakarta time.

## 📊 Monitoring and Statistics

### Get Processing Statistics

```bash
curl -X GET "http://localhost:8000/api/v1/document-processing/statistics?days_back=30" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Monitor Task Status

```bash
curl -X GET "http://localhost:8000/api/v1/document-processing/status" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## 🔧 Configuration Options

### Document Processor Configuration

```python
from app.services.document_processor import DocumentIngestionConfig

config = DocumentIngestionConfig(
    max_concurrent_downloads=5,     # Concurrent download limit
    max_concurrent_processing=3,    # Concurrent processing limit  
    batch_size=50,                  # Documents per batch
    retry_attempts=3,               # Retry failed downloads
    timeout_seconds=30,             # HTTP timeout
    daily_limit=1000,               # Max documents per day
    enable_deduplication=True,      # Check for duplicates
    enable_vectorization=True       # Generate embeddings
)
```

## 🚀 Deployment Recommendations

### 1. Environment Variables

Add these to your environment configuration:

```bash
# Document Processing Settings
DOCUMENT_PROCESSING_ENABLED=true
DAILY_PROCESSING_TIME=02:00
DAILY_PROCESSING_TIMEZONE=Asia/Jakarta
MAX_CONCURRENT_DOWNLOADS=5
BATCH_SIZE=50
ENABLE_AUTO_SCHEDULING=true

# Supreme Court Website Settings  
MAHKAMAH_AGUNG_BASE_URL=https://putusan3.mahkamahagung.go.id
SCRAPING_USER_AGENT=CourtSight-Bot/1.0
REQUEST_DELAY_SECONDS=1
```

### 2. Background Task Configuration

Ensure your FastAPI application is configured for background tasks:

```python
# In main.py
from fastapi import BackgroundTasks
from app.services.document_scheduler import document_scheduler

@app.on_event("startup")
async def startup_event():
    # Start the document scheduler
    await document_scheduler.start()

@app.on_event("shutdown") 
async def shutdown_event():
    # Stop the document scheduler
    await document_scheduler.stop()
```

### 3. Monitoring Setup

Set up monitoring for the document processing pipeline:

```python
# Custom metrics for monitoring
from prometheus_client import Counter, Histogram, Gauge

documents_processed = Counter('documents_processed_total', 'Total documents processed')
processing_duration = Histogram('document_processing_seconds', 'Time spent processing documents')
queue_size = Gauge('processing_queue_size', 'Number of tasks in processing queue')
```

## 🔗 Integration Checklist

- [ ] **Document Discovery**: Implement your team's scraping logic in `_discover_documents_for_date`
- [ ] **Content Extraction**: Implement your team's text extraction logic in `_extract_text_content`
- [ ] **API Authentication**: Set up JWT tokens for API access
- [ ] **Scheduling**: Configure daily automated processing
- [ ] **Monitoring**: Set up monitoring and alerting for failed tasks
- [ ] **Error Handling**: Test error scenarios and recovery mechanisms
- [ ] **Performance Tuning**: Optimize concurrent processing limits
- [ ] **Database Backup**: Ensure regular backups of processed documents

## 💡 Best Practices

1. **Rate Limiting**: Be respectful to the Supreme Court website with appropriate delays
2. **Error Recovery**: Implement robust retry logic for failed downloads
3. **Deduplication**: Always check for existing documents to avoid duplicates
4. **Monitoring**: Monitor processing success rates and performance metrics
5. **Logging**: Comprehensive logging for debugging and audit trails
6. **Security**: Use proper authentication and authorization for API access

## 🆘 Troubleshooting

### Common Issues

1. **Processing Stuck**: Check queue status and restart scheduler if needed
2. **High Failure Rate**: Review error logs and adjust retry settings
3. **Slow Processing**: Increase concurrent processing limits or optimize extraction logic
4. **Memory Issues**: Reduce batch size and increase processing intervals

### Debug Commands

```bash
# Check processing status
curl -X GET "http://localhost:8000/api/v1/document-processing/status"

# List recent tasks with errors
curl -X GET "http://localhost:8000/api/v1/document-processing/tasks?status_filter=failed"

# Get detailed task information
curl -X GET "http://localhost:8000/api/v1/document-processing/tasks/{task_id}"
```

This integration guide provides everything your team needs to seamlessly integrate their existing document processing workflow with the CourtSight backend server!
