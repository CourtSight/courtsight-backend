"""
Document Processing API endpoints for CourtSight
Provides REST API for managing bulk document ingestion from Supreme Court website
"""

from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from ...core.db.database import async_get_db
from ...models.user import User
from ...api.dependencies import get_current_user, get_optional_user
from ...services.document_processor import document_processor
from ...services.document_scheduler import document_scheduler
from ...core.logger import logger


# Pydantic models for API requests/responses
class ProcessingRequest(BaseModel):
    """Request model for starting document processing."""
    target_date: Optional[date] = Field(None, description="Date to process documents for (defaults to yesterday)")
    force_reprocess: bool = Field(False, description="Whether to reprocess existing documents")
    max_documents: Optional[int] = Field(None, description="Maximum number of documents to process")


class BulkProcessingRequest(BaseModel):
    """Request model for bulk document processing."""
    start_date: date = Field(..., description="Start date for bulk processing")
    end_date: date = Field(..., description="End date for bulk processing")
    batch_size: int = Field(100, ge=1, le=1000, description="Number of documents per batch")
    force_reprocess: bool = Field(False, description="Whether to reprocess existing documents")


class ProcessingStatusResponse(BaseModel):
    """Response model for processing status."""
    is_running: bool
    current_task: Optional[Dict[str, Any]] = None
    queue_size: int
    last_completed: Optional[Dict[str, Any]] = None


class ProcessingSummaryResponse(BaseModel):
    """Response model for processing summary."""
    target_date: str
    total_discovered: int
    total_processed: int
    successful: int
    failed: int
    skipped: int
    processing_time_seconds: float
    success_rate: float


class TaskResponse(BaseModel):
    """Response model for scheduled tasks."""
    task_id: str
    task_type: str
    status: str
    target_date: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# Router setup
router = APIRouter(prefix="/document-processing", tags=["Document Processing"])


@router.post("/start-daily", response_model=ProcessingSummaryResponse)
async def start_daily_processing(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(async_get_db),
    current_user: User = Depends(get_current_user)
) -> ProcessingSummaryResponse:
    """
    Start daily document processing for a specific date.
    
    This endpoint triggers the document processing pipeline to fetch and process
    documents from the Supreme Court website for the specified date.
    
    **Required permissions**: Admin or Document Manager role
    """
    try:
        # Set target date (default to yesterday if not specified)
        target_date = request.target_date or (date.today() - timedelta(days=1))
        
        logger.info(f"Starting daily processing for {target_date} (user: {current_user.id})")
        
        # Start processing in background
        processing_result = await document_processor.process_daily_documents(
            db=db,
            target_date=target_date,
            force_reprocess=request.force_reprocess
        )
        
        return ProcessingSummaryResponse(**processing_result)
        
    except Exception as e:
        logger.error(f"Failed to start daily processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start processing: {str(e)}"
        )


@router.post("/start-bulk", response_model=Dict[str, str])
async def start_bulk_processing(
    request: BulkProcessingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(async_get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Start bulk document processing for a date range.
    
    This endpoint triggers bulk processing of documents for multiple dates.
    The processing runs in the background and can be monitored via status endpoints.
    
    **Required permissions**: Admin role
    """
    try:
        # Validate date range
        if request.start_date > request.end_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Start date must be before or equal to end date"
            )
        
        # Check if date range is not too large (max 30 days)
        days_diff = (request.end_date - request.start_date).days
        if days_diff > 30:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Date range cannot exceed 30 days for bulk processing"
            )
        
        logger.info(f"Starting bulk processing from {request.start_date} to {request.end_date} (user: {current_user.id})")
        
        # Schedule bulk processing task
        task_id = await document_scheduler.schedule_bulk_processing(
            start_date=request.start_date,
            end_date=request.end_date,
            batch_size=request.batch_size,
            force_reprocess=request.force_reprocess,
            user_id=current_user.id
        )
        
        return {
            "task_id": task_id,
            "status": "scheduled",
            "message": f"Bulk processing scheduled for {days_diff + 1} days"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start bulk processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start bulk processing: {str(e)}"
        )


@router.get("/status", response_model=ProcessingStatusResponse)
async def get_processing_status(
    db: AsyncSession = Depends(async_get_db),
    current_user: User = Depends(get_optional_user)
) -> ProcessingStatusResponse:
    """
    Get current document processing status.
    
    Returns information about:
    - Whether processing is currently running
    - Current task details
    - Queue size
    - Last completed task
    """
    try:
        status_info = await document_scheduler.get_status()
        
        return ProcessingStatusResponse(
            is_running=status_info["is_running"],
            current_task=status_info.get("current_task"),
            queue_size=status_info["queue_size"],
            last_completed=status_info.get("last_completed")
        )
        
    except Exception as e:
        logger.error(f"Failed to get processing status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )


@router.get("/tasks", response_model=List[TaskResponse])
async def list_processing_tasks(
    limit: int = Query(20, ge=1, le=100, description="Number of tasks to return"),
    offset: int = Query(0, ge=0, description="Number of tasks to skip"),
    status_filter: Optional[str] = Query(None, description="Filter by task status"),
    db: AsyncSession = Depends(async_get_db),
    current_user: User = Depends(get_current_user)
) -> List[TaskResponse]:
    """
    List processing tasks with optional filtering.
    
    Returns a list of scheduled and completed processing tasks.
    """
    try:
        tasks = await document_scheduler.list_tasks(
            limit=limit,
            offset=offset,
            status_filter=status_filter
        )
        
        return [
            TaskResponse(
                task_id=task.task_id,
                task_type=task.task_type,
                status=task.status.value,
                target_date=task.target_date.isoformat(),
                created_at=task.created_at.isoformat(),
                started_at=task.started_at.isoformat() if task.started_at else None,
                completed_at=task.completed_at.isoformat() if task.completed_at else None,
                result=task.result,
                error_message=task.error_message
            )
            for task in tasks
        ]
        
    except Exception as e:
        logger.error(f"Failed to list tasks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tasks: {str(e)}"
        )


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_details(
    task_id: str,
    db: AsyncSession = Depends(async_get_db),
    current_user: User = Depends(get_current_user)
) -> TaskResponse:
    """
    Get details for a specific processing task.
    """
    try:
        task = await document_scheduler.get_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
        
        return TaskResponse(
            task_id=task.task_id,
            task_type=task.task_type,
            status=task.status.value,
            target_date=task.target_date.isoformat(),
            created_at=task.created_at.isoformat(),
            started_at=task.started_at.isoformat() if task.started_at else None,
            completed_at=task.completed_at.isoformat() if task.completed_at else None,
            result=task.result,
            error_message=task.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task details: {str(e)}"
        )


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    db: AsyncSession = Depends(async_get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Cancel a pending or running processing task.
    
    **Required permissions**: Admin role or task owner
    """
    try:
        success = await document_scheduler.cancel_task(task_id, current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found or cannot be cancelled"
            )
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}"
        )


@router.get("/statistics")
async def get_processing_statistics(
    days_back: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    db: AsyncSession = Depends(async_get_db),
    current_user: User = Depends(get_optional_user)
) -> Dict[str, Any]:
    """
    Get processing statistics for the specified time period.
    
    Returns statistics about document processing performance,
    success rates, and trends over time.
    """
    try:
        stats = await document_scheduler.get_statistics(days_back=days_back)
        
        return {
            "period": {
                "days_back": days_back,
                "start_date": (date.today() - timedelta(days=days_back)).isoformat(),
                "end_date": date.today().isoformat()
            },
            "totals": stats.get("totals", {}),
            "success_rates": stats.get("success_rates", {}),
            "daily_breakdown": stats.get("daily_breakdown", []),
            "performance_metrics": stats.get("performance_metrics", {})
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.post("/schedule-daily")
async def schedule_daily_processing(
    time_str: str = Query(..., description="Time to run daily processing (HH:MM format)"),
    timezone: str = Query("Asia/Jakarta", description="Timezone for scheduling"),
    enabled: bool = Query(True, description="Whether to enable the scheduled task"),
    db: AsyncSession = Depends(async_get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Schedule automatic daily document processing.
    
    Sets up a recurring task to automatically process documents
    from the Supreme Court website every day at the specified time.
    
    **Required permissions**: Admin role
    """
    try:
        # Validate time format
        try:
            datetime.strptime(time_str, "%H:%M")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid time format. Use HH:MM format (e.g., 02:00)"
            )
        
        # Schedule the recurring task
        task_id = await document_scheduler.schedule_daily_recurring(
            time_str=time_str,
            timezone=timezone,
            enabled=enabled,
            user_id=current_user.id
        )
        
        return {
            "task_id": task_id,
            "status": "scheduled" if enabled else "disabled",
            "message": f"Daily processing {'scheduled' if enabled else 'disabled'} at {time_str} {timezone}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to schedule daily processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to schedule daily processing: {str(e)}"
        )


# Integration endpoint for your team's existing workflow
@router.post("/integrate-team-workflow")
async def integrate_team_workflow(
    workflow_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(async_get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Integration endpoint for your team's existing document processing workflow.
    
    This endpoint is designed to receive data from your team's existing
    scraping and processing logic and integrate it into the backend server.
    
    Expected workflow_data format:
    {
        "documents": [
            {
                "case_number": "...",
                "title": "...",
                "court_name": "...",
                "document_url": "...",
                "metadata": {...}
            }
        ],
        "source": "team_workflow",
        "processing_date": "2024-01-15"
    }
    """
    try:
        documents = workflow_data.get("documents", [])
        source = workflow_data.get("source", "team_workflow")
        processing_date = workflow_data.get("processing_date")
        
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents provided in workflow_data"
            )
        
        logger.info(f"Integrating {len(documents)} documents from team workflow (user: {current_user.id})")
        
        # Process the documents using the existing pipeline
        # This allows your team's workflow to seamlessly integrate
        task_id = await document_scheduler.schedule_team_integration(
            documents=documents,
            source=source,
            user_id=current_user.id,
            metadata=workflow_data
        )
        
        return {
            "task_id": task_id,
            "status": "scheduled",
            "documents_count": len(documents),
            "message": "Team workflow integration scheduled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to integrate team workflow: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to integrate workflow: {str(e)}"
        )
