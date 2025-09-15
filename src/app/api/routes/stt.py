import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.database import async_get_db
from app.models.stt import STTJobStatus as STTJobStatusModel
from app.schemas.stt import (
    STTRequest,
    STTResponse,
    STTJobStatusResponse,
    STTJobList,
    HealthCheckResponse,
    JobStatus as STTJobStatus
)
from app.services.stt import TranscriptionProcessor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stt", tags=["Speech-to-Text"])

# Global transcription processor instance
transcription_processor = TranscriptionProcessor()


async def get_transcription_processor() -> TranscriptionProcessor:
    """Dependency to get transcription processor."""
    return transcription_processor


@router.post("/transcribe", response_model=STTJobStatusResponse)
async def create_transcription_job(
    request: STTRequest,
    background_tasks: BackgroundTasks,
    processor: TranscriptionProcessor = Depends(get_transcription_processor),
    db_session: AsyncSession = Depends(async_get_db)
):
    """
    Create a new transcription job for batch processing.
    
    This endpoint accepts audio URIs (GCS) or file uploads and starts
    a background transcription job using Google Cloud Speech-to-Text.
    """
    try:
        logger.info(f"Creating transcription job for URI: {request.audio_uri}")
        
        # Create job in database
        stt_job = await processor.create_job(
            audio_uri=request.audio_uri,
            language_code=request.language_code,
            enable_diarization=request.enable_diarization,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
            enable_word_time_offsets=request.enable_word_time_offsets,
            engine=request.engine.value,
            output_format=request.output_format.value,
            db_session=db_session
        )
        
        # Start background processing
        background_tasks.add_task(
            process_transcription_job,
            stt_job.job_id,
            processor
        )
        
        logger.info(f"Started background transcription for job: {stt_job.job_id}")
        
        return STTJobStatusResponse(
            job_id=stt_job.job_id,
            status=stt_job.status,
            progress=0.0,
            created_at=stt_job.created_at,
            started_at=stt_job.started_at,
            completed_at=stt_job.completed_at,
            error_message=stt_job.error_message,
            execution_time=stt_job.execution_time
        )
        
    except Exception as e:
        logger.error(f"Failed to create transcription job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create transcription job: {str(e)}"
        )


@router.post("/upload-and-transcribe", response_model=STTJobStatusResponse)
async def upload_and_transcribe(
    file: UploadFile = File(...),
    language_code: str = "id-ID",
    enable_diarization: bool = True,
    min_speakers: int = 1,
    max_speakers: int = 6,
    enable_word_time_offsets: bool = True,
    engine: str = "gcp_stt_v2",
    output_format: str = "json",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    processor: TranscriptionProcessor = Depends(get_transcription_processor),
    db_session: AsyncSession = Depends(async_get_db)
):
    """
    Upload an audio file and start transcription.
    
    This endpoint accepts direct file uploads, saves them temporarily,
    and starts a transcription job.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an audio file"
            )
        
        # Check file size (max 100MB by default)
        import os
        from ...core.config import settings
        
        max_size = settings.AUDIO_MAX_SIZE_MB * 1024 * 1024
        file_size = 0
        
        # Read file content to check size
        content = await file.read()
        file_size = len(content)
        
        if file_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {settings.AUDIO_MAX_SIZE_MB}MB"
            )
        
        # Save file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as temp_file:
            temp_file.write(content)
        
        logger.info(f"Uploaded file {file.filename} ({file_size} bytes)")
        
        # Create transcription request
        request = STTRequest(
            audio_uri=temp_path,  # Use local path, will be uploaded to GCS
            language_code=language_code,
            enable_diarization=enable_diarization,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            enable_word_time_offsets=enable_word_time_offsets,
            engine=engine,
            output_format=output_format
        )
        
        # Create job
        stt_job = await processor.create_job(
            audio_uri=request.audio_uri,
            language_code=request.language_code,
            enable_diarization=request.enable_diarization,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
            enable_word_time_offsets=request.enable_word_time_offsets,
            engine=request.engine.value,
            output_format=request.output_format.value,
            db_session=db_session
        )
        
        # Start background processing
        background_tasks.add_task(
            process_transcription_job,
            stt_job.job_id,
            processor
        )
        
        return STTJobStatusResponse(
            job_id=stt_job.job_id,
            status=stt_job.status,
            progress=0.0,
            created_at=stt_job.created_at,
            started_at=stt_job.started_at,
            completed_at=stt_job.completed_at,
            error_message=stt_job.error_message,
            execution_time=stt_job.execution_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload and transcribe: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload and transcribe: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=STTJobStatusResponse)
async def get_job_status(
    job_id: str,
    processor: TranscriptionProcessor = Depends(get_transcription_processor),
    db_session: AsyncSession = Depends(async_get_db)
):
    """
    Get the status of a transcription job.
    
    Returns current status, progress, and metadata for the specified job.
    """
    try:
        stt_job = await processor.get_job_status(job_id, db_session)
        
        if not stt_job:
            raise HTTPException(
                status_code=404,
                detail=f"Job not found: {job_id}"
            )
        
        # Calculate progress based on status
        progress = 0.0
        if stt_job.status == STTJobStatus.PROCESSING:
            progress = 0.5
        elif stt_job.status == STTJobStatus.COMPLETED:
            progress = 1.0
        elif stt_job.status == STTJobStatus.FAILED:
            progress = 0.0
        
        return STTJobStatusResponse(
            job_id=stt_job.job_id,
            status=stt_job.status,
            progress=progress,
            created_at=stt_job.created_at,
            started_at=stt_job.started_at,
            completed_at=stt_job.completed_at,
            error_message=stt_job.error_message,
            execution_time=stt_job.execution_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.get("/jobs/{job_id}/results", response_model=STTResponse)
async def get_job_results(
    job_id: str,
    processor: TranscriptionProcessor = Depends(get_transcription_processor),
    db_session: AsyncSession = Depends(async_get_db)
):
    """
    Get the complete results of a completed transcription job.
    
    Returns full transcript, segments, word timestamps, and metadata.
    """
    try:
        results = await processor.get_job_results(job_id, db_session)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"Job not found or not completed: {job_id}"
            )
        
        return STTResponse(**results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job results for {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job results: {str(e)}"
        )


@router.get("/jobs", response_model=STTJobList)
async def list_jobs(
    skip: int = Query(0, ge=0, description="Number of jobs to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of jobs to return"),
    status: Optional[STTJobStatus] = Query(None, description="Filter by job status"),
    processor: TranscriptionProcessor = Depends(get_transcription_processor),
    db_session: AsyncSession = Depends(async_get_db)
):
    """
    List transcription jobs with pagination and optional status filtering.
    
    Returns a paginated list of jobs with their current status and metadata.
    """
    try:
        results = await processor.list_jobs(
            db_session=db_session,
            skip=skip,
            limit=limit,
            status_filter=status
        )
        
        return STTJobList(**results)
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list jobs: {str(e)}"
        )


@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    processor: TranscriptionProcessor = Depends(get_transcription_processor),
    db_session: AsyncSession = Depends(async_get_db)
):
    """
    Cancel a pending or processing transcription job.
    
    Note: Once transcription has started in GCP, it cannot be cancelled,
    but the job status will be updated to failed.
    """
    try:
        stt_job = await processor.get_job_status(job_id, db_session)
        
        if not stt_job:
            raise HTTPException(
                status_code=404,
                detail=f"Job not found: {job_id}"
            )
        
        if stt_job.status in [STTJobStatus.COMPLETED, STTJobStatus.FAILED]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job in status: {stt_job.status}"
            )
        
        # Update job status to cancelled (failed with specific message)
        stt_job.status = STTJobStatus.FAILED
        stt_job.error_message = "Job cancelled by user"
        stt_job.completed_at = datetime.now()
        
        await db_session.commit()
        
        logger.info(f"Cancelled job: {job_id}")
        
        return {"message": f"Job {job_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel job: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    processor: TranscriptionProcessor = Depends(get_transcription_processor),
    db_session: AsyncSession = Depends(async_get_db)
):
    """
    Check the health of the STT service and its dependencies.
    
    Returns status of GCP services, database connection, and active jobs count.
    """
    try:
        # Get health status from processor
        health_status = await processor.health_check()
        
        # Count active jobs
        from sqlalchemy import select, func
        from app.models.stt import STTJob
        
        active_jobs_result = await db_session.execute(
            select(func.count(STTJob.job_id)).where(
                STTJob.status.in_([STTJobStatus.PENDING, STTJobStatus.PROCESSING])
            )
        )
        active_jobs = active_jobs_result.scalar() or 0
        
        health_status["active_jobs"] = active_jobs
        
        return HealthCheckResponse(**health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            service="stt",
            status="unhealthy",
            timestamp=datetime.now(),
            gcp_connection=False,
            gcs_connection=False,
            database_connection=False,
            active_jobs=0
        )


async def process_transcription_job(job_id: str, processor: TranscriptionProcessor):
    """
    Background task to process transcription job.
    
    This function runs asynchronously to avoid blocking the API response.
    """
    try:
        logger.info(f"Starting background processing for job: {job_id}")
        
        # Get database session for background task
        from ...core.db.database import local_session
        
        async with local_session() as db_session:
            await processor.process_job(job_id, db_session)
            
        logger.info(f"Completed background processing for job: {job_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for job {job_id}: {e}")
        # Error handling is done in processor.process_job
