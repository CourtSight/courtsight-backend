import logging
import uuid
from datetime import datetime, UTC
from typing import Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.stt import STTJob, TranscriptSegment, WordTimestamp, STTJobStatus
from .gcp_stt_service import GCPSTTService

logger = logging.getLogger(__name__)


class TranscriptionProcessor:
    """Handles transcription job processing and database operations."""

    def __init__(self):
        """Initialize transcription processor."""
        self.gcp_service = GCPSTTService()

    async def create_job(
        self,
        audio_uri: str,
        language_code: str = "id-ID",
        enable_diarization: bool = True,
        min_speakers: int = 1,
        max_speakers: int = 6,
        enable_word_time_offsets: bool = True,
        engine: str = "gcp_stt_v2",
        output_format: str = "json",
    ) -> STTJob:
        """
        Create a new STT job in the database.
        
        Args:
            audio_uri: URI to audio file
            language_code: Language code for transcription
            enable_diarization: Enable speaker diarization
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            enable_word_time_offsets: Enable word-level timestamps
            engine: STT engine to use
            output_format: Output format
            db_session: Database session
            
        Returns:
            Created STT job
        """
        try:
            # Generate unique job ID
            job_id = f"stt_{uuid.uuid4().hex[:8]}"
            
            # Create STT job record
            stt_job = STTJob(
                job_id=job_id,
                source_uri=audio_uri,
                engine=engine,
                language_code=language_code,
                status=STTJobStatus.PENDING,
                enable_diarization=enable_diarization,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                enable_word_time_offsets=enable_word_time_offsets,
                output_format=output_format,
                created_at=datetime.now(UTC)
            )
            
            db_session.add(stt_job)
            await db_session.commit()
            await db_session.refresh(stt_job)
            
            logger.info(f"Created STT job: {job_id}")
            return stt_job
            
        except Exception as e:
            logger.error(f"Failed to create STT job: {e}")
            await db_session.rollback()
            raise

    async def process_job(self, job_id: str, db_session: AsyncSession) -> Dict:
        """
        Process an STT job asynchronously.
        
        Args:
            job_id: Job identifier
            db_session: Database session
            
        Returns:
            Processing result
        """
        try:
            # Get job from database
            from sqlalchemy import select
            
            result = await db_session.execute(
                select(STTJob).where(STTJob.job_id == job_id)
            )
            stt_job = result.scalar_one_or_none()
            
            if not stt_job:
                raise ValueError(f"STT job not found: {job_id}")
            
            # Update job status to processing
            stt_job.status = STTJobStatus.PROCESSING
            stt_job.started_at = datetime.now(UTC)
            await db_session.commit()
            
            logger.info(f"Starting processing for job: {job_id}")
            
            # Process transcription based on engine
            if stt_job.engine == "gcp_stt_v2":
                result_data = await self._process_with_gcp(stt_job)
            else:
                raise ValueError(f"Unsupported STT engine: {stt_job.engine}")
            
            # Save results to database
            await self._save_results_to_db(stt_job, result_data, db_session)
            
            # Update job status to completed
            stt_job.status = STTJobStatus.COMPLETED
            stt_job.completed_at = datetime.now(UTC)
            stt_job.execution_time = (stt_job.completed_at - stt_job.started_at).total_seconds()
            
            await db_session.commit()
            
            logger.info(f"Completed processing for job: {job_id}")
            return result_data
            
        except Exception as e:
            logger.error(f"Failed to process job {job_id}: {e}")
            
            # Update job status to failed
            try:
                result = await db_session.execute(
                    select(STTJob).where(STTJob.job_id == job_id)
                )
                stt_job = result.scalar_one_or_none()
                
                if stt_job:
                    stt_job.status = STTJobStatus.FAILED
                    stt_job.error_message = str(e)
                    stt_job.completed_at = datetime.now(UTC)
                    await db_session.commit()
                    
            except Exception as db_error:
                logger.error(f"Failed to update job status to failed: {db_error}")
            
            raise

    async def _process_with_gcp(self, stt_job: STTJob) -> Dict:
        """
        Process transcription using GCP Speech-to-Text.
        
        Args:
            stt_job: STT job object
            
        Returns:
            Transcription result
        """
        try:
            # If source_uri is not a GCS URI, upload it first
            audio_uri = stt_job.source_uri
            if not audio_uri.startswith("gs://"):
                audio_uri = await self.gcp_service.upload_to_gcs(
                    stt_job.source_uri, 
                    stt_job.job_id
                )
            
            # Perform transcription
            result_data = await self.gcp_service.transcribe_audio(
                audio_uri=audio_uri,
                job_id=stt_job.job_id,
                language_code=stt_job.language_code,
                enable_diarization=stt_job.enable_diarization,
                min_speakers=stt_job.min_speakers,
                max_speakers=stt_job.max_speakers,
                enable_word_time_offsets=stt_job.enable_word_time_offsets,
            )
            
            # Save transcript to GCS
            storage_url = await self.gcp_service.save_transcript_to_gcs(
                result_data, 
                stt_job.job_id
            )
            result_data["storage_url"] = storage_url
            
            return result_data
            
        except Exception as e:
            logger.error(f"GCP transcription failed for job {stt_job.job_id}: {e}")
            raise

    async def _save_results_to_db(
        self, 
        stt_job: STTJob, 
        result_data: Dict, 
        db_session: AsyncSession
    ):
        """
        Save transcription results to database.
        
        Args:
            stt_job: STT job object
            result_data: Transcription result data
            db_session: Database session
        """
        try:
            # Update job with results
            stt_job.transcript = result_data.get("transcript", "")
            stt_job.confidence = result_data.get("confidence")
            stt_job.storage_url = result_data.get("storage_url")
            stt_job.job_metadata = result_data.get("metadata", {})
            
            # Save transcript segments
            for segment_data in result_data.get("segments", []):
                segment = TranscriptSegment(
                    job_id=stt_job.job_id,
                    speaker=segment_data.get("speaker"),
                    start_time=segment_data.get("start_time", 0.0),
                    end_time=segment_data.get("end_time", 0.0),
                    text=segment_data.get("text", ""),
                    confidence=segment_data.get("confidence"),
                    created_at=datetime.now(UTC)
                )
                db_session.add(segment)
            
            # Save word timestamps if enabled
            if stt_job.enable_word_time_offsets:
                for word_data in result_data.get("words", []):
                    word_timestamp = WordTimestamp(
                        job_id=stt_job.job_id,
                        word=word_data.get("word", ""),
                        start_time=word_data.get("start_time", 0.0),
                        end_time=word_data.get("end_time", 0.0),
                        confidence=word_data.get("confidence"),
                        speaker=word_data.get("speaker"),
                        created_at=datetime.now(UTC)
                    )
                    db_session.add(word_timestamp)
            
            await db_session.commit()
            logger.info(f"Saved transcription results to database for job: {stt_job.job_id}")
            
        except Exception as e:
            logger.error(f"Failed to save results to database for job {stt_job.job_id}: {e}")
            await db_session.rollback()
            raise

    async def get_job_status(self, job_id: str, db_session: AsyncSession) -> Optional[STTJob]:
        """
        Get job status from database.
        
        Args:
            job_id: Job identifier
            db_session: Database session
            
        Returns:
            STT job object or None if not found
        """
        try:
            from sqlalchemy import select
            
            result = await db_session.execute(
                select(STTJob).where(STTJob.job_id == job_id)
            )
            return result.scalar_one_or_none()
            
        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {e}")
            raise

    async def get_job_results(self, job_id: str, db_session: AsyncSession) -> Optional[Dict]:
        """
        Get complete job results including segments and words.
        
        Args:
            job_id: Job identifier
            db_session: Database session
            
        Returns:
            Complete job results or None if not found
        """
        try:
            from sqlalchemy import select
            
            # Get job
            job_result = await db_session.execute(
                select(STTJob).where(STTJob.job_id == job_id)
            )
            stt_job = job_result.scalar_one_or_none()
            
            if not stt_job or stt_job.status != STTJobStatus.COMPLETED:
                return None
            
            # Get segments
            segments_result = await db_session.execute(
                select(TranscriptSegment).where(TranscriptSegment.job_id == job_id)
                .order_by(TranscriptSegment.start_time)
            )
            segments = segments_result.scalars().all()
            
            # Get words
            words_result = await db_session.execute(
                select(WordTimestamp).where(WordTimestamp.job_id == job_id)
                .order_by(WordTimestamp.start_time)
            )
            words = words_result.scalars().all()
            
            # Format response
            return {
                "job_id": stt_job.job_id,
                "transcript": stt_job.transcript,
                "segments": [
                    {
                        "speaker": segment.speaker,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "text": segment.text,
                        "confidence": segment.confidence
                    }
                    for segment in segments
                ],
                "words": [
                    {
                        "word": word.word,
                        "start_time": word.start_time,
                        "end_time": word.end_time,
                        "confidence": word.confidence,
                        "speaker": word.speaker
                    }
                    for word in words
                ] if stt_job.enable_word_time_offsets else [],
                "storage_url": stt_job.storage_url,
                "execution_time": stt_job.execution_time,
                "confidence": stt_job.confidence,
                "status": stt_job.status,
                "created_at": stt_job.created_at,
                "completed_at": stt_job.completed_at
            }
            
        except Exception as e:
            logger.error(f"Failed to get job results for {job_id}: {e}")
            raise

    async def list_jobs(
        self, 
        db_session: AsyncSession,
        skip: int = 0,
        limit: int = 10,
        status_filter: Optional[STTJobStatus] = None
    ) -> Dict:
        """
        List STT jobs with pagination.
        
        Args:
            db_session: Database session
            skip: Number of jobs to skip
            limit: Maximum number of jobs to return
            status_filter: Filter by job status
            
        Returns:
            Paginated job list
        """
        try:
            from sqlalchemy import select, func
            
            # Build query
            query = select(STTJob)
            count_query = select(func.count(STTJob.job_id))
            
            if status_filter:
                query = query.where(STTJob.status == status_filter)
                count_query = count_query.where(STTJob.status == status_filter)
            
            # Get total count
            total_result = await db_session.execute(count_query)
            total_count = total_result.scalar()
            
            # Get jobs with pagination
            jobs_result = await db_session.execute(
                query.order_by(STTJob.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            jobs = jobs_result.scalars().all()
            
            return {
                "jobs": [
                    {
                        "job_id": job.job_id,
                        "status": job.status,
                        "progress": 1.0 if job.status == STTJobStatus.COMPLETED else 
                                  0.5 if job.status == STTJobStatus.PROCESSING else 0.0,
                        "created_at": job.created_at,
                        "started_at": job.started_at,
                        "completed_at": job.completed_at,
                        "error_message": job.error_message,
                        "execution_time": job.execution_time
                    }
                    for job in jobs
                ],
                "total_count": total_count,
                "page": (skip // limit) + 1,
                "page_size": limit
            }
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise

    async def health_check(self) -> Dict:
        """
        Check health of STT services.
        
        Returns:
            Health status dictionary
        """
        try:
            # Check GCP services
            gcp_health = await self.gcp_service.health_check()
            
            return {
                "service": "stt",
                "status": "healthy" if all(gcp_health.values()) else "degraded",
                "timestamp": datetime.now(UTC),
                "gcp_connection": gcp_health.get("gcp_speech_api", False),
                "gcs_connection": gcp_health.get("gcp_storage", False),
                "database_connection": True,  # Will be updated by caller
                "active_jobs": 0  # Will be updated by caller
            }
            
        except Exception as e:
            logger.error(f"STT health check failed: {e}")
            return {
                "service": "stt",
                "status": "unhealthy",
                "timestamp": datetime.now(UTC),
                "gcp_connection": False,
                "gcs_connection": False,
                "database_connection": False,
                "active_jobs": 0,
                "error": str(e)
            }
