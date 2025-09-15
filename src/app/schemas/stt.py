from datetime import datetime
from typing import List, Optional
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class STTEngineType(str, Enum):
    """Available STT engines for schemas."""
    GCP_STT_V2 = "gcp_stt_v2"
    WHISPER = "whisper"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"


class FileFormat(str, Enum):
    """Available output formats for schemas."""
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"


class STTRequest(BaseModel):
    """Request schema for batch transcription."""
    
    audio_uri: str = Field(
        ..., 
        description="GCS URI (gs://bucket/file.wav) or file upload URL",
        min_length=1,
        max_length=512
    )
    language_code: str = Field(
        default="id-ID",
        description="Language code for transcription",
        pattern=r"^[a-z]{2}-[A-Z]{2}$"
    )
    enable_diarization: bool = Field(
        default=True,
        description="Enable speaker diarization"
    )
    min_speakers: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum number of speakers"
    )
    max_speakers: int = Field(
        default=6,
        ge=1,
        le=10,
        description="Maximum number of speakers"
    )
    enable_word_time_offsets: bool = Field(
        default=True,
        description="Enable word-level timestamps"
    )
    engine: STTEngineType = Field(
        default=STTEngineType.GCP_STT_V2,
        description="STT engine to use"
    )
    output_format: FileFormat = Field(
        default=FileFormat.JSON,
        description="Output format for transcript"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "audio_uri": "gs://courtsight-stt/sample.wav",
                "language_code": "id-ID",
                "enable_diarization": True,
                "min_speakers": 2,
                "max_speakers": 6,
                "enable_word_time_offsets": True,
                "engine": "gcp_stt_v2",
                "output_format": "json"
            }
        }


class TranscriptSegmentSchema(BaseModel):
    """Schema for transcript segments with speaker and timing info."""
    
    speaker: Optional[str] = Field(None, description="Speaker identifier (e.g., Speaker_1)")
    start_time: float = Field(..., description="Segment start time in seconds")
    end_time: float = Field(..., description="Segment end time in seconds")
    text: str = Field(..., description="Transcript text for this segment")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")

    class Config:
        json_schema_extra = {
            "example": {
                "speaker": "Speaker_1",
                "start_time": 0.0,
                "end_time": 5.2,
                "text": "Dengan ini saya buka sidang pengadilan negeri.",
                "confidence": 0.95
            }
        }


class WordTimestampSchema(BaseModel):
    """Schema for word-level timestamps."""
    
    word: str = Field(..., description="Individual word")
    start_time: float = Field(..., description="Word start time in seconds")
    end_time: float = Field(..., description="Word end time in seconds")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Word confidence score")
    speaker: Optional[str] = Field(None, description="Speaker for this word")

    class Config:
        json_schema_extra = {
            "example": {
                "word": "dengan",
                "start_time": 0.0,
                "end_time": 0.5,
                "confidence": 0.98,
                "speaker": "Speaker_1"
            }
        }


class STTResponse(BaseModel):
    """Response schema for completed transcription."""
    
    job_id: str = Field(..., description="Unique job identifier")
    transcript: str = Field(..., description="Full transcript text")
    segments: List[TranscriptSegmentSchema] = Field(
        default_factory=list,
        description="Transcript segments with speaker diarization"
    )
    words: List[WordTimestampSchema] = Field(
        default_factory=list,
        description="Word-level timestamps"
    )
    storage_url: Optional[str] = Field(None, description="GCS URL for stored transcript")
    execution_time: float = Field(..., description="Processing time in seconds")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence score")
    status: JobStatus = Field(..., description="Job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "stt_12345",
                "transcript": "Dengan ini saya buka sidang pengadilan negeri Jakarta Selatan.",
                "segments": [
                    {
                        "speaker": "Speaker_1",
                        "start_time": 0.0,
                        "end_time": 5.2,
                        "text": "Dengan ini saya buka sidang pengadilan negeri Jakarta Selatan.",
                        "confidence": 0.95
                    }
                ],
                "words": [
                    {
                        "word": "dengan",
                        "start_time": 0.0,
                        "end_time": 0.5,
                        "confidence": 0.98,
                        "speaker": "Speaker_1"
                    }
                ],
                "storage_url": "gs://courtsight-stt/transcripts/stt_12345.json",
                "execution_time": 2.5,
                "confidence": 0.94,
                "status": "completed",
                "created_at": "2025-09-16T10:00:00Z",
                "completed_at": "2025-09-16T10:00:02Z"
            }
        }


class STTJobStatusResponse(BaseModel):
    """Response schema for job status check."""
    
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Processing progress (0-1)")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time: Optional[float] = Field(None, description="Processing time in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "stt_12345",
                "status": "processing",
                "progress": 0.75,
                "created_at": "2025-09-16T10:00:00Z",
                "started_at": "2025-09-16T10:00:01Z",
                "completed_at": None,
                "error_message": None,
                "execution_time": None
            }
        }


class STTJobList(BaseModel):
    """Response schema for listing STT jobs."""
    
    jobs: List[STTJobStatusResponse] = Field(..., description="List of STT jobs")
    total_count: int = Field(..., description="Total number of jobs")
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=10, description="Number of jobs per page")

    class Config:
        json_schema_extra = {
            "example": {
                "jobs": [
                    {
                        "job_id": "stt_12345",
                        "status": "completed",
                        "progress": 1.0,
                        "created_at": "2025-09-16T10:00:00Z",
                        "started_at": "2025-09-16T10:00:01Z",
                        "completed_at": "2025-09-16T10:00:03Z",
                        "error_message": None,
                        "execution_time": 2.5
                    }
                ],
                "total_count": 1,
                "page": 1,
                "page_size": 10
            }
        }


class HealthCheckResponse(BaseModel):
    """STT service health check response."""
    
    service: str = Field(default="stt", description="Service name")
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    gcp_connection: bool = Field(..., description="GCP Speech-to-Text API connectivity")
    gcs_connection: bool = Field(..., description="Google Cloud Storage connectivity")
    database_connection: bool = Field(..., description="Database connectivity")
    active_jobs: int = Field(..., description="Number of active transcription jobs")

    class Config:
        json_schema_extra = {
            "example": {
                "service": "stt",
                "status": "healthy",
                "timestamp": "2025-09-16T10:00:00Z",
                "gcp_connection": True,
                "gcs_connection": True,
                "database_connection": True,
                "active_jobs": 3
            }
        }
