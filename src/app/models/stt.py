import uuid as uuid_pkg
from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import DateTime, JSON, String, Text, Float, Integer, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from ..core.db.database import Base


class STTEngine(str, Enum):
    """Available STT engines."""
    GCP_STT_V2 = "gcp_stt_v2"
    WHISPER = "whisper"


class STTJobStatus(str, Enum):
    """STT Job status."""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"


class OutputFormat(str, Enum):
    """Available output formats."""
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"


class STTJob(Base):
    """STT Job model for tracking transcription jobs."""
    __tablename__ = "stt_jobs"

    # Primary identifier (required)
    job_id: Mapped[str] = mapped_column(String(50), primary_key=True, unique=True)
    
    # Required fields first
    source_uri: Mapped[str] = mapped_column(String(512))  # GCS URI or file path
    
    # Optional fields with defaults
    uuid: Mapped[uuid_pkg.UUID] = mapped_column(default_factory=uuid_pkg.uuid4, unique=True, index=True)
    engine: Mapped[STTEngine] = mapped_column(String(20), default=STTEngine.GCP_STT_V2)
    language_code: Mapped[str] = mapped_column(String(10), default="id-ID")
    status: Mapped[STTJobStatus] = mapped_column(String(20), default=STTJobStatus.PENDING, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default_factory=lambda: datetime.now(UTC))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    enable_diarization: Mapped[bool] = mapped_column(Boolean, default=True)
    min_speakers: Mapped[int] = mapped_column(Integer, default=1)
    max_speakers: Mapped[int] = mapped_column(Integer, default=6)
    enable_word_time_offsets: Mapped[bool] = mapped_column(Boolean, default=True)
    output_format: Mapped[OutputFormat] = mapped_column(String(10), default=OutputFormat.JSON)
    transcript: Mapped[str | None] = mapped_column(Text, default=None)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    execution_time: Mapped[float | None] = mapped_column(Float, default=None)  # seconds
    storage_url: Mapped[str | None] = mapped_column(String(512), default=None)  # GCS URL for results
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    job_metadata: Mapped[dict | None] = mapped_column(JSON, default=None)  # Renamed from 'metadata' to avoid conflict


class TranscriptSegment(Base):
    """Individual transcript segments with speaker and timing information."""
    __tablename__ = "transcript_segments"
    
    # Primary identifier (auto-generated)
    id: Mapped[int] = mapped_column(autoincrement=True, nullable=False, unique=True, primary_key=True, init=False)
    
    # Required fields
    job_id: Mapped[str] = mapped_column(String(50), index=True)  # FK to stt_jobs.job_id
    start_time: Mapped[float] = mapped_column(Float)  # seconds
    end_time: Mapped[float] = mapped_column(Float)    # seconds
    text: Mapped[str] = mapped_column(Text)
    
    # Optional fields with defaults
    uuid: Mapped[uuid_pkg.UUID] = mapped_column(default_factory=uuid_pkg.uuid4, unique=True, index=True)
    speaker: Mapped[str | None] = mapped_column(String(50), default=None)  # Speaker_1, Speaker_2, etc.
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default_factory=lambda: datetime.now(UTC))


class WordTimestamp(Base):
    """Word-level timestamps for precise transcription."""
    __tablename__ = "word_timestamps"
    
    # Primary identifier (auto-generated)
    id: Mapped[int] = mapped_column(autoincrement=True, nullable=False, unique=True, primary_key=True, init=False)
    
    # Required fields
    job_id: Mapped[str] = mapped_column(String(50), index=True)  # FK to stt_jobs.job_id
    word: Mapped[str] = mapped_column(String(100))
    start_time: Mapped[float] = mapped_column(Float)  # seconds
    end_time: Mapped[float] = mapped_column(Float)    # seconds
    
    # Optional fields with defaults
    uuid: Mapped[uuid_pkg.UUID] = mapped_column(default_factory=uuid_pkg.uuid4, unique=True, index=True)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    speaker: Mapped[str | None] = mapped_column(String(50), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default_factory=lambda: datetime.now(UTC))
