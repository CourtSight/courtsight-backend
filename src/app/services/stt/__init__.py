"""STT service package for Speech-to-Text functionality."""

from .gcp_stt_service import GCPSTTService
from .transcription_processor import TranscriptionProcessor

__all__ = ["GCPSTTService", "TranscriptionProcessor"]
