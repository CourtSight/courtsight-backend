import asyncio
import logging
import os
import uuid
from datetime import datetime, UTC
from typing import Dict, List, Optional, Tuple

import aiofiles
from google.cloud import speech_v2 as speech
from google.cloud import storage
from google.oauth2 import service_account

from app.core.config import settings
from app.models.stt import STTEngine, STTJobStatus, OutputFormat

logger = logging.getLogger(__name__)


class GCPSTTService:
    """Google Cloud Platform Speech-to-Text service integration."""

    def __init__(self):
        """Initialize GCP STT service with authentication."""
        self.project_id = settings.GCP_PROJECT
        self.bucket_name = settings.GCS_BUCKET
        
        # Initialize GCP clients
        self._speech_client = None
        self._storage_client = None
        
        # Setup credentials if provided
        if settings.GOOGLE_APPLICATION_CREDENTIALS:
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    settings.GOOGLE_APPLICATION_CREDENTIALS
                )
                self._speech_client = speech.SpeechClient(credentials=credentials)
                self._storage_client = storage.Client(credentials=credentials)
                logger.info("GCP STT service initialized with service account credentials")
            except Exception as e:
                logger.error(f"Failed to initialize GCP credentials: {e}")
                # Fall back to default credentials
                self._init_default_clients()
        else:
            self._init_default_clients()

    def _init_default_clients(self):
        """Initialize GCP clients with default credentials."""
        try:
            self._speech_client = speech.SpeechClient()
            self._storage_client = storage.Client(project=self.project_id)
            logger.info("GCP STT service initialized with default credentials")
        except Exception as e:
            logger.error(f"Failed to initialize GCP clients: {e}")
            raise

    @property
    def speech_client(self) -> speech.SpeechClient:
        """Get Speech-to-Text client."""
        if self._speech_client is None:
            raise RuntimeError("GCP Speech client not initialized")
        return self._speech_client

    @property
    def storage_client(self) -> storage.Client:
        """Get Cloud Storage client."""
        if self._storage_client is None:
            raise RuntimeError("GCP Storage client not initialized")
        return self._storage_client

    async def upload_to_gcs(self, file_path: str, job_id: str) -> str:
        """
        Upload audio file to Google Cloud Storage.
        
        Args:
            file_path: Local path to audio file
            job_id: Unique job identifier
            
        Returns:
            GCS URI of uploaded file
        """
        try:
            # Generate GCS object name
            file_extension = os.path.splitext(file_path)[1]
            blob_name = f"audio/{job_id}{file_extension}"
            
            # Get bucket and upload
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            
            # Upload file asynchronously
            async with aiofiles.open(file_path, 'rb') as file:
                content = await file.read()
                
            # Run upload in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: blob.upload_from_string(content, content_type='audio/wav')
            )
            
            gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
            logger.info(f"Uploaded audio file to: {gcs_uri}")
            return gcs_uri
            
        except Exception as e:
            logger.error(f"Failed to upload file to GCS: {e}")
            raise

    async def transcribe_audio(
        self,
        audio_uri: str,
        job_id: str,
        language_code: str = "id-ID",
        enable_diarization: bool = True,
        min_speakers: int = 1,
        max_speakers: int = 6,
        enable_word_time_offsets: bool = True,
    ) -> Dict:
        """
        Transcribe audio using Google Cloud Speech-to-Text v2.
        
        Args:
            audio_uri: GCS URI of audio file
            job_id: Unique job identifier
            language_code: Language code for transcription
            enable_diarization: Enable speaker diarization
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            enable_word_time_offsets: Enable word-level timestamps
            
        Returns:
            Transcription result dictionary
        """
        try:
            # Configure recognition request
            config = speech.RecognitionConfig(
                auto_decoding_config=speech.AutoDetectDecodingConfig(),
                language_codes=[language_code],
                model="latest_long",  # Best for long-form audio
                features=speech.RecognitionFeatures(
                    enable_word_time_offsets=enable_word_time_offsets,
                    enable_word_confidence=True,
                    enable_automatic_punctuation=True,
                    enable_spoken_punctuation=False,
                    enable_spoken_emojis=False,
                ),
            )
            
            # Configure diarization if enabled
            if enable_diarization:
                config.features.diarization_config = speech.SpeakerDiarizationConfig(
                    enable_speaker_diarization=True,
                    min_speaker_count=min_speakers,
                    max_speaker_count=max_speakers,
                )

            # Create transcription request
            request = speech.BatchRecognizeRequest(
                recognizer=f"projects/{self.project_id}/locations/global/recognizers/_",
                config=config,
                files=[
                    speech.BatchRecognizeFileMetadata(
                        uri=audio_uri,
                    )
                ],
            )

            # Start transcription operation
            logger.info(f"Starting transcription for job {job_id} with URI: {audio_uri}")
            
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            operation = await loop.run_in_executor(
                None,
                lambda: self.speech_client.batch_recognize(request=request)
            )
            
            # Wait for operation to complete
            result = await loop.run_in_executor(None, operation.result)
            
            logger.info(f"Transcription completed for job {job_id}")
            return self._process_transcription_result(result, job_id)
            
        except Exception as e:
            logger.error(f"Transcription failed for job {job_id}: {e}")
            raise

    def _process_transcription_result(self, result, job_id: str) -> Dict:
        """
        Process GCP transcription result into structured format.
        
        Args:
            result: GCP transcription result
            job_id: Unique job identifier
            
        Returns:
            Processed transcription data
        """
        try:
            processed_result = {
                "job_id": job_id,
                "transcript": "",
                "segments": [],
                "words": [],
                "confidence": 0.0,
                "language_code": "",
                "metadata": {}
            }

            if not result.results:
                logger.warning(f"No transcription results for job {job_id}")
                return processed_result

            # Extract transcript and metadata
            all_words = []
            all_segments = []
            total_confidence = 0.0
            result_count = 0
            
            for batch_result in result.results.values():
                for transcript_result in batch_result.transcript.results:
                    if not transcript_result.alternatives:
                        continue
                        
                    alternative = transcript_result.alternatives[0]
                    
                    # Add to full transcript
                    processed_result["transcript"] += alternative.transcript + " "
                    total_confidence += alternative.confidence
                    result_count += 1
                    
                    # Extract language code
                    if transcript_result.language_code:
                        processed_result["language_code"] = transcript_result.language_code
                    
                    # Process words with timestamps
                    for word_info in alternative.words:
                        word_data = {
                            "word": word_info.word,
                            "start_time": self._duration_to_seconds(word_info.start_offset),
                            "end_time": self._duration_to_seconds(word_info.end_offset),
                            "confidence": word_info.confidence,
                            "speaker": getattr(word_info, 'speaker_label', None)
                        }
                        all_words.append(word_data)
                    
                    # Create segments from continuous speaker sections
                    # This is a simplified approach - in production, you'd want more sophisticated segmentation
                    if all_words:
                        segment_data = {
                            "speaker": getattr(alternative.words[0], 'speaker_label', 'Speaker_1') if alternative.words else 'Speaker_1',
                            "start_time": self._duration_to_seconds(alternative.words[0].start_offset) if alternative.words else 0.0,
                            "end_time": self._duration_to_seconds(alternative.words[-1].end_offset) if alternative.words else 0.0,
                            "text": alternative.transcript.strip(),
                            "confidence": alternative.confidence
                        }
                        all_segments.append(segment_data)

            # Calculate average confidence
            if result_count > 0:
                processed_result["confidence"] = total_confidence / result_count

            processed_result["transcript"] = processed_result["transcript"].strip()
            processed_result["segments"] = all_segments
            processed_result["words"] = all_words
            
            # Add metadata
            processed_result["metadata"] = {
                "total_words": len(all_words),
                "total_segments": len(all_segments),
                "processing_time": datetime.now(UTC).isoformat(),
                "engine": "gcp_stt_v2"
            }
            
            logger.info(f"Processed transcription result for job {job_id}: "
                       f"{len(all_words)} words, {len(all_segments)} segments")
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Failed to process transcription result for job {job_id}: {e}")
            raise

    def _duration_to_seconds(self, duration) -> float:
        """Convert protobuf Duration to seconds."""
        if duration is None:
            return 0.0
        return duration.total_seconds()

    async def save_transcript_to_gcs(self, transcript_data: Dict, job_id: str) -> str:
        """
        Save transcript data to Google Cloud Storage.
        
        Args:
            transcript_data: Processed transcript data
            job_id: Unique job identifier
            
        Returns:
            GCS URI of saved transcript
        """
        try:
            import json
            
            # Generate GCS object name
            blob_name = f"transcripts/{job_id}.json"
            
            # Get bucket and save
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            
            # Convert to JSON and upload
            json_data = json.dumps(transcript_data, indent=2, ensure_ascii=False)
            
            # Run upload in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: blob.upload_from_string(
                    json_data.encode('utf-8'),
                    content_type='application/json'
                )
            )
            
            gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
            logger.info(f"Saved transcript to: {gcs_uri}")
            return gcs_uri
            
        except Exception as e:
            logger.error(f"Failed to save transcript to GCS: {e}")
            raise

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of GCP services.
        
        Returns:
            Dictionary with service health status
        """
        health_status = {
            "gcp_speech_api": False,
            "gcp_storage": False
        }
        
        try:
            # Test Speech-to-Text API
            loop = asyncio.get_event_loop()
            recognizers = await loop.run_in_executor(
                None,
                lambda: self.speech_client.list_recognizers(
                    parent=f"projects/{self.project_id}/locations/global"
                )
            )
            health_status["gcp_speech_api"] = True
            logger.debug("GCP Speech-to-Text API health check passed")
            
        except Exception as e:
            logger.error(f"GCP Speech-to-Text API health check failed: {e}")

        try:
            # Test Cloud Storage
            bucket = self.storage_client.bucket(self.bucket_name)
            await loop.run_in_executor(None, bucket.reload)
            health_status["gcp_storage"] = True
            logger.debug("GCP Cloud Storage health check passed")
            
        except Exception as e:
            logger.error(f"GCP Cloud Storage health check failed: {e}")

        return health_status
