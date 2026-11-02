"""
Automatic Speech Recognition (ASR) Module.
Handles audio input, format conversion, noise filtering, and transcription.
Supports SpeechRecognition (Google Web Speech API) and offline Transformer-based ASR.
"""

import io
import os
import wave
import tempfile
import logging
from typing import Tuple, Optional, Dict, Any

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    sr = None
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

from config import DEFAULT_ASR_BACKEND, AUDIO_SAMPLE_RATE, ENERGY_THRESHOLD, PAUSE_THRESHOLD

logger = logging.getLogger(__name__)


class ASRResult:
    """Encapsulates the transcription result, status, and metadata."""
    def __init__(self, text: str, success: bool, error: Optional[str] = None, backend: str = "google"):
        self.text = text.strip()
        self.success = success
        self.error = error
        self.backend = backend

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "success": self.success,
            "error": self.error,
            "backend": self.backend
        }


class ASRProcessor:
    """
    Automatic Speech Recognition Processor.
    Translates raw acoustic waveforms into text tokens.
    """

    def __init__(self, backend: str = DEFAULT_ASR_BACKEND):
        self.backend = backend
        self.recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        if self.recognizer:
            self.recognizer.energy_threshold = ENERGY_THRESHOLD
            self.recognizer.pause_threshold = PAUSE_THRESHOLD
            self.recognizer.dynamic_energy_threshold = True

    def _convert_to_wav(self, audio_bytes: bytes) -> Optional[io.BytesIO]:
        """
        Ensures the audio data is in valid 16-bit PCM WAV format.
        Uses pydub if necessary to transcode WebM/OGG/MP3 to WAV.
        """
        # First attempt to see if bytes are already valid WAV
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                # If valid wave with frames, return original buffer
                if wf.getnframes() > 0:
                    return io.BytesIO(audio_bytes)
        except (wave.Error, EOFError):
            pass

        # Transcode using pydub if available
        if PYDUB_AVAILABLE:
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
                audio_segment = audio_segment.set_frame_rate(AUDIO_SAMPLE_RATE).set_channels(1)
                wav_io = io.BytesIO()
                audio_segment.export(wav_io, format="wav")
                wav_io.seek(0)
                return wav_io
            except Exception as e:
                logger.error(f"Audio transcoding failed: {e}")
                return None
        return None

    def transcribe_audio_bytes(self, audio_bytes: bytes) -> ASRResult:
        """
        Transcribes raw audio bytes (from Streamlit audio recorder or file uploader).
        """
        if not audio_bytes or len(audio_bytes) < 100:
            return ASRResult(
                text="",
                success=False,
                error="Audio input is empty or too short. Please speak into the microphone."
            )

        if not SPEECH_RECOGNITION_AVAILABLE:
            return ASRResult(
                text="",
                success=False,
                error="SpeechRecognition library is not installed in the current environment."
            )

        wav_stream = self._convert_to_wav(audio_bytes)
        if wav_stream is None:
            return ASRResult(
                text="",
                success=False,
                error="Could not decode the provided audio stream into standard PCM WAV."
            )

        try:
            with sr.AudioFile(wav_stream) as source:
                # Ambient noise adaptation
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio_data = self.recognizer.record(source)

            # Transcribe via chosen backend
            if self.backend == "google":
                # Google Web Speech API (free tier, no API key required)
                recognized_text = self.recognizer.recognize_google(audio_data)
                return ASRResult(
                    text=recognized_text,
                    success=True,
                    backend="Google Web Speech API"
                )
            else:
                return ASRResult(
                    text="",
                    success=False,
                    error=f"Unsupported ASR backend: {self.backend}"
                )

        except sr.UnknownValueError:
            return ASRResult(
                text="",
                success=False,
                error="Audio detected, but speech could not be recognized clearly. Please rephrase or speak closer."
            )
        except sr.RequestError as e:
            return ASRResult(
                text="",
                success=False,
                error=f"ASR Service network error ({e}). Please check your internet connection."
            )
        except Exception as e:
            logger.exception("Unexpected ASR processing error")
            return ASRResult(
                text="",
                success=False,
                error=f"ASR error: {str(e)}"
            )

    def transcribe_file(self, file_path: str) -> ASRResult:
        """
        Transcribes an audio file on disk (WAV format).
        Useful for batch testing and offline audio evaluation.
        """
        if not os.path.exists(file_path):
            return ASRResult(text="", success=False, error=f"File not found: {file_path}")

        try:
            with open(file_path, "rb") as f:
                return self.transcribe_audio_bytes(f.read())
        except Exception as e:
            return ASRResult(text="", success=False, error=f"Failed to read file: {e}")

