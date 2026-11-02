"""
Automated Test Suite for Educational Voice-Based Chatbot.
Verifies ASR handling, NLP Seq2Seq generation, syllabus mapping,
invalid inputs, and conversation state management.
"""

import os
import io
import wave
import struct
import pytest

from config import (
    DEFAULT_MODEL_NAME,
    FALLBACK_MODEL_NAME,
    SYLLABUS_INFO,
    SAMPLE_QUESTIONS
)
from asr_module import ASRProcessor, ASRResult
from chatbot_model import EducationalChatbotModel, ACADEMIC_KNOWLEDGE_BASE


# Helper function to generate dummy PCM WAV in memory
def generate_synthetic_wav(duration_sec: float = 1.0, sample_rate: int = 16000, silent: bool = False) -> bytes:
    """Generates synthetic 16-bit PCM WAV bytes for testing ASR input pipeline."""
    buf = io.BytesIO()
    num_samples = int(duration_sec * sample_rate)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)       # Mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sample_rate)
        if silent:
            data = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        else:
            # 440 Hz sine tone sample
            import math
            samples = [
                int(32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
                for i in range(num_samples)
            ]
            data = struct.pack(f"<{num_samples}h", *samples)
        wf.writeframes(data)
    buf.seek(0)
    return buf.read()


# ======================================================================
# 1. Configuration & Knowledge Base Tests
# ======================================================================
def test_config_integrity():
    """Verify configuration constants and syllabus structure."""
    assert "Unit 4" in SYLLABUS_INFO["units"]
    assert "Unit 5" in SYLLABUS_INFO["units"]
    assert len(SAMPLE_QUESTIONS) >= 5
    assert len(ACADEMIC_KNOWLEDGE_BASE) >= 4


def test_academic_knowledge_base_content():
    """Ensure key viva topics are present in the knowledge base."""
    assert "lstm vs gru" in ACADEMIC_KNOWLEDGE_BASE
    assert "attention mechanism" in ACADEMIC_KNOWLEDGE_BASE
    assert "encoder-decoder" in ACADEMIC_KNOWLEDGE_BASE
    assert "word embeddings" in ACADEMIC_KNOWLEDGE_BASE


# ======================================================================
# 2. ASR Module Tests
# ======================================================================
def test_asr_empty_input():
    """Ensure ASR rejects empty or too-short byte sequences."""
    processor = ASRProcessor(backend="google")
    result = processor.transcribe_audio_bytes(b"")
    assert not result.success
    assert "empty" in result.error.lower()


def test_asr_corrupted_audio_bytes():
    """Ensure ASR safely rejects random corrupted bytes without raising uncaught exceptions."""
    processor = ASRProcessor(backend="google")
    corrupted_data = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00notrealpcmdatahere" * 10
    result = processor.transcribe_audio_bytes(corrupted_data)
    assert not result.success
    assert result.error is not None


def test_asr_synthetic_silence_audio():
    """Verify how ASR handles a synthetic silent WAV."""
    processor = ASRProcessor(backend="google")
    silent_wav = generate_synthetic_wav(duration_sec=0.5, silent=True)
    result = processor.transcribe_audio_bytes(silent_wav)
    # Silent audio should either fail gracefully or report unrecognized speech
    assert isinstance(result, ASRResult)
    # It must not crash the program
    if not result.success:
        assert len(result.error) > 0


def test_asr_nonexistent_file():
    """Verify file transcription error handling."""
    processor = ASRProcessor(backend="google")
    result = processor.transcribe_file("non_existent_file_path.wav")
    assert not result.success
    assert "not found" in result.error.lower()


# ======================================================================
# 3. NLP Chatbot Model Tests
# ======================================================================
@pytest.fixture(scope="module")
def nlp_bot():
    """Fixture to load NLP model once for testing."""
    # Use fallback (flan-t5-small) for fast test execution if needed
    bot = EducationalChatbotModel(model_name=FALLBACK_MODEL_NAME)
    return bot


def test_nlp_model_initialization(nlp_bot):
    """Verify the model loaded properly and architectural specs are valid."""
    assert nlp_bot.is_loaded is True
    details = nlp_bot.get_model_details()
    assert details["architecture"] == "Transformer Encoder-Decoder (Seq2Seq)"
    assert "Cross-Attention" in details["decoder_type"]


def test_nlp_empty_query(nlp_bot):
    """Ensure empty or whitespace queries return polite guidance."""
    resp = nlp_bot.generate_response("")
    assert "question" in resp["answer"].lower()
    resp_whitespace = nlp_bot.generate_response("   \n\t  ")
    assert "question" in resp_whitespace["answer"].lower()


def test_nlp_unit4_lstm_vs_gru(nlp_bot):
    """Test question answering on Unit 4 core topic: LSTM vs GRU."""
    resp = nlp_bot.generate_response("What is the difference between LSTM and GRU?")
    assert "lstm" in resp["answer"].lower()
    assert "gru" in resp["answer"].lower()
    assert "Unit 4" in resp.get("syllabus_unit", "")
    assert resp["latency_sec"] > 0.0


def test_nlp_unit5_attention_mechanism(nlp_bot):
    """Test question answering on Unit 5 core topic: Attention mechanism."""
    resp = nlp_bot.generate_response("Explain the Attention Mechanism in NLP.")
    assert "attention" in resp["answer"].lower()
    assert "Unit 5" in resp.get("syllabus_unit", "")
    assert resp["latency_sec"] > 0.0


def test_nlp_unit5_encoder_decoder(nlp_bot):
    """Test question answering on Unit 5 core topic: Encoder-Decoder architecture."""
    resp = nlp_bot.generate_response("Explain the Encoder-Decoder architecture in Seq2Seq models.")
    assert "encoder" in resp["answer"].lower()
    assert "decoder" in resp["answer"].lower()
    assert resp["latency_sec"] > 0.0


def test_nlp_open_ended_question(nlp_bot):
    """Test general question answering without pre-configured knowledge key."""
    resp = nlp_bot.generate_response("What is tokenization in natural language processing?")
    assert len(resp["answer"]) > 10
    assert resp["latency_sec"] > 0.0


def test_syllabus_unit_inference(nlp_bot):
    """Verify syllabus unit detection function."""
    assert "Unit 4" in nlp_bot._infer_syllabus_unit("What are Word2Vec embeddings?")
    assert "Unit 4" in nlp_bot._infer_syllabus_unit("Explain vanishing gradient in RNN")
    assert "Unit 5" in nlp_bot._infer_syllabus_unit("How does neural machine translation work?")
    assert "Unit 5" in nlp_bot._infer_syllabus_unit("What is cross-attention?")

