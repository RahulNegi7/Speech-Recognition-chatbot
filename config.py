"""
Configuration settings for the Voice-Based Educational Chatbot.
Covers NLP model parameters, ASR settings, and syllabus domain mapping.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DOCS_DIR = BASE_DIR / "docs"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

# Ensure directories exist
for directory in [ASSETS_DIR, DOCS_DIR, SCREENSHOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# NLP Model Configuration
# ---------------------------------------------------------
# Google's Flan-T5-base is a Seq2Seq Transformer Encoder-Decoder model (~248M params).
# Flan-T5-small (~77M params) is an ultra-lightweight alternative for slower CPUs.
DEFAULT_MODEL_NAME = "google/flan-t5-base"
FALLBACK_MODEL_NAME = "google/flan-t5-small"

# Generation Hyperparameters
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.95
DEFAULT_NUM_BEAMS = 2
DEFAULT_REPETITION_PENALTY = 1.15

# ---------------------------------------------------------
# ASR (Automatic Speech Recognition) Configuration
# ---------------------------------------------------------
# Supported backends: 'google' (via SpeechRecognition), 'whisper_tiny' (offline transformer)
DEFAULT_ASR_BACKEND = "google"
AUDIO_SAMPLE_RATE = 16000  # 16 kHz standard for speech processing
ENERGY_THRESHOLD = 300     # Sensitivity for silence detection
PAUSE_THRESHOLD = 0.8      # Seconds of non-speaking audio before phrase is complete

# ---------------------------------------------------------
# Syllabus Mapping (Academic Curriculum Context)
# ---------------------------------------------------------
SYLLABUS_INFO = {
    "title": "Design and Development of a Voice-Based Educational Chatbot",
    "course": "Natural Language Processing / Deep Learning",
    "units": {
        "Unit 4": {
            "title": "Language Models, Word Embeddings & Sequence Processing",
            "topics": [
                "N-gram Language Models",
                "Word Embeddings (Word2Vec, GloVe, FastText)",
                "Recurrent Neural Networks (RNN) and Vanishing Gradient",
                "Long Short-Term Memory (LSTM) - Forget, Input, Output Gates",
                "Gated Recurrent Units (GRU) - Reset and Update Gates",
                "Bidirectional Sequence Processing",
            ],
            "role": "Foundational sequence modeling background; contrasts recurrent bottlenecks with modern attention mechanisms."
        },
        "Unit 5": {
            "title": "Encoder-Decoder Models, Seq2Seq & Attention Mechanism",
            "topics": [
                "Encoder-Decoder Architecture (Sutskever et al., Cho et al.)",
                "Sequence-to-Sequence (Seq2Seq) Information Bottleneck",
                "Additive (Bahdanau) and Multiplicative (Luong) Attention",
                "Self-Attention and Multi-Head Attention",
                "Transformer Encoder-Decoder (Vaswani et al.)",
                "Neural Machine Translation (NMT) and Text Generation",
            ],
            "role": "Directly implemented by the FLAN-T5 Seq2Seq Transformer core powering this educational chatbot."
        }
    }
}

# Sample academic questions for student demonstration
SAMPLE_QUESTIONS = [
    "What is the difference between LSTM and GRU?",
    "Explain the Encoder-Decoder architecture in Seq2Seq models.",
    "What is the purpose of the Attention Mechanism in NLP?",
    "How does Self-Attention differ from Cross-Attention?",
    "Why do vanilla RNNs suffer from vanishing gradients?",
    "What is the role of Word Embeddings in language modeling?",
    "Explain Neural Machine Translation and how Seq2Seq is used.",
    "What are the gates in an LSTM cell and what do they do?"
]

