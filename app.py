"""
Streamlit Web Application: Voice-Based Educational Chatbot.
Integrates Automatic Speech Recognition (ASR Only) with Seq2Seq Transformer NLP.
Designed for college students studying NLP and Deep Learning (Units 4 & 5).
"""

import os
import io
import time
import logging
import warnings
from datetime import datetime

# Silence benign third-party warnings in terminal
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import streamlit as st

from config import (
    DEFAULT_MODEL_NAME,
    FALLBACK_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_NUM_BEAMS,
    MAX_NEW_TOKENS,
    SYLLABUS_INFO,
    SAMPLE_QUESTIONS
)
from asr_module import ASRProcessor, ASRResult
from chatbot_model import EducationalChatbotModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Page Configuration & Styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="EduVoice NLP Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished academic UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .badge-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }
    .badge {
        background-color: #EEF2FF;
        color: #4338CA;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #C7D2FE;
    }
    .badge-asr {
        background-color: #ECFDF5;
        color: #065F46;
        border-color: #A7F3D0;
    }
    .badge-unit {
        background-color: #FEF3C7;
        color: #92400E;
        border-color: #FDE68A;
    }
    .transcript-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 0.95rem;
    }
    .meta-chip {
        font-size: 0.78rem;
        color: #6B7280;
        background: #F3F4F6;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
        display: inline-block;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Resource Caching
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Seq2Seq Transformer Model (FLAN-T5)...")
def load_cached_nlp_model(model_name: str, cache_key: str = "v3_rag") -> EducationalChatbotModel:
    """Loads and caches the Hugging Face NLP model in memory."""
    return EducationalChatbotModel(model_name=model_name)


@st.cache_resource
def get_asr_processor() -> ASRProcessor:
    """Instantiates the ASR processor."""
    return ASRProcessor(backend="google")


# ----------------------------------------------------------------------
# Session State Initialization
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to your Voice-Based Educational Chatbot!**\n\n"
                "I am your academic assistant for **Natural Language Processing & Deep Learning**.\n"
                "- Ask questions using **voice** (via microphone) or **text**.\n"
                "- Receive structured **text explanations**, architecture breakdowns, and formulas.\n"
                "- Directly aligned with **Unit 4 (RNN/LSTM/GRU & Embeddings)** and **Unit 5 (Seq2Seq & Attention)**.\n\n"
                "*(Note: By project specifications, responses are strictly provided as TEXT; no Text-to-Speech audio is generated).* \n\n"
                "Try picking a sample question or record your voice below!"
            ),
            "meta": {
                "model": "System Welcome",
                "latency": 0.0,
                "unit": "Course Overview"
            }
        }
    ]

if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = None

if "audio_processed_key" not in st.session_state:
    st.session_state.audio_processed_key = None

if "last_topic_id" not in st.session_state:
    st.session_state.last_topic_id = None


# ----------------------------------------------------------------------
# Sidebar: Academic Syllabus & Model Controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("Academic Dashboard")
    st.caption("NLP & Deep Learning Project viva reference")

    # Model Configuration
    st.subheader("⚙️ NLP Model Settings")
    selected_model = st.selectbox(
        "Seq2Seq Model",
        options=[DEFAULT_MODEL_NAME, FALLBACK_MODEL_NAME],
        index=0,
        help="Select the pretrained Encoder-Decoder model. Both run locally on CPU."
    )

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05,
                                help="0 = deterministic greedy decoding, higher = creative sampling.")
    with col2:
        top_p = st.slider("Top-P (Nucleus)", 0.5, 1.0, DEFAULT_TOP_P, 0.05,
                          help="Nucleus sampling probability threshold.")

    col3, col4 = st.columns(2)
    with col3:
        num_beams = st.slider("Beam Search", 1, 4, DEFAULT_NUM_BEAMS, 1,
                              help="Number of search paths during Seq2Seq autoregressive generation.")
    with col4:
        max_tokens = st.slider("Max New Tokens", 64, 384, MAX_NEW_TOKENS, 32)

    # Syllabus Alignment Section
    st.divider()
    st.subheader("📚 Syllabus Alignment")
    with st.expander("Unit 4: Sequence Processing & Embeddings", expanded=False):
        st.markdown("""
        **Curriculum Topics:**
        - Word Embeddings (Word2Vec, GloVe, FastText)
        - Recurrent Neural Networks (RNN) & BPTT
        - Vanishing Gradient Problem
        - LSTM (Forget, Input, Output Gates, Cell State)
        - GRU (Reset, Update Gates)
        
        *Project Role:* Contrasted with modern non-recurrent Transformer encoders.
        """)

    with st.expander("Unit 5: Seq2Seq & Attention Mechanism", expanded=True):
        st.markdown("""
        **Curriculum Topics:**
        - Encoder-Decoder Architecture
        - Seq2Seq Information Bottleneck
        - Additive (Bahdanau) & Scaled Dot-Product Attention
        - Transformer (Self-Attention & Cross-Attention)
        - Neural Machine Translation (NMT)
        
        *Project Role:* **Directly implemented** via the FLAN-T5 Encoder-Decoder architecture.
        """)

    st.divider()
    st.subheader("🎙️ ASR Architecture")
    st.info(
        "**ASR Only Mode Active:**\n\n"
        "1. Captures acoustic speech waveforms.\n"
        "2. Normalizes sample rate to 16 kHz.\n"
        "3. Decodes speech phonemes into text tokens.\n"
        "4. Passes recognized text to NLP Seq2Seq model.\n\n"
        "*(No TTS / spoken audio is generated)*."
    )

    st.divider()
    st.subheader("📚 Syllabus Knowledge Base")
    st.caption("Unit 4 (Sequence Models) & Unit 5 (Seq2Seq & Attention)")
    st.success("✅ 25 Curriculum Topics Indexed")

    # Custom University Slides Status & Upload
    with st.expander("📂 University Lecture Slides (PPT)", expanded=False):
        st.markdown(
            "If your professor provides specific Unit 4 / Unit 5 lecture slides, "
            "place `.pptx` or `.txt` files in `knowledge_base/slides/` or upload here:"
        )
        uploaded_slide = st.file_uploader("Upload slide notes (.pptx, .txt)", type=["pptx", "txt", "md"], key="slide_uploader")
        if uploaded_slide is not None:
            from knowledge_base.knowledge_base_manager import SLIDES_DIR
            save_path = SLIDES_DIR / uploaded_slide.name
            with open(save_path, "wb") as f:
                f.write(uploaded_slide.read())
            st.success(f"Uploaded and indexed {uploaded_slide.name}!")
            st.info("Re-running to apply slide updates...")
            st.rerun()

    # Actions
    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.last_transcription = None
        st.session_state.last_topic_id = None
        st.rerun()

    # Export Chat for Assignment Submission
    if len(st.session_state.messages) > 1:
        chat_export_text = f"# Academic Chatbot Interaction Log\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        for msg in st.session_state.messages:
            role = "Student" if msg["role"] == "user" else "Educational Chatbot"
            chat_export_text += f"### {role}:\n{msg['content']}\n\n"
        st.download_button(
            label="📥 Export Chat History (Markdown)",
            data=chat_export_text,
            file_name="nlp_chatbot_session.md",
            mime="text/markdown",
            use_container_width=True
        )


# ----------------------------------------------------------------------
# Main Application Header
# ----------------------------------------------------------------------
st.markdown('<div class="main-header">🎓 Voice-Based Educational Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Natural Language Processing & Deep Learning Assistant | Automatic Speech Recognition (ASR Only)</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="badge-container">
    <span class="badge badge-asr">🎙️ ASR Integrated (Speech-to-Text)</span>
    <span class="badge">🚫 Text-Only Output (No TTS)</span>
    <span class="badge badge-unit">📘 Unit 4: Embeddings & Sequence Models</span>
    <span class="badge badge-unit">📙 Unit 5: Encoder-Decoder & Attention</span>
    <span class="badge">⚡ FLAN-T5 Seq2Seq (Vaswani et al.)</span>
</div>
""", unsafe_allow_html=True)


# Initialize Model and ASR
nlp_model = load_cached_nlp_model(selected_model)
asr_processor = get_asr_processor()


# ----------------------------------------------------------------------
# Quick Sample Questions for Demonstration
# ----------------------------------------------------------------------
st.markdown("**💡 Quick Viva / Syllabus Questions:**")
cols = st.columns(4)
sample_prompt = None
for i, sample in enumerate(SAMPLE_QUESTIONS[:4]):
    with cols[i]:
        if st.button(sample, key=f"sample_{i}", use_container_width=True):
            sample_prompt = sample


# ----------------------------------------------------------------------
# Chat Display
# ----------------------------------------------------------------------
for message in st.session_state.messages:
    avatar = "🧑‍🎓" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "meta" in message and message["meta"]:
            meta = message["meta"]
            meta_str = ""
            if "latency" in meta and meta["latency"] > 0:
                meta_str += f"<span class='meta-chip'>⏱️ {meta['latency']}s</span>"
            if "model" in meta:
                meta_str += f"<span class='meta-chip'>🧠 {meta['model']}</span>"
            if "unit" in meta and meta["unit"]:
                meta_str += f"<span class='meta-chip'>📚 {meta['unit']}</span>"
            if meta_str:
                st.markdown(meta_str, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Input Section: Tabs for Audio Input & Text Input
# ----------------------------------------------------------------------
input_tab1, input_tab2 = st.tabs(["🎙️ Voice Input (Microphone / Audio)", "⌨️ Text Input"])

pending_query = None
input_source = None

with input_tab1:
    st.markdown("##### Record Question via Microphone")
    audio_val = st.audio_input("Speak your academic question:")

    if audio_val is not None:
        audio_bytes = audio_val.read()
        # Hash or identify audio bytes to avoid re-triggering on each rerun
        current_audio_id = hash(audio_bytes)
        if st.session_state.audio_processed_key != current_audio_id:
            with st.spinner("🎙️ Converting speech to text via ASR..."):
                asr_result: ASRResult = asr_processor.transcribe_audio_bytes(audio_bytes)

            st.session_state.audio_processed_key = current_audio_id

            if asr_result.success and asr_result.text:
                st.success(f"**Recognized Speech:** \"{asr_result.text}\"")
                pending_query = asr_result.text
                input_source = f"Voice ({asr_result.backend})"
            else:
                st.error(f"❌ **ASR Error:** {asr_result.error or 'Speech could not be understood'}")
                st.info("Tip: Speak clearly, close to the microphone, and ensure microphone permissions are granted.")

    # Alternative Audio File Upload
    with st.expander("Or upload pre-recorded audio file (.wav, .mp3)"):
        uploaded_audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])
        if uploaded_audio is not None:
            if st.button("Transcribe & Submit Uploaded Audio"):
                with st.spinner("Processing uploaded audio..."):
                    asr_result = asr_processor.transcribe_audio_bytes(uploaded_audio.read())
                if asr_result.success and asr_result.text:
                    st.success(f"**Recognized Speech:** \"{asr_result.text}\"")
                    pending_query = asr_result.text
                    input_source = "Uploaded Audio"
                else:
                    st.error(f"❌ ASR Error: {asr_result.error}")

with input_tab2:
    with st.form("text_input_form", clear_on_submit=True):
        typed_text = st.text_input("Type your academic question:", placeholder="e.g. What is an RNN? or Explain LSTM in simple words")
        submitted = st.form_submit_button("Submit Question", use_container_width=True)
        if submitted and typed_text.strip():
            pending_query = typed_text.strip()
            input_source = "Typed Input"

# Bottom global chat bar (always accessible)
bottom_chat_text = st.chat_input("Type your question here (or record voice above)...")
if bottom_chat_text and bottom_chat_text.strip():
    pending_query = bottom_chat_text.strip()
    input_source = "Chat Input"

# Check if a sample question was clicked
if sample_prompt:
    pending_query = sample_prompt
    input_source = "Quick Sample"


# ----------------------------------------------------------------------
# Query Processing & NLP Generation
# ----------------------------------------------------------------------
if pending_query:
    # 1. Add student question to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": pending_query,
        "meta": {"source": input_source}
    })

    # Display student message immediately
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(pending_query)
        if input_source:
            st.markdown(f"<span class='meta-chip'>📥 Source: {input_source}</span>", unsafe_allow_html=True)

    # 2. Generate response via Seq2Seq Transformer model & Syllabus RAG
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner(f"🧠 Retrieving syllabus context & generating with {selected_model}..."):
            try:
                response_data = nlp_model.generate_response(
                    question=pending_query,
                    last_topic_id=st.session_state.get("last_topic_id"),
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_tokens
                )
            except TypeError:
                response_data = nlp_model.generate_response(
                    question=pending_query,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_tokens
                )

        # Update last active topic for follow-up questions
        if response_data.get("retrieved_topic_id"):
            st.session_state.last_topic_id = response_data["retrieved_topic_id"]

        answer_text = response_data["answer"]
        st.markdown(answer_text)

        # Meta tags
        meta_chips = (
            f"<span class='meta-chip'>⏱️ {response_data['latency_sec']}s</span>"
            f"<span class='meta-chip'>🧠 {response_data['model']}</span>"
        )
        if response_data.get("syllabus_unit"):
            meta_chips += f"<span class='meta-chip'>📚 {response_data['syllabus_unit']}</span>"
        if response_data.get("topic_name"):
            meta_chips += f"<span class='meta-chip'>🎯 Topic: {response_data['topic_name']}</span>"
        if response_data.get("confidence_score") and response_data["confidence_score"] > 0:
            meta_chips += f"<span class='meta-chip'>🔍 Score: {response_data['confidence_score']}</span>"

        st.markdown(meta_chips, unsafe_allow_html=True)

    # 3. Save assistant response to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "meta": {
            "latency": response_data["latency_sec"],
            "model": response_data["model"],
            "unit": response_data.get("syllabus_unit", "NLP"),
            "topic": response_data.get("topic_name")
        }
    })

    # Rerun to cleanly refresh the chat UI
    st.rerun()

