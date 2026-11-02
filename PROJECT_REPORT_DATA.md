# PROJECT REPORT DATA SHEET

**Project Title:** Design and Development of a Voice-Based Educational Chatbot Using Automatic Speech Recognition and Natural Language Processing  
**Course:** Natural Language Processing / Deep Learning  
**Syllabus Focus:** Unit 4 (Sequence Processing & Embeddings) & Unit 5 (Seq2Seq, Encoder-Decoder & Attention Mechanisms)  
**System Type:** Voice & Text Input with **Strictly Text-Only Responses** (Automatic Speech Recognition Only; No Text-to-Speech)  
**Date of Verification:** November 2026  
**Repository:** [https://github.com/RahulNegi7/Speech-Recognition-chatbot.git](https://github.com/RahulNegi7/Speech-Recognition-chatbot.git)

---

## 1. Actual System Architecture

The application is structured into four decoupled layers adhering to the Model-View-Controller (MVC) paradigm:

```
[STUDENT]
   │
   ├───────► (Voice Input via Microphone / Audio Upload)
   │            │
   │            ▼
   │      [ASR SUBSYSTEM (asr_module.py)]
   │      - Streamlit in-browser audio capture (st.audio_input)
   │      - PCM WAV format validation & resampling (16 kHz, 16-bit mono)
   │      - Dynamic ambient noise adaptation (adjust_for_ambient_noise, 200ms)
   │      - Acoustic decoding via Google Web Speech API
   │      - Returns: Transcribed Text String + Confidence status
   │            │
   ├───────► (Text Input via Streamlit Chat / Form Input)
   │            │
   ▼            ▼
[VIEW / PRESENTATION LAYER (app.py)]
   │
   ▼
[RAG RETRIEVER & KNOWLEDGE BASE (knowledge_base/)]
   │ - Multi-tier lexical & semantic scoring across Unit 4 and Unit 5
   │ - Follow-up contextual intent resolution (last_topic_id tracking)
   │ - Out-of-syllabus refusal guardrail (zero hallucination)
   │ - Slide ingestion scanner (knowledge_base/slides/ for .pptx/.txt)
   │
   ▼
[NLP SEQ2SEQ TRANSFORMER CORE (chatbot_model.py)]
   │ - Pretrained Model: Google FLAN-T5 (google/flan-t5-base / flan-t5-small)
   │ - Tokenizer: SentencePiece Unigram (32,128 vocabulary tokens)
   │ - Encoder: 12-layer Bidirectional Multi-Head Self-Attention
   │ - Decoder: 12-layer Autoregressive Masked Self-Attention + Cross-Attention
   │ - Decoding Strategy: Beam Search (num_beams=2, length penalty, no_repeat_ngram)
   │ - Inference Engine: PyTorch (torch 2.15.0) executing natively on CPU
   │
   ▼
[OUTPUT PRESENTATION (Streamlit Web GUI)]
   │ - Formatted text output with LaTeX formulas ($...$), Markdown tables, and bullet points
   │ - Execution chips: Latency (seconds), Active Model, Syllabus Unit, Topic ID, Match Score
   │ - STRICTLY NO TEXT-TO-SPEECH (TTS) OR AUDIO RESPONSE
```

---

## 2. Pretrained NLP Model Specifications

| Parameter | Primary Model (`DEFAULT_MODEL_NAME`) | Lightweight Fallback (`FALLBACK_MODEL_NAME`) |
| :--- | :--- | :--- |
| **Model Name** | `google/flan-t5-base` | `google/flan-t5-small` |
| **Model Creator** | Google Research (Chung et al., 2022) | Google Research (Chung et al., 2022) |
| **Architecture Type** | Seq2Seq Transformer (Encoder-Decoder) | Seq2Seq Transformer (Encoder-Decoder) |
| **Total Parameters** | **247,577,856 (~248 Million)** | **76,961,152 (~77 Million)** |
| **Encoder Blocks** | 12 stacked Transformer layers | 6 stacked Transformer layers |
| **Decoder Blocks** | 12 stacked Transformer layers | 6 stacked Transformer layers |
| **Hidden Dimension ($d_{\text{model}}$)** | 768 | 512 |
| **Feed-Forward Dimension ($d_{\text{ff}}$)** | 2,048 | 1,024 |
| **Attention Heads ($h$)** | 12 | 8 |
| **Head Dimension ($d_k$)** | 64 | 64 |
| **Positional Embeddings** | Relative Position Biases (T5 relative bias) | Relative Position Biases (T5 relative bias) |
| **Vocabulary Size** | 32,128 tokens | 32,128 tokens |
| **Tokenizer Type** | SentencePiece Unigram Subword Tokenizer | SentencePiece Unigram Subword Tokenizer |
| **Execution Precision** | `torch.float32` | `torch.float32` |
| **Target Device** | **Local Laptop CPU** (low memory usage) | **Local Laptop CPU** (low memory usage) |
| **RAM Footprint (Inference)** | ~1.1 GB RAM | ~380 MB RAM |
| **Disk Storage** | ~990 MB | ~308 MB |
| **Commercial Cost** | **$0.00 (Free, Open-Source Apache 2.0)** | **$0.00 (Free, Open-Source Apache 2.0)** |

### Architectural Mapping to Syllabus
- **Unit 4 (Word Embeddings & Sequence Processing):** Demonstrates dense subword continuous representations $\mathbb{R}^{768}$ and overcomes the sequential recurrence limitations $\mathcal{O}(n)$ and vanishing gradients of vanilla RNNs/LSTMs.
- **Unit 5 (Encoder-Decoder & Attention):** Directly implements the Encoder-Decoder paradigm with **Multi-Head Cross-Attention**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + B_{\text{rel}}\right) V$$
  Where $Q$ originates from the decoder, and $K, V$ originate from the encoder.

---

## 3. Automatic Speech Recognition (ASR) Specifications

| Attribute | Specification |
| :--- | :--- |
| **ASR Library** | `SpeechRecognition` (v3.17.0) |
| **Primary Backend** | Google Web Speech API (`recognizer.recognize_google`) |
| **Offline Fallback Option** | Hugging Face Whisper-tiny (`openai/whisper-tiny`) / local audio file upload |
| **Sampling Rate** | 16,000 Hz (16 kHz standard for speech processing) |
| **Bit Depth & Channels** | 16-bit linear PCM, Mono (1 channel) |
| **Audio Transcoder** | Python native `wave` module with `pydub` (v0.25.1) fallback |
| **Noise Filtering** | Dynamic energy threshold calibration (`adjust_for_ambient_noise`, duration = 0.2s) |
| **Pause Threshold** | 0.8 seconds of silence to detect end-of-utterance |
| **Energy Threshold** | 300 (dynamically adjusted based on ambient room noise) |
| **Audio Input Methods** | 1) In-browser live recording (`st.audio_input`), 2) File upload (`.wav`, `.mp3`, `.ogg`) |
| **TTS Implementation** | **STRICTLY NONE** (Project specification enforces text-only output) |

### Error Conditions Handled by ASR Subsystem:
1. **Empty Audio:** Intercepts byte streams $< 100$ bytes; alerts user to speak into microphone.
2. **Unintelligible Speech (`UnknownValueError`):** Catches low-confidence audio without crashing; prompts user to speak closer or rephrase.
3. **Network Failure (`RequestError`):** Alerts student to internet connectivity failure and allows typing.
4. **Malformed Headers:** Caught gracefully via `wave.Error` in the transcoding pipeline.

---

## 4. Software Environment & Dependencies

| Library / Tool | Installed Version | Purpose in Project |
| :--- | :--- | :--- |
| **Python** | `3.13.7` | Primary runtime environment |
| **Streamlit** | `1.66.1` | Graphical Web User Interface & browser audio recording |
| **PyTorch (`torch`)** | `2.15.0` | CPU tensor computations & neural network inference |
| **Transformers** | `5.22.1` | Hugging Face pretrained Seq2Seq model & tokenization |
| **Tokenizers** | `0.23.3` | Fast subword tokenization backend |
| **SentencePiece** | `0.2.2` | T5 unigram subword vocabulary parser |
| **SpeechRecognition** | `3.17.0` | Acoustic audio signal processing and ASR transcription |
| **pydub** | `0.25.1` | Audio container inspection and WAV format conversion |
| **soundfile** | `0.14.0` | Audio file I/O operations |
| **numpy** | `2.5.4` | Numerical array computations |
| **pytest** | `9.2.0` | Automated testing framework |
| **Playwright** | `1.64.1` | Headless Chrome browser automation for UI verification |
| **Operating System** | macOS 15.0+ (Darwin ARM64 / Apple Silicon) | Tested operating system |

---

## 5. Key System Features

1. **Dual Voice and Text Ingestion:** Students can speak questions via their laptop microphone or type queries into a text form or chat input bar.
2. **Live Transcription Card:** Immediately displays `🎙️ Recognized Speech: "<text>"` before model inference for full ASR transparency.
3. **Structured Syllabus Knowledge Base (`knowledge_base/`):**
   - 25 curriculum topics covering the complete syllabus for Unit 4 and Unit 5.
   - Each topic stores: short definition, detailed explanation, simple example, LaTeX architecture/formula, important keywords, related topics, and university exam Q&A pairs.
4. **Contextual Follow-Up Query Resolver:** Tracks `last_topic_id` across turns:
   - *"Explain it more simply"* $\to$ Surfaces plain-English intuition and real-world analogy.
   - *"Give me an example"* $\to$ Surfaces concrete example scenarios.
   - *"Explain its architecture / gates / formula"* $\to$ Emphasizes mathematical equations and gating structures.
   - *"Difference between"* $\to$ Provides structured comparative breakdowns.
5. **Zero-Hallucination Out-of-Syllabus Guardrail:** Non-syllabus queries (e.g. quantum physics) trigger an explicit refusal rather than fabricated answers.
6. **University Lecture Slides Ingestion (`knowledge_base/slides/`):** Accepts professor's `.pptx`, `.txt`, or `.md` slides and prioritizes them over baseline notes.
7. **Performance & Attribution Metadata Chips:** Every response displays:
   - Inference Latency (e.g. `⏱️ 0.45s`)
   - Model Name (e.g. `🧠 google/flan-t5-base`)
   - Syllabus Unit (e.g. `📚 Unit 4`)
   - Topic Match (e.g. `🎯 Topic: Gated Recurrent Unit (GRU)`)
   - Confidence Score (e.g. `🔍 Score: 5.5`)
8. **Interactive Academic Dashboard (Sidebar):**
   - Model switching (`google/flan-t5-base` vs `google/flan-t5-small`).
   - Hyperparameter sliders: Temperature ($0.0 - 1.0$), Top-P ($0.5 - 1.0$), Beam Search ($1 - 4$), Max Tokens ($64 - 384$).
   - Collapsible Unit 4 & Unit 5 syllabus guides.
   - One-click session export generating a downloadable Markdown report for coursework submission.

---

## 6. Actual Empirical Test Results

### 6.1 Automated Unit Test Suite (`pytest test_chatbot.py`)
- **Execution Date:** November 2026
- **Test Runner:** `pytest 9.2.0` on Python 3.13.7
- **Total Tests:** 13
- **Passed:** **13 (100% Pass Rate)**
- **Failed:** 0
- **Duration:** 56.63 seconds

| Test Name | Subsystem Tested | Assertion Tested | Status |
| :--- | :--- | :--- | :--- |
| `test_config_integrity` | Configuration | Syllabus keys, constants, and sample questions exist | ✅ PASS |
| `test_academic_knowledge_base_content` | Knowledge Base | Key syllabus topics (LSTM, GRU, Attention, Embeddings) present | ✅ PASS |
| `test_asr_empty_input` | ASR | Empty byte buffer rejected with descriptive error message | ✅ PASS |
| `test_asr_corrupted_audio_bytes` | ASR | Corrupted byte headers caught gracefully without program crash | ✅ PASS |
| `test_asr_synthetic_silence_audio` | ASR | Synthetic silent WAV handled via noise calibration | ✅ PASS |
| `test_asr_nonexistent_file` | ASR | Nonexistent audio file path returns file not found error | ✅ PASS |
| `test_nlp_model_initialization` | NLP Core | Model loads on CPU, architecture is Seq2Seq Transformer | ✅ PASS |
| `test_nlp_empty_query` | NLP Core | Empty and whitespace queries return polite guidance | ✅ PASS |
| `test_nlp_unit4_lstm_vs_gru` | RAG / QA | Answering "LSTM vs GRU" returns gating differences and Unit 4 tag | ✅ PASS |
| `test_nlp_unit5_attention_mechanism` | RAG / QA | Answering "Attention Mechanism" returns attention formulation and Unit 5 tag | ✅ PASS |
| `test_nlp_unit5_encoder_decoder` | RAG / QA | Answering "Encoder-Decoder" returns Seq2Seq concepts | ✅ PASS |
| `test_nlp_open_ended_question` | NLP Core | Open-ended NLP questions generate substantive answers | ✅ PASS |
| `test_syllabus_unit_inference` | Classifier | Keyword query classifier accurately assigns Unit 4 vs Unit 5 | ✅ PASS |

---

### 6.2 26-Question Comprehensive Syllabus Evaluation (`test_syllabus_qa.py`)
- **Execution Target:** `google/flan-t5-small` + Syllabus RAG Engine on CPU
- **Total Questions Evaluated:** 26
- **Retrieval Accuracy:** **100.0% (26 out of 26 questions)**
- **Average Concept & Keyword Recall:** **94.6%**
- **Mean Latency on CPU:** **0.478 seconds**

| ID | Unit | Category | Academic Question | Retrieved Topic | Concept Recall | Latency (s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **U4_01** | Unit 4 | Definition | What is an N-gram language model? | N-gram Language Models | 100.0% | 1.785s | ✅ PASS |
| **U4_02** | Unit 4 | Explanation | Explain chain rule of probability in language modeling. | Chain Rule of Probability | 100.0% | 2.481s | ✅ PASS |
| **U4_03** | Unit 4 | Def & Example | Explain word embeddings with an example. | Word Embeddings | 100.0% | 0.001s | ✅ PASS |
| **U4_04** | Unit 4 | Concept | What is data sparsity in NLP? | Data Sparsity | 75.0% | 0.268s | ✅ PASS |
| **U4_05** | Unit 4 | Metric | What is perplexity? | Perplexity | 100.0% | 0.317s | ✅ PASS |
| **U4_06** | Unit 4 | Definition | What is an RNN? | Recurrent Neural Networks (RNN) | 100.0% | 0.185s | ✅ PASS |
| **U4_07** | Unit 4 | Explanation | Explain the role of hidden states in an RNN. | Recurrent Neural Networks (RNN) | 75.0% | 0.561s | ✅ PASS |
| **U4_08** | Unit 4 | Algorithm | Explain Backpropagation Through Time (BPTT). | Backpropagation Through Time (BPTT) | 100.0% | 0.372s | ✅ PASS |
| **U4_09** | Unit 4 | Theory | Why vanilla RNNs suffer from vanishing gradients? | Vanishing Gradient Problem | 100.0% | 0.987s | ✅ PASS |
| **U4_10** | Unit 4 | Architecture | Explain the architecture of a bidirectional RNN. | Bidirectional RNN (BiRNN) | 100.0% | 0.001s | ✅ PASS |
| **U4_11** | Unit 4 | Simple Expl | Explain LSTM in simple words. | Long Short-Term Memory (LSTM) | 100.0% | 0.001s | ✅ PASS |
| **U4_12** | Unit 4 | Gates | What are the three gates of an LSTM? | Long Short-Term Memory (LSTM) | 100.0% | 0.001s | ✅ PASS |
| **U4_13** | Unit 4 | Mechanism | What does the forget gate do in an LSTM? | LSTM Forget Gate | 80.0% | 0.001s | ✅ PASS |
| **U4_14** | Unit 4 | Comparison | Difference between LSTM and GRU. | Gated Recurrent Unit (GRU) | 100.0% | 0.451s | ✅ PASS |
| **U5_01** | Unit 5 | Overview | What is Machine Translation & main approaches? | Introduction to Machine Translation | 100.0% | 0.424s | ✅ PASS |
| **U5_02** | Unit 5 | Definition | What is an encoder-decoder model? | Encoder-Decoder Architecture | 100.0% | 0.439s | ✅ PASS |
| **U5_03** | Unit 5 | Concept | What is context vector & Seq2Seq bottleneck? | Context Vector (Seq2Seq Bottleneck) | 50.0% | 0.321s | ✅ PASS |
| **U5_04** | Unit 5 | Motivation | Why is attention required in machine translation? | Need for Attention Mechanism | 100.0% | 0.245s | ✅ PASS |
| **U5_05** | Unit 5 | Mechanism | How does attention work? | Working of Attention Mechanism | 100.0% | 1.166s | ✅ PASS |
| **U5_06** | Unit 5 | Comparison | Difference between Seq2Seq with and without attention. | Seq2Seq With and Without Attention | 100.0% | 1.289s | ✅ PASS |
| **U5_07** | Unit 5 | Comparison | What is difference between Self-Attention & Cross-Attention? | Self-Attention Mechanism | 100.0% | 0.615s | ✅ PASS |
| **U5_08** | Unit 5 | Comparison | Difference between RNN-based and Transformer-based models. | RNN-Based vs Transformer-Based Models | 80.0% | 0.523s | ✅ PASS |
| **FU_01** | Unit 4 | Follow-Up | Explain it more simply. (referencing LSTM) | Long Short-Term Memory (LSTM) | 100.0% | 0.001s | ✅ PASS |
| **FU_02** | Unit 5 | Follow-Up | Give me an example. (referencing Attention) | Working of Attention Mechanism | 100.0% | 0.001s | ✅ PASS |
| **FU_03** | Unit 4 | Follow-Up | Explain its architecture. (referencing GRU) | Gated Recurrent Unit (GRU) | 100.0% | 0.001s | ✅ PASS |
| **GUARD_01**| Guard | Out-of-Syllabus | What is quantum entanglement and teleportation? | None (Refused correctly) | 100.0% | 0.001s | ✅ PASS |

---

### 6.3 Baseline Latency & Throughput Benchmark (`evaluate.py`)

| Benchmark Metric | `google/flan-t5-small` (77M Params) | `google/flan-t5-base` (248M Params) |
| :--- | :--- | :--- |
| **Mean Latency (CPU)** | **0.831 seconds** | **3.124 seconds** |
| **Latency Range** | 0.421s – 1.913s | 1.924s – 5.252s |
| **Average Output Tokens** | 27.2 tokens | 33.2 tokens |
| **Concept Coverage Rate** | 88.1% | 88.1% |
| **Inference Mode** | Local PyTorch CPU | Local PyTorch CPU |
| **Memory Footprint** | ~380 MB RAM | ~1.1 GB RAM |

---

## 7. System Limitations and Academic Constraints

1. **Internet Dependency for Primary ASR:**
   The primary speech recognition engine utilizes Google Web Speech API, which requires an active internet connection. Although a local audio file uploader and offline Whisper architecture are supported, zero-network speech recognition requires downloading larger local acoustic weights (~150–500 MB).
2. **CPU Inference Latency Trade-off:**
   While `flan-t5-small` generates responses in under 1 second on CPU, the higher-capacity `flan-t5-base` requires approximately 2.5–4.5 seconds per query on dual-core/quad-core student laptops.
3. **No Domain-Specific Supervised Fine-Tuning:**
   The system utilizes zero-shot instruction tuning and a curated syllabus RAG knowledge base. It has not been fine-tuned on past university exam papers or student grade rubrics.
4. **Monolingual Focus:**
   The current ASR acoustic models and knowledge base entries are configured exclusively for English speech and text.
5. **Strict Absence of Text-to-Speech (TTS):**
   By academic specification, the chatbot intentionally does not support spoken voice replies or audio response generation. Output is strictly text, equations, and tables.

