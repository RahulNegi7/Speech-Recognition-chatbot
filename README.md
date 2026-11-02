# Design and Development of a Voice-Based Educational Chatbot Using Automatic Speech Recognition and Natural Language Processing

🎓 **Academic Mini-Project / Coursework Assignment**  
**Course:** Natural Language Processing (NLP) & Deep Learning  
**Curriculum Alignment:** Unit 4 (Language Models & Sequence Processing) & Unit 5 (Seq2Seq, Encoder-Decoder & Attention)  
**System Type:** Voice & Text Input with **Strictly Text-Only Responses** (Automatic Speech Recognition Only; No Text-to-Speech)

---

## 📌 Project Overview & Objectives

In modern higher education, students frequently require interactive, self-paced explanations of complex mathematical and algorithmic concepts in Natural Language Processing and Deep Learning. Traditional chatbots either rely on rigid rule-based systems or require costly cloud subscriptions.

This project delivers a **voice-enabled educational chatbot** designed to run efficiently on standard consumer laptop CPUs without requiring paid API tokens or a dedicated GPU. Students can query academic topics either by **speaking into a microphone** or typing via text, receiving clear, structured, and syllabus-aligned text responses.

### Key Objectives
1. **ASR Integration (Speech-to-Text Only):** Provide reliable voice transcription supporting live microphone recordings and audio file uploads, with robust handling for ambient noise, silent recordings, and audio format conversion.
2. **Strictly Text Output:** Strictly adhere to the project constraint that **no Text-to-Speech (TTS)** or audio playback is implemented; all responses are rendered as formatted text, mathematical formulas, and bullet points.
3. **Seq2Seq Transformer Core:** Deploy a pretrained Seq2Seq Transformer model (`google/flan-t5-base` / `google/flan-t5-small`) running locally on CPU.
4. **Curriculum Grounding:** Directly demonstrate core concepts from **Unit 4** (Word Embeddings, Recurrent Networks, LSTMs, GRUs) and **Unit 5** (Encoder-Decoder architecture, Seq2Seq bottleneck, Multi-Head Self-Attention, and Cross-Attention).

---

## 📚 Curriculum & Theoretical Mapping

| Curriculum Unit | Topics Covered in Syllabus | How the Project Integrates the Concept |
| :--- | :--- | :--- |
| **Unit 4: Sequence Processing & Language Models** | Word Embeddings (Word2Vec, GloVe), N-grams, Recurrent Neural Networks (RNN), Vanishing Gradient, LSTM, GRU | **Theoretical Contrast & Embeddings:** The chatbot uses dense subword embeddings (SentencePiece vocabulary of 32,128 tokens) mapping discrete tokens into continuous semantic vectors $\mathbb{R}^{d}$. The theoretical limitations of classical RNN/LSTM/GRU sequential recurrence (information bottleneck and sequential computational dependency $\mathcal{O}(n)$) serve as the motivating background for the Transformer architecture used here. |
| **Unit 5: Encoder-Decoder & Attention Mechanism** | Seq2Seq Architectures, Additive (Bahdanau) Attention, Scaled Dot-Product (Luong) Attention, Transformer Multi-Head Attention, Neural Machine Translation (NMT) | **Direct Architectural Implementation:** The chatbot core (`FLAN-T5`) is an **Encoder-Decoder Seq2Seq Transformer**. The bidirectional encoder creates contextual representations, and the autoregressive decoder applies **Multi-Head Cross-Attention** to dynamically align generated output tokens with the question representation. |

> ⚠️ **Academic Distinction:**  
> This project does **not** falsely claim that Transformers are RNNs or LSTMs. Instead, Transformers replace recurrent step-by-step state passing with parallel multi-head self-attention and cross-attention, completely bypassing the vanishing gradient and recurrence bottlenecks studied in Unit 4.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ / 3.13 | Core programming language |
| **Web GUI** | Streamlit | Modern, student-friendly web interface with live audio recording |
| **ASR Engine** | SpeechRecognition / PyAudio / pydub | Acoustic waveform capture, ambient noise compensation, speech-to-text |
| **NLP Framework** | Hugging Face Transformers | Pretrained Seq2Seq model loading, tokenization, beam search |
| **Tensor Computation** | PyTorch (`torch`) | Neural network inference optimized for CPU |
| **NLP Model** | `google/flan-t5-base` / `google/flan-t5-small` | Instruction-tuned Seq2Seq Transformer (Encoder-Decoder) |
| **Testing & Evaluation** | pytest, custom benchmark suite | Automated verification and empirical latency/accuracy benchmarking |

---

## 🏛️ System Architecture

```
                                  [STUDENT]
                                   /      \
                      (Voice Input)        (Text Input)
                            /                    \
              [Microphone / Audio Upload]         \
                           |                       \
            [ASR Module: Audio Preprocessing]       \
             (Sample Rate 16kHz, Noise Filter)      |
                           |                        |
            [ASR Engine: Google Web Speech]         |
                           |                        |
                   [Recognized Text]                |
                           \                       /
                            \                     /
                             ▼                   ▼
                         +--------------------------+
                         |     Streamlit Web GUI    |
                         +--------------------------+
                                      |
                                      ▼
                      +--------------------------------+
                      |   Educational Chatbot Core     |
                      |   (chatbot_model.py)           |
                      +--------------------------------+
                                      |
                        +-------------+-------------+
                        |                           |
                        ▼                           ▼
            [Syllabus Knowledge Base]    [FLAN-T5 Tokenizer]
            (Unit 4 & 5 Concept Bank)    (SentencePiece Subwords)
                        \                           /
                         \                         /
                          ▼                       ▼
            +-----------------------------------------------+
            |        Seq2Seq Transformer (FLAN-T5)          |
            |                                               |
            |   +---------------------------------------+   |
            |   |   ENCODER: Bidirectional Self-Attn   |   |
            |   +---------------------------------------+   |
            |                       |                       |
            |                Cross-Attention                |
            |                       ▼                       |
            |   +---------------------------------------+   |
            |   |   DECODER: Masked Self-Attn + Cross   |   |
            |   +---------------------------------------+   |
            +-----------------------------------------------+
                                      |
                         (Autoregressive Generation)
                                      |
                                      ▼
                           [Formulated Text Answer]
                                      |
                      +--------------------------------+
                      |   Rendered on Streamlit UI     |
                      |   (Strictly Text; NO TTS)      |
                      +--------------------------------+
```

---

## 🚀 Installation & Execution Guide

### 1. Clone or Open the Project
```bash
cd "speech chatbot"
```

### 2. Set Up a Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Chatbot Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎙️ How Automatic Speech Recognition (ASR) Works

1. **Acoustic Waveform Capture:** Audio input is recorded through the browser's microphone widget (`st.audio_input`) or uploaded as a `.wav` file.
2. **Audio Preprocessing & Transcoding:** Using `pydub` and `wave`, audio streams are validated, normalized, and converted into standard 16-bit PCM WAV at 16,000 Hz (mono).
3. **Ambient Noise Calibration:** The recognizer calibrates its `energy_threshold` dynamically using the first 200ms of audio (`adjust_for_ambient_noise`) to prevent quiet background humming from interfering with speech.
4. **Phoneme & Sequence Decoding:** The audio data is processed via Google Web Speech API (online, zero subscription required) to translate acoustic spectrographic features into textual word sequences.
5. **Fail-Safe Error Handling:**
   - **Empty / Short Audio:** Detected before sending, reporting: *"Audio input is empty or too short."*
   - **Unclear Speech / Distorted Noise:** Catches `UnknownValueError`, reporting: *"Audio detected, but speech could not be recognized clearly."*
   - **Network Interruption:** Catches `RequestError`, alerting the student to check internet connectivity.

---

## 🧠 Why FLAN-T5 Was Chosen Over Causal Decoder-Only Models

Many beginners default to causal language models like GPT-2 for conversational projects. However, for an academic syllabus focused on **Unit 5 (Encoder-Decoder & Seq2Seq)**, FLAN-T5 is mathematically and pedagogically superior:

1. **Direct Alignment with Unit 5:** Causal models (GPT-2) are **Decoder-Only** architectures. In contrast, FLAN-T5 is a pure **Encoder-Decoder** architecture, providing a concrete demonstration of:
   - Encoder bidirectional contextualization.
   - Information transfer via Cross-Attention:
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
     where queries $Q$ come from the decoder, and keys $K$ and values $V$ come from the encoder.
2. **Instruction Fine-Tuning (FLAN):** Unlike raw base models that ramble or complete sentences arbitrarily, FLAN-T5 was fine-tuned on thousands of academic and instructional tasks, allowing it to produce concise, textbook-style answers without multi-billion parameter overhead.
3. **CPU Friendly:** `flan-t5-base` (~248M parameters) and `flan-t5-small` (~77M parameters) operate smoothly within 1–2 GB of system RAM, generating answers in 1–3 seconds on standard dual/quad-core laptop CPUs.

---

## 🧪 Testing & Evaluation

### Running 26-Question Comprehensive Syllabus Evaluation
To evaluate whether the chatbot retrieves relevant material and accurately answers questions across every topic in Unit 4 and Unit 5 (definitions, formulas, comparisons, follow-ups, and guardrails):

```bash
python test_syllabus_qa.py
```
This evaluates all 26 syllabus questions and saves results to `docs/syllabus_qa_report.md` (**100.0% retrieval accuracy, 94.6% concept recall**).

### Running Automated Pytest Suite
```bash
pytest test_chatbot.py -v
```

---

## 📁 Project Directory Structure

```
speech_chatbot/
├── app.py                     # Streamlit frontend & interactive chat interface
├── asr_module.py              # ASR processing, noise adjustment & audio conversion
├── chatbot_model.py           # Seq2Seq Transformer model & RAG query handler
├── config.py                  # Model hyperparameters, syllabus definitions & paths
├── evaluate.py                # Quantitative evaluation benchmark on academic QA
├── test_chatbot.py            # Automated unit & integration test suite (13 tests)
├── test_syllabus_qa.py        # 26-question syllabus evaluation benchmark
├── knowledge_base/            # Structured syllabus knowledge base
│   ├── unit4_sequence_processing.json  # N-grams, Word2Vec, RNN, BPTT, LSTM, GRU
│   ├── unit5_machine_translation.json  # RBMT, SMT, NMT, Seq2Seq, Attention, Transformers
│   ├── knowledge_base_manager.py       # RAG retrieval engine & follow-up intent handler
│   └── slides/                         # Directory for custom university lecture slides (.pptx/.txt)
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive project documentation
├── assets/                    # Audio samples and test assets
│   └── sample_audio.wav       # Generated test audio file
├── screenshots/               # Verified running application UI screenshots
└── docs/                      # Academic report, benchmark logs, and viva guides
    ├── syllabus_qa_report.md  # 26-question evaluation report (100% accuracy)
    ├── syllabus_qa_results.json# Raw evaluation measurements
    ├── evaluation_report.md   # Latency & throughput benchmark report
    ├── evaluation_results.json# Raw benchmark measurements
    └── academic_project_report.md # Full 20–25 page academic project report
```

---

## 🎓 Viva Voce Preparation (Q&A for Examiners)

**Q1: Why did your project use an Encoder-Decoder model instead of an RNN or LSTM?**  
*Answer:* RNNs and LSTMs process sequences sequentially ($t_1, t_2, \dots, t_T$), which prevents parallel computation during training and creates a memory bottleneck when long inputs are compressed into a single hidden vector $h_T$. The Encoder-Decoder Transformer processes all tokens in parallel using Self-Attention and dynamic Cross-Attention, preserving long-range dependencies without vanishing gradients.

**Q2: What is the exact mathematical difference between Self-Attention and Cross-Attention?**  
*Answer:* In Self-Attention, the Queries ($Q$), Keys ($K$), and Values ($V$) are all computed from linear projections of the same sequence (e.g. within the encoder). In Cross-Attention, Queries ($Q$) are projected from the decoder's previous layer, while Keys ($K$) and Values ($V$) are projected from the final encoder representation.

**Q3: Why is there no Text-to-Speech (TTS) in this project?**  
*Answer:* The project specification strictly focuses on **Automatic Speech Recognition (ASR)** to explore speech-to-text acoustic modeling. The output is intentionally constrained to text so students can read formulas, copy code snippets, and review structured bullet points.

---

## 📄 License
Academic and educational use only under the MIT License.

