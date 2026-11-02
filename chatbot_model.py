"""
Educational NLP Chatbot Model Module.
Implements Seq2Seq Transformer (Google FLAN-T5) with tokenization,
attention mechanisms, educational prompt formatting, and syllabus knowledge mapping.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import (
    DEFAULT_MODEL_NAME,
    FALLBACK_MODEL_NAME,
    MAX_INPUT_LENGTH,
    MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_NUM_BEAMS,
    DEFAULT_REPETITION_PENALTY,
    SYLLABUS_INFO
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Curated Academic Syllabus Knowledge Base
# Provides authoritative academic definitions and comparisons for
# Unit 4 and Unit 5 viva topics to augment student learning.
# ----------------------------------------------------------------------
ACADEMIC_KNOWLEDGE_BASE = {
    "lstm vs gru": {
        "title": "Comparison: Long Short-Term Memory (LSTM) vs Gated Recurrent Unit (GRU)",
        "unit": "Unit 4 (Sequence Processing & Recurrent Models)",
        "content": (
            "**Key Differences Between LSTM and GRU:**\n\n"
            "1. **Number of Gates:**\n"
            "   - **LSTM (Hochreiter & Schmidhuber, 1997):** Contains **3 gates**: Forget Gate ($f_t$), Input Gate ($i_t$), and Output Gate ($o_t$).\n"
            "   - **GRU (Cho et al., 2014):** Contains **2 gates**: Reset Gate ($r_t$) and Update Gate ($z_t$).\n\n"
            "2. **State Vectors:**\n"
            "   - **LSTM:** Maintains two separate state vectors: Cell State ($c_t$, long-term memory) and Hidden State ($h_t$, short-term memory).\n"
            "   - **GRU:** Merges cell state and hidden state into a single Hidden State vector ($h_t$).\n\n"
            "3. **Computational Complexity & Efficiency:**\n"
            "   - **GRU** has fewer parameters (~25% fewer than LSTM), making it faster to train and computationally lighter on smaller datasets.\n"
            "   - **LSTM** has greater expressive capacity and handles longer sequential dependencies more reliably.\n\n"
            "4. **Syllabus Context (Unit 4):** Both architectures address the **vanishing gradient problem** of vanilla RNNs using additive state paths."
        )
    },
    "vanishing gradient": {
        "title": "Vanishing Gradient Problem in RNNs",
        "unit": "Unit 4 (Recurrent Neural Networks)",
        "content": (
            "**The Vanishing Gradient Problem:**\n\n"
            "1. **Mathematical Cause:** During Backpropagation Through Time (BPTT), gradients are multiplied across time steps:\n"
            "   $$\\frac{\\partial L}{\\partial h_0} = \\prod_{k=1}^T \\frac{\\partial h_k}{\\partial h_{k-1}} \\frac{\\partial L}{\\partial h_T}$$\n"
            "   When weights $W$ have eigenvalues $< 1$ or activation derivatives (e.g. $\\tanh' \\le 1$) are repeated, the product exponentially decays towards zero as $T \\to \\infty$.\n\n"
            "2. **Consequence:** The network cannot learn long-range temporal dependencies; early inputs are effectively forgotten.\n\n"
            "3. **Solution:** LSTMs and GRUs introduce linear additive paths (constant error carousels) so gradients can flow backward through time without repeated matrix multiplication decay."
        )
    },
    "encoder-decoder": {
        "title": "Encoder-Decoder Architecture & Seq2Seq",
        "unit": "Unit 5 (Encoder-Decoder Models & Seq2Seq)",
        "content": (
            "**Encoder-Decoder Architecture (Seq2Seq):**\n\n"
            "1. **Core Concept (Sutskever et al. 2014 / Cho et al. 2014):**\n"
            "   Designed to map an input sequence of variable length $X = (x_1, ..., x_T)$ to an output sequence of variable length $Y = (y_1, ..., y_{T'})$.\n\n"
            "2. **Encoder:**\n"
            "   Processes the input tokens sequentially or bidirectionally, compressing the input meaning into a dense representation (context vector or matrix of hidden states).\n\n"
            "3. **Decoder:**\n"
            "   An autoregressive generator that takes the context vector and previously generated tokens to predict the next token $y_t$.\n\n"
            "4. **The Classical Bottleneck:** In early RNN Seq2Seq models, the entire input had to be squeezed into a single fixed-length vector. This bottleneck led directly to the development of the **Attention Mechanism**."
        )
    },
    "attention mechanism": {
        "title": "Attention Mechanism in NLP",
        "unit": "Unit 5 (Attention Mechanism & Neural Machine Translation)",
        "content": (
            "**The Attention Mechanism:**\n\n"
            "1. **Origin:** Proposed by Bahdanau et al. (2014) and Luong et al. (2015) for Neural Machine Translation to overcome the fixed-length vector bottleneck in Seq2Seq.\n\n"
            "2. **Mathematical Formulation (Scaled Dot-Product Attention):**\n"
            "   $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V$$\n"
            "   - **Query ($Q$):** What the current decoder position is looking for.\n"
            "   - **Key ($K$):** Identifiers for each token in the source sequence.\n"
            "   - **Value ($V$):** The actual semantic information of each source token.\n"
            "   - $\\sqrt{d_k}$ is a scaling factor to prevent dot products from growing excessively large and pushing the softmax into regions with vanishing gradients.\n\n"
            "3. **Self-Attention vs Cross-Attention:**\n"
            "   - **Self-Attention:** $Q, K, V$ originate from the *same* sequence (learns intra-sequence token relations).\n"
            "   - **Cross-Attention:** $Q$ originates from the decoder, while $K, V$ originate from the encoder (aligns output tokens to source context)."
        )
    },
    "word embeddings": {
        "title": "Word Embeddings & Dense Vector Representations",
        "unit": "Unit 4 (Word Embeddings & Language Models)",
        "content": (
            "**Word Embeddings:**\n\n"
            "1. **Definition:** Continuous dense vector representations where semantically related words are placed close to each other in vector space ($d \\ll V$, typically 100–768 dimensions), overcoming the curse of dimensionality and sparsity of one-hot vectors.\n\n"
            "2. **Core Approaches:**\n"
            "   - **Word2Vec (Mikolov et al., 2013):** Continuous Bag of Words (CBOW) predicts target word from context; Skip-Gram predicts context words from target.\n"
            "   - **GloVe (Pennington et al., 2014):** Factorizes global word co-occurrence matrix.\n"
            "   - **FastText (Bojanowski et al., 2017):** Utilizes character n-grams to handle Out-Of-Vocabulary (OOV) words.\n"
            "   - **Subword / Contextual Embeddings (Transformer / T5):** SentencePiece / BPE embeddings where representations change dynamically based on surrounding sentence context."
        )
    },
    "neural machine translation": {
        "title": "Neural Machine Translation (NMT)",
        "unit": "Unit 5 (Neural Machine Translation & Seq2Seq)",
        "content": (
            "**Neural Machine Translation (NMT):**\n\n"
            "1. **Definition:** An end-to-end deep learning framework for translating text from a source language to a target language without pipeline components (like phrase tables or language models in statistical MT).\n\n"
            "2. **Evolution:**\n"
            "   - Stage 1: Recurrent Seq2Seq (RNN/LSTM with fixed context vector).\n"
            "   - Stage 2: Attention-augmented Seq2Seq (Bahdanau alignment).\n"
            "   - Stage 3: Transformer models (Vaswani et al. 2017) using multi-head self-attention, which became the modern foundation for models like T5, BART, and MarianMT."
        )
    }
}


class EducationalChatbotModel:
    """
    NLP Educational Chatbot powered by Hugging Face Transformers.
    Directly demonstrates Unit 4 (Embeddings, Sequence Modeling) and
    Unit 5 (Encoder-Decoder Seq2Seq & Multi-Head Cross-Attention).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.load_model()

    def load_model(self):
        """Loads the pretrained Seq2Seq model and tokenizer on CPU/GPU."""
        logger.info(f"Loading NLP Model '{self.model_name}' on device '{self.device}'...")
        # Try normal load first, then local_files_only fallback
        for local_only in [False, True]:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local_only)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    local_files_only=local_only
                )
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                logger.info(f"Model and Tokenizer successfully loaded (local_only={local_only}).")
                return
            except Exception as e:
                if local_only:
                    logger.warning(f"Could not load {self.model_name}: {e}. Trying fallback model...")

        # If primary failed, try fallback model (flan-t5-small)
        for local_only in [False, True]:
            try:
                self.model_name = FALLBACK_MODEL_NAME
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local_only)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    local_files_only=local_only
                )
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                logger.info(f"Fallback model {self.model_name} loaded successfully (local_only={local_only}).")
                return
            except Exception as ex:
                if local_only:
                    logger.error(f"Failed to load fallback model: {ex}")
                    self.is_loaded = False

    def _find_curated_knowledge(self, query: str) -> Optional[Dict[str, str]]:
        """
        Checks whether the student query aligns directly with foundational
        Unit 4 / Unit 5 syllabus concepts to provide authoritative reference.
        """
        q = query.lower().strip()
        for key, entry in ACADEMIC_KNOWLEDGE_BASE.items():
            # Check key phrases
            tokens = key.split(" vs ") if " vs " in key else [key]
            if all(token in q for token in tokens):
                return entry
            # Check individual keywords
            if key in q:
                return entry

        # Additional alias mappings
        if "gru" in q and "lstm" in q:
            return ACADEMIC_KNOWLEDGE_BASE["lstm vs gru"]
        if "vanishing" in q and "gradient" in q:
            return ACADEMIC_KNOWLEDGE_BASE["vanishing gradient"]
        if "seq2seq" in q or "encoder decoder" in q or "encoder-decoder" in q:
            return ACADEMIC_KNOWLEDGE_BASE["encoder-decoder"]
        if "self-attention" in q or "cross-attention" in q or "attention" in q:
            return ACADEMIC_KNOWLEDGE_BASE["attention mechanism"]
        if "embedding" in q or "word2vec" in q:
            return ACADEMIC_KNOWLEDGE_BASE["word embeddings"]
        if "nmt" in q or "machine translation" in q:
            return ACADEMIC_KNOWLEDGE_BASE["neural machine translation"]

        return None

    def _format_prompt(self, user_question: str) -> str:
        """
        Wraps the question in an instruction-tuned academic prompt.
        Helps FLAN-T5 generate structured, informative educational explanations.
        """
        return (
            f"You are a helpful university professor teaching Natural Language Processing and Deep Learning. "
            f"Provide a clear, detailed, and student-friendly explanation for the following academic question. "
            f"Include definition, working mechanism, and key characteristics:\n\n"
            f"Question: {user_question}\n\n"
            f"Answer:"
        )

    def generate_response(
        self,
        question: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        num_beams: int = DEFAULT_NUM_BEAMS,
        max_new_tokens: int = MAX_NEW_TOKENS
    ) -> Dict[str, Any]:
        """
        Generates an educational text response for a user query.
        Returns response text along with syllabus context and latency metrics.
        """
        if not question or not question.strip():
            return {
                "answer": "Please ask a question related to your NLP or Deep Learning studies.",
                "latency_sec": 0.0,
                "model": self.model_name,
                "syllabus_unit": None,
                "architecture_type": "Seq2Seq Transformer (Encoder-Decoder)"
            }

        start_time = time.time()
        knowledge_match = self._find_curated_knowledge(question)

        # If model is not loaded, rely on syllabus knowledge or offline fallback
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            if knowledge_match:
                latency = round(time.time() - start_time, 3)
                return {
                    "answer": f"### {knowledge_match['title']}\n\n{knowledge_match['content']}",
                    "latency_sec": latency,
                    "model": "Academic Syllabus Knowledge Base (Offline)",
                    "syllabus_unit": knowledge_match["unit"],
                    "architecture_type": "Curated Educational Syllabus Base"
                }
            return {
                "answer": "The NLP model could not be initialized. Please check that dependencies and models are loaded.",
                "latency_sec": 0.0,
                "model": "Error",
                "syllabus_unit": None,
                "architecture_type": "None"
            }

        # Format input for FLAN-T5 Seq2Seq model
        prompt = self._format_prompt(question)

        try:
            # Tokenization (demonstrating SentencePiece subword tokenization)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                padding=True
            ).to(self.device)

            input_token_count = inputs["input_ids"].shape[1]

            # Model Generation (Seq2Seq generation with beam search / sampling)
            with torch.no_grad():
                do_sample = temperature > 0.0
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "num_beams": num_beams,
                    "repetition_penalty": DEFAULT_REPETITION_PENALTY,
                    "early_stopping": True
                }
                if do_sample:
                    gen_kwargs["temperature"] = max(temperature, 0.1)
                    gen_kwargs["top_p"] = top_p
                    gen_kwargs["do_sample"] = True
                else:
                    gen_kwargs["do_sample"] = False

                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **gen_kwargs
                )

            # Decode generated tokens back to text string
            raw_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            output_token_count = outputs.shape[1]
            latency = round(time.time() - start_time, 3)

            # Determine syllabus alignment
            syllabus_unit = knowledge_match["unit"] if knowledge_match else self._infer_syllabus_unit(question)

            # If the query matches a core syllabus concept, enrich the answer with
            # the structured academic reference note so the student receives a
            # comprehensive textbook-level response suitable for viva preparation.
            if knowledge_match:
                combined_answer = (
                    f"### {knowledge_match['title']}\n\n"
                    f"{knowledge_match['content']}\n\n"
                    f"---\n"
                    f"**Model Summary ({self.model_name}):**\n"
                    f"{raw_answer}"
                )
            else:
                combined_answer = raw_answer

            return {
                "answer": combined_answer,
                "latency_sec": latency,
                "model": self.model_name,
                "device": self.device,
                "input_tokens": input_token_count,
                "output_tokens": output_token_count,
                "syllabus_unit": syllabus_unit,
                "architecture_type": "Seq2Seq Transformer (Encoder-Decoder with Cross-Attention)"
            }

        except Exception as e:
            logger.exception("Error during model generation")
            # If generation fails, return knowledge match if available
            if knowledge_match:
                return {
                    "answer": f"### {knowledge_match['title']}\n\n{knowledge_match['content']}",
                    "latency_sec": round(time.time() - start_time, 3),
                    "model": "Fallback Syllabus Base",
                    "syllabus_unit": knowledge_match["unit"],
                    "architecture_type": "Knowledge Base Fallback"
                }
            return {
                "answer": f"An error occurred while generating the response: {str(e)}",
                "latency_sec": round(time.time() - start_time, 3),
                "model": self.model_name,
                "syllabus_unit": None,
                "architecture_type": "Error"
            }

    def _infer_syllabus_unit(self, question: str) -> Optional[str]:
        """Infers whether an open-ended question belongs to Unit 4 or Unit 5."""
        q = question.lower()
        unit4_keywords = ["rnn", "lstm", "gru", "embedding", "word2vec", "glove", "n-gram", "recurrent", "vanishing"]
        unit5_keywords = ["encoder", "decoder", "seq2seq", "attention", "transformer", "nmt", "translation", "cross-attention", "beam search"]

        if any(k in q for k in unit4_keywords):
            return "Unit 4 (Word Embeddings & Recurrent Models)"
        if any(k in q for k in unit5_keywords):
            return "Unit 5 (Encoder-Decoder Models & Attention)"
        return "General NLP / Deep Learning"

    def get_model_details(self) -> Dict[str, Any]:
        """Returns architectural specifications for documentation and viva."""
        return {
            "model_name": self.model_name,
            "architecture": "Transformer Encoder-Decoder (Seq2Seq)",
            "encoder_type": "Bidirectional Multi-Head Self-Attention",
            "decoder_type": "Autoregressive Masked Self-Attention + Multi-Head Cross-Attention",
            "device": self.device,
            "vocabulary_size": getattr(self.tokenizer, "vocab_size", 32128) if self.tokenizer else 32128,
            "max_position_embeddings": MAX_INPUT_LENGTH,
            "unit_4_relation": "Demonstrates word embeddings & sequence processing without recurrent bottleneck.",
            "unit_5_relation": "Direct implementation of Encoder-Decoder Seq2Seq architecture with Cross-Attention."
        }
