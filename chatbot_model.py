"""
Educational NLP Chatbot Model Module with Syllabus-Grounded RAG.
Combines Google FLAN-T5 Seq2Seq Transformer with an extensive
Unit 4 and Unit 5 curriculum Knowledge Base and follow-up query engine.
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
from knowledge_base.knowledge_base_manager import KnowledgeBaseManager, RAGRetriever

logger = logging.getLogger(__name__)

# Re-expose knowledge base dictionary for backwards compatibility and tests
ACADEMIC_KNOWLEDGE_BASE = KnowledgeBaseManager().topic_map


class EducationalChatbotModel:
    """
    NLP Educational Chatbot powered by Hugging Face Transformers and
    Syllabus-Based Retrieval-Augmented Generation (RAG).
    Covers Unit 4 (Sequence Modeling) and Unit 5 (Seq2Seq & Attention).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.is_loaded = False

        # Initialize Knowledge Base & RAG Retriever
        self.kb_manager = KnowledgeBaseManager()
        self.retriever = RAGRetriever(self.kb_manager)

        self.load_model()

    def load_model(self):
        """Loads the pretrained Seq2Seq model and tokenizer on CPU/GPU."""
        logger.info(f"Loading NLP Model '{self.model_name}' on device '{self.device}'...")
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

    def _generate_neural_summary(
        self,
        question: str,
        context: str,
        temperature: float,
        top_p: float,
        num_beams: int,
        max_new_tokens: int
    ) -> str:
        """
        Uses FLAN-T5 Seq2Seq model to generate a concise neural synthesis
        grounded in the retrieved syllabus context.
        """
        if not self.is_loaded or not self.model or not self.tokenizer:
            return ""

        prompt = (
            f"Context: {context[:600]}\n\n"
            f"Based on the curriculum context above, explain this concept in 2 clear, concise sentences:\n"
            f"Question: {question}\n\n"
            f"Summary:"
        )

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                do_sample = temperature > 0.0
                gen_kwargs = {
                    "max_new_tokens": min(max_new_tokens, 128),
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

            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return summary
        except Exception as e:
            logger.warning(f"Neural summary generation failed: {e}")
            return ""

    def generate_response(
        self,
        question: str,
        last_topic_id: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        num_beams: int = DEFAULT_NUM_BEAMS,
        max_new_tokens: int = MAX_NEW_TOKENS
    ) -> Dict[str, Any]:
        """
        Retrieval-Augmented Generation (RAG) educational query handler.
        Retrieves relevant curriculum content and synthesizes a structured student response.
        """
        if not question or not question.strip():
            return {
                "answer": "Please ask a question related to your NLP or Deep Learning studies.",
                "latency_sec": 0.0,
                "model": self.model_name,
                "syllabus_unit": None,
                "retrieved_topic_id": None,
                "confidence_score": 0.0,
                "architecture_type": "Seq2Seq Transformer (Encoder-Decoder)"
            }

        start_time = time.time()

        # 1. Retrieve relevant knowledge base entry using RAG
        retrieval = self.retriever.retrieve(question, last_topic_id=last_topic_id)

        # 2. If topic is NOT found in syllabus knowledge base, return honest disclaimer
        if not retrieval["found"]:
            latency = round(time.time() - start_time, 3)
            return {
                "answer": retrieval["message"],
                "latency_sec": latency,
                "model": "Syllabus RAG Guard",
                "syllabus_unit": None,
                "retrieved_topic_id": None,
                "confidence_score": 0.0,
                "architecture_type": "Knowledge Base Filter"
            }

        # 3. Format response based on retrieval and intent
        topic = retrieval["topic"]
        topic_id = topic["id"]
        unit = topic["unit_number"]
        intent = retrieval.get("follow_up_intent")

        # -------------------------------------------------------------
        # A. Follow-Up: "Explain it more simply"
        # -------------------------------------------------------------
        if intent == "simpler":
            answer_text = (
                f"### 💡 {topic['topic_name']} — Explained in Simple Words\n\n"
                f"**In plain English:**\n"
                f"{topic['short_definition']}\n\n"
                f"**Intuition & Real-World Analogy:**\n"
                f"{topic['simple_example']}\n\n"
                f"---\n"
                f"**Key Takeaway:** {topic['detailed_explanation'][:250]}...\n\n"
                f"*(Unit: {unit} | Topic ID: `{topic_id}`)*"
            )

        # -------------------------------------------------------------
        # B. Follow-Up: "Give me an example"
        # -------------------------------------------------------------
        elif intent == "example":
            answer_text = (
                f"### 📌 Real-World Example: {topic['topic_name']}\n\n"
                f"**Example Scenario:**\n"
                f"{topic['simple_example']}\n\n"
                f"**Why this illustrates {topic['topic_name']}:**\n"
                f"{topic['short_definition']}\n\n"
                f"*(Unit: {unit} | Topic ID: `{topic_id}`)*"
            )

        # -------------------------------------------------------------
        # C. Follow-Up: "Explain its architecture / gates / formula"
        # -------------------------------------------------------------
        elif intent == "architecture" or "architecture" in question.lower() or "formula" in question.lower() or "gate" in question.lower():
            formula = topic.get("architecture_or_formula", "Architecture details summarized in explanation.")
            answer_text = (
                f"### 🏛️ Architecture & Mathematical Formulation: {topic['topic_name']}\n\n"
                f"**Mathematical Formulation / Structural Equations:**\n"
                f"{formula}\n\n"
                f"**Architectural Flow:**\n"
                f"{topic['detailed_explanation']}\n\n"
                f"*(Unit: {unit} | Topic ID: `{topic_id}`)*"
            )

        # -------------------------------------------------------------
        # D. Standard Comprehensive Educational Answer
        # -------------------------------------------------------------
        else:
            qa_section = ""
            if topic.get("common_questions_and_answers"):
                qa = topic["common_questions_and_answers"][0]
                qa_section = (
                    f"\n\n---\n"
                    f"**🎓 Common Viva Question & Answer:**\n"
                    f"- **Q:** *{qa['question']}*\n"
                    f"- **A:** {qa['answer']}"
                )

            formula_section = ""
            if topic.get("architecture_or_formula"):
                formula_section = f"\n\n**Mathematical Formulation / Architecture:**\n{topic['architecture_or_formula']}"

            # Generate neural summary via FLAN-T5
            neural_summary = self._generate_neural_summary(
                question=question,
                context=topic["detailed_explanation"],
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens
            )

            neural_section = ""
            if neural_summary:
                neural_section = f"\n\n---\n**🤖 Seq2Seq Model Summary ({self.model_name}):**\n{neural_summary}"

            answer_text = (
                f"### 📚 {topic['topic_name']} ({unit} — {topic['category']})\n\n"
                f"**Definition:**\n"
                f"{topic['short_definition']}\n\n"
                f"**Detailed Explanation:**\n"
                f"{topic['detailed_explanation']}\n\n"
                f"**Simple Example:**\n"
                f"{topic['simple_example']}"
                f"{formula_section}"
                f"{qa_section}"
                f"{neural_section}"
            )

        latency = round(time.time() - start_time, 3)

        return {
            "answer": answer_text,
            "latency_sec": latency,
            "model": self.model_name,
            "device": self.device,
            "syllabus_unit": unit,
            "retrieved_topic_id": topic_id,
            "topic_name": topic["topic_name"],
            "confidence_score": retrieval["score"],
            "architecture_type": "Seq2Seq Transformer + Syllabus RAG"
        }

    def _infer_syllabus_unit(self, question: str) -> Optional[str]:
        """Infers whether an open-ended question belongs to Unit 4 or Unit 5."""
        ret = self.retriever.retrieve(question)
        if ret.get("found") and ret.get("unit"):
            return ret["unit"]
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
            "knowledge_base_topics": len(self.kb_manager.topics),
            "custom_slides_indexed": self.kb_manager.has_custom_slides,
            "vocabulary_size": getattr(self.tokenizer, "vocab_size", 32128) if self.tokenizer else 32128,
            "unit_4_relation": "Demonstrates word embeddings & sequence processing without recurrent bottleneck.",
            "unit_5_relation": "Direct implementation of Encoder-Decoder Seq2Seq architecture with Cross-Attention."
        }
