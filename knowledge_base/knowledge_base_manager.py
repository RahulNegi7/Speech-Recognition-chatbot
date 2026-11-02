"""
Knowledge Base Manager and RAG Retriever for Unit 4 and Unit 5.
Manages structured syllabus topics, handles follow-up intents,
and checks for custom university lecture slides.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

KB_DIR = Path(__file__).resolve().parent
SLIDES_DIR = KB_DIR / "slides"
SLIDES_DIR.mkdir(parents=True, exist_ok=True)


class KnowledgeBaseManager:
    """
    Loads and manages syllabus knowledge base items for Unit 4 and Unit 5.
    Also parses any custom university lecture slides (.pptx, .txt, .md) if placed in slides/.
    """

    def __init__(self):
        self.topics: List[Dict[str, Any]] = []
        self.topic_map: Dict[str, Dict[str, Any]] = {}
        self.custom_slides: List[Dict[str, Any]] = []
        self.has_custom_slides: bool = False
        self.load_all()

    def load_all(self):
        """Loads default JSON syllabus files and scans for custom university slides."""
        self.topics = []
        self.topic_map = {}

        # 1. Load Unit 4 & Unit 5 JSON files
        for filename in ["unit4_sequence_processing.json", "unit5_machine_translation.json"]:
            file_path = KB_DIR / filename
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for item in data.get("topics", []):
                            self.topics.append(item)
                            self.topic_map[item["id"]] = item
                            # Also map lower-case topic name
                            self.topic_map[item["topic_name"].lower()] = item
                    logger.info(f"Loaded {len(data.get('topics', []))} topics from {filename}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        # Add common aliases for quick direct access
        if "u4_gated_recurrent_unit_gru" in self.topic_map:
            self.topic_map["lstm vs gru"] = self.topic_map["u4_gated_recurrent_unit_gru"]
        if "u5_working_of_attention" in self.topic_map:
            self.topic_map["attention mechanism"] = self.topic_map["u5_working_of_attention"]
        if "u5_encoder_decoder_architecture" in self.topic_map:
            self.topic_map["encoder-decoder"] = self.topic_map["u5_encoder_decoder_architecture"]
        if "u4_word_embeddings" in self.topic_map:
            self.topic_map["word embeddings"] = self.topic_map["u4_word_embeddings"]

        # 2. Check for uploaded university lecture slides / notes in slides/
        self._load_custom_slides()

    def _load_custom_slides(self):
        """Scans knowledge_base/slides/ for custom university PPTX/TXT/MD files."""
        self.custom_slides = []
        if not SLIDES_DIR.exists():
            self.has_custom_slides = False
            return

        supported_files = list(SLIDES_DIR.glob("*.txt")) + list(SLIDES_DIR.glob("*.md"))
        pptx_files = list(SLIDES_DIR.glob("*.pptx"))

        if pptx_files:
            try:
                from pptx import Presentation
                for ppt_file in pptx_files:
                    prs = Presentation(str(ppt_file))
                    slide_texts = []
                    for idx, slide in enumerate(prs.slides):
                        text = " ".join([shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()])
                        if text:
                            slide_texts.append(f"[Slide {idx+1}]: {text}")
                    if slide_texts:
                        self.custom_slides.append({
                            "source_file": ppt_file.name,
                            "type": "pptx",
                            "content": "\n".join(slide_texts)
                        })
            except ImportError:
                logger.info("python-pptx not installed; .pptx files detected but raw slide parser inactive.")

        for text_file in supported_files:
            try:
                with open(text_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        self.custom_slides.append({
                            "source_file": text_file.name,
                            "type": text_file.suffix.replace(".", ""),
                            "content": content
                        })
            except Exception as e:
                logger.error(f"Error reading slide note {text_file}: {e}")

        self.has_custom_slides = len(self.custom_slides) > 0


class RAGRetriever:
    """
    Retrieval-Augmented Generation (RAG) engine.
    Performs keyword & semantic relevance ranking, handles follow-up intents,
    and returns authoritative educational context.
    """

    def __init__(self, kb_manager: KnowledgeBaseManager):
        self.kb = kb_manager
        # Stopwords for scoring
        self.stopwords = {
            "what", "is", "an", "a", "the", "in", "of", "and", "or", "for",
            "to", "how", "does", "do", "explain", "describe", "tell", "me",
            "about", "why", "are", "with", "between", "work", "can", "you",
            "please", "give", "definition", "concept", "define"
        }

    def _tokenize(self, text: str) -> List[str]:
        """Cleans and extracts meaningful query tokens."""
        clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        tokens = [w for w in clean.split() if w and w not in self.stopwords]
        return tokens

    def _detect_follow_up_intent(self, query: str) -> Optional[str]:
        """Detects whether the query is a follow-up request referencing the previous topic."""
        q = query.lower().strip()
        if re.search(r"\b(simply|simpler|simple words|simple terms|easier|layman|simplify)\b", q):
            return "simpler"
        if re.search(r"\b(example|real world|analogy|instance|sample)\b", q):
            return "example"
        if re.search(r"\b(architecture|formula|equation|mathematics|gates|math|how it is built|structure)\b", q):
            return "architecture"
        if re.search(r"\b(difference|compare|versus|vs|contrast)\b", q) and len(self._tokenize(query)) <= 3:
            return "difference"
        return None

    def retrieve(
        self,
        query: str,
        last_topic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves the most relevant knowledge base item for a query.
        Also resolves follow-up questions referencing previous turns.
        """
        follow_up_intent = self._detect_follow_up_intent(query)
        q_tokens = self._tokenize(query)

        # -------------------------------------------------------------
        # 1. Handle Follow-Up Questions (referencing last active topic)
        # -------------------------------------------------------------
        if follow_up_intent and last_topic_id and len(q_tokens) <= 3:
            matched_item = None
            for item in self.kb.topics:
                if item["id"] == last_topic_id:
                    matched_item = item
                    break

            if matched_item:
                return {
                    "found": True,
                    "is_follow_up": True,
                    "follow_up_intent": follow_up_intent,
                    "topic": matched_item,
                    "score": 1.0,
                    "unit": matched_item["unit_number"]
                }

        # -------------------------------------------------------------
        # 2. Comparative Query Matching (e.g. LSTM vs GRU)
        # -------------------------------------------------------------
        q_lower = query.lower()
        if "lstm" in q_lower and "gru" in q_lower:
            for item in self.kb.topics:
                if item["id"] == "u4_gated_recurrent_unit_gru":
                    return {"found": True, "is_follow_up": False, "topic": item, "score": 1.0, "unit": "Unit 4"}
        if "without attention" in q_lower or ("seq2seq" in q_lower and "attention" in q_lower and "difference" in q_lower):
            for item in self.kb.topics:
                if item["id"] == "u5_seq2seq_with_and_without_attention":
                    return {"found": True, "is_follow_up": False, "topic": item, "score": 1.0, "unit": "Unit 5"}
        if "rnn" in q_lower and "transformer" in q_lower:
            for item in self.kb.topics:
                if item["id"] == "u5_difference_between_rnn_and_transformer":
                    return {"found": True, "is_follow_up": False, "topic": item, "score": 1.0, "unit": "Unit 5"}
        if "self-attention" in q_lower and "cross-attention" in q_lower:
            for item in self.kb.topics:
                if item["id"] == "u5_self_attention":
                    return {"found": True, "is_follow_up": False, "topic": item, "score": 1.0, "unit": "Unit 5"}

        # -------------------------------------------------------------
        # 3. Keyword and Semantic Scoring across Knowledge Base
        # -------------------------------------------------------------
        best_item = None
        best_score = 0.0

        for item in self.kb.topics:
            score = 0.0
            topic_name_lower = item["topic_name"].lower()
            keywords = [k.lower() for k in item.get("important_keywords", [])]

            # Exact topic name phrase match
            if topic_name_lower in q_lower:
                score += 5.0

            # Match keywords
            for kw in keywords:
                if kw in q_lower:
                    score += 2.5
                else:
                    # Token-level overlap
                    kw_tokens = set(self._tokenize(kw))
                    overlap = kw_tokens.intersection(set(q_tokens))
                    if overlap:
                        score += 0.8 * len(overlap)

            # Match in common questions
            for qa in item.get("common_questions_and_answers", []):
                q_text = qa["question"].lower()
                if any(phrase in q_lower for phrase in [q_text, q_text.replace("?", "")]):
                    score += 4.0
                overlap = set(self._tokenize(q_text)).intersection(set(q_tokens))
                score += 0.5 * len(overlap)

            # Match in short definition
            def_tokens = set(self._tokenize(item.get("short_definition", "")))
            def_overlap = def_tokens.intersection(set(q_tokens))
            score += 0.3 * len(def_overlap)

            if score > best_score:
                best_score = score
                best_item = item

        # Confidence Threshold Check
        # If the highest score is too low, the query is outside Unit 4/5 knowledge base
        CONFIDENCE_THRESHOLD = 1.2
        if best_item and best_score >= CONFIDENCE_THRESHOLD:
            return {
                "found": True,
                "is_follow_up": False,
                "follow_up_intent": follow_up_intent,
                "topic": best_item,
                "score": round(best_score, 2),
                "unit": best_item["unit_number"]
            }

        # -------------------------------------------------------------
        # 4. Out-of-Syllabus / Unavailable Fallback
        # -------------------------------------------------------------
        slide_notice = (
            "Custom university slides are currently indexed."
            if self.kb.has_custom_slides
            else "Standard curriculum knowledge base active. (If you have your professor's lecture slides or PPTX files, place them in the 'knowledge_base/slides/' directory to index them)."
        )

        return {
            "found": False,
            "is_follow_up": False,
            "message": (
                "⚠️ **Topic Not Found in Syllabus Knowledge Base**\n\n"
                f"I could not locate an authoritative answer for your query in the current **Unit 4 (Sequence Processing)** or **Unit 5 (Machine Translation & Attention)** syllabus topics.\n\n"
                f"*{slide_notice}*\n\n"
                "**Please try asking about core curriculum topics, such as:**\n"
                "- *Unit 4:* N-gram models, Word Embeddings, Perplexity, RNN, BPTT, LSTM, Gates (Forget, Input, Output), GRU, Vanishing Gradient.\n"
                "- *Unit 5:* Machine Translation (RBMT, SMT, NMT), Encoder-Decoder, Context Vector, Attention Mechanism, Self-Attention, Transformers."
            ),
            "score": 0.0,
            "unit": None
        }
