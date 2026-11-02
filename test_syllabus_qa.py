"""
Comprehensive Syllabus Q&A Evaluation Suite (Units 4 and 5).
Tests at least 20 academic questions covering definitions, comparisons,
architectures, formulas, follow-up queries, and out-of-syllabus guardrails.
Saves actual test results to docs/syllabus_qa_results.json and docs/syllabus_qa_report.md.
"""

import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from config import DEFAULT_MODEL_NAME, FALLBACK_MODEL_NAME, DOCS_DIR
from chatbot_model import EducationalChatbotModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 26 Comprehensive Academic Syllabus Questions
SYLLABUS_TEST_QUESTIONS = [
    # --- UNIT 4: Sequence Processing & Language Models ---
    {
        "id": "U4_01",
        "unit": "Unit 4",
        "category": "Definition",
        "question": "What is an N-gram language model?",
        "expected_topic": "N-gram Language Models",
        "expected_keywords": ["markov", "previous", "probability", "n-1"]
    },
    {
        "id": "U4_02",
        "unit": "Unit 4",
        "category": "Explanation",
        "question": "Explain the chain rule of probability in language modeling.",
        "expected_topic": "Chain Rule of Probability",
        "expected_keywords": ["conditional", "joint", "product"]
    },
    {
        "id": "U4_03",
        "unit": "Unit 4",
        "category": "Definition & Example",
        "question": "Explain word embeddings with an example.",
        "expected_topic": "Word Embeddings",
        "expected_keywords": ["dense", "vector", "word2vec", "semantic"]
    },
    {
        "id": "U4_04",
        "unit": "Unit 4",
        "category": "Concept",
        "question": "What is data sparsity in NLP?",
        "expected_topic": "Data Sparsity",
        "expected_keywords": ["zero", "probability", "smoothing", "corpus"]
    },
    {
        "id": "U4_05",
        "unit": "Unit 4",
        "category": "Evaluation Metric",
        "question": "What is perplexity?",
        "expected_topic": "Perplexity",
        "expected_keywords": ["cross-entropy", "branching", "lower", "metric"]
    },
    {
        "id": "U4_06",
        "unit": "Unit 4",
        "category": "Definition",
        "question": "What is an RNN?",
        "expected_topic": "Recurrent Neural Networks (RNN)",
        "expected_keywords": ["recurrent", "hidden state", "cycle", "sequential"]
    },
    {
        "id": "U4_07",
        "unit": "Unit 4",
        "category": "Explanation",
        "question": "Explain the role of hidden states in an RNN.",
        "expected_topic": "Hidden States in RNN",
        "expected_keywords": ["memory", "history", "context", "vector"]
    },
    {
        "id": "U4_08",
        "unit": "Unit 4",
        "category": "Training Algorithm",
        "question": "Explain Backpropagation Through Time (BPTT).",
        "expected_topic": "Backpropagation Through Time (BPTT)",
        "expected_keywords": ["unroll", "gradient", "chain rule", "time"]
    },
    {
        "id": "U4_09",
        "unit": "Unit 4",
        "category": "Theoretical Problem",
        "question": "Why do vanilla RNNs suffer from the vanishing gradient problem?",
        "expected_topic": "Vanishing Gradient Problem",
        "expected_keywords": ["jacobian", "multiply", "decay", "tanh", "exponential"]
    },
    {
        "id": "U4_10",
        "unit": "Unit 4",
        "category": "Architecture",
        "question": "Explain the architecture of a bidirectional RNN.",
        "expected_topic": "Bidirectional RNN (BiRNN)",
        "expected_keywords": ["forward", "backward", "future", "concatenated"]
    },
    {
        "id": "U4_11",
        "unit": "Unit 4",
        "category": "Simplified Explanation",
        "question": "Explain LSTM in simple words.",
        "expected_topic": "Long Short-Term Memory (LSTM)",
        "expected_keywords": ["cell state", "gate", "forget", "memory"]
    },
    {
        "id": "U4_12",
        "unit": "Unit 4",
        "category": "Architecture & Gates",
        "question": "What are the three gates of an LSTM?",
        "expected_topic": "Long Short-Term Memory (LSTM)",
        "expected_keywords": ["forget", "input", "output", "gates"]
    },
    {
        "id": "U4_13",
        "unit": "Unit 4",
        "category": "Gate Mechanism",
        "question": "What does the forget gate do in an LSTM?",
        "expected_topic": "LSTM Forget Gate",
        "expected_keywords": ["sigmoid", "0", "1", "discard", "retain"]
    },
    {
        "id": "U4_14",
        "unit": "Unit 4",
        "category": "Comparison",
        "question": "Difference between LSTM and GRU.",
        "expected_topic": "Gated Recurrent Unit (GRU)",
        "expected_keywords": ["gates", "cell state", "reset", "update", "parameters"]
    },

    # --- UNIT 5: Machine Translation & Encoder-Decoder Models ---
    {
        "id": "U5_01",
        "unit": "Unit 5",
        "category": "Overview",
        "question": "What is Machine Translation and what are its main approaches?",
        "expected_topic": "Introduction to Machine Translation",
        "expected_keywords": ["rbmt", "smt", "nmt", "translation"]
    },
    {
        "id": "U5_02",
        "unit": "Unit 5",
        "category": "Definition",
        "question": "What is an encoder-decoder model?",
        "expected_topic": "Encoder-Decoder Architecture",
        "expected_keywords": ["encoder", "decoder", "context vector", "seq2seq"]
    },
    {
        "id": "U5_03",
        "unit": "Unit 5",
        "category": "Core Concept",
        "question": "What is the context vector and what is the Seq2Seq bottleneck?",
        "expected_topic": "Context Vector (Seq2Seq Bottleneck)",
        "expected_keywords": ["fixed-length", "bottleneck", "compress", "long sentence"]
    },
    {
        "id": "U5_04",
        "unit": "Unit 5",
        "category": "Motivation",
        "question": "Why is attention required in machine translation?",
        "expected_topic": "Need for Attention Mechanism",
        "expected_keywords": ["bottleneck", "long sentences", "look-back", "degradation"]
    },
    {
        "id": "U5_05",
        "unit": "Unit 5",
        "category": "Mechanism",
        "question": "How does attention work?",
        "expected_topic": "Working of Attention Mechanism",
        "expected_keywords": ["alignment", "weights", "softmax", "dynamic context"]
    },
    {
        "id": "U5_06",
        "unit": "Unit 5",
        "category": "Comparison",
        "question": "Difference between Seq2Seq with and without attention.",
        "expected_topic": "Difference Between Seq2Seq With and Without Attention",
        "expected_keywords": ["static", "dynamic", "fixed", "bleu"]
    },
    {
        "id": "U5_07",
        "unit": "Unit 5",
        "category": "Comparison",
        "question": "What is the difference between Self-Attention and Cross-Attention?",
        "expected_topic": "Self-Attention Mechanism",
        "expected_keywords": ["query", "key", "value", "same sequence", "encoder", "decoder"]
    },
    {
        "id": "U5_08",
        "unit": "Unit 5",
        "category": "Comparison",
        "question": "Difference between RNN-based and Transformer-based models.",
        "expected_topic": "Difference Between RNN-Based and Transformer-Based Models",
        "expected_keywords": ["sequential", "parallel", "o(1)", "o(n)", "recurrence"]
    },

    # --- Follow-Up Intent Handling ---
    {
        "id": "FU_01",
        "unit": "Unit 4",
        "category": "Follow-Up (Simple)",
        "question": "Explain it more simply.",
        "prior_topic_id": "u4_long_short_term_memory_lstm",
        "expected_topic": "Long Short-Term Memory (LSTM)",
        "expected_keywords": ["plain english", "intuition", "memory"]
    },
    {
        "id": "FU_02",
        "unit": "Unit 5",
        "category": "Follow-Up (Example)",
        "question": "Give me an example.",
        "prior_topic_id": "u5_working_of_attention",
        "expected_topic": "Working of Attention Mechanism",
        "expected_keywords": ["example", "area", "european"]
    },
    {
        "id": "FU_03",
        "unit": "Unit 4",
        "category": "Follow-Up (Architecture)",
        "question": "Explain its architecture.",
        "prior_topic_id": "u4_gated_recurrent_unit_gru",
        "expected_topic": "Gated Recurrent Unit (GRU)",
        "expected_keywords": ["reset gate", "update gate", "formula"]
    },

    # --- Out-of-Syllabus Guardrail ---
    {
        "id": "GUARD_01",
        "unit": "Out of Syllabus",
        "category": "Guardrail (Not Found)",
        "question": "What is quantum entanglement and quantum teleportation?",
        "expected_topic": None,
        "expected_keywords": ["not found in", "syllabus knowledge base"]
    }
]


def run_syllabus_evaluation(model_name: str = FALLBACK_MODEL_NAME) -> Dict[str, Any]:
    """Runs rigorous test suite across 26 questions, measuring accuracy and retrieval."""
    logger.info(f"Initializing Chatbot Model ({model_name}) for 26-question syllabus evaluation...")
    bot = EducationalChatbotModel(model_name=model_name)

    results = []
    latencies = []
    retrieval_success_count = 0
    keyword_match_scores = []

    for test in SYLLABUS_TEST_QUESTIONS:
        q_id = test["id"]
        q_text = test["question"]
        prior_topic = test.get("prior_topic_id")
        expected_topic = test.get("expected_topic")
        expected_keywords = test["expected_keywords"]

        logger.info(f"Evaluating {q_id} [{test['category']}]: '{q_text}'")
        resp = bot.generate_response(q_text, last_topic_id=prior_topic)

        answer = resp["answer"].lower()
        latency = resp["latency_sec"]
        latencies.append(latency)

        retrieved_topic = resp.get("topic_name")

        # Check retrieval correctness
        if expected_topic is None:
            # Should correctly say NOT found
            retrieval_ok = "not found in" in answer or "syllabus knowledge base" in answer
        else:
            retrieval_ok = (retrieved_topic == expected_topic) or any(k in answer for k in expected_keywords[:2])

        if retrieval_ok:
            retrieval_success_count += 1

        # Check keyword matches
        matched_kw = [k for k in expected_keywords if k in answer]
        kw_score = round(len(matched_kw) / len(expected_keywords) * 100, 1)
        keyword_match_scores.append(kw_score)

        results.append({
            "id": q_id,
            "unit": test["unit"],
            "category": test["category"],
            "question": q_text,
            "expected_topic": expected_topic or "None (Guardrail)",
            "retrieved_topic": retrieved_topic or "None",
            "retrieval_success": retrieval_ok,
            "keyword_match_percent": kw_score,
            "matched_keywords": matched_kw,
            "latency_sec": latency,
            "answer_preview": resp["answer"][:160].replace("\n", " ") + "..."
        })

    avg_latency = round(sum(latencies) / len(latencies), 3)
    retrieval_accuracy = round(retrieval_success_count / len(SYLLABUS_TEST_QUESTIONS) * 100, 1)
    avg_keyword_coverage = round(sum(keyword_match_scores) / len(keyword_match_scores), 1)

    summary = {
        "model_name": model_name,
        "device": bot.device,
        "total_test_questions": len(SYLLABUS_TEST_QUESTIONS),
        "retrieval_accuracy_percent": retrieval_accuracy,
        "average_keyword_coverage_percent": avg_keyword_coverage,
        "average_latency_sec": avg_latency,
        "results": results
    }

    # Save JSON results
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DOCS_DIR / "syllabus_qa_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save Markdown report
    md_path = DOCS_DIR / "syllabus_qa_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Unit 4 & Unit 5 Syllabus Knowledge Base Evaluation Report\n\n")
        f.write(f"- **Evaluated Model:** `{model_name}`\n")
        f.write(f"- **Total Questions Evaluated:** {len(SYLLABUS_TEST_QUESTIONS)}\n")
        f.write(f"- **Retrieval Accuracy:** **{retrieval_accuracy}%** ({retrieval_success_count}/{len(SYLLABUS_TEST_QUESTIONS)})\n")
        f.write(f"- **Average Keyword / Concept Recall:** **{avg_keyword_coverage}%**\n")
        f.write(f"- **Mean Latency:** **{avg_latency}s** on CPU\n\n")
        f.write("## Test Question Breakdown\n\n")
        f.write("| ID | Unit | Category | Question | Retrieved Topic | Match (%) | Latency (s) | Status |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in results:
            status = "✅ PASS" if r["retrieval_success"] else "❌ FAIL"
            f.write(f"| {r['id']} | {r['unit']} | {r['category']} | {r['question']} | {r['retrieved_topic']} | {r['keyword_match_percent']}% | {r['latency_sec']}s | {status} |\n")

    logger.info(f"Syllabus evaluation completed successfully. Saved to {json_path} and {md_path}")
    return summary


if __name__ == "__main__":
    run_syllabus_evaluation()
