"""
Academic Evaluation and Benchmarking Script.
Evaluates the educational chatbot on a curated dataset of NLP/DL questions,
recording real-world latency, token counts, and syllabus concept alignment.
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

# Curated academic evaluation dataset
EVALUATION_DATASET = [
    {
        "id": "EVAL_01",
        "unit": "Unit 4",
        "question": "What is the difference between LSTM and GRU?",
        "expected_concepts": ["gate", "forget", "hidden", "cell state", "parameter"]
    },
    {
        "id": "EVAL_02",
        "unit": "Unit 4",
        "question": "Why do vanilla Recurrent Neural Networks suffer from the vanishing gradient problem?",
        "expected_concepts": ["gradient", "time", "backpropagation", "decay", "eigenvalue"]
    },
    {
        "id": "EVAL_03",
        "unit": "Unit 4",
        "question": "Explain Word2Vec and how CBOW differs from Skip-gram.",
        "expected_concepts": ["vector", "cbow", "skip-gram", "context", "target"]
    },
    {
        "id": "EVAL_04",
        "unit": "Unit 5",
        "question": "Explain the Encoder-Decoder architecture used in Seq2Seq models.",
        "expected_concepts": ["encoder", "decoder", "representation", "sequence", "bottleneck"]
    },
    {
        "id": "EVAL_05",
        "unit": "Unit 5",
        "question": "What is the Attention Mechanism and how does it solve the Seq2Seq bottleneck?",
        "expected_concepts": ["query", "key", "value", "softmax", "weight"]
    },
    {
        "id": "EVAL_06",
        "unit": "Unit 5",
        "question": "What is the difference between Self-Attention and Cross-Attention in Transformers?",
        "expected_concepts": ["self-attention", "cross-attention", "encoder", "decoder"]
    },
    {
        "id": "EVAL_07",
        "unit": "Unit 5",
        "question": "Explain Neural Machine Translation and how Seq2Seq is applied.",
        "expected_concepts": ["translation", "source", "target", "sequence"]
    },
    {
        "id": "EVAL_08",
        "unit": "General NLP",
        "question": "What is tokenization and why is subword tokenization like SentencePiece important?",
        "expected_concepts": ["token", "subword", "vocabulary", "out-of-vocabulary"]
    }
]


def run_benchmark(model_name: str = FALLBACK_MODEL_NAME) -> Dict[str, Any]:
    """Runs evaluation benchmark on the academic dataset and computes metrics."""
    logger.info(f"Starting benchmark evaluation with model: {model_name}...")
    bot = EducationalChatbotModel(model_name=model_name)

    results = []
    latencies = []
    output_tokens_list = []
    concept_recall_scores = []

    for item in EVALUATION_DATASET:
        q_id = item["id"]
        q_text = item["question"]
        expected_concepts = item["expected_concepts"]

        logger.info(f"Evaluating {q_id}: {q_text}")
        response = bot.generate_response(q_text)

        answer = response["answer"].lower()
        latency = response["latency_sec"]
        out_tokens = response.get("output_tokens", len(answer.split()))

        # Calculate concept recall (percentage of expected syllabus keywords matched)
        matched_concepts = [c for c in expected_concepts if c in answer]
        concept_recall = round(len(matched_concepts) / len(expected_concepts) * 100, 1)

        latencies.append(latency)
        output_tokens_list.append(out_tokens)
        concept_recall_scores.append(concept_recall)

        results.append({
            "id": q_id,
            "unit": item["unit"],
            "question": q_text,
            "latency_sec": latency,
            "output_tokens": out_tokens,
            "matched_concepts": matched_concepts,
            "concept_recall_percent": concept_recall,
            "syllabus_unit_inferred": response.get("syllabus_unit", "Unknown"),
            "answer_preview": response["answer"][:180] + "..." if len(response["answer"]) > 180 else response["answer"]
        })

    # Summary Statistics
    avg_latency = round(sum(latencies) / len(latencies), 3)
    min_latency = round(min(latencies), 3)
    max_latency = round(max(latencies), 3)
    avg_tokens = round(sum(output_tokens_list) / len(output_tokens_list), 1)
    avg_concept_recall = round(sum(concept_recall_scores) / len(concept_recall_scores), 1)

    benchmark_summary = {
        "model_name": model_name,
        "device": bot.device,
        "total_test_cases": len(EVALUATION_DATASET),
        "average_latency_sec": avg_latency,
        "min_latency_sec": min_latency,
        "max_latency_sec": max_latency,
        "average_output_tokens": avg_tokens,
        "average_concept_recall_percent": avg_concept_recall,
        "details": results
    }

    # Save to docs folder
    output_json_path = DOCS_DIR / "evaluation_results.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    # Save Markdown report
    output_md_path = DOCS_DIR / "evaluation_report.md"
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Academic Chatbot Evaluation & Performance Benchmark Report\n\n")
        f.write(f"- **Evaluated Model:** `{model_name}`\n")
        f.write(f"- **Execution Device:** `{bot.device.upper()}`\n")
        f.write(f"- **Total Test Questions:** {len(EVALUATION_DATASET)}\n")
        f.write(f"- **Mean Latency:** {avg_latency} seconds\n")
        f.write(f"- **Latency Range:** [{min_latency}s, {max_latency}s]\n")
        f.write(f"- **Mean Concept Coverage:** {avg_concept_recall}%\n")
        f.write(f"- **Mean Output Tokens:** {avg_tokens}\n\n")
        f.write(f"## Detailed Question-by-Question Results\n\n")
        f.write("| ID | Curriculum Unit | Question | Latency (s) | Concept Coverage | Answer Preview |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            clean_preview = r["answer_preview"].replace("\n", " ").replace("|", "/")
            f.write(f"| {r['id']} | {r['unit']} | {r['question']} | {r['latency_sec']}s | {r['concept_recall_percent']}% | {clean_preview} |\n")

    logger.info(f"Benchmark finished. Results saved to {output_json_path} and {output_md_path}")
    return benchmark_summary


if __name__ == "__main__":
    import sys
    model_to_run = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_NAME
    run_benchmark(model_to_run)
