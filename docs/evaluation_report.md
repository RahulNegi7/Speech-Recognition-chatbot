# Academic Chatbot Evaluation & Performance Benchmark Report

- **Evaluated Model:** `google/flan-t5-base`
- **Execution Device:** `CPU`
- **Total Test Questions:** 8
- **Mean Latency:** 3.124 seconds
- **Latency Range:** [1.924s, 5.252s]
- **Mean Concept Coverage:** 88.1%
- **Mean Output Tokens:** 33.2

## Detailed Question-by-Question Results

| ID | Curriculum Unit | Question | Latency (s) | Concept Coverage | Answer Preview |
|---|---|---|---|---|---|
| EVAL_01 | Unit 4 | What is the difference between LSTM and GRU? | 5.011s | 100.0% | ### Comparison: Long Short-Term Memory (LSTM) vs Gated Recurrent Unit (GRU)  **Key Differences Between LSTM and GRU:**  1. **Number of Gates:**    - **LSTM (Hochreiter & Schmidhube... |
| EVAL_02 | Unit 4 | Why do vanilla Recurrent Neural Networks suffer from the vanishing gradient problem? | 2.227s | 100.0% | ### Vanishing Gradient Problem in RNNs  **The Vanishing Gradient Problem:**  1. **Mathematical Cause:** During Backpropagation Through Time (BPTT), gradients are multiplied across ... |
| EVAL_03 | Unit 4 | Explain Word2Vec and how CBOW differs from Skip-gram. | 5.252s | 100.0% | ### Word Embeddings & Dense Vector Representations  **Word Embeddings:**  1. **Definition:** Continuous dense vector representations where semantically related words are placed clo... |
| EVAL_04 | Unit 5 | Explain the Encoder-Decoder architecture used in Seq2Seq models. | 2.159s | 100.0% | ### Encoder-Decoder Architecture & Seq2Seq  **Encoder-Decoder Architecture (Seq2Seq):**  1. **Core Concept (Sutskever et al. 2014 / Cho et al. 2014):**    Designed to map an input ... |
| EVAL_05 | Unit 5 | What is the Attention Mechanism and how does it solve the Seq2Seq bottleneck? | 3.253s | 80.0% | ### Attention Mechanism in NLP  **The Attention Mechanism:**  1. **Origin:** Proposed by Bahdanau et al. (2014) and Luong et al. (2015) for Neural Machine Translation to overcome t... |
| EVAL_06 | Unit 5 | What is the difference between Self-Attention and Cross-Attention in Transformers? | 2.788s | 100.0% | ### Attention Mechanism in NLP  **The Attention Mechanism:**  1. **Origin:** Proposed by Bahdanau et al. (2014) and Luong et al. (2015) for Neural Machine Translation to overcome t... |
| EVAL_07 | Unit 5 | Explain Neural Machine Translation and how Seq2Seq is applied. | 1.924s | 75.0% | ### Neural Machine Translation (NMT)  **Neural Machine Translation (NMT):**  1. **Definition:** An end-to-end deep learning framework for translating text from a source language to... |
| EVAL_08 | General NLP | What is tokenization and why is subword tokenization like SentencePiece important? | 2.376s | 50.0% | The SentencePiece is a subword tokenization mechanism that is used in Natural Language Processing and Deep Learning. |
