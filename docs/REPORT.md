# Retrieval-Augmented Generation (RAG) Based Document Question Answering System

## Abstract
Large language models store knowledge in parameters, which makes it hard to update and prone to unsupported
statements (hallucination). We implement a fully local Retrieval-Augmented Generation system for question
answering over user-supplied documents and study how the retrieval strategy affects evidence quality and
answer quality. The system combines dense retrieval (Sentence-Transformers + FAISS), lexical retrieval (BM25),
Reciprocal Rank Fusion, and an optional cross-encoder reranker, followed by an instruction-tuned FLAN-T5
generator with a confidence-based abstention gate. A reproducible script evaluates four retrieval configurations
with retrieval metrics (Hit@k, MRR, P@k) and answer metrics (EM, F1, Contains).

## 1. Introduction
Closed-book QA with parametric models cannot cite sources, cannot be updated without retraining, and may
fabricate answers. RAG (Lewis et al., 2020) conditions generation on passages retrieved from an external corpus.
Our goals: (i) a transparent, reproducible reference implementation that runs on a CPU; (ii) a controlled
comparison of retrieval strategies; (iii) a user interface showing the evidence behind each answer.

## 2. Related Work
RAG (Lewis et al., 2020) introduced joint retrieval-generation. Dense sentence embeddings (Reimers & Gurevych, 2019)
enable semantic search; FAISS (Johnson et al., 2019) provides efficient vector search; BM25 (Robertson & Zaragoza, 2009)
remains a strong lexical baseline; Reciprocal Rank Fusion (Cormack et al., 2009) combines rankings without score
calibration; cross-encoders (Nogueira & Cho, 2019) rerank candidates with joint query-passage attention;
instruction-tuned T5 models (Raffel et al., 2020; Chung et al., 2022) follow prompts for context-grounded answering.

## 3. Method
1. **Ingestion.** PDF/TXT/MD files are read per page, whitespace-normalised and split into overlapping windows of 150 words
   (30-word overlap) so answers spanning a boundary are not lost. Each chunk keeps source and page metadata.
2. **Indexing.** Chunks are embedded with all-MiniLM-L6-v2 (384-d, L2-normalised) and stored in a FAISS `IndexFlatIP`
   (exact cosine search). A BM25Okapi index is built over the same chunks.
3. **Retrieval modes.** *dense*: cosine top-k. *bm25*: lexical top-k. *hybrid*: RRF,
   score(d) = sum over rankers of 1/(60 + rank_r(d)), over 20 candidates per ranker. *hybrid_rerank*: hybrid candidates
   rescored by the ms-marco-MiniLM-L-6-v2 cross-encoder.
4. **Abstention.** If the best retrieved chunk has cosine similarity below 0.20, the system answers that it does not know.
5. **Generation.** FLAN-T5-base receives the top chunks (as many as fit in 1000 tokens, best first) with an instruction to
   answer briefly using only the context; decoding uses beam search (4 beams, max 64 new tokens).

## 4. Experimental Setup
- **Data:** three short original documents (RAG, vector search, Transformers/embeddings), 6 chunks in total.
- **Questions:** 14 factoid questions, each with a gold answer and a verbatim *evidence phrase*.
- **Relevance judgement:** a chunk is relevant iff it contains the evidence phrase (case/whitespace-insensitive).
- **Metrics:** Hit@1, Hit@k, MRR@k, P@k (retrieval); EM, F1, Contains, Abstain rate (generation); latency.
- **Reproduction:** `python evaluate.py`; outputs `results/metrics.csv`, `results/results.md`, `results/predictions.json`.
  Inference is deterministic (beam search, no sampling), so reruns on the same hardware/library versions match.

## 5. Results
Numbers are produced by `evaluate.py` and written to `results/results.md`; copy that table here when reporting your run.
Results depend on library versions and hardware, so they are not hard-coded in this document.

## 6. Discussion and Limitations
- With only six chunks, top-3 retrieval covers half the corpus, so Hit@3 is expected to saturate and differences between
  modes will mainly show in Hit@1 and MRR. Meaningful retrieval comparisons require a larger corpus: run `evaluate.py`
  on your own documents and evaluation file.
- Substring-based relevance may under-count valid evidence that is paraphrased, and over-count coincidental matches.
- EM is harsh for generative answers; F1 and Contains are more informative. LLM-as-judge or human evaluation would
  further improve validity.
- FLAN-T5-base is small; it handles short factoid answers but is weak at multi-hop reasoning and long synthesis.
- Fixed word-window chunking ignores document structure (headings, tables); PDFs with scanned pages need OCR.
- The abstention threshold (0.20) is a heuristic and should be tuned per corpus.

## 7. Future Work
Semantic/structure-aware chunking, query rewriting, multi-hop retrieval, larger generators (FLAN-T5-large or other
open models), faithfulness metrics (citation precision, NLI-based support), and benchmark evaluation on public QA sets.

## References
- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP.
- Johnson, J., Douze, M., & Jegou, H. (2019). Billion-scale similarity search with GPUs. IEEE Trans. Big Data.
- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. FnTIR.
- Cormack, G., Clarke, C., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods. SIGIR.
- Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. arXiv:1901.04085.
- Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR.
- Chung, H. W. et al. (2022). Scaling Instruction-Finetuned Language Models. arXiv:2210.11416.
