# RAG-DocQA: Retrieval-Augmented Generation Based Document Question Answering

A complete, locally-runnable RAG system. It ingests PDF/TXT/MD documents, indexes them with dense
embeddings (FAISS) and BM25, retrieves evidence with dense / lexical / hybrid (RRF) / reranked strategies,
and generates grounded answers with FLAN-T5. **No API key, no cloud, no GPU required.**

```
 Documents ─► Chunking ─► Embeddings (MiniLM) ─► FAISS index ─┐
 (PDF/TXT/MD)  150 words,    + BM25 lexical index              │
               30 overlap                                      ▼
 Question ─► Retriever: dense | bm25 | hybrid (RRF) | hybrid + cross-encoder rerank
                                  │  top-k chunks
                                  ▼
            Confidence gate (abstain if max cosine < 0.20)
                                  ▼
            FLAN-T5 prompt: "answer only from context" ─► Answer + cited sources
```

## Project structure
```
rag-docqa/
├── rag/              # core library (config, ingest, index, retriever, generator, pipeline, metrics)
├── app.py            # Streamlit web UI
├── cli.py            # command-line interface (index / ask / chat)
├── evaluate.py       # reproducible evaluation + ablation
├── data/sample_docs/ # 3 bundled demo documents
├── data/eval/        # 14 hand-written QA pairs with gold evidence phrases
├── tests/            # unit tests (no model downloads needed)
├── docs/REPORT.md    # academic write-up (method, protocol, limitations, references)
└── requirements.txt
```

## Beginner step-by-step

**Requirements:** Python 3.9 - 3.12, ~3 GB free disk, internet for the *first* run (models download once).

1. **Open a terminal** in the project folder (`cd rag-docqa`).
2. **Create a virtual environment**
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
3. **Install dependencies:** `pip install -r requirements.txt`
4. **Run the unit tests (optional sanity check):** `python -m pytest -q`
5. **Build the index from the sample documents:** `python cli.py index --docs data/sample_docs`
6. **Ask a question:** `python cli.py ask "What is FAISS?" --mode hybrid_rerank`
7. **Launch the web app:** `streamlit run app.py` (opens http://localhost:8501). Upload your own PDFs in
   the sidebar, click *Index uploaded files*, then ask questions.
8. **Run the evaluation:** `python evaluate.py` (add `--no_generation` for a fast retrieval-only run).
   Results are written to `results/metrics.csv`, `results/results.md`, `results/predictions.json`.

First run downloads: all-MiniLM-L6-v2 (~90 MB), FLAN-T5-base (~1 GB), cross-encoder reranker (~90 MB).

## Using your own documents
Put `.pdf`, `.txt` or `.md` files in a folder and run `python cli.py index --docs path/to/folder`.
Scanned (image-only) PDFs contain no extractable text and need OCR first.
To evaluate on your data, create a JSON list of `{"question", "answer", "evidence"}` objects
(`evidence` = short verbatim phrase that must appear in a relevant chunk) and pass `--eval your.json --docs folder`.

## Configuration
Everything is in `rag/config.py`: models, chunk size/overlap, `top_k`, RRF constant, abstention threshold,
generation length. For better answers swap `gen_model` for `google/flan-t5-large` (slower, ~3 GB).

## Evaluation protocol
| Metric | Meaning |
|---|---|
| Hit@1 / Hit@k | a retrieved chunk contains the gold evidence phrase at rank 1 / within top-k |
| MRR@k | mean reciprocal rank of the first relevant chunk |
| P@k | fraction of top-k chunks that are relevant |
| EM / F1 | SQuAD-style exact match and token F1 of the generated answer vs gold |
| Contains | gold answer string appears inside the generated answer (lenient) |
| Abstain | fraction of questions where the confidence gate refused to answer |

`evaluate.py` runs the ablation dense vs BM25 vs hybrid vs hybrid+rerank. The bundled set is a small smoke-test
benchmark (14 questions, 3 short documents); see `docs/REPORT.md` for limitations and how to scale it.

## Troubleshooting
| Problem | Fix |
|---|---|
| `No index found` | run `python cli.py index --docs data/sample_docs` first |
| `ModuleNotFoundError` | activate the venv and re-run `pip install -r requirements.txt` |
| Slow answers | normal on CPU (a few seconds); lower `num_beams` in `config.py` |
| Model download blocked | download once on a connected machine; set `HF_HOME` to copy the cache |
| `faiss` install fails | use Python 3.10-3.12 or `conda install -c conda-forge faiss-cpu` |

## License
MIT (see `LICENSE`).
