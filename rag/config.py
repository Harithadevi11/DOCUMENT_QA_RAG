"""Central configuration for the RAG system."""
from dataclasses import dataclass


@dataclass
class Config:
    # Models (all run locally; no API key required)
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    gen_model: str = "google/flan-t5-base"

    # Chunking (measured in words)
    chunk_words: int = 150
    chunk_overlap: int = 30

    # Retrieval
    top_k: int = 3
    candidate_k: int = 20      # candidates fetched before fusion / reranking
    rrf_k: int = 60            # constant in Reciprocal Rank Fusion
    min_dense_score: float = -1.0  # below this cosine score the system abstains

    # Generation
    max_input_tokens: int = 1000
    max_new_tokens: int = 64
    num_beams: int = 4

    # Paths
    docs_dir: str = "data/sample_docs"
    index_dir: str = "storage/index"
