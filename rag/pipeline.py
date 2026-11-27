"""End-to-end RAG pipeline: ingest -> index -> retrieve -> generate."""
import time
from typing import Dict, List, Optional, Tuple

from .config import Config
from .ingest import chunk_documents, load_documents
from .index import VectorIndex
from .retriever import Retriever

ABSTAIN = "I don't know based on the provided documents."


class RAGPipeline:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.index = VectorIndex(self.cfg.embed_model)
        self.retriever: Optional[Retriever] = None
        self._generator = None

    @property
    def generator(self):
        if self._generator is None:
            from .generator import Generator

            self._generator = Generator(self.cfg)
        return self._generator

    def build_index(self, docs_dir: Optional[str] = None) -> int:
        docs = load_documents(docs_dir or self.cfg.docs_dir)
        chunks = chunk_documents(docs, self.cfg.chunk_words, self.cfg.chunk_overlap)
        self.index = VectorIndex(self.cfg.embed_model)
        self.index.build(chunks)
        self.index.save(self.cfg.index_dir)
        self.retriever = Retriever(self.index, self.cfg)
        return len(chunks)

    def load_index(self) -> int:
        self.index = VectorIndex(self.cfg.embed_model)
        self.index.load(self.cfg.index_dir)
        self.retriever = Retriever(self.index, self.cfg)
        return len(self.index.chunks)

    def retrieve(self, question: str, mode: str = "hybrid", top_k: Optional[int] = None) -> List[Dict]:
        if self.retriever is None:
            raise RuntimeError("No index loaded. Call build_index() or load_index() first.")
        return self.retriever.search(question, mode=mode, top_k=top_k or self.cfg.top_k)

    def generate_answer(self, question: str, hits: List[Dict]) -> Tuple[str, bool]:
        """Returns (answer, abstained)."""
        if not hits or max(h["dense_score"] for h in hits) < self.cfg.min_dense_score:
            return ABSTAIN, True
        return self.generator.generate(question, hits), False

    def ask(self, question: str, mode: str = "hybrid", top_k: Optional[int] = None) -> Dict:
        t0 = time.perf_counter()
        hits = self.retrieve(question, mode, top_k)
        t1 = time.perf_counter()
        answer, abstained = self.generate_answer(question, hits)
        t2 = time.perf_counter()
        return {
            "question": question,
            "answer": answer,
            "abstained": abstained,
            "mode": mode,
            "sources": hits,
            "retrieval_seconds": round(t1 - t0, 3),
            "generation_seconds": round(t2 - t1, 3),
        }
