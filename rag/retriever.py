"""Dense, lexical (BM25), hybrid (RRF) and reranked retrieval."""
import re
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from .config import Config
from .index import VectorIndex

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self, index: VectorIndex, cfg: Config):
        self.index = index
        self.cfg = cfg
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in index.chunks])
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.cfg.rerank_model)
        return self._reranker

    def _dense_ranking(self, qvec: np.ndarray, n: int) -> List[int]:
        _, ids = self.index.search(qvec, n)
        return [int(i) for i in ids]

    def _bm25_ranking(self, query: str, n: int) -> List[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)[:n]]

    def _rrf(self, rankings: List[List[int]]) -> List[tuple]:
        fused: Dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.cfg.rrf_k + rank + 1)
        return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    def search(self, query: str, mode: str = "hybrid", top_k: int = 3) -> List[Dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        chunks = self.index.chunks
        n_cand = min(len(chunks), max(self.cfg.candidate_k, top_k))
        qvec = self.index.embed([query])[0]
        dense_all = self.index.embeddings @ qvec  # cosine similarity to every chunk

        if mode == "dense":
            ranked = [(i, float(dense_all[i])) for i in self._dense_ranking(qvec, n_cand)]
        elif mode == "bm25":
            scores = self.bm25.get_scores(tokenize(query))
            ranked = [(i, float(scores[i])) for i in self._bm25_ranking(query, n_cand)]
        else:
            ranked = self._rrf([self._dense_ranking(qvec, n_cand), self._bm25_ranking(query, n_cand)])[:n_cand]
            if mode == "hybrid_rerank":
                pairs = [(query, chunks[i]["text"]) for i, _ in ranked]
                ce_scores = self.reranker.predict(pairs)
                ranked = sorted(
                    [(i, float(s)) for (i, _), s in zip(ranked, ce_scores)],
                    key=lambda kv: kv[1],
                    reverse=True,
                )

        hits = []
        for rank, (i, score) in enumerate(ranked[:top_k], start=1):
            hit = dict(chunks[i])
            hit.update({"rank": rank, "score": float(score), "dense_score": float(dense_all[i])})
            hits.append(hit)
        return hits
