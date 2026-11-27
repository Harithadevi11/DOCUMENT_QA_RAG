"""Dense vector index built on FAISS and Sentence-Transformers."""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorIndex:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self.index = None
        self.chunks: List[Dict] = []
        self.embeddings: np.ndarray = np.zeros((0, 0), dtype="float32")

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype="float32")

    def build(self, chunks: List[Dict]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        self.chunks = chunks
        self.embeddings = self.embed([c["text"] for c in chunks], show_progress=True)
        # Unit-length vectors: inner product == cosine similarity.
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self.chunks))
        scores, ids = self.index.search(query_vec.reshape(1, -1).astype("float32"), k)
        return scores[0], ids[0]

    def save(self, directory: str) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "faiss.index"))
        (out / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
        (out / "meta.json").write_text(json.dumps({"embed_model": self.model_name}), encoding="utf-8")

    def load(self, directory: str) -> None:
        src = Path(directory)
        if not (src / "faiss.index").exists():
            raise FileNotFoundError(f"No index found in {directory}. Build one first.")
        self.index = faiss.read_index(str(src / "faiss.index"))
        self.chunks = json.loads((src / "chunks.json").read_text(encoding="utf-8"))
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        self.model_name = meta["embed_model"]
        self._model = None
        self.embeddings = self.index.reconstruct_n(0, self.index.ntotal)
