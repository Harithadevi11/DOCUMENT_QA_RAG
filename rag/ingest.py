"""Document loading and chunking."""
import re
from pathlib import Path
from typing import Dict, List

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_file(path: Path) -> List[Dict]:
    """Return a list of {source, page, text} records for one file."""
    path = Path(path)
    suffix = path.suffix.lower()
    records: List[Dict] = []
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for page_no, page in enumerate(reader.pages, start=1):
            text = _clean(page.extract_text() or "")
            if text:
                records.append({"source": path.name, "page": page_no, "text": text})
    elif suffix in {".txt", ".md"}:
        text = _clean(path.read_text(encoding="utf-8", errors="ignore"))
        if text:
            records.append({"source": path.name, "page": 1, "text": text})
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return records


def load_documents(folder: str) -> List[Dict]:
    """Load every supported file in a folder (non-recursive)."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Document folder not found: {folder}")
    docs: List[Dict] = []
    for path in sorted(folder_path.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(read_file(path))
    if not docs:
        raise ValueError(f"No readable .pdf/.txt/.md documents found in {folder}")
    return docs


def chunk_documents(docs: List[Dict], chunk_words: int = 150, overlap: int = 30) -> List[Dict]:
    """Split page-level records into overlapping word windows."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap < 0 or overlap >= chunk_words:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_words")

    chunks: List[Dict] = []
    for doc in docs:
        words = doc["text"].split()
        start, n = 0, 0
        while start < len(words):
            end = start + chunk_words
            piece = words[start:end]
            chunks.append(
                {
                    "id": f"{Path(doc['source']).stem}-p{doc['page']}-c{n}",
                    "source": doc["source"],
                    "page": doc["page"],
                    "text": " ".join(piece),
                }
            )
            n += 1
            if end >= len(words):
                break
            start = end - overlap
    return chunks
