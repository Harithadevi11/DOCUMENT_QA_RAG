import pytest

from rag.ingest import chunk_documents, load_documents


def make_doc(n_words):
    return [{"source": "a.txt", "page": 1, "text": " ".join(f"w{i}" for i in range(n_words))}]


def test_chunk_sizes_and_overlap():
    chunks = chunk_documents(make_doc(100), chunk_words=40, overlap=10)
    assert [len(c["text"].split()) for c in chunks] == [40, 40, 40]
    assert chunks[0]["text"].split()[-10:] == chunks[1]["text"].split()[:10]


def test_short_document_single_chunk():
    chunks = chunk_documents(make_doc(5), chunk_words=40, overlap=10)
    assert len(chunks) == 1 and chunks[0]["id"] == "a-p1-c0"


def test_every_word_is_covered():
    chunks = chunk_documents(make_doc(103), chunk_words=30, overlap=7)
    covered = set(w for c in chunks for w in c["text"].split())
    assert covered == {f"w{i}" for i in range(103)}


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_documents(make_doc(10), chunk_words=5, overlap=5)


def test_load_sample_documents():
    docs = load_documents("data/sample_docs")
    assert {d["source"] for d in docs} == {"rag_overview.txt", "vector_search.txt", "transformers.txt"}
