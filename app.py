"""Streamlit web interface.  Run:  streamlit run app.py"""
from pathlib import Path

import streamlit as st

from rag.pipeline import RAGPipeline
from rag.retriever import MODES

UPLOAD_DIR = Path("data/uploads")

st.set_page_config(page_title="RAG Document QA", page_icon="📚", layout="wide")
st.title("📚 RAG Document Question Answering")
st.caption("Retrieval (FAISS + BM25 + optional reranker) → FLAN-T5 generation. Runs fully offline after first model download.")


@st.cache_resource(show_spinner="Loading models...")
def get_pipeline() -> RAGPipeline:
    pipe = RAGPipeline()
    try:
        pipe.load_index()
    except FileNotFoundError:
        pass
    return pipe


pipe = get_pipeline()

with st.sidebar:
    st.header("1. Index documents")
    uploads = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Index uploaded files", disabled=not uploads):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for old in UPLOAD_DIR.iterdir():
            if old.is_file():
                old.unlink()
        for f in uploads:
            (UPLOAD_DIR / Path(f.name).name).write_bytes(f.getbuffer())
        with st.spinner("Indexing..."):
            n = pipe.build_index(str(UPLOAD_DIR))
        st.success(f"Indexed {n} chunks.")
    if st.button("Index bundled sample documents"):
        with st.spinner("Indexing..."):
            n = pipe.build_index("data/sample_docs")
        st.success(f"Indexed {n} chunks.")

    st.header("2. Settings")
    mode = st.selectbox("Retrieval mode", MODES, index=2)
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 6, 3)

question = st.text_input("Ask a question about your documents")
if st.button("Ask", type="primary") and question.strip():
    if pipe.retriever is None:
        st.warning("Index some documents first (see sidebar).")
    else:
        with st.spinner("Thinking..."):
            result = pipe.ask(question.strip(), mode, top_k)
        st.subheader("Answer")
        st.write(result["answer"])
        if result["abstained"]:
            st.info("Retrieval confidence was too low, so the system abstained instead of guessing.")
        st.caption(f"Retrieval {result['retrieval_seconds']}s · Generation {result['generation_seconds']}s")
        st.subheader("Retrieved evidence")
        for s in result["sources"]:
            with st.expander(f"#{s['rank']} · {s['source']} · page {s['page']} · cosine {s['dense_score']:.3f}"):
                st.write(s["text"])
