"""Command-line interface.

Examples:
  python cli.py index --docs data/sample_docs
  python cli.py ask "What is FAISS?" --mode hybrid_rerank
  python cli.py chat
"""
import argparse
import sys

from rag.config import Config
from rag.pipeline import RAGPipeline
from rag.retriever import MODES


def show(result: dict) -> None:
    print(f"\nAnswer: {result['answer']}")
    if result["abstained"]:
        print("(retrieval confidence too low - the system abstained)")
    print(f"[retrieval {result['retrieval_seconds']}s | generation {result['generation_seconds']}s | mode={result['mode']}]")
    print("\nSources:")
    for s in result["sources"]:
        preview = s["text"][:140] + ("..." if len(s["text"]) > 140 else "")
        print(f"  #{s['rank']} {s['source']} (page {s['page']}) score={s['score']:.3f} cos={s['dense_score']:.3f}\n     {preview}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG Document QA")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="build the vector index from a folder")
    p_index.add_argument("--docs", default=Config().docs_dir)

    for name in ("ask", "chat"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=MODES, default="hybrid")
        p.add_argument("--top_k", type=int, default=Config().top_k)
        if name == "ask":
            p.add_argument("question")

    args = parser.parse_args()
    pipe = RAGPipeline()

    if args.cmd == "index":
        n = pipe.build_index(args.docs)
        print(f"Indexed {n} chunks from '{args.docs}' into '{pipe.cfg.index_dir}'.")
        return 0

    try:
        n = pipe.load_index()
    except FileNotFoundError:
        print("No index found. Run: python cli.py index --docs data/sample_docs")
        return 1
    print(f"Loaded index with {n} chunks.")

    if args.cmd == "ask":
        show(pipe.ask(args.question, args.mode, args.top_k))
    else:
        print("Type a question (or 'exit').")
        while True:
            q = input("\n> ").strip()
            if q.lower() in {"exit", "quit", ""}:
                break
            show(pipe.ask(q, args.mode, args.top_k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
