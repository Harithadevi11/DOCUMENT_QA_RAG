"""Reproducible evaluation: retrieval ablation + end-to-end answer quality.

Usage:
  python evaluate.py                       # all 4 retrieval modes, with generation
  python evaluate.py --no_generation       # retrieval metrics only (fast)
  python evaluate.py --docs my_docs --eval my_eval.json --top_k 5
"""
import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

from rag import metrics as M
from rag.config import Config
from rag.pipeline import RAGPipeline
from rag.retriever import MODES


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/sample_docs")
    ap.add_argument("--eval", default="data/eval/eval_set.json")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--no_generation", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = replace(Config(), index_dir="storage/eval_index", top_k=args.top_k)
    pipe = RAGPipeline(cfg)
    n_chunks = pipe.build_index(args.docs)
    data = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    print(f"Indexed {n_chunks} chunks | {len(data)} evaluation questions | top_k={args.top_k}\n")

    rows, predictions = [], []
    for mode in args.modes:
        hit1, hitk, rr, prec, em, f1, cont, abst, t_ret, t_gen = ([] for _ in range(10))
        for item in data:
            t0 = time.perf_counter()
            hits = pipe.retrieve(item["question"], mode, args.top_k)
            t_ret.append(time.perf_counter() - t0)

            relevant = [M.evidence_in_text(item["evidence"], h["text"]) for h in hits]
            rank = next((i + 1 for i, ok in enumerate(relevant) if ok), None)
            hit1.append(float(rank == 1))
            hitk.append(float(rank is not None))
            rr.append(M.reciprocal_rank(rank))
            prec.append(sum(relevant) / len(relevant) if relevant else 0.0)

            record = {"mode": mode, "question": item["question"], "gold": item["answer"], "first_relevant_rank": rank}
            if not args.no_generation:
                t1 = time.perf_counter()
                answer, abstained = pipe.generate_answer(item["question"], hits)
                t_gen.append(time.perf_counter() - t1)
                em.append(M.exact_match(answer, item["answer"]))
                f1.append(M.f1_score(answer, item["answer"]))
                cont.append(M.contains_gold(answer, item["answer"]))
                abst.append(float(abstained))
                record.update({"prediction": answer, "abstained": abstained})
            predictions.append(record)

        row = {
            "mode": mode,
            f"Hit@1": mean(hit1),
            f"Hit@{args.top_k}": mean(hitk),
            f"MRR@{args.top_k}": mean(rr),
            f"P@{args.top_k}": mean(prec),
            "retrieval_ms": 1000 * mean(t_ret),
        }
        if not args.no_generation:
            row.update({"EM": mean(em), "F1": mean(f1), "Contains": mean(cont),
                        "Abstain": mean(abst), "generation_s": mean(t_gen)})
        rows.append(row)

    headers = list(rows[0].keys())
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    table = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    table += ["| " + " | ".join(fmt(r[h]) for h in headers) + " |" for r in rows]
    print("\n".join(table))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    (out / "results.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    (out / "predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved metrics.csv, results.md, predictions.json to '{out}/'")


if __name__ == "__main__":
    main()
