"""Retrieval evaluation.

    python -m src.evaluate            # current backend (RAG_BACKEND, default tfidf)
    python -m src.evaluate --compare  # tfidf vs embeddings, prints a markdown table

Metrics (hit@k): is the expected *source file* / *section* among the top-k chunks?
Section-level is the stricter one, since it checks that the exact passage was found.
"""
import json
import sys
from .ingest import load_chunks, DOCS_DIR, EVAL_PATH
from .retriever import Retriever


def hit_rate(retriever, questions, k=3, level="source"):
    hits = 0
    for item in questions:
        res = [c for c, _ in retriever.search(item["q"], k=k, min_score=0.0)]
        if level == "source":
            hits += any(c.source == item["expected_source"] for c in res)
        else:
            hits += any(c.source == item["expected_source"]
                        and c.section == item["expected_section"] for c in res)
    return hits / len(questions)


def evaluate(backend):
    with open(EVAL_PATH, encoding="utf-8") as f:
        qs = json.load(f)
    r = Retriever(load_chunks(DOCS_DIR), backend=backend)
    return {f"{lvl}@{k}": hit_rate(r, qs, k, lvl)
            for lvl in ("source", "section") for k in (1, 3)}


def table(results):
    cols = ["source@1", "source@3", "section@1", "section@3"]
    lines = ["| backend | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, res in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{res[c]:.0%}" for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    import os
    if "--compare" in sys.argv:
        out = {}
        for b in ("tfidf", "embeddings"):
            try:
                out[b] = evaluate(b)
            except ImportError:
                print(f"Skipping {b}: pip install sentence-transformers", file=sys.stderr)
            except Exception as e:  # e.g. model download failed (offline)
                print(f"Skipping {b}: {type(e).__name__}: {e}", file=sys.stderr)
        if not out:
            sys.exit("No backend could be evaluated.")
        print(table(out))
    else:
        b = os.getenv("RAG_BACKEND", "tfidf")
        print(table({b: evaluate(b)}))
