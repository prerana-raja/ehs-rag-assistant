"""Command line: python -m src.cli "your question" """
import sys
from .ingest import load_chunks, DOCS_DIR
from .retriever import Retriever
from .generate import answer


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m src.cli "your question"')
    q = " ".join(sys.argv[1:])
    r = Retriever(load_chunks(DOCS_DIR))
    hits = r.search(q)
    print(answer(q, hits))
    print("\nSources:")
    for i, (c, s) in enumerate(hits, 1):
        print(f"  [{i}] {c.label} (score {s:.2f})")


if __name__ == "__main__":
    main()
