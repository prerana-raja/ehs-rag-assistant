import json
from src.ingest import chunk_words, split_sections, load_chunks, DOCS_DIR, EVAL_PATH
from src.retriever import Retriever
from src.evaluate import hit_rate
from src.generate import build_prompt, answer


def test_chunk_overlap_and_size():
    words = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_words(words, size=100, overlap=20)
    assert all(len(c.split()) <= 100 for c in chunks)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_split_sections_keeps_headings():
    assert split_sections("# A\none\n## B\ntwo") == [("A", "one"), ("B", "two")]


def test_retrieves_right_source():
    r = Retriever(load_chunks(DOCS_DIR), backend="tfidf")
    top = r.search("who can remove a lockout lock", k=1)[0][0]
    assert top.source == "lockout_tagout.md"


def test_eval_hit_rate_reasonable():
    r = Retriever(load_chunks(DOCS_DIR), backend="tfidf")
    qs = json.load(open(EVAL_PATH, encoding="utf-8"))
    assert hit_rate(r, qs, k=3) >= 0.7


def test_unrelated_question_says_not_found(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = Retriever(load_chunks(DOCS_DIR), backend="tfidf")
    hits = r.search("best pizza topping in naples", k=3, min_score=0.15)
    assert "could not find" in answer("best pizza topping in naples", hits)


def test_prompt_contains_numbered_sources():
    r = Retriever(load_chunks(DOCS_DIR), backend="tfidf")
    p = build_prompt("q", r.search("hot work permit"))
    assert "[1]" in p and "Question: q" in p


def test_works_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(load_chunks()) > 0


def test_llm_failure_falls_back_to_passage(monkeypatch):
    import sys, types
    class Boom:
        def __init__(self, *a, **k): raise RuntimeError("no network")
    fake = types.SimpleNamespace(Anthropic=Boom)
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    r = Retriever(load_chunks(DOCS_DIR), backend="tfidf")
    out = answer("hot work permit", r.search("hot work permit"))
    assert "LLM call failed" in out and "Source:" in out
