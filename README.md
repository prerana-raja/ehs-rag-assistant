# EHS Procedure Assistant (RAG)

A small retrieval-augmented generation (RAG) app that answers Environment, Health & Safety
questions **only from your own documents** and cites the passages it used.

> All sample documents in `data/docs` are **fictional** and written for this demo.
> Do not use them as real safety guidance.

## Why this project
EHS teams keep procedures in long PDFs that are hard to search during a job. This prototype shows
the core pattern for a grounded assistant: ingest documents, retrieve the relevant passages,
and let an LLM answer with citations or admit when the documents don't contain the answer.

## How it works
```
docs (.md/.txt/.pdf) -> heading-aware chunks (overlap) -> index
question -> top-k retrieval -> prompt with numbered context -> LLM answer with [1] [2] citations
```
| Part | File | Notes |
|---|---|---|
| Ingestion and chunking | `src/ingest.py` | Splits on headings, then ~120-word chunks with 25-word overlap; PDF support via `pypdf` |
| Retrieval | `src/retriever.py` | TF-IDF by default (no downloads or keys). `RAG_BACKEND=embeddings` switches to sentence embeddings |
| Generation | `src/generate.py` | Claude via the Anthropic API with a "context only, cite sources, say so if unknown" prompt. Without an API key it shows the top 3 matching passages |
| Interfaces | `app.py`, `src/cli.py` | Streamlit UI and command line |
| Evaluation | `src/evaluate.py` | hit@k of the correct source file and section on `data/eval_questions.json`; `--compare` runs both backends |
| Tests | `tests/` | chunking, retrieval, refusal on unrelated questions, prompt format, LLM-failure fallback, running from any directory |

## Run it
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.cli "Who can remove my lock during lockout tagout?"
streamlit run app.py
python -m pytest -q
python -m src.evaluate
python -m src.evaluate --compare   # needs sentence-transformers and internet on first run
```
For LLM answers set `ANTHROPIC_API_KEY` (and optionally `RAG_MODEL`; default `claude-sonnet-5-5`). If the API call fails, the app falls back to showing the top passages. For embeddings:
`pip install sentence-transformers` then `RAG_BACKEND=embeddings`.

To use your own documents, drop `.md`, `.txt` or `.pdf` files into `data/docs/`.

## Results: TF-IDF vs embeddings
Retrieval hit rate on 16 labelled questions in `data/eval_questions.json` (written by me, half of them
deliberately paraphrased so they share few words with the documents). *source* = correct file in the
top-k chunks, *section* = correct file **and** heading in the top-k chunks.

Reproduce with `python -m src.evaluate --compare`.

| backend | source@1 | source@3 | section@1 | section@3 |
|---|---|---|---|---|
| tfidf | 100% | 100% | 50% | 94% |
| embeddings (all-MiniLM-L6-v2) | _run it and paste here_ | | | |

What the TF-IDF numbers show: it almost always finds the right *document*, but picks the right
*passage* first only half the time. Example: "How long must the fire watch stay after welding?"
returns the hot-work *Scope* section (it contains "welding") rather than *After finishing*.
TF-IDF matches words, not meaning. Passing the top 3-4 chunks to the LLM hides this in practice.

Caveats: 16 questions is a smoke test, not a benchmark, and the corpus is five short documents.

## Known limitations
- TF-IDF matches words, not meaning, so it often ranks the right document first but the wrong passage (see results above).
- No reranking, no hybrid search, no conversation memory, no scanned-PDF OCR.
- PDFs have no markdown headings, so they are chunked by length only and have no section label.
- The evaluation set is small and written by me; treat it as a smoke test.

## Ideas to extend
- A larger, independently written question set
- Hybrid retrieval (TF-IDF + embeddings) and a reranker
- Store vectors in a database (FAISS, pgvector) and add a REST API with FastAPI
- A feedback button in the UI to log unhelpful answers
