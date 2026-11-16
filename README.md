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
| Generation | `src/generate.py` | Claude via the Anthropic API with a "context only, cite sources, say so if unknown" prompt. Without an API key, or if the API call fails, it shows the top 3 matching passages |
| Interfaces | `app.py`, `src/cli.py` | Streamlit UI and command line |
| Evaluation | `src/evaluate.py` | hit@k of the correct source file and section on `data/eval_questions.json`; `--compare` runs both backends |
| Tests | `tests/` | chunking, retrieval, refusal on unrelated questions, prompt format, LLM-failure fallback, running from any directory |

## Run it

Setup (Windows PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Setup (Mac / Linux):
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then:
```bash
python -m src.cli "Who can remove my lock during lockout tagout?"
streamlit run app.py
python -m pytest -q
python -m src.evaluate
```

**LLM answers:** set the `ANTHROPIC_API_KEY` environment variable (and optionally `RAG_MODEL`;
default `claude-sonnet-5-5`). Without a key, or if the API call fails, the app falls back to
showing the top matching passages, so everything else runs with no key and no cost.

**Embeddings backend (optional):**
```bash
python -m pip install sentence-transformers
python -m src.evaluate --compare      # needs internet on the first run to download the model
```
To use embeddings in the app or CLI, set `RAG_BACKEND=embeddings`.
On Windows, if PyTorch fails to load with a `WinError 126` error, install the
[Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) and retry.

To use your own documents, drop `.md`, `.txt` or `.pdf` files into `data/docs/`.

## Results: TF-IDF vs embeddings
Retrieval hit rate on 16 labelled questions in `data/eval_questions.json` (written by me, half of them
deliberately paraphrased so they share few words with the documents). *source* = correct file in the
top-k chunks, *section* = correct file **and** heading in the top-k chunks.

Reproduce with `python -m src.evaluate --compare`.

| backend | source@1 | source@3 | section@1 | section@3 |
|---|---|---|---|---|
| tfidf | 100% | 100% | 50% | 94% |
| embeddings (all-MiniLM-L6-v2) | 100% | 100% | 75% | 100% |

**What the numbers show**
- Both backends always find the right *document* (source@1 and source@3 are 100% for both), so
  that metric does not separate them on this corpus.
- The difference is at the *passage* level. TF-IDF ranks the correct section first on 8 of 16
  questions; embeddings do so on 12 of 16 (section@1: 50% -> 75%). At top-3, TF-IDF misses 1
  question and embeddings miss none (94% -> 100%).
- This fits how the two methods work. TF-IDF matches words, so a question like "How long must the
  fire watch stay after welding?" returns the hot-work *Scope* section (it contains "welding")
  rather than *After finishing*. Embeddings match meaning, which helps most on the paraphrased questions.
- Passing the top 3-4 chunks to the LLM hides most of the TF-IDF ranking errors in practice, but
  better first-rank precision means fewer irrelevant passages in the prompt.

**Caveats:** 16 questions is a smoke test, not a benchmark. One question is worth 6.25 percentage
points, so the section@1 gap is 4 questions. The corpus is five short documents, the questions
were written by me, and each number comes from a single run. Embeddings still missed 4 of 16
questions at rank 1.

## Known limitations
- TF-IDF matches words, not meaning, so it often ranks the right document first but the wrong passage.
- Embeddings are better at passage ranking here but still not perfect (75% section@1).
- No reranking, no hybrid search, no conversation memory, no scanned-PDF OCR.
- PDFs have no markdown headings, so they are chunked by length only and have no section label.
- The evaluation set is small and written by me; treat it as a smoke test.
- The LLM answer path depends on an Anthropic API key; the retrieval, evaluation and tests do not.

## Ideas to extend
- A larger, independently written question set, and a look at which questions still fail
- Hybrid retrieval (TF-IDF + embeddings) and a reranker
- Store vectors in a database (FAISS, pgvector) and add a REST API with FastAPI
- A feedback button in the UI to log unhelpful answers
