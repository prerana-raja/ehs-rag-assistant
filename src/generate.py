"""Answer generation. Uses Claude if ANTHROPIC_API_KEY is set, else an extractive fallback."""
from __future__ import annotations
import os
from .ingest import Chunk

SYSTEM = (
    "You are an EHS (Environment, Health & Safety) assistant. Answer ONLY from the "
    "numbered context passages. Cite the passages you used like [1] or [2]. If the "
    "context does not contain the answer, say you could not find it in the provided "
    "documents and suggest contacting the site EHS team. Never invent procedures, "
    "limits or numbers. Keep answers short and practical."
)


def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    ctx = "\n\n".join(f"[{i}] ({c.label})\n{c.text}" for i, (c, _) in enumerate(hits, 1))
    return f"Context:\n{ctx}\n\nQuestion: {question}"


def _extractive(hits, note):
    body = "\n\n".join(f"[{i}] {c.text}\nSource: {c.label}" for i, (c, _) in enumerate(hits[:3], 1))
    return f"({note})\n\n{body}"


def answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    if not hits:
        return ("I could not find anything relevant in the provided documents. "
                "Please contact your site EHS team.")
    if not os.getenv("ANTHROPIC_API_KEY"):
        return _extractive(hits, "No API key set, showing the best matching passages.")
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=os.getenv("RAG_MODEL", "claude-sonnet-5-5"),
            max_tokens=600,
            system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(question, hits)}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        if text:
            return text
        return _extractive(hits, "The model returned no text, showing the best matching passages.")
    except Exception as e:  # network, auth, rate limit, bad model name, missing package
        return _extractive(hits, f"LLM call failed ({type(e).__name__}), showing the best matching passages.")
