"""Retrieval. Default: TF-IDF (no downloads, no keys). Optional: sentence embeddings."""
from __future__ import annotations
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .ingest import Chunk


class Retriever:
    def __init__(self, chunks: list[Chunk], backend: str | None = None):
        if not chunks:
            raise ValueError("No chunks to index. Add .md/.txt/.pdf files to data/docs.")
        self.chunks = chunks
        self.backend = backend or os.getenv("RAG_BACKEND", "tfidf")
        texts = [f"{c.section}. {c.text}" for c in chunks]
        if self.backend == "embeddings":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._matrix = self._model.encode(texts, normalize_embeddings=True)
        else:
            self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                        sublinear_tf=True)
            self._matrix = self._vec.fit_transform(texts)

    def search(self, query: str, k: int = 4, min_score: float = 0.05):
        """Return [(Chunk, score)] sorted best-first, dropping weak matches."""
        if self.backend == "embeddings":
            q = self._model.encode([query], normalize_embeddings=True)[0]
            scores = self._matrix @ q
        else:
            scores = (self._matrix @ self._vec.transform([query]).T).toarray().ravel()
        idx = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in idx if scores[i] >= min_score]
