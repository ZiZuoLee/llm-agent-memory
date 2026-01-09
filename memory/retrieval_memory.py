"""Retrieval memory module for storing and searching text memories."""

from __future__ import annotations

import math
from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class RetrievalMemory:
    """Store text memories and retrieve them using cosine similarity."""

    def __init__(self) -> None:
        """Initialize in-memory storage and optional embedding model."""
        self._texts: List[str] = []
        self._embeddings: List[List[float]] = []
        self._model: Optional[SentenceTransformer] = None
        if SentenceTransformer is not None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_memory(self, text: str) -> None:
        """Add a text memory and its embedding to storage."""
        embedding = self._embed_text(text)
        self._texts.append(text)
        self._embeddings.append(embedding)

    def retrieve(self, query: str, k: int) -> List[str]:
        """Retrieve top-k most similar memories for a query."""
        if k <= 0 or not self._texts:
            return []
        query_embedding = self._embed_text(query)
        scored = []
        for text, emb in zip(self._texts, self._embeddings):
            similarity = _cosine_similarity(query_embedding, emb)
            scored.append((similarity, text))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:k]]

    def _embed_text(self, text: str) -> List[float]:
        """Embed text using sentence-transformers or a simple fallback."""
        if self._model is not None:
            return self._model.encode([text])[0].tolist()
        return _mock_embedding(text)


def _mock_embedding(text: str, dim: int = 64) -> List[float]:
    """Create a deterministic mock embedding from character codes."""
    vector = [0.0] * dim
    for idx, char in enumerate(text):
        bucket = idx % dim
        vector[bucket] += float(ord(char))
    return _normalize(vector)


def _normalize(vector: List[float]) -> List[float]:
    """Normalize a vector to unit length."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(length))
    norm_a = math.sqrt(sum(a[i] * a[i] for i in range(length)))
    norm_b = math.sqrt(sum(b[i] * b[i] for i in range(length)))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
