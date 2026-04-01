"""TF-IDF based context retrieval with cosine similarity."""

import logging
import re

import numpy as np

from cortex.memory.context_store import ContextEvent

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant context events using TF-IDF cosine similarity.

    Lightweight alternative to embedding models — no external dependencies
    beyond numpy. Suitable for on-device retrieval.
    """

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}

    def _tokenize(self, text: str) -> list[str]:
        """Split text into lowercase tokens."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_vocab(self, documents: list[str]) -> None:
        """Build vocabulary from all documents."""
        self._vocabulary = {}
        idx = 0
        for doc in documents:
            for token in self._tokenize(doc):
                if token not in self._vocabulary:
                    self._vocabulary[token] = idx
                    idx += 1

    def _tfidf_vector(
        self, text: str, doc_count: int, doc_freqs: dict[str, int]
    ) -> np.ndarray:
        """Compute TF-IDF vector for a single document."""
        tokens = self._tokenize(text)
        vec = np.zeros(len(self._vocabulary), dtype=np.float32)

        if not tokens:
            return vec

        # Term frequency
        for token in tokens:
            if token in self._vocabulary:
                vec[self._vocabulary[token]] += 1
        vec /= len(tokens)  # normalize TF

        # Inverse document frequency
        for token, idx in self._vocabulary.items():
            if vec[idx] > 0:
                df = doc_freqs.get(token, 1)
                vec[idx] *= np.log((doc_count + 1) / (df + 1)) + 1

        return vec

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def search(
        self,
        query: str,
        events: list[ContextEvent],
        top_k: int = 3,
    ) -> list[tuple[ContextEvent, float]]:
        """Search for events most relevant to the query.

        Args:
            query: Search query string.
            events: List of context events to search through.
            top_k: Number of top results to return.

        Returns:
            List of (event, similarity_score) tuples, sorted by relevance.
        """
        if not events or not query.strip():
            return []

        documents = [e.description for e in events]
        documents.append(query)

        # Build vocabulary
        self._build_vocab(documents)

        if not self._vocabulary:
            return []

        # Document frequencies
        doc_freqs: dict[str, int] = {}
        for doc in documents:
            seen = set()
            for token in self._tokenize(doc):
                if token not in seen:
                    doc_freqs[token] = doc_freqs.get(token, 0) + 1
                    seen.add(token)

        doc_count = len(documents)

        # Compute vectors
        query_vec = self._tfidf_vector(query, doc_count, doc_freqs)
        event_vecs = [
            self._tfidf_vector(doc, doc_count, doc_freqs)
            for doc in documents[:-1]
        ]

        # Score and rank
        scored = []
        for event, vec in zip(events, event_vecs):
            sim = self._cosine_similarity(query_vec, vec)
            scored.append((event, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = scored[:top_k]
        logger.debug(
            "search query='%s' top_k=%d results=%d best=%.3f",
            query[:50],
            top_k,
            len(results),
            results[0][1] if results else 0,
        )
        return results
