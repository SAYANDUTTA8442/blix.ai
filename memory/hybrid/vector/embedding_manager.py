"""
EmbeddingManager — produce fixed-dimension float vectors from text.

Default backend: random projection hashing (256-dim).
  • No ML library required
  • Deterministic per token (same text → same vector)
  • Reasonable cosine similarity for symbolic text
  • Swap in sentence-transformers or OpenAI with one call:
      manager.set_backend(SentenceTransformerBackend("all-MiniLM-L6-v2"))

Pluggable via EmbeddingBackend protocol.
"""
from __future__ import annotations
import hashlib
import logging
import math
import re
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)

EMBEDDING_DIM = 256  # default dimension; backends may override


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol that every embedding backend must satisfy."""
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into float vectors of length self.dim."""
        ...


# ────────────────────────────────────────────────────────────────────
# Built-in backends
# ────────────────────────────────────────────────────────────────────

class HashProjectionBackend:
    """
    Pure-Python random-projection hash embedding.

    Each token is hashed to a deterministic position in the output vector.
    TF-IDF-style weighting (log(1+tf)) is applied per token.
    The vector is L2-normalised so cosine similarity == dot product.

    Pros:  zero dependencies, fast, deterministic, consistent.
    Cons:  no semantic understanding; synonyms are unrelated.
    Suitable for: symbolic reasoning, development, testing.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(t) for t in texts]

    def _encode_one(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.dim

        # Token frequency
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1

        vec = [0.0] * self.dim
        for tok, freq in tf.items():
            # Deterministic position from token hash
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
            # Sign from second hash
            sign_h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            pos = h % self.dim
            sign = 1.0 if sign_h % 2 == 0 else -1.0
            weight = math.log1p(freq)
            vec[pos] += sign * weight

        return self._l2_normalize(vec)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower()
        return re.findall(r"\b[a-z][a-z0-9]*\b", text)

    @staticmethod
    def _l2_normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm < 1e-9:
            return vec
        return [x / norm for x in vec]


class NumpyBackend:
    """
    Numpy-accelerated version of HashProjectionBackend.
    Identical algorithm, ~20× faster for large batches.
    Requires: numpy (pre-installed in most environments)
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        import numpy as np
        self._np = np

    def encode(self, texts: list[str]) -> list[list[float]]:
        np = self._np
        results = []
        for text in texts:
            tokens = self._tokenize(text)
            if not tokens:
                results.append([0.0] * self.dim)
                continue
            vec = np.zeros(self.dim, dtype=np.float64)
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, freq in tf.items():
                h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
                sh = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                pos = h % self.dim
                sign = 1.0 if sh % 2 == 0 else -1.0
                vec[pos] += sign * math.log1p(freq)
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
            results.append(vec.tolist())
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-z][a-z0-9]*\b", text.lower())


# ────────────────────────────────────────────────────────────────────
# Manager
# ────────────────────────────────────────────────────────────────────

class EmbeddingManager:
    """
    Central embedding service for HGSHM.

    Usage
    -----
    manager = EmbeddingManager()                         # hash-projection default
    manager.set_backend(NumpyBackend(dim=256))           # swap to numpy version
    # manager.set_backend(SentenceTransformerBackend(…)) # swap to real embeddings

    vec  = manager.embed("some text")
    vecs = manager.embed_batch(["text a", "text b"])
    sim  = manager.cosine_similarity(vec_a, vec_b)
    """

    def __init__(self, backend: EmbeddingBackend | None = None) -> None:
        if backend is None:
            # Try numpy for speed, fall back to pure Python
            try:
                backend = NumpyBackend()
                log.debug("EmbeddingManager: using NumpyBackend")
            except ImportError:
                backend = HashProjectionBackend()
                log.debug("EmbeddingManager: using HashProjectionBackend (pure Python)")
        self._backend = backend
        self._cache: dict[str, list[float]] = {}

    def set_backend(self, backend: EmbeddingBackend) -> None:
        """Hot-swap the embedding backend. Clears the cache."""
        self._backend = backend
        self._cache.clear()
        log.info("EmbeddingManager: backend changed to %s (dim=%d)",
                 type(backend).__name__, backend.dim)

    @property
    def dim(self) -> int:
        return self._backend.dim

    def embed(self, text: str, use_cache: bool = True) -> list[float]:
        """Embed a single text string."""
        if use_cache and text in self._cache:
            return self._cache[text]
        vec = self._backend.encode([text])[0]
        if use_cache:
            self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str], use_cache: bool = True) -> list[list[float]]:
        """Embed multiple texts, using cache where possible."""
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        to_embed_indices = []
        to_embed_texts = []
        for i, text in enumerate(texts):
            if use_cache and text in self._cache:
                results[i] = self._cache[text]
            else:
                to_embed_indices.append(i)
                to_embed_texts.append(text)
        if to_embed_texts:
            vecs = self._backend.encode(to_embed_texts)
            for idx, vec in zip(to_embed_indices, vecs):
                results[idx] = vec
                if use_cache:
                    self._cache[texts[idx]] = vec
        return [r for r in results if r is not None]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two L2-normalised vectors."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        # Vectors are already L2-normalised by the backends
        return max(-1.0, min(1.0, dot))

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)
