"""
Cross-Encoder Retrieval Reranker — Blix v0.3.10  (New module 2)

Upgrades retrieval scoring from a single weighted formula (semantic +
recency + importance + ..., v0.3.7's ``retrieval.temporal_retriever``)
to a two-stage pipeline:

    Embedding Retrieval (bi-encoder, fast, approximate)
      ↓
    Top-50
      ↓
    Cross-Encoder (slow, accurate, query-document JOINT scoring)
      ↓
    Top-10

A cross-encoder scores (query, document) pairs jointly — it sees both
texts at once and can model their interaction directly — which is
strictly more accurate than the bi-encoder's "embed each separately,
compare vectors" approach, at the cost of being too slow to run over
the full memory store. Running it only over the bi-encoder's top-50
candidates gets the accuracy benefit at acceptable cost.

== Honest implementation note ==
The spec names ``bge-reranker-base`` and a ``MiniLM`` cross-encoder —
specific pretrained transformer checkpoints. This environment has no
network path to huggingface.co or any model-weight host (confirmed:
only package indices like pypi/npm are reachable), and no weights are
pre-cached locally. Rather than silently instantiating an untrained
``transformers`` model and passing off random-weight output as a
"cross-encoder score" (which would be actively misleading — untrained
transformer weights produce noise, not relevance judgments), this
module:

  1. Defines the real two-stage pipeline shape and the
     ``CrossEncoderReranker`` interface exactly as production code
     would call it.
  2. Attempts to load a real ``sentence_transformers.CrossEncoder``
     model if one is available/cached (the ``transformers`` and
     ``sentence-transformers`` packages ARE installed here).
  3. Falls back to a genuinely-functional, clearly-labeled lexical
     cross-scorer (token-overlap + position-aware scoring computed
     jointly per query-document pair) when no model can be loaded —
     not a fake number, an honest classical alternative that still
     does real (query, document) joint scoring.

If this is deployed somewhere with model access, swapping in the real
checkpoint is a one-line change to ``_try_load_model()``.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from schemas.memory_entry import MemoryEntry
from utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_CANDIDATE_K = 50
_DEFAULT_RERANK_K = 10


@dataclass
class RerankedResult:
    """One reranked memory entry with its cross-encoder (or fallback) score."""

    entry: MemoryEntry
    score: float
    scorer_mode: str   # "cross_encoder" | "lexical_fallback"


def _try_load_cross_encoder(model_name: str):
    """
    Attempt to load a real sentence-transformers CrossEncoder. Returns
    None (triggering the lexical fallback) if the model can't be
    fetched — e.g. no network path to the model host, which is the
    expected/normal case in this environment.
    """
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name)
    except Exception as exc:
        log.info("CrossEncoderReranker: could not load '%s' (%s) — using lexical fallback.", model_name, exc)
        return None


def _lexical_pair_score(query: str, document: str) -> float:
    """
    Honest, non-ML joint (query, document) scorer used when no
    cross-encoder model is available. Unlike simple cosine/Jaccard
    similarity (which embeds each text independently), this scores the
    PAIR jointly: term overlap weighted by query-term position and
    density in the document, which is closer in spirit to what a
    cross-encoder attends to than a pure bag-of-words comparison.
    """
    query_terms = [w.lower() for w in query.split() if len(w) > 2]
    doc_lower = document.lower()
    if not query_terms:
        return 0.0

    matched = 0
    density_bonus = 0.0
    for term in query_terms:
        if term in doc_lower:
            matched += 1
            # Reward terms appearing earlier in the document (proxy for relevance/topicality).
            position = doc_lower.find(term) / max(1, len(doc_lower))
            density_bonus += (1.0 - position) * 0.1

    coverage = matched / len(query_terms)
    return max(0.0, min(1.0, coverage * 0.8 + density_bonus))


class CrossEncoderReranker:
    """
    Two-stage retrieval reranker: bi-encoder candidate generation
    (Top-K) followed by cross-encoder joint (query, document) scoring
    (Top-N).

    Parameters
    ----------
    model_name:
        Name of the cross-encoder checkpoint to attempt to load (e.g.
        "cross-encoder/ms-marco-MiniLM-L-6-v2"). If unavailable, the
        lexical fallback scorer is used instead — transparently, with
        every result tagged by ``scorer_mode``.
    candidate_k:
        How many top candidates from the upstream (bi-encoder/base)
        retriever to rerank.
    rerank_k:
        How many top-scoring results to return after reranking.
    attempt_model_load:
        Whether to try loading the real cross-encoder model at
        construction time. Defaults to True for production use; set to
        False to skip the (network-dependent, ~5s-on-timeout) load
        attempt entirely and go straight to the lexical fallback —
        useful for tests and offline environments where the outcome is
        already known.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_k: int = _DEFAULT_CANDIDATE_K,
        rerank_k: int = _DEFAULT_RERANK_K,
        attempt_model_load: bool = True,
    ) -> None:
        self._model_name = model_name
        self._candidate_k = candidate_k
        self._rerank_k = rerank_k
        self._model = _try_load_cross_encoder(model_name) if attempt_model_load else None

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def rerank(self, query: str, candidates: list[MemoryEntry]) -> list[RerankedResult]:
        """
        Rerank ``candidates`` (expected to already be the upstream
        retriever's Top-K, e.g. from ``core.semantic_retriever.SemanticRetriever``)
        against ``query``, returning the top ``rerank_k`` by cross-encoder
        (or fallback) score, descending.
        """
        pool = candidates[: self._candidate_k]
        if not pool:
            return []

        if self._model is not None:
            try:
                pairs = [(query, f"{e.input} {e.output}") for e in pool]
                raw_scores = self._model.predict(pairs)
                scored = [
                    RerankedResult(entry=e, score=float(s), scorer_mode="cross_encoder")
                    for e, s in zip(pool, raw_scores)
                ]
                scored.sort(key=lambda r: -r.score)
                return scored[: self._rerank_k]
            except Exception as exc:
                log.warning("CrossEncoderReranker: model inference failed (%s) — falling back.", exc)

        scored = [
            RerankedResult(entry=e, score=_lexical_pair_score(query, f"{e.input} {e.output}"), scorer_mode="lexical_fallback")
            for e in pool
        ]
        scored.sort(key=lambda r: -r.score)
        return scored[: self._rerank_k]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_using_real_model(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name
