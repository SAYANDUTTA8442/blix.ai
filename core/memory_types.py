"""
Memory Type Separation — Blix v0.3.1  (Issue 9)

Addresses: "No episodic vs semantic separation — all memories become summaries."

Human cognition distinguishes three memory types:
    Episodic   — specific events ("On Tuesday I debugged the embedding store")
    Semantic   — abstract knowledge ("Cosine similarity measures vector angle")
    Procedural — skills/how-to ("To run evals: python -m blix.evaluation.cli")

Mixing them causes retrieval confusion at scale because:
    - Episodic memories are highly specific but go stale quickly.
    - Semantic memories are generalised and stay valid long-term.
    - Procedural memories should be retrieved for task queries, not knowledge queries.

This module:
1. Defines the ``MemoryType`` enum and adds it to the retrieval pipeline.
2. Implements ``MemoryTypeClassifier`` — heuristic + LLM-upgradeable.
3. Implements ``TypeAwareRetriever`` that can filter or weight by type.

Python 3.10 compatible.
"""
# DEPRECATED — core.memory_types (ISSUE-009)
#
# This module is superseded by memory.hybrid.models.memory_node.
# The class ``MemoryType`` here is the v0.3.x implementation;
# ``memory.hybrid.models.memory_node.MemoryType`` is the v0.3.15+ HGSHM implementation.
#
# These are different classes with different APIs. Callers that need
# the v0.3.15+ version must update their imports:
#
#     # Old (this file — legacy):
#     from core.memory_types import MemoryType
#
#     # New (HGSHM-backed):
#     from memory.hybrid.models.memory_node import MemoryType
#
# This file will be removed in v0.4. Do not add new callers.
# Issue: https://github.com/blix/blix/issues/9
#


from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Memory type enum
# ---------------------------------------------------------------------------


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------

# Patterns strongly indicative of each type
_EPISODIC_RE = re.compile(
    r"\b(today|yesterday|last week|this morning|on monday|on tuesday|on wednesday|"
    r"on thursday|on friday|just now|earlier|i worked|i fixed|i debugged|i built|"
    r"i ran|we discussed|session|meeting|i tried|i noticed|i realized)\b",
    re.IGNORECASE,
)

_SEMANTIC_RE = re.compile(
    r"\b(is defined as|means|refers to|in general|always|never|typically|"
    r"the concept|theory|principle|fundamentally|by definition|"
    r"the idea|explains|formula|equation|proof|theorem|algorithm)\b",
    re.IGNORECASE,
)

_PROCEDURAL_RE = re.compile(
    r"\b(how to|step by step|to run|to install|to configure|to build|"
    r"command|script|workflow|process|procedure|run:|execute:|usage:|"
    r"python |pip |npm |git |make |docker )\b",
    re.IGNORECASE,
)


class MemoryTypeClassifier:
    """
    Classifies a MemoryEntry into Episodic / Semantic / Procedural.

    Strategy: heuristic pattern matching (fast, offline).
    Upgradeable to an NLI model or LLM classifier by overriding
    ``classify_text``.

    Parameters
    ----------
    episodic_weight, semantic_weight, procedural_weight:
        Per-class multipliers for the respective retrieval scores.
        Allows retrieval to down-weight episodic memories for knowledge queries.
    """

    def __init__(
        self,
        episodic_weight: float = 0.8,
        semantic_weight: float = 1.0,
        procedural_weight: float = 0.9,
    ) -> None:
        self._weights = {
            MemoryType.EPISODIC:   episodic_weight,
            MemoryType.SEMANTIC:   semantic_weight,
            MemoryType.PROCEDURAL: procedural_weight,
            MemoryType.UNKNOWN:    1.0,
        }

    def classify(self, memory: object) -> MemoryType:
        """Classify a MemoryEntry object."""
        text = (
            getattr(memory, "input", "")
            + " "
            + getattr(memory, "output", "")
        )
        return self.classify_text(text)

    def classify_text(self, text: str) -> MemoryType:
        """
        Classify raw text into a MemoryType.

        Override this method to plug in an LLM or NLI classifier.
        """
        ep = len(_EPISODIC_RE.findall(text))
        se = len(_SEMANTIC_RE.findall(text))
        pr = len(_PROCEDURAL_RE.findall(text))

        if ep == 0 and se == 0 and pr == 0:
            return MemoryType.UNKNOWN

        scores = {
            MemoryType.EPISODIC:   ep,
            MemoryType.SEMANTIC:   se,
            MemoryType.PROCEDURAL: pr,
        }
        return max(scores, key=lambda k: scores[k])

    def type_weight(self, memory_type: MemoryType) -> float:
        """Return the retrieval weight for this type."""
        return self._weights.get(memory_type, 1.0)

    def set_weight(self, memory_type: MemoryType, weight: float) -> None:
        self._weights[memory_type] = max(0.0, min(2.0, weight))


# ---------------------------------------------------------------------------
# Type-aware retrieval post-processor
# ---------------------------------------------------------------------------


class TypeAwareRetriever:
    """
    Post-processes a retrieval result by:
    1. Classifying each candidate's memory type.
    2. Adjusting the score by the type-specific weight.
    3. Optionally filtering to a specific type.

    Parameters
    ----------
    classifier:
        ``MemoryTypeClassifier`` instance.
    query_type:
        If provided, boost memories of this type and penalise others.
    """

    def __init__(
        self,
        classifier: Optional[MemoryTypeClassifier] = None,
        query_type: Optional[MemoryType] = None,
    ) -> None:
        self._clf = classifier or MemoryTypeClassifier()
        self._query_type = query_type

    def detect_query_type(self, query: str) -> MemoryType:
        """
        Infer the memory type most likely to answer a query.

        Used to automatically set ``query_type`` at retrieval time.
        """
        return self._clf.classify_text(query)

    def rerank(
        self,
        memories: list,
        scores: dict[int, float],
        query: Optional[str] = None,
    ) -> list:
        """
        Re-rank memories by type-adjusted scores.

        Parameters
        ----------
        memories:
            Candidate list.
        scores:
            memory_id → current score.
        query:
            If provided, auto-detect query type to boost matching memories.
        """
        effective_query_type = self._query_type
        if query is not None:
            effective_query_type = self.detect_query_type(query)
            log.debug("TypeAwareRetriever: inferred query type %s", effective_query_type.value)

        type_map: dict[int, MemoryType] = {}
        adjusted: dict[int, float] = {}

        for mem in memories:
            mid = getattr(mem, "id")
            mtype = self._clf.classify(mem)
            type_map[mid] = mtype
            weight = self._clf.type_weight(mtype)

            # Extra boost when memory type matches inferred query type
            if effective_query_type and mtype == effective_query_type:
                weight = min(2.0, weight * 1.2)

            adjusted[mid] = min(1.0, scores.get(mid, 0.0) * weight)

        return sorted(memories, key=lambda m: -adjusted.get(getattr(m, "id"), 0.0))

    def filter_by_type(self, memories: list, memory_type: MemoryType) -> list:
        """Return only memories of the specified type."""
        return [m for m in memories if self._clf.classify(m) == memory_type]

    def annotate(self, memories: list) -> list[tuple[object, MemoryType]]:
        """Return (memory, type) pairs for all memories."""
        return [(m, self._clf.classify(m)) for m in memories]
