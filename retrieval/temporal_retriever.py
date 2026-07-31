"""
Temporal Retriever — Blix v0.3.7  (New module 6)

Fixes the second bug the spec calls out: retrieval currently treats
old and current facts as equally competing on semantic + recency +
importance alone. A memory from 2024 saying "I use Python" can outscore
a 2026 memory saying "I use Rust" on pure semantic similarity, even
though Python is now ``HISTORICAL`` and Rust is ``ACTIVE``.

``TemporalRetriever`` extends the v0.3 scoring formula with two new
components:

    score = semantic + recency + importance + state_relevance + belief_confidence

* ``state_relevance`` — boosts memories that support the CURRENTLY
  ACTIVE state for any (entity, attribute) they mention, and penalises
  memories that only support SUPERSEDED/HISTORICAL state.
* ``belief_confidence`` — boosts memories backing high-confidence,
  ACTIVE beliefs; penalises memories backing low-confidence or
  CONFLICTING beliefs.

This module does NOT replace ``core.memory_scorer.MemoryScorer`` or
``core.semantic_retriever.SemanticRetriever`` — it wraps them, adding
the two new weighted terms on top of whatever base score they already
produce, so v0.3–v0.3.6 retrieval keeps working unchanged when this
retriever isn't used.

Python 3.10 compatible.
"""
# DEPRECATED — retrieval.temporal_retriever (ISSUE-009)
#
# This module is superseded by memory.hybrid.retrieval.hybrid_retriever.
# The class ``TemporalRetriever`` here is the v0.3.x implementation;
# ``memory.hybrid.retrieval.hybrid_retriever.TemporalRetriever`` is the v0.3.15+ HGSHM implementation.
#
# These are different classes with different APIs. Callers that need
# the v0.3.15+ version must update their imports:
#
#     # Old (this file — legacy):
#     from retrieval.temporal_retriever import TemporalRetriever
#
#     # New (HGSHM-backed):
#     from memory.hybrid.retrieval.hybrid_retriever import TemporalRetriever
#
# This file will be removed in v0.4. Do not add new callers.
# Issue: https://github.com/blix/blix/issues/9
#


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.state_tracker import StateTracker
from core.truth_manager import TruthManager, TruthStatus
from memory.beliefs import BeliefStore
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------


@dataclass
class TemporalScoringWeights:
    """
    Weights for the v0.3.7 five-component formula.

        score = semantic*w1 + recency*w2 + importance*w3
              + state_relevance*w4 + belief_confidence*w5

    Defaults keep the original three components dominant (0.8 combined)
    while giving meaningful but not overwhelming weight to the two new
    temporal-truth signals (0.2 combined).
    """

    semantic: float = 0.35
    recency: float = 0.2
    importance: float = 0.25
    state_relevance: float = 0.12
    belief_confidence: float = 0.08

    def total(self) -> float:
        return (self.semantic + self.recency + self.importance
                + self.state_relevance + self.belief_confidence)


@dataclass
class TemporalScore:
    """Full scoring breakdown for one memory under the v0.3.7 formula."""

    memory_id: int
    final_score: float
    base_score: float            # semantic+recency+importance portion only
    state_relevance: float
    belief_confidence: float
    truth_status_note: str = ""

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "final_score": round(self.final_score, 4),
            "base_score": round(self.base_score, 4),
            "state_relevance": round(self.state_relevance, 4),
            "belief_confidence": round(self.belief_confidence, 4),
            "truth_status_note": self.truth_status_note,
        }


# ---------------------------------------------------------------------------
# Temporal Retriever
# ---------------------------------------------------------------------------


# How much to scale state_relevance by TruthStatus
_STATE_STATUS_MULTIPLIER: dict[TruthStatus, float] = {
    TruthStatus.ACTIVE: 1.0,
    TruthStatus.CONFLICTING: 0.5,
    TruthStatus.HISTORICAL: 0.2,
    TruthStatus.SUPERSEDED: 0.0,
    TruthStatus.ARCHIVED: 0.0,
}


class TemporalRetriever:
    """
    Wraps base memory scoring with temporal-truth-aware adjustments.

    Parameters
    ----------
    state_tracker:
        ``StateTracker`` — used to determine whether a memory's
        mentioned entity/attribute is currently ACTIVE or stale.
    truth_manager:
        ``TruthManager`` — used to look up TruthStatus for snapshots/beliefs.
    belief_store:
        ``BeliefStore`` — used to compute belief_confidence.
    weights:
        ``TemporalScoringWeights``.
    """

    def __init__(
        self,
        state_tracker: Optional[StateTracker] = None,
        truth_manager: Optional[TruthManager] = None,
        belief_store: Optional[BeliefStore] = None,
        weights: Optional[TemporalScoringWeights] = None,
    ) -> None:
        self._tracker = state_tracker
        self._truth = truth_manager
        self._beliefs = belief_store
        self._w = weights or TemporalScoringWeights()

    # ------------------------------------------------------------------
    # Component: state_relevance
    # ------------------------------------------------------------------

    def _state_relevance(
        self,
        entity: Optional[str],
        attribute: Optional[str],
        snapshot_id: Optional[str] = None,
    ) -> tuple[float, str]:
        """
        How relevant is this memory given the CURRENT truth status of the
        state it supports?

        Returns (component_value 0-1, explanatory note).
        """
        if self._tracker is None or entity is None or attribute is None:
            return 0.5, "no state context"

        current = self._tracker.current(entity, attribute)
        if current is None:
            return 0.5, "no tracked state for this entity/attribute"

        status = TruthStatus.ACTIVE
        if self._truth is not None and snapshot_id is not None:
            status = self._truth.status_of(snapshot_id)
        elif self._truth is not None:
            status = self._truth.status_of(current.snapshot_id)

        multiplier = _STATE_STATUS_MULTIPLIER.get(status, 0.5)
        return multiplier, f"state status={status.value}"

    # ------------------------------------------------------------------
    # Component: belief_confidence
    # ------------------------------------------------------------------

    def _belief_confidence(self, belief_id: Optional[str]) -> tuple[float, str]:
        """How much should a memory's score be adjusted by belief confidence/status?"""
        if self._beliefs is None or belief_id is None:
            return 0.5, "no belief context"

        belief = self._beliefs.get(belief_id)
        if belief is None:
            return 0.5, "belief not found"

        status_multiplier = {
            TruthStatus.ACTIVE: 1.0,
            TruthStatus.CONFLICTING: 0.4,
            TruthStatus.HISTORICAL: 0.3,
            TruthStatus.SUPERSEDED: 0.05,
            TruthStatus.ARCHIVED: 0.0,
        }.get(belief.status, 0.5)

        value = belief.confidence * status_multiplier
        return value, f"belief confidence={belief.confidence:.2f} status={belief.status.value}"

    # ------------------------------------------------------------------
    # Public scoring API
    # ------------------------------------------------------------------

    def score(
        self,
        memory_id: int,
        *,
        semantic: float,
        recency: float,
        importance: float,
        entity: Optional[str] = None,
        attribute: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        belief_id: Optional[str] = None,
    ) -> TemporalScore:
        """
        Compute the full v0.3.7 score for one memory.

        ``semantic``/``recency``/``importance`` are expected pre-computed
        0-1 values (e.g. from ``SemanticRetriever`` and ``MemoryScorer``).
        ``entity``/``attribute``/``snapshot_id``/``belief_id`` are
        optional hints about what state/belief this memory supports —
        when absent, both new components default to a neutral 0.5
        (no penalty, no boost) so memories unrelated to tracked state
        aren't unfairly disadvantaged.
        """
        w = self._w
        base = w.semantic * semantic + w.recency * recency + w.importance * importance

        state_val, state_note = self._state_relevance(entity, attribute, snapshot_id)
        belief_val, belief_note = self._belief_confidence(belief_id)

        final = base + w.state_relevance * state_val + w.belief_confidence * belief_val
        final = max(0.0, min(1.0, final))

        note = "; ".join(n for n in (state_note, belief_note) if n)
        return TemporalScore(
            memory_id=memory_id, final_score=final, base_score=base,
            state_relevance=state_val, belief_confidence=belief_val,
            truth_status_note=note,
        )

    def rank(
        self,
        candidates: list[dict],
    ) -> list[TemporalScore]:
        """
        Score and rank a batch of candidate memories.

        Each dict in ``candidates`` should have keys matching ``score()``'s
        kwargs (``memory_id``, ``semantic``, ``recency``, ``importance``,
        and optionally ``entity``/``attribute``/``snapshot_id``/``belief_id``).

        Returns scores sorted descending by final_score.
        """
        scores = [self.score(**c) for c in candidates]
        return sorted(scores, key=lambda s: -s.final_score)

    # ------------------------------------------------------------------
    # Convenience: re-rank an already-retrieved memory list
    # ------------------------------------------------------------------

    def prioritize_current_over_historical(
        self, memories: list, entity_attribute_map: Optional[dict] = None,
    ) -> list:
        """
        Given a list of MemoryEntry-like objects (must have ``.id``), return
        them re-sorted so memories supporting currently-ACTIVE state come
        before memories supporting SUPERSEDED/HISTORICAL state, while
        preserving relative order within each tier (stable sort).

        ``entity_attribute_map``: optional dict of memory_id → (entity, attribute)
        to look up tracked state for. Memories without an entry are treated
        as neutral (tier 1, alongside ACTIVE).
        """
        if self._tracker is None:
            return list(memories)

        entity_attribute_map = entity_attribute_map or {}

        def tier(memory) -> int:
            mapping = entity_attribute_map.get(memory.id)
            if mapping is None:
                return 1  # neutral — no tracked state context
            entity, attribute = mapping
            current = self._tracker.current(entity, attribute)
            if current is None:
                return 1
            status = TruthStatus.ACTIVE
            if self._truth is not None:
                status = self._truth.status_of(current.snapshot_id)
            if status == TruthStatus.ACTIVE:
                return 0  # highest priority
            if status == TruthStatus.CONFLICTING:
                return 1
            return 2  # HISTORICAL / SUPERSEDED / ARCHIVED

        return sorted(memories, key=tier)
