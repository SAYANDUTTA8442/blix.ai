"""
Attention Manager — Blix v0.3.9  (New module 2)

Before this module, everything Blix considered was treated as equally
important — there was no mechanism to decide "this matters more right
now than that." ``AttentionManager`` scores arbitrary candidate items
(memories, beliefs, plan steps, failures, anything with the right
signals available) and decides which ones are important enough to
enter the ``workspace.global_workspace.GlobalWorkspace``.

    attention_score = 0.4*relevance + 0.3*urgency + 0.2*novelty + 0.1*confidence

Inspired by ACT-R's activation equation and Global Workspace Theory's
attention-gated entry into conscious workspace — items below the
attention threshold simply don't compete for the workspace's limited
capacity.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Attention scoring weights — module-level constants matching the spec's
# literal formula. Exposed as a dataclass so callers can override per
# AttentionManager instance without touching this module.
# ---------------------------------------------------------------------------


@dataclass
class AttentionWeights:
    relevance: float = 0.4
    urgency: float = 0.3
    novelty: float = 0.2
    confidence: float = 0.1

    def total(self) -> float:
        return self.relevance + self.urgency + self.novelty + self.confidence


@dataclass
class AttentionCandidate:
    """One item competing for attention/workspace entry."""

    ref_id: str
    source: str                    # which subsystem this came from, e.g. "memory", "planner"
    content_summary: str              # short human-readable description
    relevance: float = 0.5
    urgency: float = 0.5
    novelty: float = 0.5
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "ref_id": self.ref_id, "source": self.source, "content_summary": self.content_summary,
            "relevance": round(self.relevance, 3), "urgency": round(self.urgency, 3),
            "novelty": round(self.novelty, 3), "confidence": round(self.confidence, 3),
        }


@dataclass
class AttentionScore:
    """Result of scoring one ``AttentionCandidate``."""

    candidate: AttentionCandidate
    score: float
    scored_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {"candidate": self.candidate.to_dict(), "score": round(self.score, 4), "scored_at": self.scored_at}


# ---------------------------------------------------------------------------
# Attention Manager
# ---------------------------------------------------------------------------


class AttentionManager:
    """
    Scores candidates for workspace entry using a weighted blend of
    relevance, urgency, novelty, and confidence.

    Parameters
    ----------
    weights:
        Optional ``AttentionWeights`` override. Defaults to the spec's
        0.4/0.3/0.2/0.1 split.
    entry_threshold:
        Minimum score for a candidate to be considered "attended to"
        (i.e. eligible for workspace entry).
    capacity:
        Maximum number of candidates that can hold attention
        simultaneously — mirrors the well-known limited-capacity
        property of biological/GWT working memory.
    """

    def __init__(
        self,
        weights: Optional[AttentionWeights] = None,
        entry_threshold: float = 0.5,
        capacity: int = 7,
    ) -> None:
        self._weights = weights or AttentionWeights()
        self._threshold = entry_threshold
        self._capacity = capacity
        self._recent_novelty_seen: set[str] = set()   # ref_ids seen before, for novelty decay

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, candidate: AttentionCandidate) -> AttentionScore:
        """Compute the weighted attention score for one candidate."""
        w = self._weights
        raw = (
            w.relevance * candidate.relevance
            + w.urgency * candidate.urgency
            + w.novelty * candidate.novelty
            + w.confidence * candidate.confidence
        )
        score = max(0.0, min(1.0, raw / w.total())) if w.total() > 0 else 0.0
        return AttentionScore(candidate=candidate, score=score)

    def score_many(self, candidates: list[AttentionCandidate]) -> list[AttentionScore]:
        """Score a batch of candidates, sorted descending by score."""
        scored = [self.score(c) for c in candidates]
        return sorted(scored, key=lambda s: -s.score)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_for_workspace(self, candidates: list[AttentionCandidate]) -> list[AttentionScore]:
        """
        Score, threshold-filter, and capacity-limit candidates — this
        is the "only the most important information enters the
        workspace" gate from the spec.
        """
        scored = self.score_many(candidates)
        above_threshold = [s for s in scored if s.score >= self._threshold]
        return above_threshold[: self._capacity]

    # ------------------------------------------------------------------
    # Novelty tracking
    # ------------------------------------------------------------------

    def novelty_for(self, ref_id: str) -> float:
        """
        Convenience: returns 1.0 (fully novel) if ``ref_id`` hasn't been
        seen before by this manager, 0.2 (mostly stale) otherwise.
        Callers can use this to derive a candidate's ``novelty`` field
        rather than hand-computing it.
        """
        if ref_id in self._recent_novelty_seen:
            return 0.2
        return 1.0

    def mark_seen(self, ref_id: str) -> None:
        self._recent_novelty_seen.add(ref_id)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def capacity(self) -> int:
        return self._capacity
