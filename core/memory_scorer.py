"""
Memory importance scoring — Blix v0.3  (Feature 2)

Every memory is scored by a weighted formula:

    score = 0.4 * relevance + 0.3 * importance + 0.2 * recency + 0.1 * frequency

All weights are configurable.  The scorer produces a ``MemoryScore`` with a
full explanation dict for debugging.

Python 3.10 compatible.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Score model
# ---------------------------------------------------------------------------


class ScoringWeights(BaseModel):
    """Configurable weights for the memory scoring formula (must sum to 1.0)."""

    relevance: float = Field(default=0.4, ge=0.0, le=1.0)
    importance: float = Field(default=0.3, ge=0.0, le=1.0)
    recency: float = Field(default=0.2, ge=0.0, le=1.0)
    frequency: float = Field(default=0.1, ge=0.0, le=1.0)

    def validate_sum(self) -> bool:
        total = self.relevance + self.importance + self.recency + self.frequency
        return abs(total - 1.0) < 1e-6


class MemoryScore(BaseModel):
    """
    Full scoring result for a single memory entry.

    ``explanation`` maps component name → (raw_value, weighted_contribution).
    """

    memory_id: int
    final_score: float = Field(..., ge=0.0, le=1.0)
    explanation: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="component → (raw_value, weighted_contribution)",
    )

    def debug_str(self) -> str:
        lines = [f"Memory {self.memory_id}: final={self.final_score:.4f}"]
        for name, (raw, contrib) in self.explanation.items():
            lines.append(f"  {name:12s}  raw={raw:.4f}  contrib={contrib:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class MemoryScorer:
    """
    Computes a composite importance score for a memory entry at query time.

    Parameters
    ----------
    weights:
        ``ScoringWeights`` instance controlling the formula.
    recency_half_life_days:
        How quickly the recency component decays.  Default 30 days means
        a 30-day-old memory has half the recency score of a fresh one.
    """

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        recency_half_life_days: float = 30.0,
    ) -> None:
        self._w = weights or ScoringWeights()
        self._half_life = recency_half_life_days

        if not self._w.validate_sum():
            log.warning(
                "ScoringWeights do not sum to 1.0 (%.4f) — scores may exceed 1.",
                sum([self._w.relevance, self._w.importance, self._w.recency, self._w.frequency]),
            )

    # ------------------------------------------------------------------
    # Component calculators
    # ------------------------------------------------------------------

    def _recency(self, timestamp: datetime) -> float:
        """Exponential decay: recency=1 now, recency→0 as age→∞."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0)
        return math.exp(-math.log(2) * age_days / self._half_life)

    @staticmethod
    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        memory_id: int,
        *,
        relevance: float,
        importance: float,
        timestamp: datetime,
        access_count: int = 1,
        max_access_count: int = 10,
    ) -> MemoryScore:
        """
        Compute and return a ``MemoryScore`` for the given inputs.

        Parameters
        ----------
        memory_id:
            ``MemoryEntry.id``.
        relevance:
            Cosine similarity or retrieval score (0–1).
        importance:
            Extraction-assigned importance (0–1); defaults to 0.5 if None.
        timestamp:
            When this memory was created.
        access_count:
            How many times this memory has been retrieved.
        max_access_count:
            Used to normalise frequency to 0–1.
        """
        r = self._clamp(relevance)
        imp = self._clamp(importance if importance is not None else 0.5)
        rec = self._recency(timestamp)
        freq = self._clamp(access_count / max(1, max_access_count))

        w = self._w
        final = (
            w.relevance * r
            + w.importance * imp
            + w.recency * rec
            + w.frequency * freq
        )
        final = self._clamp(final)

        explanation = {
            "relevance":  (r,   w.relevance  * r),
            "importance": (imp, w.importance * imp),
            "recency":    (rec, w.recency    * rec),
            "frequency":  (freq, w.frequency * freq),
        }
        return MemoryScore(memory_id=memory_id, final_score=final, explanation=explanation)

    def score_batch(
        self,
        entries: list[dict],
    ) -> list[MemoryScore]:
        """
        Score a list of entry dicts.

        Each dict must have keys: ``id``, ``relevance``, ``importance``,
        ``timestamp``.  Optional: ``access_count``, ``max_access_count``.

        Returns list sorted by ``final_score`` descending.
        """
        scores = [
            self.score(
                e["id"],
                relevance=e.get("relevance", 0.0),
                importance=e.get("importance", 0.5),
                timestamp=e["timestamp"],
                access_count=e.get("access_count", 1),
                max_access_count=e.get("max_access_count", 10),
            )
            for e in entries
        ]
        scores.sort(key=lambda s: s.final_score, reverse=True)
        return scores
