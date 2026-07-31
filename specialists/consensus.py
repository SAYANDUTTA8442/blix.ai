"""
Consensus — Blix v0.3.9  (Part of New module 5, Internal Specialists)

Completes the pipeline from the spec:

    Workspace
      ↓
    Specialists
      ↓
    Consensus

Polls every registered ``specialists.base.BaseSpecialist`` for its
opinion on a topic, then aggregates the opinions (which may disagree)
into a single ``ConsensusResult`` — the beginning of Society-of-Mind:
no single specialist dictates the outcome, but their weighted
agreement does.

Python 3.10 compatible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from specialists.base import BaseSpecialist, SpecialistOpinion


@dataclass
class ConsensusResult:
    """Aggregated outcome of polling all specialists about one topic."""

    topic: str
    opinions: list[SpecialistOpinion] = field(default_factory=list)
    majority_verdict: str = "no_opinion"
    agreement_ratio: float = 0.0          # fraction of opinionated specialists agreeing with majority_verdict
    mean_confidence: float = 0.0
    decided_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "opinions": [o.to_dict() for o in self.opinions],
            "majority_verdict": self.majority_verdict,
            "agreement_ratio": round(self.agreement_ratio, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "decided_at": self.decided_at,
        }

    @property
    def is_contested(self) -> bool:
        """True if specialists meaningfully disagreed (majority below 2/3)."""
        return 0.0 < self.agreement_ratio < (2.0 / 3.0)


class SpecialistConsensus:
    """
    Polls registered specialists and aggregates their opinions.

    Parameters
    ----------
    specialists:
        Optional initial list of ``BaseSpecialist`` instances.
    """

    def __init__(self, specialists: list[BaseSpecialist] | None = None) -> None:
        self._specialists: list[BaseSpecialist] = list(specialists) if specialists else []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, specialist: BaseSpecialist) -> None:
        self._specialists.append(specialist)

    def registered_names(self) -> list[str]:
        return [s.name for s in self._specialists]

    # ------------------------------------------------------------------
    # Consultation + aggregation
    # ------------------------------------------------------------------

    def consult_all(self, topic: str, **context) -> list[SpecialistOpinion]:
        """Poll every registered specialist; returns one opinion each (including no_opinion)."""
        return [s.consult(topic, **context) for s in self._specialists]

    def decide(self, topic: str, **context) -> ConsensusResult:
        """
        Poll all specialists and aggregate into a ``ConsensusResult``.

        Aggregation ignores "no_opinion" verdicts for the majority
        calculation (a specialist with nothing to say shouldn't dilute
        consensus among those that do), but the full opinion list
        (including no_opinion ones) is preserved for transparency.
        """
        opinions = self.consult_all(topic, **context)
        opinionated = [o for o in opinions if o.has_opinion]

        if not opinionated:
            return ConsensusResult(topic=topic, opinions=opinions, majority_verdict="no_opinion", agreement_ratio=0.0, mean_confidence=0.0)

        verdict_counts = Counter(o.verdict for o in opinionated)
        majority_verdict, majority_count = verdict_counts.most_common(1)[0]
        agreement_ratio = majority_count / len(opinionated)
        mean_confidence = sum(o.confidence for o in opinionated) / len(opinionated)

        return ConsensusResult(
            topic=topic, opinions=opinions, majority_verdict=majority_verdict,
            agreement_ratio=agreement_ratio, mean_confidence=mean_confidence,
        )
