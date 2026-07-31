"""
Capability Metrics — Blix v0.3.8  (New module 10b)

Measures self-AWARENESS specifically: not "is Blix good at coding" (that's
``metacognition.capability_tracker.CapabilityTracker.accuracy()``), but
"does Blix's BELIEF about its coding ability match its ACTUAL coding
ability". A system can be skilled but miscalibrated about its own
skill — this module catches that gap.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from metacognition.capability_tracker import CapabilityTracker
from metacognition.self_model import SelfModelStore


@dataclass
class SelfAwarenessGap:
    """Gap between believed and actual capability for one domain."""

    domain: str
    believed: float
    actual: float

    @property
    def gap(self) -> float:
        return self.believed - self.actual

    @property
    def abs_gap(self) -> float:
        return abs(self.gap)

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "believed": round(self.believed, 4),
            "actual": round(self.actual, 4),
            "gap": round(self.gap, 4),
        }


class CapabilityMetrics:
    """Self-awareness and capability-tracking quality metrics."""

    @staticmethod
    def self_awareness_gaps(
        self_model_store: SelfModelStore, capability_tracker: CapabilityTracker,
    ) -> list[SelfAwarenessGap]:
        """
        For every domain tracked by ``CapabilityTracker`` with enough
        samples to trust its accuracy, compare it against what
        ``SelfModel`` currently believes.
        """
        gaps = []
        for rec in capability_tracker.all_records():
            if not capability_tracker.is_confident(rec.domain):
                continue
            believed = self_model_store.capability(rec.domain)
            gaps.append(SelfAwarenessGap(domain=rec.domain, believed=believed, actual=rec.accuracy))
        return gaps

    @staticmethod
    def mean_self_awareness_gap(gaps: list[SelfAwarenessGap]) -> float:
        """Mean absolute gap across domains — lower means better self-awareness."""
        if not gaps:
            return 0.0
        return round(sum(g.abs_gap for g in gaps) / len(gaps), 4)

    @staticmethod
    def self_awareness_score(gaps: list[SelfAwarenessGap]) -> float:
        """1 - mean_abs_gap, clamped to [0,1] — higher is better, 1.0 = perfect self-knowledge."""
        if not gaps:
            return 1.0
        mean_gap = CapabilityMetrics.mean_self_awareness_gap(gaps)
        return round(max(0.0, 1.0 - mean_gap), 4)

    @staticmethod
    def overestimated_domains(gaps: list[SelfAwarenessGap], threshold: float = 0.15) -> list[SelfAwarenessGap]:
        """Domains where believed capability significantly exceeds actual (overconfidence in self-knowledge)."""
        return [g for g in gaps if g.gap > threshold]

    @staticmethod
    def underestimated_domains(gaps: list[SelfAwarenessGap], threshold: float = 0.15) -> list[SelfAwarenessGap]:
        """Domains where believed capability significantly trails actual (underselling itself)."""
        return [g for g in gaps if g.gap < -threshold]

    @staticmethod
    def capability_coverage(self_model_store: SelfModelStore, capability_tracker: CapabilityTracker) -> float:
        """
        Fraction of confidently-measured capability-tracker domains that
        are ALSO present in the Self Model — i.e. how much of what Blix
        has track record on has actually been synced into self-knowledge.
        """
        confident_domains = [r.domain for r in capability_tracker.all_records() if capability_tracker.is_confident(r.domain)]
        if not confident_domains:
            return 1.0
        tracked_in_model = sum(1 for d in confident_domains if d in self_model_store.model.capabilities)
        return round(tracked_in_model / len(confident_domains), 4)
