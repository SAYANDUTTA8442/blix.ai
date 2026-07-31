"""
Curiosity Engine — Blix v0.3.13  (New module 1, "Curiosity + Active Experimentation")

Produces ``CuriositySignal`` objects — prioritized exploration targets —
from five symbolic triggers drawn entirely from existing v0.3.x infrastructure:

    LOW CONFIDENCE      — beliefs with confidence below threshold
    CONTRADICTIONS      — pairs of conflicting beliefs (BeliefStore.find_conflicting_candidates)
    SPARSE EVIDENCE     — beliefs/cause edges with very few observations
    FREQUENT FAILURES   — FailureMemory recurring failure domains
    UNKNOWN DOMAINS     — KnowledgeGapTracker gaps needing exploration

Each trigger produces a ``CuriositySignal(target, reason, novelty,
uncertainty, expected_information_gain)`` that ``ExperimentPlanner``
can act on by generating an experiment, or that ``HypothesisManager``
can turn into a hypothesis.

No RL, no intrinsic reward shaping, no self-play. This is symbolic
curiosity: a structured scan over existing cognitive state that surfaces
what deserves investigation, ranked by expected information gain.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.4
_DEFAULT_SPARSE_EVIDENCE_THRESHOLD = 3


class CuriosityTrigger(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    CONTRADICTION = "contradiction"
    SPARSE_EVIDENCE = "sparse_evidence"
    FREQUENT_FAILURES = "frequent_failures"
    UNKNOWN_DOMAIN = "unknown_domain"


@dataclass
class CuriositySignal:
    """One exploration target identified by the CuriosityEngine."""

    target: str                      # what to explore (belief statement, domain, edge, etc.)
    trigger: CuriosityTrigger
    reason: str
    novelty: float = 0.5             # 0-1; how new/unexpected this gap is
    uncertainty: float = 0.5         # 0-1; how uncertain Blix is about this target
    expected_information_gain: float = 0.5  # 0-1; how much learning a resolved signal would produce
    conflicting_belief_ids: list[str] = field(default_factory=list)  # populated for CONTRADICTION signals
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "target": self.target, "trigger": self.trigger.value, "reason": self.reason,
            "novelty": round(self.novelty, 4), "uncertainty": round(self.uncertainty, 4),
            "expected_information_gain": round(self.expected_information_gain, 4),
            "conflicting_belief_ids": self.conflicting_belief_ids,
            "generated_at": self.generated_at,
        }

    @property
    def priority_score(self) -> float:
        """Combined score for ranking: weighted uncertainty + information gain."""
        return 0.4 * self.uncertainty + 0.4 * self.expected_information_gain + 0.2 * self.novelty


class CuriosityEngine:
    """
    Scans existing cognitive state and produces ranked CuriositySignals.

    Parameters
    ----------
    belief_store:
        ``memory.beliefs.BeliefStore`` — scanned for low-confidence and
        contradicting beliefs.
    failure_memory:
        Optional ``agents.failure_memory.FailureMemory`` — scanned for
        recurring failure domains.
    cause_graph:
        Optional ``causality.cause_graph.CauseGraph`` — scanned for
        low-evidence edges.
    knowledge_gap_tracker:
        Optional ``knowledge.knowledge_gap_tracker.KnowledgeGapTracker``
        — scanned for known gaps needing exploration.
    low_confidence_threshold:
        Beliefs below this confidence trigger LOW_CONFIDENCE signals.
    sparse_evidence_threshold:
        Beliefs/edges with fewer observations than this trigger SPARSE_EVIDENCE.
    """

    def __init__(
        self,
        belief_store,
        failure_memory=None,
        cause_graph=None,
        knowledge_gap_tracker=None,
        low_confidence_threshold: float = _DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        sparse_evidence_threshold: int = _DEFAULT_SPARSE_EVIDENCE_THRESHOLD,
    ) -> None:
        self._belief_store = belief_store
        self._failure_memory = failure_memory
        self._cause_graph = cause_graph
        self._knowledge_gap_tracker = knowledge_gap_tracker
        self._low_conf_threshold = low_confidence_threshold
        self._sparse_ev_threshold = sparse_evidence_threshold

    # ------------------------------------------------------------------
    # Top-level scan
    # ------------------------------------------------------------------

    def generate_signals(self, top_k: int = 10) -> list[CuriositySignal]:
        """
        Run all five triggers and return the top-k signals by priority score.
        """
        signals: list[CuriositySignal] = []
        signals.extend(self._low_confidence_signals())
        signals.extend(self._contradiction_signals())
        signals.extend(self._sparse_evidence_signals())
        signals.extend(self._frequent_failure_signals())
        signals.extend(self._unknown_domain_signals())

        signals.sort(key=lambda s: -s.priority_score)
        return signals[:top_k]

    # ------------------------------------------------------------------
    # Trigger 1 — Low confidence
    # ------------------------------------------------------------------

    def _low_confidence_signals(self) -> list[CuriositySignal]:
        from core.truth_manager import TruthStatus
        signals = []
        for belief in self._belief_store.all_with_status(TruthStatus.ACTIVE):
            if belief.confidence < self._low_conf_threshold:
                uncertainty = 1.0 - belief.confidence
                eig = uncertainty * (1.0 - belief.confidence)
                signals.append(CuriositySignal(
                    target=belief.statement[:120],
                    trigger=CuriosityTrigger.LOW_CONFIDENCE,
                    reason=f"Belief confidence {belief.confidence:.2f} is below threshold {self._low_conf_threshold}.",
                    novelty=0.3, uncertainty=uncertainty, expected_information_gain=eig,
                ))
        return signals

    # ------------------------------------------------------------------
    # Trigger 2 — Contradictions
    # ------------------------------------------------------------------

    def _contradiction_signals(self) -> list[CuriositySignal]:
        from core.truth_manager import TruthStatus
        signals = []
        seen: set[tuple[str, str]] = set()
        for belief in self._belief_store.all_with_status(TruthStatus.ACTIVE):
            conflicts = self._belief_store.find_conflicting_candidates(belief.statement, min_overlap=0.35)
            for conflict in conflicts:
                pair = tuple(sorted([belief.belief_id, conflict.belief_id]))
                if pair in seen:
                    continue
                seen.add(pair)
                signals.append(CuriositySignal(
                    target=f"{belief.statement[:60]} vs {conflict.statement[:60]}",
                    trigger=CuriosityTrigger.CONTRADICTION,
                    reason=f"Belief '{belief.statement[:60]}' conflicts with '{conflict.statement[:60]}'.",
                    novelty=0.7, uncertainty=0.8, expected_information_gain=0.9,
                    conflicting_belief_ids=[belief.belief_id, conflict.belief_id],
                ))
        return signals

    # ------------------------------------------------------------------
    # Trigger 3 — Sparse evidence
    # ------------------------------------------------------------------

    def _sparse_evidence_signals(self) -> list[CuriositySignal]:
        from core.truth_manager import TruthStatus
        signals = []
        for belief in self._belief_store.all_with_status(TruthStatus.ACTIVE):
            if belief.evidence_count < self._sparse_ev_threshold:
                eig = 0.5 + 0.1 * (self._sparse_ev_threshold - belief.evidence_count)
                signals.append(CuriositySignal(
                    target=belief.statement[:120],
                    trigger=CuriosityTrigger.SPARSE_EVIDENCE,
                    reason=f"Belief has only {belief.evidence_count} evidence observation(s) — needs more support.",
                    novelty=0.4, uncertainty=0.6, expected_information_gain=min(1.0, eig),
                ))
        return signals

    # ------------------------------------------------------------------
    # Trigger 4 — Frequent failures
    # ------------------------------------------------------------------

    def _frequent_failure_signals(self) -> list[CuriositySignal]:
        if self._failure_memory is None:
            return []
        signals = []
        records = self._failure_memory.most_common_failures(top_k=self._failure_memory.count)
        by_tool: dict[str, int] = {}
        for r in records:
            tool = r.tool or "unknown"
            by_tool[tool] = by_tool.get(tool, 0) + r.occurrences
        for tool, occurrences in by_tool.items():
            if occurrences >= 3:
                eig = min(1.0, 0.4 + 0.06 * occurrences)
                signals.append(CuriositySignal(
                    target=f"tool:{tool}",
                    trigger=CuriosityTrigger.FREQUENT_FAILURES,
                    reason=f"Tool '{tool}' has {occurrences} recorded failures — root cause unclear.",
                    novelty=0.5, uncertainty=0.75, expected_information_gain=eig,
                ))
        return signals

    # ------------------------------------------------------------------
    # Trigger 5 — Unknown domains (KnowledgeGapTracker)
    # ------------------------------------------------------------------

    def _unknown_domain_signals(self) -> list[CuriositySignal]:
        if self._knowledge_gap_tracker is None:
            return []
        signals = []
        for gap in self._knowledge_gap_tracker.needs_exploration():
            signals.append(CuriositySignal(
                target=gap.domain,
                trigger=CuriosityTrigger.UNKNOWN_DOMAIN,
                reason=gap.gap_reason or f"Knowledge gap in domain '{gap.domain}' (severity={gap.severity.value}).",
                novelty=0.6, uncertainty=gap.uncertainty,
                expected_information_gain=min(1.0, gap.uncertainty * 0.9),
            ))
        return signals
