"""
Hypothesis Manager — Blix v0.3.13  (New module 2, "Curiosity + Active Experimentation")

Manages the full lifecycle of hypotheses — from an initial curious
observation to supported or rejected knowledge:

    Observation
      ↓
    Hypothesis (PENDING)
      ↓
    Experiment
      ↓
    Evidence
      ↓
    SUPPORTED or REJECTED

Schema:

    Hypothesis(
        statement,
        confidence,
        evidence,       # list of supporting evidence strings
        source,         # what generated this hypothesis
        status,         # PENDING | SUPPORTED | REJECTED | UNKNOWN
    )

This is distinct from ``memory.beliefs.BeliefStore.add_hypothesis()``
(v0.3.11) — that is a staging slot for a single low-trust belief
waiting for one confirmation. ``HypothesisManager`` is a richer
lifecycle store: it tracks multiple evidence observations over time,
explicit rejection, and the link back to the experiment that tested it.
When a hypothesis reaches SUPPORTED status, ``HypothesisManager``
optionally promotes it to a real OBSERVED belief via
``BeliefStore.confirm_observation()`` — the same promotion path used
everywhere in the codebase, not a new bypass.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from causality.epistemic_status import EpistemicStatus
from utils.logger import get_logger

log = get_logger(__name__)


class HypothesisStatus(str, Enum):
    PENDING = "pending"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass
class Hypothesis:
    """
    One hypothesis tracked through its full evidence lifecycle.

    ``status`` starts PENDING and transitions to SUPPORTED, REJECTED,
    or UNKNOWN based on experimental evidence. ``epistemic_status``
    is always HYPOTHESIS until the hypothesis is SUPPORTED and
    promoted to a real belief — at which point the Belief in
    BeliefStore carries OBSERVED, but this record retains SUPPORTED
    as a documentation trail.
    """

    statement: str
    confidence: float = 0.3
    evidence: list[str] = field(default_factory=list)
    source: str = ""               # e.g. "CuriosityEngine:low_confidence", "user", "experiment_xyz"
    status: HypothesisStatus = HypothesisStatus.PENDING
    epistemic_status: EpistemicStatus = EpistemicStatus.HYPOTHESIS
    hypothesis_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    linked_experiment_ids: list[str] = field(default_factory=list)
    linked_belief_id: Optional[str] = None   # set when promoted to BeliefStore
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id, "statement": self.statement,
            "confidence": round(self.confidence, 4), "evidence": self.evidence,
            "source": self.source, "status": self.status.value,
            "epistemic_status": self.epistemic_status.value,
            "linked_experiment_ids": self.linked_experiment_ids,
            "linked_belief_id": self.linked_belief_id,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        return cls(
            statement=d["statement"], confidence=d.get("confidence", 0.3),
            evidence=d.get("evidence", []), source=d.get("source", ""),
            status=HypothesisStatus(d.get("status", "pending")),
            epistemic_status=EpistemicStatus(d.get("epistemic_status", EpistemicStatus.HYPOTHESIS.value)),
            hypothesis_id=d.get("hypothesis_id", uuid.uuid4().hex[:10]),
            linked_experiment_ids=d.get("linked_experiment_ids", []),
            linked_belief_id=d.get("linked_belief_id"),
            created_at=d.get("created_at", ""), updated_at=d.get("updated_at", ""),
        )


class HypothesisManager:
    """
    Stores and manages Hypothesis lifecycle from PENDING to
    SUPPORTED/REJECTED.

    Parameters
    ----------
    hypothesis_file:
        Path to ``hypotheses.json``.
    belief_store:
        Optional ``BeliefStore`` — when provided, SUPPORTED hypotheses
        are automatically promoted to OBSERVED beliefs via
        ``confirm_observation()``.
    support_threshold:
        Confidence level at which a hypothesis is considered SUPPORTED.
    rejection_threshold:
        Confidence level below which a hypothesis is considered REJECTED.
    """

    def __init__(
        self,
        hypothesis_file: Path,
        belief_store=None,
        support_threshold: float = 0.7,
        rejection_threshold: float = 0.2,
    ) -> None:
        self._file = hypothesis_file
        self._belief_store = belief_store
        self._support_threshold = support_threshold
        self._rejection_threshold = rejection_threshold
        self._hypotheses: dict[str, Hypothesis] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for item in raw:
                h = Hypothesis.from_dict(item)
                self._hypotheses[h.hypothesis_id] = h
        except Exception as exc:
            log.warning("HypothesisManager: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([h.to_dict() for h in self._hypotheses.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def propose(self, statement: str, confidence: float = 0.3, source: str = "") -> Hypothesis:
        """Propose a new hypothesis in PENDING status."""
        hyp = Hypothesis(statement=statement, confidence=max(0.0, min(1.0, confidence)), source=source)
        self._hypotheses[hyp.hypothesis_id] = hyp
        self._save()
        log.info("HypothesisManager: new hypothesis '%s' (id=%s)", statement[:60], hyp.hypothesis_id)
        return hyp

    # ------------------------------------------------------------------
    # Evidence recording
    # ------------------------------------------------------------------

    def add_evidence(self, hypothesis_id: str, evidence: str, confidence_delta: float = 0.1) -> Optional[Hypothesis]:
        """
        Record one piece of supporting evidence and adjust confidence.
        Positive delta = supporting, negative = contradicting.
        Automatically transitions to SUPPORTED or REJECTED when thresholds cross.
        """
        hyp = self._hypotheses.get(hypothesis_id)
        if hyp is None or hyp.status in (HypothesisStatus.SUPPORTED, HypothesisStatus.REJECTED):
            return hyp

        hyp.evidence.append(evidence)
        hyp.confidence = max(0.0, min(1.0, hyp.confidence + confidence_delta))
        hyp.updated_at = datetime.now(timezone.utc).isoformat()

        if hyp.confidence >= self._support_threshold:
            self._support(hyp)
        elif hyp.confidence <= self._rejection_threshold:
            hyp.status = HypothesisStatus.REJECTED
            log.info("HypothesisManager: REJECTED hypothesis %s (confidence=%.2f)", hypothesis_id, hyp.confidence)

        self._save()
        return hyp

    def _support(self, hyp: Hypothesis) -> None:
        """Mark a hypothesis SUPPORTED and optionally promote to BeliefStore."""
        hyp.status = HypothesisStatus.SUPPORTED
        hyp.epistemic_status = EpistemicStatus.OBSERVED  # evidence now considered real
        log.info("HypothesisManager: SUPPORTED hypothesis %s (confidence=%.2f)", hyp.hypothesis_id, hyp.confidence)

        if self._belief_store is not None:
            # Stage it first, then confirm — follows the established pipeline
            belief = self._belief_store.add_hypothesis(hyp.statement, confidence=hyp.confidence, basis=f"hypothesis:{hyp.hypothesis_id}")
            confirmed = self._belief_store.confirm_observation(belief.belief_id)
            if confirmed:
                hyp.linked_belief_id = confirmed.belief_id
                log.info("HypothesisManager: promoted hypothesis %s -> belief %s", hyp.hypothesis_id, confirmed.belief_id)

    # ------------------------------------------------------------------
    # Linking to experiments
    # ------------------------------------------------------------------

    def link_experiment(self, hypothesis_id: str, experiment_id: str) -> Optional[Hypothesis]:
        hyp = self._hypotheses.get(hypothesis_id)
        if hyp is None:
            return None
        if experiment_id not in hyp.linked_experiment_ids:
            hyp.linked_experiment_ids.append(experiment_id)
            hyp.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()
        return hyp

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, hypothesis_id: str) -> Optional[Hypothesis]:
        return self._hypotheses.get(hypothesis_id)

    def by_status(self, status: HypothesisStatus) -> list[Hypothesis]:
        return [h for h in self._hypotheses.values() if h.status == status]

    def pending(self) -> list[Hypothesis]:
        return self.by_status(HypothesisStatus.PENDING)

    def supported(self) -> list[Hypothesis]:
        return self.by_status(HypothesisStatus.SUPPORTED)

    def rejected(self) -> list[Hypothesis]:
        return self.by_status(HypothesisStatus.REJECTED)

    def expire_stale(self, max_age_days: float = 30.0) -> list[Hypothesis]:
        """
        Transition PENDING hypotheses that have been waiting longer than
        ``max_age_days`` to UNKNOWN status — they haven't been tested and
        shouldn't block the pipeline indefinitely.

        Returns the list of hypotheses that were expired.
        """
        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        expired = []
        for h in list(self._hypotheses.values()):
            if h.status != HypothesisStatus.PENDING:
                continue
            try:
                created = datetime.fromisoformat(h.created_at)
                if created < cutoff:
                    h.status = HypothesisStatus.UNKNOWN
                    h.updated_at = datetime.now(timezone.utc).isoformat()
                    expired.append(h)
            except (ValueError, TypeError):
                pass
        if expired:
            self._save()
            log.info("HypothesisManager: expired %d stale PENDING hypothesis/hypotheses -> UNKNOWN", len(expired))
        return expired

    def repeatedly_failed(self, min_evidence: int = 2) -> list[Hypothesis]:
        """Hypotheses that were rejected with multiple contradicting evidence pieces — useful for MetaCausalReflection."""
        return [h for h in self.rejected() if len(h.evidence) >= min_evidence]

    @property
    def count(self) -> int:
        return len(self._hypotheses)
