"""
Predictive Memory — Blix v0.3.10  (New module 10)

Brings the FUTURE into memory: rather than only storing what has
happened (every memory layer through v0.3.9), ``FutureMemoryStore``
stores explicit predictions about what is EXPECTED to happen, with a
confidence and a target date:

    ExpectedState(
        confidence=0.63,
        predicted_date=...,
    )

Example from the spec: ``paper_acceptance=0.63``.

This is intentionally pure data modeling — no ML model is needed to
store and later check predictions, only the bookkeeping discipline of
recording WHAT was predicted, WHEN, and WITH WHAT CONFIDENCE, so it can
later be scored against what actually happened (feeding calibration
metrics like ``evaluation.confidence_metrics.ConfidenceMetrics``, v0.3.8).

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class ExpectedState:
    """One prediction about a future state of affairs."""

    subject: str                  # e.g. "paper_acceptance", "deploy_success"
    confidence: float
    predicted_date: Optional[str] = None   # ISO date the outcome is expected to resolve by
    expected_state_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    rationale: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved: bool = False
    actual_outcome: Optional[bool] = None
    resolved_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "expected_state_id": self.expected_state_id, "subject": self.subject,
            "confidence": round(self.confidence, 4), "predicted_date": self.predicted_date,
            "rationale": self.rationale, "created_at": self.created_at,
            "resolved": self.resolved, "actual_outcome": self.actual_outcome, "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExpectedState":
        return cls(
            subject=d["subject"], confidence=d.get("confidence", 0.5),
            predicted_date=d.get("predicted_date"), expected_state_id=d.get("expected_state_id", uuid.uuid4().hex[:10]),
            rationale=d.get("rationale", ""), created_at=d.get("created_at", ""),
            resolved=d.get("resolved", False), actual_outcome=d.get("actual_outcome"), resolved_at=d.get("resolved_at"),
        )

    @property
    def was_correct(self) -> Optional[bool]:
        """Whether the prediction's confidence-implied direction matched the actual outcome, once resolved."""
        if not self.resolved or self.actual_outcome is None:
            return None
        predicted_yes = self.confidence >= 0.5
        return predicted_yes == self.actual_outcome


class FutureMemoryStore:
    """
    Persists ``ExpectedState`` predictions and tracks their later resolution.

    Parameters
    ----------
    future_memory_file:
        Path to ``future_memory.json``.
    """

    def __init__(self, future_memory_file: Path) -> None:
        self._file = future_memory_file
        self._states: dict[str, ExpectedState] = {}
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
                s = ExpectedState.from_dict(item)
                self._states[s.expected_state_id] = s
        except Exception:
            pass

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([s.to_dict() for s in self._states.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Recording predictions
    # ------------------------------------------------------------------

    def predict(self, subject: str, confidence: float, predicted_date: Optional[str] = None, rationale: str = "") -> ExpectedState:
        """Record a new prediction about a future state."""
        state = ExpectedState(
            subject=subject, confidence=max(0.0, min(1.0, confidence)),
            predicted_date=predicted_date, rationale=rationale,
        )
        self._states[state.expected_state_id] = state
        self._save()
        return state

    # ------------------------------------------------------------------
    # Resolving predictions
    # ------------------------------------------------------------------

    def resolve(self, expected_state_id: str, actual_outcome: bool) -> Optional[ExpectedState]:
        """Mark a prediction as resolved with its actual outcome, for later calibration scoring."""
        state = self._states.get(expected_state_id)
        if state is None:
            return None
        state.resolved = True
        state.actual_outcome = actual_outcome
        state.resolved_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return state

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, expected_state_id: str) -> Optional[ExpectedState]:
        return self._states.get(expected_state_id)

    def pending(self) -> list[ExpectedState]:
        return [s for s in self._states.values() if not s.resolved]

    def resolved(self) -> list[ExpectedState]:
        return [s for s in self._states.values() if s.resolved]

    def by_subject(self, subject: str) -> list[ExpectedState]:
        return [s for s in self._states.values() if s.subject == subject]

    def calibration_accuracy(self) -> float:
        """Fraction of resolved predictions whose confidence-implied direction was correct."""
        resolved = [s for s in self.resolved() if s.was_correct is not None]
        if not resolved:
            return 1.0
        correct = sum(1 for s in resolved if s.was_correct)
        return round(correct / len(resolved), 4)

    # ------------------------------------------------------------------
    # v0.3.13 — Experiment storage alongside predictions
    # ------------------------------------------------------------------

    def record_experiment(self, subject: str, confidence: float, predicted_date: Optional[str] = None, rationale: str = "") -> "ExpectedState":
        """
        Store an experiment's expected outcome alongside regular predictions —
        uses the same ``ExpectedState`` pattern so the existing calibration
        and resolution machinery applies automatically to experiment outcomes.
        ``subject`` should be prefixed with ``experiment:`` to distinguish
        experiment records from regular predictions in queries.
        """
        return self.predict(subject=subject, confidence=confidence, predicted_date=predicted_date, rationale=rationale)

    def resolve_experiment(self, expected_state_id: str, actual_outcome: bool) -> Optional["ExpectedState"]:
        """Resolve an experiment's outcome — identical to resolve(), convenience alias for clarity."""
        return self.resolve(expected_state_id, actual_outcome=actual_outcome)

    def experiments(self) -> list["ExpectedState"]:
        """All states whose subject starts with 'experiment:' — recorded via record_experiment()."""
        return [s for s in self._states.values() if s.subject.startswith("experiment:")]

    @property
    def count(self) -> int:
        return len(self._states)
