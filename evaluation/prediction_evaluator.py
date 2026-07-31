"""
Prediction Evaluator — Blix v0.3.12  (wiring, not new storage — "Imagination + Search")

Implements the spec's Items 6 ("Prediction Memory") and 10
("Prediction Evaluator") WITHOUT adding a new memory subsystem, per
the explicit "v0.3.12 should not add more memory" constraint:

    Predicted: 0.8 success
    Actual: failure
      ↓
    Brier score / Calibration / Prediction drift

Storage for predictions and their resolved outcomes already exists —
``memory.future_memory.FutureMemoryStore`` (v0.3.10, ``ExpectedState``
records with ``confidence``/``actual_outcome``/``was_correct``).
Calibration math already exists —
``evaluation.confidence_metrics.ConfidenceMetrics`` (v0.3.8, Brier
score / expected calibration error / over/under-confidence rate). This
module is the missing thin adapter between them: it converts resolved
``ExpectedState`` records into ``CalibrationCase`` objects and reports
the full metric suite, plus one genuinely new metric — prediction
drift over time — that neither existing module computes.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from evaluation.confidence_metrics import CalibrationCase, ConfidenceMetrics
from memory.future_memory import ExpectedState, FutureMemoryStore
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PredictionDrift:
    """How prediction calibration is trending over time — split older vs. more recent resolved predictions."""

    earlier_brier: float
    recent_brier: float
    drift: float   # recent_brier - earlier_brier; negative means improving (lower Brier = better)
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "earlier_brier": round(self.earlier_brier, 4), "recent_brier": round(self.recent_brier, 4),
            "drift": round(self.drift, 4), "improving": self.drift < 0, "sample_count": self.sample_count,
        }


class PredictionEvaluator:
    """
    Evaluates calibration quality over ``FutureMemoryStore``'s resolved predictions.

    Parameters
    ----------
    future_memory:
        ``FutureMemoryStore`` (v0.3.10) — source of predictions and their resolved outcomes.
    """

    def __init__(self, future_memory: FutureMemoryStore) -> None:
        self._future_memory = future_memory

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _calibration_cases(self) -> list[CalibrationCase]:
        resolved = [s for s in self._future_memory.resolved() if s.was_correct is not None]
        return [CalibrationCase(confidence=s.confidence, was_correct=s.was_correct) for s in resolved]

    # ------------------------------------------------------------------
    # Calibration suite (wires ConfidenceMetrics, v0.3.8)
    # ------------------------------------------------------------------

    def calibration_report(self) -> dict:
        """Full calibration report: Brier score, ECE, over/under-confidence rate, per-bucket breakdown."""
        cases = self._calibration_cases()
        if not cases:
            return {"sample_count": 0, "brier_score": None, "expected_calibration_error": None}

        return {
            "sample_count": len(cases),
            "brier_score": ConfidenceMetrics.brier_score(cases),
            "expected_calibration_error": ConfidenceMetrics.expected_calibration_error(cases),
            "overconfidence_rate": ConfidenceMetrics.overconfidence_rate(cases),
            "underconfidence_rate": ConfidenceMetrics.underconfidence_rate(cases),
            "buckets": [b.to_dict() for b in ConfidenceMetrics.calibration_buckets(cases)],
        }

    # ------------------------------------------------------------------
    # Prediction drift — genuinely new metric, not present in ConfidenceMetrics
    # ------------------------------------------------------------------

    def prediction_drift(self, min_samples_per_half: int = 3) -> Optional[PredictionDrift]:
        """
        Split resolved predictions chronologically (by resolved_at) into
        an earlier half and a more recent half, and compare Brier
        scores between them — a coarse signal of whether calibration is
        improving or degrading over time.

        Returns ``None`` if there aren't enough resolved predictions in
        each half to compare meaningfully.
        """
        resolved = [s for s in self._future_memory.resolved() if s.was_correct is not None and s.resolved_at]
        resolved.sort(key=lambda s: s.resolved_at)

        if len(resolved) < 2 * min_samples_per_half:
            return None

        midpoint = len(resolved) // 2
        earlier, recent = resolved[:midpoint], resolved[midpoint:]

        earlier_cases = [CalibrationCase(confidence=s.confidence, was_correct=s.was_correct) for s in earlier]
        recent_cases = [CalibrationCase(confidence=s.confidence, was_correct=s.was_correct) for s in recent]

        earlier_brier = ConfidenceMetrics.brier_score(earlier_cases)
        recent_brier = ConfidenceMetrics.brier_score(recent_cases)

        return PredictionDrift(
            earlier_brier=earlier_brier, recent_brier=recent_brier,
            drift=recent_brier - earlier_brier, sample_count=len(resolved),
        )

    # ------------------------------------------------------------------
    # Per-subject breakdown
    # ------------------------------------------------------------------

    def calibration_for_subject(self, subject: str) -> dict:
        """Calibration report scoped to predictions about one subject."""
        resolved = [s for s in self._future_memory.by_subject(subject) if s.was_correct is not None]
        cases = [CalibrationCase(confidence=s.confidence, was_correct=s.was_correct) for s in resolved]
        if not cases:
            return {"subject": subject, "sample_count": 0, "brier_score": None}
        return {
            "subject": subject, "sample_count": len(cases),
            "brier_score": ConfidenceMetrics.brier_score(cases),
            "expected_calibration_error": ConfidenceMetrics.expected_calibration_error(cases),
        }
