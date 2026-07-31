"""
Confidence Prediction Model — Blix v0.3.10  (New module 3)

Upgrades ``reasoning.confidence_reasoner.ConfidenceReasoner.answer_confidence()``
(v0.3.8, a fixed hand-tuned formula over evidence/source counts) to a
learned model: P(answer_correct), conditioned on whatever features are
available, fit on Blix's own track record of which answers turned out
to be right.

Used by:
    verification.verifier.VerificationEngine   — as an additional signal alongside rule-based checks
    planning.plan_evaluator.PlanQualityEvaluator — feeds into plan confidence
    core.truth_manager.TruthManager                — informs confidence on TruthRecord updates

Real scikit-learn logistic regression on real Blix outcomes (verified
answers, executed plans with known results), with the existing
``ConfidenceReasoner.answer_confidence()`` heuristic as the cold-start
fallback. This module does not replace ``ConfidenceReasoner`` — it is
an additional, optional signal source that callers can blend in.

Python 3.10 compatible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from learning.ml_base import PredictionResult, TrainableModel
from reasoning.confidence_reasoner import ConfidenceReasoner
from utils.logger import get_logger

log = get_logger(__name__)

_FEATURE_NAMES = [
    "evidence_count",
    "source_count",
    "contradicting_evidence_count",
    "heuristic_confidence",
    "verification_passed",
]

_DEFAULT_MIN_SAMPLES = 15


def _estimator_factory():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=500)


class ConfidenceModel:
    """
    Learns P(answer_correct) from evidence/verification features.

    Parameters
    ----------
    examples_file:
        Path to persist accumulated training examples.
    confidence_reasoner:
        ``ConfidenceReasoner`` — supplies the cold-start fallback via
        its existing hand-tuned ``answer_confidence()`` heuristic.
    min_samples_to_train:
        Minimum examples before switching from fallback to learned mode.
    """

    def __init__(
        self,
        examples_file: Path,
        confidence_reasoner: Optional[ConfidenceReasoner] = None,
        min_samples_to_train: int = _DEFAULT_MIN_SAMPLES,
    ) -> None:
        self._reasoner = confidence_reasoner or ConfidenceReasoner()
        self._model = TrainableModel(
            examples_file=examples_file, feature_names=_FEATURE_NAMES,
            min_samples_to_train=min_samples_to_train, estimator_factory=_estimator_factory,
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _features(
        self, evidence_count: int, source_count: int,
        contradicting_evidence_count: int, verification_passed: Optional[bool],
    ) -> tuple[dict[str, float], float]:
        heuristic_estimate = self._reasoner.answer_confidence(
            evidence_count=evidence_count, source_count=source_count,
            contradicting_evidence_count=contradicting_evidence_count,
        )
        features = {
            "evidence_count": float(evidence_count),
            "source_count": float(source_count),
            "contradicting_evidence_count": float(contradicting_evidence_count),
            "heuristic_confidence": heuristic_estimate.score,
            "verification_passed": (1.0 if verification_passed else 0.0) if verification_passed is not None else 0.5,
        }
        return features, heuristic_estimate.score

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_correctness(
        self,
        evidence_count: int = 0,
        source_count: int = 0,
        contradicting_evidence_count: int = 0,
        verification_passed: Optional[bool] = None,
    ) -> PredictionResult:
        """Predict P(answer_correct) given evidence/verification features."""
        features, heuristic_score = self._features(evidence_count, source_count, contradicting_evidence_count, verification_passed)
        return self._model.predict(
            features, fallback=heuristic_score,
            fallback_explanation=f"Cold start — using ConfidenceReasoner heuristic ({heuristic_score:.2f}).",
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def observe_outcome(
        self,
        was_correct: bool,
        evidence_count: int = 0,
        source_count: int = 0,
        contradicting_evidence_count: int = 0,
        verification_passed: Optional[bool] = None,
    ) -> None:
        """Record a real (features) -> was_correct observation as a training example."""
        features, _ = self._features(evidence_count, source_count, contradicting_evidence_count, verification_passed)
        self._model.add_example(features, label=1.0 if was_correct else 0.0)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._model.is_trained

    @property
    def sample_count(self) -> int:
        return self._model.sample_count
