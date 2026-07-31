"""
Tool Success Predictor — Blix v0.3.10  (New module 4)

Upgrades tool-success estimation from a flat historical success rate
(``agents.tool_reliability.ToolReliabilityRegistry.success_rate()``,
v0.3.6) to a feature-conditioned prediction: given the TASK, the TOOL,
and the surrounding CONTEXT/HISTORY, how likely is this specific
invocation to succeed — not just "how often has this tool succeeded
historically, averaged over everything."

    Input:  task, tool, context, history
    Output: success probability

This is real, fit-on-Blix's-own-data logistic regression
(``learning.ml_base.TrainableModel``), not a placeholder. Training
examples accumulate automatically from
``agents.tool_reliability.ToolReliabilityRegistry.record()`` calls (via
``observe_outcome()`` below) and from ``agents.execution_feedback``.
Until enough examples exist, predictions fall back to the existing
v0.3.6 flat success-rate heuristic — clearly labeled as such — rather
than fabricating false precision from an undertrained model.

Python 3.10 compatible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agents.tool_reliability import ToolReliabilityRegistry
from learning.ml_base import PredictionResult, TrainableModel
from utils.logger import get_logger

log = get_logger(__name__)

_FEATURE_NAMES = [
    "tool_historical_success_rate",
    "tool_sample_confident",
    "task_complexity_hint",
    "is_repeated_attempt",
    "context_confidence",
]

_DEFAULT_MIN_SAMPLES = 15


def _estimator_factory():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=500)


class ToolSuccessPredictor:
    """
    Predicts the probability that invoking ``tool`` on ``task`` will
    succeed, given context/history features.

    Parameters
    ----------
    examples_file:
        Path to persist accumulated training examples.
    tool_reliability:
        ``ToolReliabilityRegistry`` — supplies both the cold-start
        fallback value and the ``tool_historical_success_rate`` feature.
    min_samples_to_train:
        Minimum examples before switching from fallback to learned mode.
    """

    def __init__(
        self,
        examples_file: Path,
        tool_reliability: Optional[ToolReliabilityRegistry] = None,
        min_samples_to_train: int = _DEFAULT_MIN_SAMPLES,
    ) -> None:
        self._tool_reliability = tool_reliability or ToolReliabilityRegistry(examples_file.parent / "_tool_reliability_internal.json")
        self._model = TrainableModel(
            examples_file=examples_file, feature_names=_FEATURE_NAMES,
            min_samples_to_train=min_samples_to_train, estimator_factory=_estimator_factory,
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _features(
        self, tool: str, task_complexity_hint: float = 0.5,
        is_repeated_attempt: bool = False, context_confidence: float = 0.5,
    ) -> dict[str, float]:
        record = self._tool_reliability.get(tool)
        return {
            "tool_historical_success_rate": record.success_rate if record else 0.5,
            "tool_sample_confident": 1.0 if self._tool_reliability.is_confident(tool) else 0.0,
            "task_complexity_hint": task_complexity_hint,
            "is_repeated_attempt": 1.0 if is_repeated_attempt else 0.0,
            "context_confidence": context_confidence,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        task: str,
        tool: str,
        task_complexity_hint: float = 0.5,
        is_repeated_attempt: bool = False,
        context_confidence: float = 0.5,
    ) -> PredictionResult:
        """Predict P(success) for invoking ``tool`` on ``task``."""
        features = self._features(tool, task_complexity_hint, is_repeated_attempt, context_confidence)
        fallback = self._tool_reliability.success_rate(tool)
        return self._model.predict(
            features, fallback=fallback,
            fallback_explanation=f"Cold start — using ToolReliabilityRegistry flat rate for '{tool}' ({fallback:.2f}).",
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def observe_outcome(
        self,
        tool: str,
        success: bool,
        task_complexity_hint: float = 0.5,
        is_repeated_attempt: bool = False,
        context_confidence: float = 0.5,
    ) -> None:
        """
        Record a real (task, tool, context) -> outcome observation as a
        training example. Call this alongside
        ``ToolReliabilityRegistry.record()`` so both the flat rate and
        the conditional model stay in sync with real activity.
        """
        features = self._features(tool, task_complexity_hint, is_repeated_attempt, context_confidence)
        self._model.add_example(features, label=1.0 if success else 0.0)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._model.is_trained

    @property
    def sample_count(self) -> int:
        return self._model.sample_count
