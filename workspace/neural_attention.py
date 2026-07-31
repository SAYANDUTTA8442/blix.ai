"""
Neural Attention System — Blix v0.3.10  (New module 5)

Upgrades ``workspace.attention_manager.AttentionManager`` (v0.3.9,
fixed weighted formula: ``0.4*relevance + 0.3*urgency + 0.2*novelty +
0.1*confidence``) to a learned scoring network: given the SAME input
signals, predict which candidates actually deserved attention, based
on Blix's own track record of which workspace entries turned out to
matter (e.g. led to a successful adaptation, were referenced again, or
were flagged important by a specialist consensus).

Inspired by ACT-R activation, same as the v0.3.9 formula it extends.

This module does NOT replace ``AttentionManager`` — it wraps it.
``AttentionManager.score()`` (the v0.3.9 fixed formula) IS the
cold-start fallback here. ``NeuralAttentionScorer`` only takes over
once enough labeled (candidate features -> was this actually
important) examples exist, exactly mirroring the
``learning.ml_base.TrainableModel`` cold-start pattern used throughout
v0.3.10.

Python 3.10 compatible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from learning.ml_base import PredictionResult, TrainableModel
from workspace.attention_manager import AttentionCandidate, AttentionManager, AttentionScore

_FEATURE_NAMES = ["relevance", "urgency", "novelty", "confidence"]

_DEFAULT_MIN_SAMPLES = 25


def _estimator_factory():
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(hidden_layer_sizes=(8,), max_iter=2000, random_state=42)


class NeuralAttentionScorer:
    """
    Learned attention scoring, falling back to
    ``AttentionManager.score()``'s fixed v0.3.9 formula when undertrained.

    Parameters
    ----------
    attention_manager:
        ``AttentionManager`` — supplies the cold-start fallback score
        (its existing fixed-weight formula, unmodified).
    examples_file:
        Path to persist accumulated (candidate features, was_important) examples.
    min_samples_to_train:
        Minimum examples before the learned scorer is used.
    """

    def __init__(
        self,
        attention_manager: AttentionManager,
        examples_file: Path,
        min_samples_to_train: int = _DEFAULT_MIN_SAMPLES,
    ) -> None:
        self._attention_manager = attention_manager
        self._model = TrainableModel(
            examples_file=examples_file, feature_names=_FEATURE_NAMES,
            min_samples_to_train=min_samples_to_train, estimator_factory=_estimator_factory,
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _features(candidate: AttentionCandidate) -> dict[str, float]:
        return {
            "relevance": candidate.relevance, "urgency": candidate.urgency,
            "novelty": candidate.novelty, "confidence": candidate.confidence,
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, candidate: AttentionCandidate) -> PredictionResult:
        """Score a candidate, learned model if trained, else the v0.3.9 fixed-weight formula."""
        fallback_score = self._attention_manager.score(candidate).score
        features = self._features(candidate)
        return self._model.predict(
            features, fallback=fallback_score,
            fallback_explanation=f"Cold start — using AttentionManager fixed-weight formula ({fallback_score:.2f}).",
        )

    def score_many(self, candidates: list[AttentionCandidate]) -> list[tuple[AttentionCandidate, PredictionResult]]:
        """Score a batch of candidates, sorted descending by score."""
        scored = [(c, self.score(c)) for c in candidates]
        return sorted(scored, key=lambda t: -t[1].value)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def observe_importance(self, candidate: AttentionCandidate, was_important: bool) -> None:
        """
        Record a real (candidate features, was_important) observation —
        e.g. ``was_important=True`` if this workspace entry led to a
        useful adaptation, was referenced again, or specialist consensus
        confirmed its relevance; ``False`` if it turned out to be noise.
        """
        features = self._features(candidate)
        self._model.add_example(features, label=1.0 if was_important else 0.0)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._model.is_trained

    @property
    def sample_count(self) -> int:
        return self._model.sample_count
