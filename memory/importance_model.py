"""
Memory Importance Predictor — Blix v0.3.10  (New module 14)

Upgrades importance scoring from the hand-crafted heuristic used by
the v0.2 ``MemoryExtractor`` (length, keyword presence, recency-style
rules) to a learned model: given features of a memory entry, predict
its importance score directly from Blix's own track record of which
memories turned out to matter (e.g. were retrieved often, referenced
in successful plans, or explicitly marked important).

The existing hand-crafted importance value (already stored on every
``schemas.memory_entry.MemoryEntry.importance``) is reused as BOTH a
cold-start fallback AND a feature — this model refines an existing
signal rather than discarding it.

Python 3.10 compatible.
"""

from __future__ import annotations

from pathlib import Path

from learning.ml_base import PredictionResult, TrainableModel
from utils.logger import get_logger

log = get_logger(__name__)

_FEATURE_NAMES = [
    "heuristic_importance",
    "input_length",
    "output_length",
    "retrieval_count",
    "has_embedding",
]

_DEFAULT_MIN_SAMPLES = 20


def _estimator_factory():
    from sklearn.linear_model import LinearRegression
    return LinearRegression()


class MemoryImportancePredictor:
    """
    Learns to predict memory importance from entry features, refining
    (not replacing) the existing hand-crafted importance heuristic.

    Parameters
    ----------
    examples_file:
        Path to persist accumulated training examples.
    min_samples_to_train:
        Minimum examples before switching from fallback to learned mode.
    """

    def __init__(self, examples_file: Path, min_samples_to_train: int = _DEFAULT_MIN_SAMPLES) -> None:
        self._model = TrainableModel(
            examples_file=examples_file, feature_names=_FEATURE_NAMES,
            min_samples_to_train=min_samples_to_train, estimator_factory=_estimator_factory,
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _features(
        heuristic_importance: float, input_text: str, output_text: str,
        retrieval_count: int = 0, has_embedding: bool = False,
    ) -> dict[str, float]:
        return {
            "heuristic_importance": heuristic_importance,
            "input_length": min(1.0, len(input_text) / 500.0),
            "output_length": min(1.0, len(output_text) / 500.0),
            "retrieval_count": min(1.0, retrieval_count / 10.0),
            "has_embedding": 1.0 if has_embedding else 0.0,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self, heuristic_importance: float, input_text: str, output_text: str,
        retrieval_count: int = 0, has_embedding: bool = False,
    ) -> PredictionResult:
        """Predict refined importance score, falling back to the hand-crafted value when cold."""
        features = self._features(heuristic_importance, input_text, output_text, retrieval_count, has_embedding)
        return self._model.predict(
            features, fallback=heuristic_importance,
            fallback_explanation=f"Cold start — using existing hand-crafted importance ({heuristic_importance:.2f}).",
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def observe_true_importance(
        self, observed_importance: float, heuristic_importance: float,
        input_text: str, output_text: str, retrieval_count: int = 0, has_embedding: bool = False,
    ) -> None:
        """
        Record a real (entry features) -> observed-importance example.
        ``observed_importance`` might come from explicit user feedback,
        retrieval frequency, or downstream task success correlation —
        whatever ground-truth signal the caller has available.
        """
        features = self._features(heuristic_importance, input_text, output_text, retrieval_count, has_embedding)
        self._model.add_example(features, label=max(0.0, min(1.0, observed_importance)))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._model.is_trained

    @property
    def sample_count(self) -> int:
        return self._model.sample_count
