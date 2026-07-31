"""
ML Model Base — Blix v0.3.10  (shared infrastructure, not a numbered spec item)

Every "learned" module in v0.3.10 (Confidence Model, Tool Success
Predictor, Memory Importance Predictor, Strategy Selector, Failure
Clusterer) follows the same honest pattern: train a small,
genuinely-real scikit-learn model on Blix's OWN accumulated runtime
data, and fall back to the existing v0.3.x heuristic until there's
enough data for the learned model to be trustworthy.

This module is NOT one of the spec's 14 numbered items — it's the
shared scaffolding that keeps that pattern from being copy-pasted five
times. A real, citable constraint shaped this design: this environment
has no path to download pretrained weights (huggingface.co and similar
model hosts are not in the network allowlist) and no historical
production data to train on upfront. So every learned model here:

  1. Starts in "cold start" mode — predict() returns the supplied
     heuristic/fallback value, clearly labeled as such.
  2. Accumulates (features, label) pairs as Blix actually runs.
  3. Once `min_samples_to_train` examples exist, fits a small
     scikit-learn estimator and switches to "learned" mode.
  4. Re-fits periodically as more data arrives (simple, not online
     learning — appropriate for the small data volumes Blix produces).

This is real machine learning on real (if currently sparse) data, not
a simulated/decorative model. It is honest about being immature when
data is scarce, rather than fabricating false precision.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class TrainingExample:
    """One (features, label) pair accumulated for a learned model."""

    features: dict[str, float]
    label: float
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {"features": self.features, "label": self.label, "recorded_at": self.recorded_at}

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingExample":
        return cls(features=d["features"], label=d["label"], recorded_at=d.get("recorded_at", ""))


@dataclass
class PredictionResult:
    """
    A prediction, tagged with whether it came from the trained model or
    the cold-start fallback — callers (and tests) should never have to
    guess which mode produced a number.
    """

    value: float
    mode: str          # "learned" | "fallback"
    sample_count: int
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "value": round(self.value, 4), "mode": self.mode,
            "sample_count": self.sample_count, "explanation": self.explanation,
        }


class TrainableModel:
    """
    Minimal fit/predict wrapper around a scikit-learn-compatible
    estimator, gated on a minimum sample count, with disk persistence
    of accumulated training examples (not the fitted model itself —
    refitting from examples on load keeps this dependency-light and
    avoids pickling fragility across scikit-learn versions).

    Parameters
    ----------
    examples_file:
        Path to persist accumulated ``TrainingExample`` records.
    feature_names:
        Ordered list of feature keys — fixes the input vector layout.
    min_samples_to_train:
        Minimum examples before switching from fallback to learned mode.
    refit_every:
        Re-fit the estimator every N new examples once trained (not on
        every single example — keeps this cheap).
    estimator_factory:
        Zero-arg callable returning a fresh, unfitted scikit-learn
        estimator (e.g. ``lambda: LogisticRegression()``).
    """

    def __init__(
        self,
        examples_file: Path,
        feature_names: list[str],
        min_samples_to_train: int,
        estimator_factory,
        refit_every: int = 5,
    ) -> None:
        self._file = examples_file
        self._feature_names = feature_names
        self._min_samples = min_samples_to_train
        self._refit_every = refit_every
        self._estimator_factory = estimator_factory
        self._examples: list[TrainingExample] = []
        self._estimator = None
        self._fitted_at_count = 0
        self._load()
        if len(self._examples) >= self._min_samples:
            self._fit()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._examples = [TrainingExample.from_dict(e) for e in raw]
        except Exception:
            self._examples = []

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([e.to_dict() for e in self._examples], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def add_example(self, features: dict[str, float], label: float) -> None:
        """Record one (features, label) observation from real Blix activity."""
        self._examples.append(TrainingExample(features=features, label=label))
        self._save()
        if len(self._examples) >= self._min_samples:
            if self._estimator is None or (len(self._examples) - self._fitted_at_count) >= self._refit_every:
                self._fit()

    def _vectorize(self, features: dict[str, float]) -> list[float]:
        return [features.get(name, 0.0) for name in self._feature_names]

    def _fit(self) -> None:
        try:
            X = [self._vectorize(e.features) for e in self._examples]
            y = [e.label for e in self._examples]
            estimator = self._estimator_factory()
            estimator.fit(X, y)
            self._estimator = estimator
            self._fitted_at_count = len(self._examples)
        except Exception:
            # Fitting failures (e.g. degenerate single-class data) just
            # keep the model in fallback mode rather than crashing callers.
            self._estimator = None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: dict[str, float], fallback: float, fallback_explanation: str = "") -> PredictionResult:
        """
        Predict using the trained estimator if available, otherwise
        return the supplied heuristic fallback — always clearly tagged.
        """
        if self._estimator is None:
            return PredictionResult(
                value=fallback, mode="fallback", sample_count=len(self._examples),
                explanation=fallback_explanation or f"Cold start ({len(self._examples)}/{self._min_samples} samples) — using heuristic.",
            )
        try:
            X = [self._vectorize(features)]
            raw = self._estimator.predict(X)[0]
            value = max(0.0, min(1.0, float(raw)))
            return PredictionResult(
                value=value, mode="learned", sample_count=len(self._examples),
                explanation=f"Learned model prediction ({len(self._examples)} training examples).",
            )
        except Exception:
            return PredictionResult(
                value=fallback, mode="fallback", sample_count=len(self._examples),
                explanation="Learned model prediction failed — using heuristic fallback.",
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._estimator is not None

    @property
    def sample_count(self) -> int:
        return len(self._examples)

    @property
    def min_samples_to_train(self) -> int:
        return self._min_samples
