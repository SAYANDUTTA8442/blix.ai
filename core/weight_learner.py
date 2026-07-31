"""
Learnable Retrieval Weights — Blix v0.3.1  (Issue 1)

Addresses: "Retrieval weights are hand-tuned / heuristic."

Implements two learning strategies:

1. ``PairwiseWeightLearner``
   Records (winner_id, loser_id) feedback pairs and fits weights
   by gradient-free optimisation over the pairwise ranking loss.

2. ``BayesianWeightOptimizer``
   Uses held-out retrieval precision as the objective and tunes
   weights with random search over a Sobol sequence — a lightweight
   Bayesian-style grid that keeps the dep-free constraint.

Both strategies update a shared ``ScoringWeights`` and persist their
training log to ``memory/scorer_weights.json``.

Python 3.10 compatible — no scipy / optuna required.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.memory_scorer import MemoryScore, MemoryScorer, ScoringWeights
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feedback types
# ---------------------------------------------------------------------------


@dataclass
class RetrievalFeedback:
    """
    One preference signal: memory ``winner_id`` was more relevant than
    ``loser_id`` for ``query`` at ``timestamp``.

    Collected implicitly (user clicked / quoted winner) or explicitly
    (thumbs-up/down on a retrieved memory in the UI).
    """

    query: str
    winner_id: int
    loser_id: int
    winner_score: float = 0.0
    loser_score: float = 0.0


@dataclass
class WeightTrainingLog:
    """Persisted training history for reproducibility."""

    iterations: int = 0
    best_weights: Optional[dict] = None
    best_metric: float = 0.0
    history: list[dict] = field(default_factory=list)

    def record(self, weights: ScoringWeights, metric: float) -> None:
        entry = {"weights": weights.model_dump(), "metric": metric, "iter": self.iterations}
        self.history.append(entry)
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_weights = weights.model_dump()
        self.iterations += 1

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "best_weights": self.best_weights,
            "best_metric": self.best_metric,
            "history": self.history[-50:],  # keep last 50 for space
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeightTrainingLog":
        obj = cls()
        obj.iterations = d.get("iterations", 0)
        obj.best_weights = d.get("best_weights")
        obj.best_metric = d.get("best_metric", 0.0)
        obj.history = d.get("history", [])
        return obj


# ---------------------------------------------------------------------------
# Pairwise ranking learner
# ---------------------------------------------------------------------------


class PairwiseWeightLearner:
    """
    Fits ``ScoringWeights`` from pairwise preference feedback using
    a coordinate-descent optimiser over the pairwise ranking loss:

        L = sum_{(w,l)} max(0, 1 - (score(w) - score(l)))

    Parameters
    ----------
    weights_file:
        Path to ``scorer_weights.json``.
    n_restarts:
        Number of random restarts for coordinate descent.
    step_size:
        Gradient-free perturbation step per coordinate.
    """

    def __init__(
        self,
        weights_file: Path,
        n_restarts: int = 5,
        step_size: float = 0.05,
    ) -> None:
        self._file = weights_file
        self._n_restarts = n_restarts
        self._step = step_size
        self._feedback: list[RetrievalFeedback] = []
        self._log = self._load_log()

    # ------------------------------------------------------------------
    # Feedback collection
    # ------------------------------------------------------------------

    def record_feedback(self, fb: RetrievalFeedback) -> None:
        """Add one preference pair to the training buffer."""
        self._feedback.append(fb)
        log.debug(
            "PairwiseWeightLearner: recorded feedback winner=%d loser=%d",
            fb.winner_id, fb.loser_id,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        scorer: MemoryScorer,
        scored_entries: list[dict],
    ) -> ScoringWeights:
        """
        Re-fit weights given the accumulated feedback and current memory pool.

        Parameters
        ----------
        scorer:
            Current MemoryScorer (read for half-life; weights will be updated).
        scored_entries:
            List of dicts with keys matching ``MemoryScorer.score_batch``.

        Returns
        -------
        ScoringWeights
            Best weights found; also updates ``scorer._w`` in-place.
        """
        if len(self._feedback) < 3:
            log.debug("PairwiseWeightLearner: not enough feedback (%d < 3)", len(self._feedback))
            return scorer._w

        id_map = {e["id"]: e for e in scored_entries}
        best_w, best_loss = scorer._w, float("inf")

        for _ in range(self._n_restarts):
            w = _random_weights()
            w = self._coordinate_descent(w, id_map, scorer)
            loss = self._pairwise_loss(w, id_map, scorer)
            if loss < best_loss:
                best_loss = loss
                best_w = w

        # Update scorer in-place
        scorer._w = best_w
        self._log.record(best_w, -best_loss)  # metric = -loss (higher is better)
        self._save_log()
        log.info(
            "PairwiseWeightLearner: fitted weights r=%.2f i=%.2f rec=%.2f f=%.2f  loss=%.4f",
            best_w.relevance, best_w.importance, best_w.recency, best_w.frequency, best_loss,
        )
        return best_w

    def _coordinate_descent(
        self,
        w: ScoringWeights,
        id_map: dict,
        scorer: MemoryScorer,
        max_iter: int = 30,
    ) -> ScoringWeights:
        """One coordinate-descent pass."""
        fields = ["relevance", "importance", "recency", "frequency"]
        current_loss = self._pairwise_loss(w, id_map, scorer)
        for _ in range(max_iter):
            improved = False
            for f in fields:
                for delta in (self._step, -self._step):
                    candidate = _perturb(w, f, delta)
                    if candidate is None:
                        continue
                    loss = self._pairwise_loss(candidate, id_map, scorer)
                    if loss < current_loss:
                        w = candidate
                        current_loss = loss
                        improved = True
                        break
            if not improved:
                break
        return w

    def _pairwise_loss(
        self,
        w: ScoringWeights,
        id_map: dict,
        scorer: MemoryScorer,
    ) -> float:
        """Hinge loss over all preference pairs."""
        tmp_scorer = MemoryScorer(weights=w, recency_half_life_days=scorer._half_life)
        total = 0.0
        for fb in self._feedback:
            we = id_map.get(fb.winner_id)
            le = id_map.get(fb.loser_id)
            if we is None or le is None:
                continue
            ws = tmp_scorer.score(
                fb.winner_id,
                relevance=we.get("relevance", 0.5),
                importance=we.get("importance", 0.5),
                timestamp=we["timestamp"],
            )
            ls = tmp_scorer.score(
                fb.loser_id,
                relevance=le.get("relevance", 0.5),
                importance=le.get("importance", 0.5),
                timestamp=le["timestamp"],
            )
            total += max(0.0, 1.0 - (ws.final_score - ls.final_score))
        return total

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_log(self) -> WeightTrainingLog:
        if not self._file.exists():
            return WeightTrainingLog()
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                return WeightTrainingLog.from_dict(json.load(fh))
        except Exception as exc:
            log.warning("Could not load scorer weights log (%s)", exc)
            return WeightTrainingLog()

    def _save_log(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump(self._log.to_dict(), fh, indent=2)

    @property
    def best_weights(self) -> Optional[ScoringWeights]:
        """Return best weights found so far, or None if untrained."""
        if self._log.best_weights is None:
            return None
        return ScoringWeights(**self._log.best_weights)

    @property
    def training_log(self) -> WeightTrainingLog:
        return self._log


# ---------------------------------------------------------------------------
# Bayesian weight optimizer (random Sobol-style search)
# ---------------------------------------------------------------------------


class BayesianWeightOptimizer:
    """
    Lightweight weight search using quasi-random (Sobol-like) grid over
    the simplex {r+i+rec+f=1, all≥0.05}.

    Evaluates each candidate by computing retrieval precision on a
    held-out ``EvalDataset``.

    Parameters
    ----------
    n_trials:
        Number of weight candidates to evaluate.
    min_weight:
        Minimum value for any single weight (prevents degenerate solutions).
    """

    def __init__(self, n_trials: int = 50, min_weight: float = 0.05) -> None:
        self._n = n_trials
        self._min = min_weight

    def optimize(
        self,
        scorer: MemoryScorer,
        precision_fn: object,  # callable(ScoringWeights) → float
    ) -> ScoringWeights:
        """
        Search for weights that maximise ``precision_fn(weights) → float``.

        Parameters
        ----------
        scorer:
            Current scorer; ``_w`` is updated in-place to the best found.
        precision_fn:
            Callable that receives a ``ScoringWeights`` and returns a
            retrieval precision float (0–1).  Typically wraps an
            ``EvalDataset`` pass with ``MemoryEvaluator``.
        """
        best_w = scorer._w
        best_p = float("-inf")

        for trial in range(self._n):
            w = _sample_simplex(self._min, seed=trial)
            try:
                p = precision_fn(w)  # type: ignore[operator]
            except Exception as exc:
                log.warning("BayesianWeightOptimizer trial %d failed: %s", trial, exc)
                continue
            if p > best_p:
                best_p = p
                best_w = w
                log.debug(
                    "BayesianWeightOptimizer: new best p=%.4f  r=%.2f i=%.2f rec=%.2f f=%.2f",
                    p, w.relevance, w.importance, w.recency, w.frequency,
                )

        scorer._w = best_w
        log.info(
            "BayesianWeightOptimizer: best weights r=%.2f i=%.2f rec=%.2f f=%.2f  precision=%.4f",
            best_w.relevance, best_w.importance, best_w.recency, best_w.frequency, best_p,
        )
        return best_w


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_weights(seed: Optional[int] = None) -> ScoringWeights:
    """Sample a random weight vector on the unit simplex."""
    rng = random.Random(seed)
    v = sorted([rng.random() for _ in range(3)])
    cuts = [v[0], v[1] - v[0], v[2] - v[1], 1.0 - v[2]]
    cuts = [max(0.05, c) for c in cuts]
    total = sum(cuts)
    r, i, rec, f = [c / total for c in cuts]
    return ScoringWeights(relevance=r, importance=i, recency=rec, frequency=f)


def _sample_simplex(min_w: float = 0.05, seed: int = 0) -> ScoringWeights:
    """Quasi-random simplex sample with a minimum per-weight floor."""
    rng = random.Random(seed * 1_000_003 + 7)
    while True:
        cuts = [rng.random() for _ in range(4)]
        total = sum(cuts)
        w = [c / total for c in cuts]
        if all(wi >= min_w for wi in w):
            return ScoringWeights(
                relevance=w[0], importance=w[1], recency=w[2], frequency=w[3]
            )


def _perturb(
    w: ScoringWeights,
    field: str,
    delta: float,
) -> Optional[ScoringWeights]:
    """
    Perturb one weight by ``delta``, redistribute to another field,
    and return a new ``ScoringWeights`` if valid.
    """
    d = w.model_dump()
    d[field] = d[field] + delta
    fields = list(d.keys())
    other = [f for f in fields if f != field]
    # Subtract delta uniformly from the other three
    per = delta / len(other)
    for f in other:
        d[f] = d[f] - per
    if any(v < 0.0 or v > 1.0 for v in d.values()):
        return None
    total = sum(d.values())
    d = {k: v / total for k, v in d.items()}
    try:
        return ScoringWeights(**d)
    except Exception:
        return None
