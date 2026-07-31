"""
Strategy Selector Network — Blix v0.3.10  (New module 6)

Upgrades ``metacognition.strategy_manager.StrategyManager.decide()``
(v0.3.8, fixed if/else thresholds: ``if complexity > 0.7: TREE_OF_THOUGHT``)
to a learned classifier: given task features, predict which
``ReasoningStrategy`` is actually most likely to lead to success,
trained on Blix's own record of which strategy was used and whether
the outcome succeeded.

    task features -> Model -> {DIRECT, TREE_OF_THOUGHT, CRITIC_FIRST, DECOMPOSE_FURTHER}

This module does NOT replace ``StrategyManager`` — ``StrategyManager``
remains the source of truth for repeated-failure tracking and the
threshold-based fallback. ``StrategySelectorNetwork`` wraps it:
fixed-threshold logic IS the cold-start fallback here, exactly as
``StrategyManager.decide()`` already computes it, and the learned
classifier only takes over once enough (features, strategy_used,
outcome) examples exist to identify which strategy actually correlates
with success for a given feature profile — not just which one the
fixed heuristic would have picked.

Python 3.10 compatible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from metacognition.strategy_manager import ReasoningStrategy, StrategyDecision, StrategyManager
from planning.plan_evaluator import PlanQualityScore
from utils.logger import get_logger

log = get_logger(__name__)

_STRATEGIES = [
    ReasoningStrategy.DIRECT, ReasoningStrategy.TREE_OF_THOUGHT,
    ReasoningStrategy.CRITIC_FIRST, ReasoningStrategy.DECOMPOSE_FURTHER,
]

_FEATURE_NAMES = ["complexity", "confidence", "risk", "dependency_density", "failure_count"]

_DEFAULT_MIN_SAMPLES = 20


def _estimator_factory():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=500, multi_class="multinomial")


class StrategySelectorNetwork:
    """
    Learns which ``ReasoningStrategy`` correlates with success for a
    given task feature profile, falling back to
    ``StrategyManager.decide()``'s fixed thresholds when undertrained.

    Parameters
    ----------
    strategy_manager:
        ``StrategyManager`` — supplies the cold-start fallback decision
        and repeated-failure tracking (unchanged, still authoritative
        for failure-streak detection).
    examples_file:
        Path to persist accumulated (features, strategy, outcome) examples.
    min_samples_to_train:
        Minimum examples per class before the learned classifier is used.
    """

    def __init__(
        self,
        strategy_manager: StrategyManager,
        examples_file: Path,
        min_samples_to_train: int = _DEFAULT_MIN_SAMPLES,
    ) -> None:
        self._strategy_manager = strategy_manager
        self._file = examples_file
        self._min_samples = min_samples_to_train
        self._examples: list[dict] = []
        self._classifier = None
        self._load()
        if len(self._examples) >= self._min_samples:
            self._fit()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        import json
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                self._examples = json.load(fh)
        except Exception:
            self._examples = []

    def _save(self) -> None:
        import json
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump(self._examples[-2000:], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _features(quality: Optional[PlanQualityScore], failure_count: int) -> dict[str, float]:
        if quality is None:
            return {"complexity": 0.5, "confidence": 0.5, "risk": 0.0, "dependency_density": 0.0, "failure_count": min(1.0, failure_count / 5.0)}
        return {
            "complexity": quality.complexity, "confidence": quality.confidence,
            "risk": quality.risk, "dependency_density": quality.dependency_density,
            "failure_count": min(1.0, failure_count / 5.0),
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def observe_outcome(
        self, ref_key: str, quality: Optional[PlanQualityScore], strategy_used: ReasoningStrategy, succeeded: bool,
    ) -> None:
        """Record a real (features, strategy_used) -> succeeded observation."""
        features = self._features(quality, self._strategy_manager.failure_count(ref_key))
        self._examples.append({"features": features, "strategy": strategy_used.value, "succeeded": succeeded})
        self._save()
        if len(self._examples) >= self._min_samples:
            self._fit()

    def _fit(self) -> None:
        try:
            from sklearn.linear_model import LogisticRegression
            # Train one-vs-rest: for each strategy, predict P(success | features) when that
            # strategy was used. At decide-time we pick the strategy with highest predicted
            # P(success) — this directly targets "which strategy leads to success", not
            # merely "which strategy would the fixed heuristic have picked".
            self._classifier = {}
            for strategy in _STRATEGIES:
                strategy_examples = [e for e in self._examples if e["strategy"] == strategy.value]
                if len(strategy_examples) < 4 or len({e["succeeded"] for e in strategy_examples}) < 2:
                    continue  # not enough examples or single-class — skip, fallback covers it
                X = [[e["features"].get(f, 0.0) for f in _FEATURE_NAMES] for e in strategy_examples]
                y = [1.0 if e["succeeded"] else 0.0 for e in strategy_examples]
                model = LogisticRegression(max_iter=500)
                model.fit(X, y)
                self._classifier[strategy] = model
        except Exception:
            self._classifier = None

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_strategy(
        self, ref_key: str, quality: Optional[PlanQualityScore] = None, confidence: Optional[float] = None,
    ) -> StrategyDecision:
        """
        Select a strategy: learned per-strategy success prediction if
        trained, otherwise ``StrategyManager.decide()``'s fixed-threshold
        fallback (also always computed, to preserve repeated-failure
        priority handling regardless of training state).
        """
        fallback_decision = self._strategy_manager.decide(ref_key, quality=quality, confidence=confidence)

        # Repeated-failure handling always takes priority, learned or not —
        # this is a safety-relevant escalation path, not a preference to be learned away.
        if self._strategy_manager.is_repeated_failure(ref_key):
            return fallback_decision

        if not self._classifier:
            return fallback_decision

        features = self._features(quality, self._strategy_manager.failure_count(ref_key))
        X = [[features.get(f, 0.0) for f in _FEATURE_NAMES]]
        predictions: dict[ReasoningStrategy, float] = {}
        for strategy, model in self._classifier.items():
            try:
                predictions[strategy] = float(model.predict_proba(X)[0][1])
            except Exception:
                continue

        if not predictions:
            return fallback_decision

        best_strategy = max(predictions, key=lambda s: predictions[s])
        return StrategyDecision(
            strategy=best_strategy,
            reason=f"Learned selector predicts P(success)={predictions[best_strategy]:.2f} for {best_strategy.value} "
                   f"(trained on {len(self._examples)} examples).",
            triggers=[f"learned_selection(p_success={predictions[best_strategy]:.2f})"],
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return bool(self._classifier)

    @property
    def sample_count(self) -> int:
        return len(self._examples)
