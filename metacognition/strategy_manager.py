"""
Strategy Manager — Blix v0.3.8  (New module 4)

Replaces the implicit "always reason and execute the same way" default
with explicit, inspectable strategy selection:

    if complexity > threshold:
        use_tree_of_thought()

    if confidence < 0.5:
        invoke_critic()

    if repeated_failure:
        switch_strategy()

This module does not implement Tree-of-Thought, alternative planners,
or new execution tools (out of scope for v0.3.8 per spec) — it decides
WHICH strategy a given goal/plan/situation calls for and exposes that
decision as a structured ``StrategyDecision`` for the
``metacognition.controller.MetaCognitiveController`` (and, eventually,
the Planner/Executor) to act on. The actual "tree of thought" reasoning
mode, if/when built, would be a consumer of this decision — not
something this module performs itself.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from planning.plan_evaluator import PlanQualityScore
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Strategy vocabulary
# ---------------------------------------------------------------------------


class ReasoningStrategy(str, Enum):
    DIRECT = "direct"                    # straightforward single-pass reasoning
    TREE_OF_THOUGHT = "tree_of_thought"    # explore multiple reasoning branches (high complexity)
    CRITIC_FIRST = "critic_first"            # invoke critic before proceeding (low confidence)
    DECOMPOSE_FURTHER = "decompose_further"    # break the plan down more before executing


@dataclass
class StrategyDecision:
    """The chosen strategy for one situation, with the reasoning behind it."""

    strategy: ReasoningStrategy
    reason: str
    triggers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"strategy": self.strategy.value, "reason": self.reason, "triggers": self.triggers}


# ---------------------------------------------------------------------------
# Strategy Manager
# ---------------------------------------------------------------------------


class StrategyManager:
    """
    Chooses a reasoning/execution strategy from plan quality signals and
    observed failure patterns.

    Parameters
    ----------
    complexity_threshold:
        Plan complexity above this triggers ``TREE_OF_THOUGHT``.
    confidence_threshold:
        Plan/answer confidence below this triggers ``CRITIC_FIRST``.
    repeated_failure_threshold:
        Consecutive failures (for the same task/goal) at or above this
        count trigger a strategy switch.
    """

    def __init__(
        self,
        complexity_threshold: float = 0.7,
        confidence_threshold: float = 0.5,
        repeated_failure_threshold: int = 2,
    ) -> None:
        self._complexity_threshold = complexity_threshold
        self._confidence_threshold = confidence_threshold
        self._repeated_failure_threshold = repeated_failure_threshold
        self._failure_counts: dict[str, int] = {}
        self._last_strategy: dict[str, ReasoningStrategy] = {}

    # ------------------------------------------------------------------
    # Failure tracking
    # ------------------------------------------------------------------

    def record_failure(self, ref_key: str) -> int:
        """Record one failure for ``ref_key`` (e.g. a task title or goal). Returns the new count."""
        self._failure_counts[ref_key] = self._failure_counts.get(ref_key, 0) + 1
        return self._failure_counts[ref_key]

    def record_success(self, ref_key: str) -> None:
        """Reset the failure streak for ``ref_key`` after a success."""
        self._failure_counts[ref_key] = 0

    def failure_count(self, ref_key: str) -> int:
        return self._failure_counts.get(ref_key, 0)

    def is_repeated_failure(self, ref_key: str) -> bool:
        return self.failure_count(ref_key) >= self._repeated_failure_threshold

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    def decide(
        self,
        ref_key: str,
        quality: Optional[PlanQualityScore] = None,
        confidence: Optional[float] = None,
    ) -> StrategyDecision:
        """
        Decide the appropriate strategy for a given situation.

        Parameters
        ----------
        ref_key:
            Identifier for the thing being strategized about (task title,
            goal text, or graph_id) — used to track repeated failures.
        quality:
            Optional ``PlanQualityScore`` — supplies complexity/confidence
            signals directly when evaluating a specific plan.
        confidence:
            Optional standalone confidence value, used when no
            ``PlanQualityScore`` is available (e.g. evaluating an answer
            rather than a plan).
        """
        triggers: list[str] = []

        if self.is_repeated_failure(ref_key):
            triggers.append(f"repeated_failure(count={self.failure_count(ref_key)})")
            decision = StrategyDecision(
                strategy=ReasoningStrategy.DECOMPOSE_FURTHER,
                reason=(
                    f"'{ref_key}' has failed {self.failure_count(ref_key)} time(s) — "
                    "switching to a more granular decomposition strategy."
                ),
                triggers=triggers,
            )
            self._last_strategy[ref_key] = decision.strategy
            return decision

        effective_confidence = confidence
        if quality is not None and effective_confidence is None:
            effective_confidence = quality.confidence

        if effective_confidence is not None and effective_confidence < self._confidence_threshold:
            triggers.append(f"low_confidence({effective_confidence:.2f})")
            decision = StrategyDecision(
                strategy=ReasoningStrategy.CRITIC_FIRST,
                reason=(
                    f"Confidence {effective_confidence:.2f} is below threshold "
                    f"{self._confidence_threshold:.2f} — invoking critic before proceeding."
                ),
                triggers=triggers,
            )
            self._last_strategy[ref_key] = decision.strategy
            return decision

        if quality is not None and quality.complexity > self._complexity_threshold:
            triggers.append(f"high_complexity({quality.complexity:.2f})")
            decision = StrategyDecision(
                strategy=ReasoningStrategy.TREE_OF_THOUGHT,
                reason=(
                    f"Plan complexity {quality.complexity:.2f} exceeds threshold "
                    f"{self._complexity_threshold:.2f} — exploring multiple reasoning branches."
                ),
                triggers=triggers,
            )
            self._last_strategy[ref_key] = decision.strategy
            return decision

        decision = StrategyDecision(
            strategy=ReasoningStrategy.DIRECT,
            reason="No complexity, confidence, or failure-pattern triggers fired — proceeding directly.",
            triggers=triggers,
        )
        self._last_strategy[ref_key] = decision.strategy
        return decision

    def last_strategy_for(self, ref_key: str) -> Optional[ReasoningStrategy]:
        return self._last_strategy.get(ref_key)

    def has_switched_strategy(self, ref_key: str) -> bool:
        """True if the most recent decision for ``ref_key`` was anything other than DIRECT."""
        return self._last_strategy.get(ref_key, ReasoningStrategy.DIRECT) != ReasoningStrategy.DIRECT
