"""
Meta-Cognitive Controller — Blix v0.3.8  (New module 1)

The top of the v0.3.8 stack. Where every other module in this release
tracks ONE kind of self-knowledge (capabilities, confidence, strategy,
skills...), ``MetaCognitiveController`` is the thing that actually
looks at all of them together and decides whether Blix's current
behavior needs to change:

    Monitor:  plan_quality, retrieval_quality, execution_quality, belief_consistency
    Detect:   low_confidence, repeated_failures, hallucinations
    Adapt:    change_strategy(), replan(), switch_tool()

This module does NOT re-implement monitoring or adaptation primitives
that already exist — it composes them:

    plan_quality        <- planning.plan_evaluator.PlanQualityEvaluator   (v0.3.8)
    belief_consistency    <- evaluation.state_metrics.StateMetrics.truth_consistency (v0.3.7)
    low_confidence          <- reasoning.confidence_reasoner.ConfidenceReasoner (v0.3.8)
    repeated_failures         <- metacognition.strategy_manager.StrategyManager (v0.3.8)
    change_strategy()           <- metacognition.strategy_manager.StrategyManager.decide() (v0.3.8)
    replan() / switch_tool()      <- planning.replanner.Replanner (v0.3.6, unmodified)

``MetaCognitiveController`` is intentionally a thin coordination layer:
it reads signals from the modules above, classifies the overall
cognitive state, and returns an ``AdaptationDecision`` describing what
(if anything) should change — leaving the actual mechanics of
replanning/tool-switching to the v0.3.6 ``Replanner``, which is not
modified by this release.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from metacognition.strategy_manager import ReasoningStrategy, StrategyDecision, StrategyManager
from planning.plan_evaluator import PlanQualityEvaluator, PlanQualityScore
from reasoning.confidence_reasoner import ConfidenceEstimate, ConfidenceReasoner
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Issue vocabulary
# ---------------------------------------------------------------------------


class CognitiveIssue(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    REPEATED_FAILURES = "repeated_failures"
    HALLUCINATION_RISK = "hallucination_risk"
    LOW_BELIEF_CONSISTENCY = "low_belief_consistency"
    NONE = "none"


class AdaptationAction(str, Enum):
    NONE = "none"
    CHANGE_STRATEGY = "change_strategy"
    REPLAN = "replan"
    SWITCH_TOOL = "switch_tool"
    FLAG_FOR_REVIEW = "flag_for_review"


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass
class CognitiveMonitorReport:
    """Snapshot of the four monitored quality dimensions for one situation."""

    plan_quality: Optional[float] = None
    retrieval_quality: Optional[float] = None
    execution_quality: Optional[float] = None
    belief_consistency: Optional[float] = None
    issues: list[CognitiveIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "plan_quality": self.plan_quality,
            "retrieval_quality": self.retrieval_quality,
            "execution_quality": self.execution_quality,
            "belief_consistency": self.belief_consistency,
            "issues": [i.value for i in self.issues],
        }

    @property
    def has_issues(self) -> bool:
        return any(i != CognitiveIssue.NONE for i in self.issues)


@dataclass
class AdaptationDecision:
    """What the controller decided to do in response to a monitor report."""

    action: AdaptationAction
    reason: str
    strategy_decision: Optional[StrategyDecision] = None

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "strategy_decision": self.strategy_decision.to_dict() if self.strategy_decision else None,
        }


# ---------------------------------------------------------------------------
# Meta-Cognitive Controller
# ---------------------------------------------------------------------------


class MetaCognitiveController:
    """
    Monitors plan/retrieval/execution/belief quality, detects cognitive
    issues, and decides on adaptations.

    Parameters
    ----------
    plan_evaluator:
        ``PlanQualityEvaluator`` — supplies plan_quality.
    confidence_reasoner:
        ``ConfidenceReasoner`` — supplies low_confidence detection.
    strategy_manager:
        ``StrategyManager`` — supplies repeated_failures detection and
        strategy-change decisions.
    confidence_floor:
        Below this, LOW_CONFIDENCE is flagged.
    belief_consistency_floor:
        Below this, LOW_BELIEF_CONSISTENCY is flagged.
    hallucination_rate_ceiling:
        At/above this, HALLUCINATION_RISK is flagged.
    """

    def __init__(
        self,
        plan_evaluator: Optional[PlanQualityEvaluator] = None,
        confidence_reasoner: Optional[ConfidenceReasoner] = None,
        strategy_manager: Optional[StrategyManager] = None,
        confidence_floor: float = 0.5,
        belief_consistency_floor: float = 0.7,
        hallucination_rate_ceiling: float = 0.2,
    ) -> None:
        self._plan_evaluator = plan_evaluator or PlanQualityEvaluator()
        self._confidence_reasoner = confidence_reasoner or ConfidenceReasoner()
        self._strategy_manager = strategy_manager or StrategyManager()
        self._confidence_floor = confidence_floor
        self._belief_consistency_floor = belief_consistency_floor
        self._hallucination_ceiling = hallucination_rate_ceiling

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def monitor(
        self,
        ref_key: str,
        graph=None,
        critique=None,
        retrieval_quality: Optional[float] = None,
        execution_quality: Optional[float] = None,
        belief_consistency: Optional[float] = None,
        hallucination_rate: Optional[float] = None,
    ) -> CognitiveMonitorReport:
        """
        Produce a ``CognitiveMonitorReport`` for the current situation.

        Any signal that isn't applicable (e.g. no plan graph yet, no
        retrieval happened this turn) can be omitted — the report simply
        won't flag issues for missing dimensions.
        """
        report = CognitiveMonitorReport(
            retrieval_quality=retrieval_quality,
            execution_quality=execution_quality,
            belief_consistency=belief_consistency,
        )

        plan_quality_score: Optional[PlanQualityScore] = None
        if graph is not None:
            plan_quality_score = self._plan_evaluator.evaluate(graph, critique)
            report.plan_quality = plan_quality_score.expected_success

        issues: list[CognitiveIssue] = []

        effective_confidence = plan_quality_score.confidence if plan_quality_score else None
        if effective_confidence is not None and effective_confidence < self._confidence_floor:
            issues.append(CognitiveIssue.LOW_CONFIDENCE)

        if self._strategy_manager.is_repeated_failure(ref_key):
            issues.append(CognitiveIssue.REPEATED_FAILURES)

        if hallucination_rate is not None and hallucination_rate >= self._hallucination_ceiling:
            issues.append(CognitiveIssue.HALLUCINATION_RISK)

        if belief_consistency is not None and belief_consistency < self._belief_consistency_floor:
            issues.append(CognitiveIssue.LOW_BELIEF_CONSISTENCY)

        if not issues:
            issues.append(CognitiveIssue.NONE)

        report.issues = issues
        return report

    # ------------------------------------------------------------------
    # Adaptation
    # ------------------------------------------------------------------

    def adapt(
        self, ref_key: str, report: CognitiveMonitorReport, quality: Optional[PlanQualityScore] = None,
    ) -> AdaptationDecision:
        """
        Decide what to do in response to a ``CognitiveMonitorReport``.

        Priority order: REPEATED_FAILURES > HALLUCINATION_RISK >
        LOW_BELIEF_CONSISTENCY > LOW_CONFIDENCE > none.
        """
        if CognitiveIssue.REPEATED_FAILURES in report.issues:
            strategy_decision = self._strategy_manager.decide(ref_key, quality=quality)
            action = (
                AdaptationAction.REPLAN
                if strategy_decision.strategy == ReasoningStrategy.DECOMPOSE_FURTHER
                else AdaptationAction.CHANGE_STRATEGY
            )
            return AdaptationDecision(
                action=action,
                reason=f"Repeated failures detected for '{ref_key}' — {strategy_decision.reason}",
                strategy_decision=strategy_decision,
            )

        if CognitiveIssue.HALLUCINATION_RISK in report.issues:
            return AdaptationDecision(
                action=AdaptationAction.FLAG_FOR_REVIEW,
                reason="Hallucination risk detected — flagging output for human/critic review rather than proceeding silently.",
            )

        if CognitiveIssue.LOW_BELIEF_CONSISTENCY in report.issues:
            return AdaptationDecision(
                action=AdaptationAction.FLAG_FOR_REVIEW,
                reason="Belief consistency is below the acceptable floor — flagging for truth-maintenance review.",
            )

        if CognitiveIssue.LOW_CONFIDENCE in report.issues:
            strategy_decision = self._strategy_manager.decide(ref_key, quality=quality)
            return AdaptationDecision(
                action=AdaptationAction.CHANGE_STRATEGY,
                reason=f"Low confidence detected — {strategy_decision.reason}",
                strategy_decision=strategy_decision,
            )

        return AdaptationDecision(action=AdaptationAction.NONE, reason="No cognitive issues detected.")

    # ------------------------------------------------------------------
    # Convenience: full monitor + adapt pass
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        ref_key: str,
        graph=None,
        critique=None,
        retrieval_quality: Optional[float] = None,
        execution_quality: Optional[float] = None,
        belief_consistency: Optional[float] = None,
        hallucination_rate: Optional[float] = None,
    ) -> tuple[CognitiveMonitorReport, AdaptationDecision]:
        """Monitor then adapt in one call — the typical integration point."""
        quality = self._plan_evaluator.evaluate(graph, critique) if graph is not None else None
        report = self.monitor(
            ref_key, graph=graph, critique=critique,
            retrieval_quality=retrieval_quality, execution_quality=execution_quality,
            belief_consistency=belief_consistency, hallucination_rate=hallucination_rate,
        )
        decision = self.adapt(ref_key, report, quality=quality)
        return report, decision
