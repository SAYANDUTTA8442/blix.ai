"""
Plan Quality Evaluator — Blix v0.3.8  (New module 7)

Sits in the pipeline between the Critic and the Executor:

    Planner → Critic → Plan Evaluator → Executor

``PlanCritic`` (v0.3.6) answers a binary-ish question: is this plan
safe to run (APPROVED / APPROVED_WITH_WARNINGS / REJECTED)? It checks
for structural problems — cycles, missing tools, known failures.

``PlanQualityEvaluator`` answers a different, complementary question:
assuming the plan IS structurally sound, how GOOD is it likely to be?
It produces continuous scores — complexity, risk, confidence,
dependency density, expected_success — that the
``metacognition.strategy_manager.StrategyManager`` and
``metacognition.controller.MetaCognitiveController`` use to decide
whether to proceed as-is, request a different planning strategy, or
invoke the critic again after a revision.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agents.tool_reliability import ToolReliabilityRegistry
from agents.types import TaskGraph
from planning.critic import CritiqueReport, IssueSeverity
from reasoning.confidence_reasoner import ConfidenceReasoner
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class PlanQualityScore:
    """Full quality assessment for one plan."""

    graph_id: str
    complexity: float        # 0-1, higher = more complex
    risk: float                # 0-1, higher = riskier
    confidence: float           # 0-1, derived plan confidence
    dependency_density: float     # 0-1, how interlinked the DAG is
    expected_success: float        # 0-1, blended overall estimate
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "graph_id": self.graph_id,
            "complexity": round(self.complexity, 4),
            "risk": round(self.risk, 4),
            "confidence": round(self.confidence, 4),
            "dependency_density": round(self.dependency_density, 4),
            "expected_success": round(self.expected_success, 4),
            "notes": self.notes,
        }

    @property
    def is_high_risk(self) -> bool:
        return self.risk >= 0.6

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < 0.5


# ---------------------------------------------------------------------------
# Plan Quality Evaluator
# ---------------------------------------------------------------------------


class PlanQualityEvaluator:
    """
    Scores a plan's expected quality on continuous dimensions.

    Parameters
    ----------
    tool_reliability:
        Optional ``ToolReliabilityRegistry`` — feeds both risk and
        expected_success.
    confidence_reasoner:
        Optional ``ConfidenceReasoner`` — reused for the confidence
        dimension rather than re-deriving plan confidence independently.
    complexity_step_threshold:
        Step count above which complexity starts climbing meaningfully.
    """

    def __init__(
        self,
        tool_reliability: Optional[ToolReliabilityRegistry] = None,
        confidence_reasoner: Optional[ConfidenceReasoner] = None,
        complexity_step_threshold: int = 4,
    ) -> None:
        self._tool_reliability = tool_reliability
        self._reasoner = confidence_reasoner or ConfidenceReasoner(tool_reliability)
        self._complexity_threshold = complexity_step_threshold

    # ------------------------------------------------------------------
    # Component scores
    # ------------------------------------------------------------------

    def _complexity(self, graph: TaskGraph) -> float:
        step_count = len(graph.tasks)
        if step_count <= self._complexity_threshold:
            return min(1.0, step_count / (2 * self._complexity_threshold))
        excess = step_count - self._complexity_threshold
        return min(1.0, 0.5 + 0.05 * excess)

    def _dependency_density(self, graph: TaskGraph) -> float:
        step_count = len(graph.tasks)
        if step_count <= 1:
            return 0.0
        total_deps = sum(len(t.depends_on) for t in graph.tasks)
        max_possible = step_count * (step_count - 1) / 2
        if max_possible == 0:
            return 0.0
        return min(1.0, total_deps / max_possible)

    def _risk(self, graph: TaskGraph, critique: Optional[CritiqueReport]) -> tuple[float, list[str]]:
        notes: list[str] = []
        risk = 0.0

        if critique is not None:
            warning_count = sum(1 for i in critique.issues if i.severity == IssueSeverity.WARNING)
            critical_count = sum(1 for i in critique.issues if i.severity == IssueSeverity.CRITICAL)
            risk += 0.15 * warning_count + 0.35 * critical_count
            if warning_count:
                notes.append(f"{warning_count} critic warning(s).")
            if critical_count:
                notes.append(f"{critical_count} critic critical issue(s).")

        if self._tool_reliability is not None:
            tool_hints = [t.tool_hint for t in graph.tasks if t.tool_hint]
            unreliable = [
                h for h in tool_hints
                if self._tool_reliability.is_confident(h) and self._tool_reliability.success_rate(h) < 0.5
            ]
            if unreliable:
                risk += 0.1 * len(set(unreliable))
                notes.append(f"{len(set(unreliable))} unreliable tool(s) referenced.")

        return min(1.0, risk), notes

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(self, graph: TaskGraph, critique: Optional[CritiqueReport] = None) -> PlanQualityScore:
        """Produce a full quality assessment for ``graph``."""
        complexity = self._complexity(graph)
        dependency_density = self._dependency_density(graph)
        risk, notes = self._risk(graph, critique)
        confidence_estimate = self._reasoner.plan_confidence(graph, critique)
        confidence = confidence_estimate.score

        # Expected success blends confidence (optimistic signal) against
        # risk and complexity (pessimistic signals).
        expected_success = max(0.0, min(1.0, confidence - 0.3 * risk - 0.15 * complexity))

        if complexity >= 0.7:
            notes.append("Plan complexity is high — consider decomposition.")
        if dependency_density >= 0.6:
            notes.append("Plan is heavily interdependent — failures may cascade.")

        return PlanQualityScore(
            graph_id=graph.graph_id, complexity=complexity, risk=risk,
            confidence=confidence, dependency_density=dependency_density,
            expected_success=expected_success, notes=notes,
        )
