"""
Confidence Reasoner — Blix v0.3.8  (New module 3b)

Where ``metacognition.confidence_manager.ConfidenceManager`` is the
generic STORE for confidence values, ``ConfidenceReasoner`` is the
generic COMPUTATION layer that derives a confidence score for a plan,
an answer, or a tool choice from its underlying signals, then hands the
result to ``ConfidenceManager`` for storage.

This separation matters: storage shouldn't know how confidence is
computed for different object types (a plan's confidence depends on
critic verdict + step risk; a tool's confidence depends on historical
reliability; an answer's confidence depends on evidence convergence),
and computation shouldn't need to know how/where results are persisted.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agents.tool_reliability import ToolReliabilityRegistry
from agents.types import TaskGraph
from planning.critic import CritiqueReport, IssueSeverity, PlanVerdict
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceEstimate:
    """One derived confidence estimate, with the factors that produced it."""

    target: str                  # "plan" | "tool" | "answer"
    ref_id: str
    score: float
    factors: dict[str, float] = field(default_factory=dict)
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "ref_id": self.ref_id,
            "score": round(self.score, 4),
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Confidence Reasoner
# ---------------------------------------------------------------------------


class ConfidenceReasoner:
    """
    Derives confidence scores for plans, tool selections, and answers
    from their underlying signals.

    Parameters
    ----------
    tool_reliability:
        Optional ``ToolReliabilityRegistry`` — used for tool confidence.
    """

    def __init__(self, tool_reliability: Optional[ToolReliabilityRegistry] = None) -> None:
        self._tool_reliability = tool_reliability

    # ------------------------------------------------------------------
    # Plan confidence
    # ------------------------------------------------------------------

    def plan_confidence(
        self, graph: TaskGraph, critique: Optional[CritiqueReport] = None,
    ) -> ConfidenceEstimate:
        """
        Derive a plan's confidence from critic verdict, step count/risk,
        and (if available) per-step tool reliability.

        Heuristic factors:
            critic_verdict   — APPROVED=1.0, APPROVED_WITH_WARNINGS=0.65, REJECTED=0.1
            tool_reliability  — mean reliability of tools referenced by the plan
            size_penalty       — longer plans carry slightly more risk of partial failure
        """
        factors: dict[str, float] = {}

        if critique is not None:
            verdict_score = {
                PlanVerdict.APPROVED: 1.0,
                PlanVerdict.APPROVED_WITH_WARNINGS: 0.65,
                PlanVerdict.REJECTED: 0.1,
            }.get(critique.verdict, 0.5)
            # Further docking for each CRITICAL issue beyond verdict alone.
            critical_count = sum(1 for i in critique.issues if i.severity == IssueSeverity.CRITICAL)
            verdict_score = max(0.0, verdict_score - 0.1 * critical_count)
            factors["critic_verdict"] = verdict_score
        else:
            factors["critic_verdict"] = 0.7  # neutral-optimistic default, no critic available

        if self._tool_reliability is not None and graph.tasks:
            tool_hints = [t.tool_hint for t in graph.tasks if t.tool_hint]
            if tool_hints:
                reliabilities = [self._tool_reliability.success_rate(h) for h in tool_hints]
                factors["tool_reliability"] = sum(reliabilities) / len(reliabilities)
            else:
                factors["tool_reliability"] = 0.5
        else:
            factors["tool_reliability"] = 0.5

        step_count = len(graph.tasks)
        factors["size_penalty"] = max(0.5, 1.0 - 0.03 * max(0, step_count - 5))

        score = (
            0.5 * factors["critic_verdict"]
            + 0.3 * factors["tool_reliability"]
            + 0.2 * factors["size_penalty"]
        )
        score = max(0.0, min(1.0, score))

        explanation = (
            f"Plan confidence {score:.2f} derived from critic_verdict="
            f"{factors['critic_verdict']:.2f}, tool_reliability={factors['tool_reliability']:.2f}, "
            f"size_penalty={factors['size_penalty']:.2f}."
        )
        return ConfidenceEstimate(
            target="plan", ref_id=graph.graph_id, score=score,
            factors=factors, explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Tool confidence
    # ------------------------------------------------------------------

    def tool_confidence(self, tool_name: str) -> ConfidenceEstimate:
        """Derive a tool's confidence purely from its cross-run reliability record."""
        if self._tool_reliability is None:
            return ConfidenceEstimate(
                target="tool", ref_id=tool_name, score=0.5,
                factors={"reliability": 0.5},
                explanation="No ToolReliabilityRegistry available; neutral default.",
            )
        rate = self._tool_reliability.success_rate(tool_name)
        confident = self._tool_reliability.is_confident(tool_name)
        # Discount confidence slightly when sample size is too low to trust the rate fully.
        score = rate if confident else 0.5 * rate + 0.25
        return ConfidenceEstimate(
            target="tool", ref_id=tool_name, score=score,
            factors={"reliability": rate, "sample_confident": float(confident)},
            explanation=f"Tool '{tool_name}' success_rate={rate:.2f} (sample_confident={confident}).",
        )

    # ------------------------------------------------------------------
    # Answer confidence (evidence-convergence based)
    # ------------------------------------------------------------------

    def answer_confidence(
        self,
        evidence_count: int,
        source_count: int,
        contradicting_evidence_count: int = 0,
        base_confidence: float = 0.5,
    ) -> ConfidenceEstimate:
        """
        Derive confidence for an answer/belief from how much converging
        (vs. contradicting) evidence supports it.

        More distinct sources and more total evidence raise confidence;
        contradicting evidence pulls it back down.
        """
        evidence_factor = min(1.0, 0.15 * evidence_count)
        source_factor = min(1.0, 0.25 * source_count)
        contradiction_penalty = min(0.6, 0.2 * contradicting_evidence_count)

        score = base_confidence + 0.5 * evidence_factor + 0.3 * source_factor - contradiction_penalty
        score = max(0.0, min(1.0, score))

        factors = {
            "evidence_factor": evidence_factor,
            "source_factor": source_factor,
            "contradiction_penalty": contradiction_penalty,
        }
        explanation = (
            f"Answer confidence {score:.2f}: {evidence_count} evidence item(s) from "
            f"{source_count} source(s), {contradicting_evidence_count} contradicting."
        )
        return ConfidenceEstimate(target="answer", ref_id="", score=score, factors=factors, explanation=explanation)

    # ------------------------------------------------------------------
    # Calibration check
    # ------------------------------------------------------------------

    @staticmethod
    def is_low_confidence(estimate: ConfidenceEstimate, threshold: float = 0.5) -> bool:
        return estimate.score < threshold
