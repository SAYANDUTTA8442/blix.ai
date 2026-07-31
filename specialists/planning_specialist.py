"""
Planning Specialist — Blix v0.3.9  (Part of New module 5, Internal Specialists)

Wraps ``planning.plan_evaluator.PlanQualityEvaluator`` (v0.3.8) behind
the ``BaseSpecialist`` interface: given a plan, does this specialist
think it's viable?

Python 3.10 compatible.
"""

from __future__ import annotations

from typing import Optional

from planning.plan_evaluator import PlanQualityEvaluator
from specialists.base import BaseSpecialist, SpecialistOpinion


class PlanningSpecialist(BaseSpecialist):
    """
    Opines on plan viability using ``PlanQualityEvaluator``.

    Parameters
    ----------
    plan_evaluator:
        ``PlanQualityEvaluator`` — supplies the underlying quality score.
    """

    name = "planning_specialist"

    def __init__(self, plan_evaluator: Optional[PlanQualityEvaluator] = None, name: Optional[str] = None) -> None:
        self._evaluator = plan_evaluator or PlanQualityEvaluator()
        if name:
            self.name = name

    def consult(self, topic: str, graph=None, critique=None, **context) -> SpecialistOpinion:
        if graph is None:
            return SpecialistOpinion(
                specialist=self.name, verdict="no_opinion", confidence=0.0,
                rationale="No plan graph supplied for evaluation.",
            )
        score = self._evaluator.evaluate(graph, critique)
        if score.expected_success >= 0.65:
            verdict = "supports"
        elif score.expected_success < 0.4:
            verdict = "opposes"
        else:
            verdict = "uncertain"
        return SpecialistOpinion(
            specialist=self.name, verdict=verdict, confidence=score.expected_success,
            rationale=(
                f"expected_success={score.expected_success:.2f}, "
                f"complexity={score.complexity:.2f}, risk={score.risk:.2f}."
            ),
        )
