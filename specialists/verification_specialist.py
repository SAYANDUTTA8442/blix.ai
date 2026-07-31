"""
Verification Specialist — Blix v0.3.9  (Part of New module 5, Internal Specialists)

Wraps ``verification.verifier.VerificationEngine`` (v0.3.6) behind the
``BaseSpecialist`` interface: given a task and its execution result,
does it pass verification?

Python 3.10 compatible.
"""

from __future__ import annotations

from typing import Optional

from specialists.base import BaseSpecialist, SpecialistOpinion
from verification.verifier import VerificationEngine


class VerificationSpecialist(BaseSpecialist):
    """
    Opines on whether a task's execution result passes verification.

    Parameters
    ----------
    verification_engine:
        ``VerificationEngine`` — supplies the underlying pass/fail report.
    """

    name = "verification_specialist"

    def __init__(self, verification_engine: Optional[VerificationEngine] = None, name: Optional[str] = None) -> None:
        self._engine = verification_engine or VerificationEngine()
        if name:
            self.name = name

    def consult(self, topic: str, task=None, result=None, **context) -> SpecialistOpinion:
        if task is None or result is None:
            return SpecialistOpinion(
                specialist=self.name, verdict="no_opinion", confidence=0.0,
                rationale="No task/execution result supplied for verification.",
            )
        report = self._engine.verify(task, result)
        verdict = "supports" if report.passed else "opposes"
        confidence = 1.0 if report.passed else 0.2
        return SpecialistOpinion(specialist=self.name, verdict=verdict, confidence=confidence, rationale=report.summary())
