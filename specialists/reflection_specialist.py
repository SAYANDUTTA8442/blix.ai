"""
Reflection Specialist — Blix v0.3.9  (Part of New module 5, Internal Specialists)

Wraps ``agents.failure_memory.FailureMemory`` and/or
``reflection.reflection_engine.ReflectionEngine`` behind the
``BaseSpecialist`` interface: given a topic/task description, has this
kind of thing gone wrong before, or has a relevant behavior-change
insight already been generated?

Python 3.10 compatible.
"""

from __future__ import annotations

from typing import Optional

from agents.failure_memory import FailureMemory
from specialists.base import BaseSpecialist, SpecialistOpinion


class ReflectionSpecialist(BaseSpecialist):
    """
    Opines based on known failure history for a topic.

    Parameters
    ----------
    failure_memory:
        ``FailureMemory`` — used to check for known failure patterns
        matching the topic (treated as a task title for lookup purposes).
    """

    name = "reflection_specialist"

    def __init__(self, failure_memory: Optional[FailureMemory] = None, name: Optional[str] = None) -> None:
        self._failure_memory = failure_memory
        if name:
            self.name = name

    def consult(self, topic: str, **context) -> SpecialistOpinion:
        if self._failure_memory is None:
            return SpecialistOpinion(specialist=self.name, verdict="no_opinion", confidence=0.0, rationale="No failure memory wired.")

        if not self._failure_memory.has_known_failure(topic):
            return SpecialistOpinion(
                specialist=self.name, verdict="supports", confidence=0.6,
                rationale=f"No known failure pattern matches '{topic}'.",
            )

        similar = self._failure_memory.similar_failures(topic)
        occurrences = sum(f.occurrences for f in similar) if similar else 1
        confidence = max(0.0, 1.0 - 0.15 * occurrences)
        return SpecialistOpinion(
            specialist=self.name, verdict="opposes", confidence=confidence,
            rationale=f"Known failure pattern matches '{topic}' ({occurrences} prior occurrence(s)).",
        )
