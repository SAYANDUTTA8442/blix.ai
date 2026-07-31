"""
Memory Specialist — Blix v0.3.9  (Part of New module 5, Internal Specialists)

Wraps belief/memory lookup behind the ``BaseSpecialist`` interface:
given a topic, does Blix have relevant remembered context, and how
confident is that context?

Does not duplicate retrieval logic — delegates to whatever
``memory.beliefs.BeliefStore``-like object is provided (anything with
a ``find_by_keyword`` / similar lookup is acceptable; this module only
requires a callable lookup function to stay decoupled from any one
memory backend's exact API).

Python 3.10 compatible.
"""

from __future__ import annotations

from typing import Callable, Optional

from specialists.base import BaseSpecialist, SpecialistOpinion


class MemorySpecialist(BaseSpecialist):
    """
    Opines on whether relevant remembered context exists for a topic.

    Parameters
    ----------
    lookup_fn:
        Callable(topic: str) -> list[object with a `.confidence` attr
        or dict with a 'confidence' key], e.g. a thin wrapper around
        ``BeliefStore`` keyword search. If omitted, the specialist
        always returns "no_opinion" (degrades gracefully with nothing wired).
    name:
        Override for the specialist's reported name.
    """

    name = "memory_specialist"

    def __init__(self, lookup_fn: Optional[Callable[[str], list]] = None, name: Optional[str] = None) -> None:
        self._lookup_fn = lookup_fn
        if name:
            self.name = name

    def consult(self, topic: str, **context) -> SpecialistOpinion:
        if self._lookup_fn is None:
            return SpecialistOpinion(specialist=self.name, verdict="no_opinion", confidence=0.0, rationale="No memory lookup wired.")

        results = self._lookup_fn(topic)
        if not results:
            return SpecialistOpinion(
                specialist=self.name, verdict="no_opinion", confidence=0.0,
                rationale=f"No remembered context found for '{topic}'.",
            )

        confidences = [
            (r.confidence if hasattr(r, "confidence") else r.get("confidence", 0.5))
            for r in results
        ]
        mean_confidence = sum(confidences) / len(confidences)
        verdict = "supports" if mean_confidence >= 0.6 else "uncertain"
        return SpecialistOpinion(
            specialist=self.name, verdict=verdict, confidence=mean_confidence,
            rationale=f"Found {len(results)} relevant memory item(s), mean confidence {mean_confidence:.2f}.",
        )
