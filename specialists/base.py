"""
Specialist base — Blix v0.3.9  (Part of New module 5, Internal Specialists)

Defines the shared contract every specialist implements:

    Workspace
      ↓
    Specialists
      ↓
    Consensus

A "specialist" is a thin adapter that wraps an already-existing
subsystem (memory retrieval, planning, reflection, verification) and
exposes ONE uniform method — ``consult(topic) -> SpecialistOpinion`` —
so a consensus mechanism can poll heterogeneous subsystems without
needing bespoke per-subsystem logic. This is the beginning of a
Society-of-Mind pattern: specialists don't coordinate directly with
each other (that's still the Global Workspace's job) — they each give
an independent, scored opinion, and consensus is computed afterward.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class SpecialistOpinion:
    """
    One specialist's opinion about a topic.

    Fields
    ------
    specialist:
        Name of the specialist that produced this opinion.
    verdict:
        Short categorical verdict, e.g. "supports", "opposes", "uncertain",
        "no_opinion" — specialists are free to define their own verdict
        vocabulary, but "no_opinion" is the universal "nothing to add" signal.
    confidence:
        0-1, how confident the specialist is in its verdict.
    rationale:
        One-line human-readable explanation.
    """

    specialist: str
    verdict: str
    confidence: float
    rationale: str = ""
    produced_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "specialist": self.specialist, "verdict": self.verdict,
            "confidence": round(self.confidence, 4), "rationale": self.rationale,
            "produced_at": self.produced_at,
        }

    @property
    def has_opinion(self) -> bool:
        return self.verdict != "no_opinion"


class BaseSpecialist:
    """
    Base class for internal specialists. Subclasses implement
    ``consult(topic, **context) -> SpecialistOpinion``.

    ``name`` defaults to the class name but can be overridden, e.g. for
    multiple instances of the same specialist type specialized to
    different domains.
    """

    name: str = "specialist"

    def consult(self, topic: str, **context) -> SpecialistOpinion:
        raise NotImplementedError("Subclasses must implement consult().")
