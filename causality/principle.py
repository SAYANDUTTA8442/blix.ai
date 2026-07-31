"""
Principle — Blix v0.3.11  (New module 5a, Phase 2)

Per explicit design constraint: principles are first-class objects,
not strings. A bare string like ``"Always benchmark before
optimization"`` carries no evidence trail, no confidence, and nothing
``PrincipleGraph``/``causality.causal_reflection.CausalReflection``/
``metacognition.strategy_evolution.StrategyEvolution`` could operate
on programmatically. ``Principle`` fixes that:

    @dataclass
    class Principle:
        id: str
        statement: str
        confidence: float
        evidence_count: int
        supporting_causes: list[str]      # CauseEdge.edge_id references
        supporting_failures: list[str]    # FailureCluster identifiers / sample failure text
        status: EpistemicStatus = PRINCIPLE

This module defines the object and its store only — synthesis (turning
raw experience into a new ``Principle``) is
``causality.principle_synthesizer.PrincipleSynthesizer``'s job; this
module is purely the typed representation and persistence layer,
matching the project's established split between "thing that computes
X" and "thing that stores X" (e.g. ``ConfidenceReasoner`` vs.
``ConfidenceManager``, v0.3.8).

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from causality.epistemic_status import EpistemicStatus
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Principle:
    """
    A synthesized, reusable generalization derived from repeated
    causal/failure patterns — a first-class object, not a string.
    """

    statement: str
    confidence: float = 0.5
    evidence_count: int = 1
    supporting_causes: list[str] = field(default_factory=list)      # CauseEdge.edge_id references
    supporting_failures: list[str] = field(default_factory=list)    # cluster ids / sample failure text
    status: EpistemicStatus = EpistemicStatus.PRINCIPLE
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id, "statement": self.statement, "confidence": round(self.confidence, 4),
            "evidence_count": self.evidence_count, "supporting_causes": self.supporting_causes,
            "supporting_failures": self.supporting_failures, "status": self.status.value,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Principle":
        return cls(
            statement=d["statement"], confidence=d.get("confidence", 0.5),
            evidence_count=d.get("evidence_count", 1),
            supporting_causes=d.get("supporting_causes", []),
            supporting_failures=d.get("supporting_failures", []),
            status=EpistemicStatus(d.get("status", EpistemicStatus.PRINCIPLE.value)),
            id=d.get("id", uuid.uuid4().hex[:10]), created_at=d.get("created_at", ""), updated_at=d.get("updated_at", ""),
        )


class PrincipleStore:
    """
    Persists ``Principle`` objects.

    Parameters
    ----------
    principle_file:
        Path to ``principles.json``.
    """

    def __init__(self, principle_file: Path) -> None:
        self._file = principle_file
        self._principles: dict[str, Principle] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for item in raw:
                p = Principle.from_dict(item)
                self._principles[p.id] = p
        except Exception as exc:
            log.warning("PrincipleStore: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([p.to_dict() for p in self._principles.values()], fh, indent=2, ensure_ascii=False)

    def persist(self) -> None:
        """Public save hook for external callers that mutate a Principle in place (e.g. PrincipleGraph propagation)."""
        self._save()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(self, principle: Principle) -> Principle:
        self._principles[principle.id] = principle
        self._save()
        return principle

    def reinforce(self, principle_id: str, confidence_increment: float = 0.05) -> Optional[Principle]:
        p = self._principles.get(principle_id)
        if p is None:
            return None
        p.evidence_count += 1
        p.confidence = min(1.0, p.confidence + confidence_increment)
        p.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return p

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, principle_id: str) -> Optional[Principle]:
        return self._principles.get(principle_id)

    def all_principles(self) -> list[Principle]:
        return list(self._principles.values())

    def high_confidence(self, threshold: float = 0.7) -> list[Principle]:
        return [p for p in self._principles.values() if p.confidence >= threshold]

    @property
    def count(self) -> int:
        return len(self._principles)
