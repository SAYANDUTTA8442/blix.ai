"""
CauseGraph — Blix v0.3.11  (New module 1, Phase 1)

Upgrades graph relations from generic entity-relation-entity
(``core.memory_graph.MemoryGraph``, v0.3.1) and temporal entity-relation-entity
(``graph.temporal_graph.TemporalGraph``, v0.3.7) to a TYPED cause-effect
relation, explicitly distinguishing how a trigger affects an outcome:

    @dataclass
    class CauseEdge:
        trigger: str
        effect: str
        relation: CauseRelation
        confidence: float
        evidence_count: int

    CauseRelation: CAUSES | INCREASES | DECREASES | ENABLES | BLOCKS

Example:
    "no evaluation"  BLOCKS    "reliable optimization"
    "benchmarks"     ENABLES   "fast iteration"
    "web failures"   CAUSES    "low confidence"

== Honest scope ==
This is real, evidence-counted graph structure — NOT validated causal
inference. Edges are derived from observed co-occurrence in Blix's own
runtime data (``agents.execution_feedback.ExecutionFeedbackLoop``,
``agents.failure_memory.FailureMemory``), every edge carries an
``EpistemicStatus.DERIVED`` tag and an honest ``evidence_count``, and
the module does NOT implement do-calculus, structural causal models,
confounder adjustment, or any method that would license a genuine
causal (vs. correlational) claim. ``confidence`` here measures
"how often did we see trigger and effect co-occur", not "P(effect |
do(trigger))" in Pearl's sense. The CAUSES/INCREASES/DECREASES/ENABLES/BLOCKS
vocabulary is a structured LABEL a caller (or, later, a human) assigns
to a correlational observation — it is not derived through any
causal-discovery algorithm in this module.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from causality.epistemic_status import EpistemicStatus
from utils.logger import get_logger

log = get_logger(__name__)


class CauseRelation(str, Enum):
    CAUSES = "causes"
    INCREASES = "increases"
    DECREASES = "decreases"
    ENABLES = "enables"
    BLOCKS = "blocks"


@dataclass
class CauseEdge:
    """One typed cause -> effect relation, with evidence backing it."""

    trigger: str
    effect: str
    relation: CauseRelation
    confidence: float = 0.5
    evidence_count: int = 1
    epistemic_status: EpistemicStatus = EpistemicStatus.DERIVED
    edge_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.edge_id:
            self.edge_id = f"{self.trigger}::{self.relation.value}::{self.effect}"

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id, "trigger": self.trigger, "effect": self.effect,
            "relation": self.relation.value, "confidence": round(self.confidence, 4),
            "evidence_count": self.evidence_count, "epistemic_status": self.epistemic_status.value,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CauseEdge":
        return cls(
            trigger=d["trigger"], effect=d["effect"], relation=CauseRelation(d["relation"]),
            confidence=d.get("confidence", 0.5), evidence_count=d.get("evidence_count", 1),
            epistemic_status=EpistemicStatus(d.get("epistemic_status", EpistemicStatus.DERIVED.value)),
            edge_id=d.get("edge_id", ""), created_at=d.get("created_at", ""), updated_at=d.get("updated_at", ""),
        )


class CauseGraph:
    """
    Stores and queries typed cause-effect edges, with confidence that
    grows/shrinks as corroborating/conflicting evidence accumulates.

    Parameters
    ----------
    cause_graph_file:
        Path to ``cause_graph.json``.
    confidence_increment:
        Confidence boost per additional corroborating observation.
    """

    def __init__(self, cause_graph_file: Path, confidence_increment: float = 0.08) -> None:
        self._file = cause_graph_file
        self._increment = confidence_increment
        self._edges: dict[str, CauseEdge] = {}
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
                edge = CauseEdge.from_dict(item)
                self._edges[edge.edge_id] = edge
            log.info("CauseGraph: loaded %d edge(s).", len(self._edges))
        except Exception as exc:
            log.warning("CauseGraph: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([e.to_dict() for e in self._edges.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record_observation(
        self, trigger: str, effect: str, relation: CauseRelation, initial_confidence: float = 0.5,
    ) -> CauseEdge:
        """
        Record one observed co-occurrence of (trigger, effect) under
        ``relation``. Reinforces an existing matching edge's confidence
        and evidence_count, or creates a new edge.
        """
        edge_id = f"{trigger}::{relation.value}::{effect}"
        existing = self._edges.get(edge_id)
        if existing:
            existing.evidence_count += 1
            existing.confidence = min(1.0, existing.confidence + self._increment)
            existing.updated_at = datetime.now(timezone.utc).isoformat()
            self._save()
            return existing

        edge = CauseEdge(trigger=trigger, effect=effect, relation=relation, confidence=initial_confidence, evidence_count=1)
        self._edges[edge.edge_id] = edge
        self._save()
        return edge

    def weaken(self, edge_id: str, amount: float = 0.15) -> Optional[CauseEdge]:
        """Reduce an edge's confidence (e.g. a contradicting observation)."""
        edge = self._edges.get(edge_id)
        if edge is None:
            return None
        edge.confidence = max(0.0, edge.confidence - amount)
        edge.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return edge

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, edge_id: str) -> Optional[CauseEdge]:
        return self._edges.get(edge_id)

    def effects_of(self, trigger: str, relation: Optional[CauseRelation] = None) -> list[CauseEdge]:
        """All edges where ``trigger`` is the cause, optionally filtered to one relation type."""
        edges = [e for e in self._edges.values() if e.trigger == trigger]
        if relation is not None:
            edges = [e for e in edges if e.relation == relation]
        return sorted(edges, key=lambda e: -e.confidence)

    def causes_of(self, effect: str, relation: Optional[CauseRelation] = None) -> list[CauseEdge]:
        """All edges where ``effect`` is the outcome, optionally filtered to one relation type."""
        edges = [e for e in self._edges.values() if e.effect == effect]
        if relation is not None:
            edges = [e for e in edges if e.relation == relation]
        return sorted(edges, key=lambda e: -e.confidence)

    def all_edges(self) -> list[CauseEdge]:
        return list(self._edges.values())

    def high_confidence_edges(self, threshold: float = 0.7) -> list[CauseEdge]:
        return [e for e in self._edges.values() if e.confidence >= threshold]

    @property
    def count(self) -> int:
        return len(self._edges)
