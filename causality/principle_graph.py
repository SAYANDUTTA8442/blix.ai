"""
Principle Graph — Blix v0.3.11  (New module 5b, Phase 2)

A supports-DAG over ``causality.principle.Principle`` objects, matching
the spec's example:

    Always evaluate before optimizing
            ↓ supports
    Reliable optimization
            ↓ supports
    Faster iteration

Structurally near-identical to
``causality.belief_dependency_graph.BeliefDependencyGraph`` (same
breadth-first damped propagation), deliberately — principles and
beliefs are both confidence-bearing objects that can reinforce or
undermine each other, and reusing the same mechanism keeps the
project's two confidence-propagation graphs behaviorally consistent
rather than inventing a second, subtly different algorithm.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from causality.principle import PrincipleStore
from utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_DAMPING = 0.5
_MAX_PROPAGATION_HOPS = 5


@dataclass
class PrincipleSupportEdge:
    """One 'supports' relation between two principles (by Principle.id)."""

    source_principle_id: str
    target_principle_id: str
    strength: float = 0.5
    edge_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.edge_id:
            self.edge_id = f"{self.source_principle_id}::supports::{self.target_principle_id}"

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id, "source_principle_id": self.source_principle_id,
            "target_principle_id": self.target_principle_id, "strength": round(self.strength, 4),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PrincipleSupportEdge":
        return cls(
            source_principle_id=d["source_principle_id"], target_principle_id=d["target_principle_id"],
            strength=d.get("strength", 0.5), edge_id=d.get("edge_id", ""), created_at=d.get("created_at", ""),
        )


@dataclass
class PrinciplePropagationResult:
    principle_id: str
    old_confidence: float
    new_confidence: float
    hops_from_source: int

    def to_dict(self) -> dict:
        return {
            "principle_id": self.principle_id, "old_confidence": round(self.old_confidence, 4),
            "new_confidence": round(self.new_confidence, 4), "hops_from_source": self.hops_from_source,
        }


class PrincipleGraph:
    """
    Stores supports edges between principles and propagates confidence
    changes through the dependency DAG.

    Parameters
    ----------
    principle_graph_file:
        Path to ``principle_graph.json``.
    principle_store:
        ``PrincipleStore`` — the principles whose confidence this graph
        reads and updates during propagation.
    damping:
        Multiplicative falloff per hop.
    """

    def __init__(self, principle_graph_file: Path, principle_store: PrincipleStore, damping: float = _DEFAULT_DAMPING) -> None:
        self._file = principle_graph_file
        self._principle_store = principle_store
        self._damping = damping
        self._edges: dict[str, PrincipleSupportEdge] = {}
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
                edge = PrincipleSupportEdge.from_dict(item)
                self._edges[edge.edge_id] = edge
        except Exception as exc:
            log.warning("PrincipleGraph: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([e.to_dict() for e in self._edges.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_support(self, source_principle_id: str, target_principle_id: str, strength: float = 0.5) -> PrincipleSupportEdge:
        edge = PrincipleSupportEdge(
            source_principle_id=source_principle_id, target_principle_id=target_principle_id,
            strength=max(0.0, min(1.0, strength)),
        )
        self._edges[edge.edge_id] = edge
        self._save()
        return edge

    def supported_by(self, principle_id: str) -> list[PrincipleSupportEdge]:
        """Edges where ``principle_id`` is the source — what this principle supports."""
        return [e for e in self._edges.values() if e.source_principle_id == principle_id]

    def supports_of(self, principle_id: str) -> list[PrincipleSupportEdge]:
        """Edges where ``principle_id`` is the target — what supports this principle."""
        return [e for e in self._edges.values() if e.target_principle_id == principle_id]

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(self, changed_principle_id: str, confidence_delta: float) -> list[PrinciplePropagationResult]:
        """Walk the supports-DAG breadth-first, applying a damped confidence delta at each hop."""
        results: list[PrinciplePropagationResult] = []
        visited: set[str] = {changed_principle_id}
        queue: deque[tuple[str, float, int]] = deque([(changed_principle_id, confidence_delta, 0)])

        while queue:
            current_id, delta, hops = queue.popleft()
            if hops >= _MAX_PROPAGATION_HOPS or abs(delta) < 0.01:
                continue

            for edge in self.supported_by(current_id):
                if edge.target_principle_id in visited:
                    continue
                target = self._principle_store.get(edge.target_principle_id)
                if target is None:
                    continue

                propagated_delta = delta * edge.strength
                old_confidence = target.confidence
                new_confidence = max(0.0, min(1.0, old_confidence + propagated_delta))
                target.confidence = new_confidence
                target.updated_at = datetime.now(timezone.utc).isoformat()

                results.append(PrinciplePropagationResult(
                    principle_id=target.id, old_confidence=old_confidence,
                    new_confidence=new_confidence, hops_from_source=hops + 1,
                ))
                visited.add(edge.target_principle_id)
                queue.append((edge.target_principle_id, propagated_delta * self._damping, hops + 1))

        if results:
            self._principle_store.persist()
        return results

    @property
    def count(self) -> int:
        return len(self._edges)
