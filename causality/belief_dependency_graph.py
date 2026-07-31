"""
Belief Dependency Graph — Blix v0.3.11  (New module 2, Phase 1)

Transforms ``core.truth_manager.TruthManager`` (v0.3.7 — tracks each
belief's truth status independently) into a genuine epistemic NETWORK:
beliefs can support or weaken each other, and a confidence change in
one belief propagates to everything that depends on it.

    Belief A  --supports-->  Belief B  --supports-->  Belief C

    A weakens
      ↓
    B confidence drops
      ↓
    C confidence drops

This is real, mechanical confidence propagation over a DAG — not
probabilistic graphical-model inference (no Bayesian network, no joint
distribution is being maintained). Each edge has a ``strength`` (how
much a change in the source belief's confidence should move the
target's), and ``propagate()`` walks the DAG breadth-first from a
changed belief, applying a damped confidence delta at each hop so the
effect doesn't compound unboundedly across long dependency chains.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from memory.beliefs import BeliefStore
from utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_DAMPING = 0.5   # each hop along the DAG has half the effect of the previous one
_MAX_PROPAGATION_HOPS = 5


class DependencyRelation(str, Enum):
    SUPPORTS = "supports"
    WEAKENS = "weakens"


@dataclass
class BeliefDependencyEdge:
    """One supports/weakens relation between two beliefs (by belief_id)."""

    source_belief_id: str
    target_belief_id: str
    relation: DependencyRelation
    strength: float = 0.5   # 0-1, how strongly a change in source should move target
    edge_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.edge_id:
            self.edge_id = f"{self.source_belief_id}::{self.relation.value}::{self.target_belief_id}"

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id, "source_belief_id": self.source_belief_id,
            "target_belief_id": self.target_belief_id, "relation": self.relation.value,
            "strength": round(self.strength, 4), "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BeliefDependencyEdge":
        return cls(
            source_belief_id=d["source_belief_id"], target_belief_id=d["target_belief_id"],
            relation=DependencyRelation(d["relation"]), strength=d.get("strength", 0.5),
            edge_id=d.get("edge_id", ""), created_at=d.get("created_at", ""),
        )


@dataclass
class PropagationResult:
    """One belief's confidence change as a result of a propagation pass."""

    belief_id: str
    old_confidence: float
    new_confidence: float
    hops_from_source: int

    def to_dict(self) -> dict:
        return {
            "belief_id": self.belief_id, "old_confidence": round(self.old_confidence, 4),
            "new_confidence": round(self.new_confidence, 4), "hops_from_source": self.hops_from_source,
        }


class BeliefDependencyGraph:
    """
    Stores supports/weakens edges between beliefs and propagates
    confidence changes through the dependency DAG.

    Parameters
    ----------
    dependency_graph_file:
        Path to ``belief_dependency_graph.json``.
    belief_store:
        ``BeliefStore`` — the beliefs whose confidence this graph reads
        and updates during propagation.
    damping:
        Multiplicative falloff per hop — keeps distant beliefs from
        being swung as hard as directly-connected ones.
    """

    def __init__(self, dependency_graph_file: Path, belief_store: BeliefStore, damping: float = _DEFAULT_DAMPING) -> None:
        self._file = dependency_graph_file
        self._belief_store = belief_store
        self._damping = damping
        self._edges: dict[str, BeliefDependencyEdge] = {}
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
                edge = BeliefDependencyEdge.from_dict(item)
                self._edges[edge.edge_id] = edge
        except Exception as exc:
            log.warning("BeliefDependencyGraph: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([e.to_dict() for e in self._edges.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_dependency(
        self, source_belief_id: str, target_belief_id: str, relation: DependencyRelation, strength: float = 0.5,
    ) -> BeliefDependencyEdge:
        """Declare that ``target`` depends on ``source`` (source supports/weakens target)."""
        edge = BeliefDependencyEdge(
            source_belief_id=source_belief_id, target_belief_id=target_belief_id,
            relation=relation, strength=max(0.0, min(1.0, strength)),
        )
        self._edges[edge.edge_id] = edge
        self._save()
        return edge

    def dependents_of(self, belief_id: str) -> list[BeliefDependencyEdge]:
        """Edges where ``belief_id`` is the source — i.e. what depends on this belief."""
        return [e for e in self._edges.values() if e.source_belief_id == belief_id]

    def dependencies_of(self, belief_id: str) -> list[BeliefDependencyEdge]:
        """Edges where ``belief_id`` is the target — i.e. what this belief depends on."""
        return [e for e in self._edges.values() if e.target_belief_id == belief_id]

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(self, changed_belief_id: str, confidence_delta: float) -> list[PropagationResult]:
        """
        Walk the dependency DAG breadth-first from ``changed_belief_id``,
        applying a damped confidence delta to every dependent belief.
        SUPPORTS edges move the dependent in the same direction as the
        delta; WEAKENS edges move it in the opposite direction.

        Returns the list of beliefs actually changed, in propagation order.
        """
        results: list[PropagationResult] = []
        visited: set[str] = {changed_belief_id}
        queue: deque[tuple[str, float, int]] = deque([(changed_belief_id, confidence_delta, 0)])

        while queue:
            current_id, delta, hops = queue.popleft()
            if hops >= _MAX_PROPAGATION_HOPS or abs(delta) < 0.01:
                continue

            for edge in self.dependents_of(current_id):
                if edge.target_belief_id in visited:
                    continue
                target = self._belief_store.get(edge.target_belief_id)
                if target is None:
                    continue

                direction = 1.0 if edge.relation == DependencyRelation.SUPPORTS else -1.0
                propagated_delta = delta * edge.strength * direction

                old_confidence = target.confidence
                new_confidence = max(0.0, min(1.0, old_confidence + propagated_delta))
                target.confidence = new_confidence
                target.updated_at = datetime.now(timezone.utc).isoformat()

                results.append(PropagationResult(
                    belief_id=target.belief_id, old_confidence=old_confidence,
                    new_confidence=new_confidence, hops_from_source=hops + 1,
                ))
                visited.add(edge.target_belief_id)
                queue.append((edge.target_belief_id, propagated_delta * self._damping, hops + 1))

        if results:
            self._belief_store.persist()
        return results

    @property
    def count(self) -> int:
        return len(self._edges)
