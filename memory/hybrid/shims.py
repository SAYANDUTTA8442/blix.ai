"""
Blix v0.3.15 Compatibility Shims.

Wraps old v0.3.14 APIs (BeliefStore, CauseGraph, PrincipleStore, etc.)
so existing code continues to work unchanged while HGSHM runs underneath.

Each shim:
  1. Accepts the same __init__ signature as the original class
  2. Translates method calls to HGSHM operations
  3. Returns the same types as the original (for test compatibility)
  4. Logs a DEBUG message on first use (no user-visible deprecation spam)

Architecture
------------
All shims share a single HGSHM instance per memory_dir. This means
beliefs added via BeliefStoreShim are visible to CauseGraphShim and
PrincipleStoreShim — the graph is unified.
"""
from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory.hybrid.hgshm import HGSHM
from memory.hybrid.models.memory_node import MemoryType, HierarchyLevel, EpistemicStatus
from memory.hybrid.models.memory_edge import EdgeRelation

log = logging.getLogger(__name__)

# ── Singleton registry so all shims share one HGSHM per memory_dir ──
_HGSHM_REGISTRY: dict[str, HGSHM] = {}

def _get_hgshm(memory_dir: Path) -> HGSHM:
    key = str(memory_dir.resolve())
    if key not in _HGSHM_REGISTRY:
        _HGSHM_REGISTRY[key] = HGSHM(memory_dir)
    return _HGSHM_REGISTRY[key]


# ────────────────────────────────────────────────────────────────────
# Belief types (mirroring the originals for test compatibility)
# ────────────────────────────────────────────────────────────────────

@dataclass
class BeliefRecord:
    """Drop-in replacement for the old Belief dataclass."""
    belief_id:       str
    text:            str
    confidence:      float
    epistemic_status: Any   # EpistemicStatus from causality.epistemic_status
    status:          Any = None  # TruthStatus
    source:          str = "system"
    metadata:        dict = field(default_factory=dict)
    created_at:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Compatibility aliases
    @property
    def statement(self) -> str:
        return self.text

    def to_dict(self) -> dict:
        return {
            "belief_id": self.belief_id,
            "text": self.text,
            "confidence": self.confidence,
            "epistemic_status": getattr(self.epistemic_status, "value", str(self.epistemic_status)),
            "status": getattr(self.status, "value", str(self.status)) if self.status else None,
            "source": self.source,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class BeliefStoreShim:
    """
    Drop-in shim for memory.beliefs.BeliefStore.

    All beliefs are stored as BELIEF-type MemoryNodes in HGSHM.
    """

    def __init__(self, db_path: Path, **kwargs) -> None:
        memory_dir = db_path.parent if db_path.suffix else db_path
        self._hgshm = _get_hgshm(memory_dir)
        self._node_to_belief_id: dict[str, str] = {}  # node_id → belief_id
        self._belief_to_node_id: dict[str, str] = {}  # belief_id → node_id
        log.debug("BeliefStoreShim: using HGSHM backend at %s", memory_dir)

    def add_or_reinforce(self, statement: str, confidence: float = 0.7,
                         source: str = "system", **kwargs) -> BeliefRecord:
        node = self._hgshm.believe(statement, confidence=confidence, source=source)
        belief_id = self._node_to_belief_id.get(node.node_id)
        if not belief_id:
            belief_id = node.node_id  # use node_id as belief_id
            self._node_to_belief_id[node.node_id] = belief_id
            self._belief_to_node_id[belief_id] = node.node_id
        return self._node_to_record(node, belief_id)

    def add_hypothesis(self, statement: str, confidence: float = 0.3,
                       **kwargs) -> BeliefRecord:
        node = self._hgshm.hypothesise(statement, confidence=confidence)
        belief_id = node.node_id
        self._node_to_belief_id[node.node_id] = belief_id
        self._belief_to_node_id[belief_id] = node.node_id
        return self._node_to_record(node, belief_id)

    def get(self, belief_id: str) -> BeliefRecord | None:
        node_id = self._belief_to_node_id.get(belief_id, belief_id)
        node = self._hgshm.get_node(node_id)
        if node is None:
            return None
        return self._node_to_record(node, belief_id)

    def all_active(self) -> list[BeliefRecord]:
        nodes = self._hgshm.all_nodes(memory_type=MemoryType.BELIEF, limit=1000)
        return [self._node_to_record(n, n.node_id) for n in nodes]

    def set_status(self, belief_id: str, status: Any) -> None:
        node_id = self._belief_to_node_id.get(belief_id, belief_id)
        node = self._hgshm.get_node(node_id)
        if node:
            node.metadata["truth_status"] = getattr(status, "value", str(status))
            self._hgshm.update_node(node)

    def find_conflicting_candidates(self, statement: str,
                                     min_overlap: float = 0.3) -> list[BeliefRecord]:
        """Find beliefs that may contradict the given statement."""
        nodes = self._hgshm.recall_nodes(statement, top_k=20, memory_types=[MemoryType.BELIEF])
        candidates = []
        q_tokens = set(statement.lower().split())
        for node in nodes:
            t_tokens = set(node.text.lower().split())
            shared = q_tokens & t_tokens
            all_t  = q_tokens | t_tokens
            if not all_t:
                continue
            jaccard = len(shared) / len(all_t)
            if min_overlap <= jaccard < 0.5:  # overlap but not identical
                candidates.append(self._node_to_record(node, node.node_id))
        return candidates

    def _node_to_record(self, node: Any, belief_id: str) -> BeliefRecord:
        # Map HGSHM EpistemicStatus to old causality.epistemic_status
        try:
            from causality.epistemic_status import EpistemicStatus as OldEpistemicStatus
            old_status = OldEpistemicStatus(node.epistemic_status.value)
        except Exception:
            old_status = node.epistemic_status
        return BeliefRecord(
            belief_id=belief_id,
            text=node.text,
            confidence=node.confidence,
            epistemic_status=old_status,
            source=node.source,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
        )

    @property
    def count(self) -> int:
        return self._hgshm.graph_store.count_nodes(MemoryType.BELIEF)


# ────────────────────────────────────────────────────────────────────
# CauseGraph shim
# ────────────────────────────────────────────────────────────────────

@dataclass
class CauseEdgeRecord:
    """Drop-in replacement for the old CauseEdge dataclass."""
    edge_id:       str
    trigger:       str
    effect:        str
    relation:      Any   # CauseRelation
    confidence:    float
    evidence_count: int = 1
    source:        str  = "system"
    metadata:      dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id, "trigger": self.trigger, "effect": self.effect,
            "relation": getattr(self.relation, "value", str(self.relation)),
            "confidence": self.confidence, "evidence_count": self.evidence_count,
        }


class CauseGraphShim:
    """
    Drop-in shim for causality.cause_graph.CauseGraph.

    Causal observations are stored as CAUSE-type MemoryNodes with typed edges.
    """

    _RELATION_MAP: dict = {}

    def __init__(self, db_path: Path, **kwargs) -> None:
        memory_dir = db_path.parent if db_path.suffix else db_path
        self._hgshm = _get_hgshm(memory_dir)
        self._edge_records: dict[str, CauseEdgeRecord] = {}
        log.debug("CauseGraphShim: using HGSHM backend at %s", memory_dir)

    def record_observation(
        self, trigger: str, effect: str, relation: Any,
        initial_confidence: float = 0.6, **kwargs
    ) -> CauseEdgeRecord:
        # Map old CauseRelation to EdgeRelation
        rel_str = getattr(relation, "value", str(relation)).upper()
        edge_rel = {
            "CAUSES": EdgeRelation.CAUSES,
            "BLOCKS": EdgeRelation.BLOCKS,
            "ENABLES": EdgeRelation.ENABLES,
            "INCREASES": EdgeRelation.ENABLES,
        }.get(rel_str, EdgeRelation.CAUSES)

        t_node, e_node, edge = self._hgshm.observe_cause(
            trigger, effect, relation=edge_rel, confidence=initial_confidence)

        # Check if this is a reinforcement
        existing = next((r for r in self._edge_records.values()
                         if r.trigger == trigger and r.effect == effect and
                         getattr(r.relation, "value", str(r.relation)).upper() == rel_str), None)
        if existing:
            existing.confidence = min(1.0, existing.confidence + 0.05)
            existing.evidence_count += 1
            return existing

        record = CauseEdgeRecord(
            edge_id=edge.edge_id, trigger=trigger, effect=effect,
            relation=relation, confidence=initial_confidence, evidence_count=1)
        self._edge_records[edge.edge_id] = record
        return record

    def get(self, edge_id: str) -> CauseEdgeRecord | None:
        return self._edge_records.get(edge_id)

    def what_causes(self, effect: str) -> Any:
        """Return a simple answer object about causes of `effect`."""
        nodes = self._hgshm.recall_nodes(effect, top_k=10, memory_types=[MemoryType.CAUSE])
        causes = [n.text for n in nodes if effect.lower() not in n.text.lower()]
        return type("CauseAnswer", (), {
            "answer_summary": ", ".join(causes[:3]) if causes else "no known causes",
            "question": f"What causes {effect}?",
        })()

    @property
    def count(self) -> int:
        return len(self._edge_records)


# ────────────────────────────────────────────────────────────────────
# PrincipleStore shim
# ────────────────────────────────────────────────────────────────────

@dataclass
class PrincipleRecord:
    """Drop-in replacement for old Principle dataclass."""
    id:          str
    statement:   str
    confidence:  float = 0.8
    source:      str   = "system"
    metadata:    dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "statement": self.statement,
                "confidence": self.confidence, "source": self.source}


class PrincipleStoreShim:
    """Drop-in shim for causality.principle.PrincipleStore."""

    def __init__(self, db_path: Path, **kwargs) -> None:
        memory_dir = db_path.parent if db_path.suffix else db_path
        self._hgshm = _get_hgshm(memory_dir)
        self._records: dict[str, PrincipleRecord] = {}
        log.debug("PrincipleStoreShim: using HGSHM backend at %s", memory_dir)

    def add(self, principle: Any) -> PrincipleRecord:
        """Accept old Principle dataclass or PrincipleRecord."""
        statement  = getattr(principle, "statement", str(principle))
        confidence = getattr(principle, "confidence", 0.8)
        node = self._hgshm.add_principle(statement, confidence=confidence)
        record = PrincipleRecord(id=node.node_id, statement=statement,
                                  confidence=confidence)
        self._records[node.node_id] = record
        return record

    def get(self, principle_id: str) -> PrincipleRecord | None:
        if principle_id in self._records:
            return self._records[principle_id]
        node = self._hgshm.get_node(principle_id)
        if node and node.memory_type == MemoryType.PRINCIPLE:
            record = PrincipleRecord(id=node.node_id, statement=node.text,
                                      confidence=node.confidence)
            self._records[node.node_id] = record
            return record
        return None

    def all(self) -> list[PrincipleRecord]:
        nodes = self._hgshm.all_nodes(memory_type=MemoryType.PRINCIPLE, limit=1000)
        return [PrincipleRecord(id=n.node_id, statement=n.text, confidence=n.confidence)
                for n in nodes]

    @property
    def count(self) -> int:
        return self._hgshm.graph_store.count_nodes(MemoryType.PRINCIPLE)
