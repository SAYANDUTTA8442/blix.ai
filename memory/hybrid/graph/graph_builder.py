"""
GraphBuilder — high-level factory for populating the HGSHM knowledge graph.

Subsystems call GraphBuilder instead of GraphStore directly when they want
to express cognitive events as graph mutations. This keeps the "what happened"
logic central and consistent.
"""
from __future__ import annotations
import logging
from typing import Any

from memory.hybrid.models.memory_node import (
    MemoryNode, MemoryType, HierarchyLevel, EpistemicStatus
)
from memory.hybrid.models.memory_edge import MemoryEdge, EdgeRelation
from memory.hybrid.graph.graph_store import GraphStore

log = logging.getLogger(__name__)


class GraphBuilder:
    """
    Convenience factory for creating HGSHM graph structures.

    All cognitive events (belief formation, causal observation, hypothesis
    creation, experiment outcome, etc.) should flow through here so the
    graph stays coherent and relationship semantics are enforced consistently.
    """

    def __init__(self, graph_store: GraphStore) -> None:
        self._graph = graph_store

    # ----------------------------------------------------------------
    # Belief operations
    # ----------------------------------------------------------------

    def add_belief(
        self,
        statement: str,
        confidence: float = 0.7,
        source: str = "system",
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> MemoryNode:
        """Add or reinforce a belief. Returns the belief node."""
        # Check for text-overlap duplicates
        existing = self._find_near_duplicate(statement, MemoryType.BELIEF)
        if existing:
            existing.update_confidence(0.05)  # reinforce
            self._graph.update_node(existing)
            log.debug("GraphBuilder: belief reinforced %s", existing.node_id[:8])
            return existing

        node = self._graph.make_node(
            text=statement,
            memory_type=MemoryType.BELIEF,
            hierarchy_level=HierarchyLevel.EPISODE,
            confidence=confidence,
            importance=0.5,
            source=source,
            epistemic_status=EpistemicStatus.OBSERVED,
            metadata=metadata or {},
            tags=tags or [],
        )
        log.debug("GraphBuilder: belief created %s", node.node_id[:8])
        return node

    def add_hypothesis(
        self,
        statement: str,
        confidence: float = 0.3,
        source: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        """Add a hypothesis node (EpistemicStatus.HYPOTHESIS)."""
        node = self._graph.make_node(
            text=statement,
            memory_type=MemoryType.HYPOTHESIS,
            hierarchy_level=HierarchyLevel.EPISODE,
            confidence=confidence,
            importance=0.4,
            source=source,
            epistemic_status=EpistemicStatus.HYPOTHESIS,
            metadata=metadata or {},
        )
        return node

    def promote_hypothesis_to_belief(
        self,
        hypothesis_node_id: str,
    ) -> MemoryNode | None:
        """Promote a HYPOTHESIS node to a BELIEF node (OBSERVED status)."""
        node = self._graph.get_node(hypothesis_node_id)
        if node is None:
            return None
        node.memory_type = MemoryType.BELIEF
        node.epistemic_status = EpistemicStatus.OBSERVED
        node.hierarchy_level = HierarchyLevel.EPISODE
        self._graph.update_node(node)
        return node

    # ----------------------------------------------------------------
    # Causal observations
    # ----------------------------------------------------------------

    def add_causal_observation(
        self,
        trigger: str,
        effect: str,
        relation: EdgeRelation = EdgeRelation.CAUSES,
        confidence: float = 0.6,
        source: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[MemoryNode, MemoryNode, MemoryEdge]:
        """
        Record a causal observation: trigger --[relation]--> effect.
        Returns (trigger_node, effect_node, edge).
        """
        trigger_node = self._get_or_create_node(trigger, MemoryType.CAUSE, confidence, source)
        effect_node  = self._get_or_create_node(effect,  MemoryType.CAUSE, confidence, source)
        edge = self._graph.make_edge(
            source_id=trigger_node.node_id,
            target_id=effect_node.node_id,
            relation=relation,
            confidence=confidence,
            weight=confidence,
            provenance=source,
            metadata=metadata or {},
        )
        return trigger_node, effect_node, edge

    # ----------------------------------------------------------------
    # Principle operations
    # ----------------------------------------------------------------

    def add_principle(
        self,
        statement: str,
        confidence: float = 0.8,
        source: str = "system",
        derived_from_ids: list[str] | None = None,
    ) -> MemoryNode:
        """Add a principle node and optionally link it to source nodes."""
        node = self._graph.make_node(
            text=statement,
            memory_type=MemoryType.PRINCIPLE,
            hierarchy_level=HierarchyLevel.PRINCIPLE,
            confidence=confidence,
            importance=0.8,
            source=source,
            epistemic_status=EpistemicStatus.PRINCIPLE,
        )
        for src_id in (derived_from_ids or []):
            self._graph.make_edge(
                source_id=node.node_id,
                target_id=src_id,
                relation=EdgeRelation.DERIVED_FROM,
                confidence=confidence,
                provenance=source,
            )
        return node

    def link_principles(
        self,
        source_id: str,
        target_id: str,
        strength: float = 0.8,
    ) -> MemoryEdge:
        """Add a SUPPORTS edge between two principle nodes."""
        return self._graph.make_edge(
            source_id=source_id,
            target_id=target_id,
            relation=EdgeRelation.SUPPORTS,
            confidence=strength,
            weight=strength,
            provenance="principle_graph",
        )

    # ----------------------------------------------------------------
    # Concept / cluster operations
    # ----------------------------------------------------------------

    def add_concept(
        self,
        name: str,
        description: str = "",
        member_node_ids: list[str] | None = None,
        source: str = "system",
    ) -> MemoryNode:
        """Create a concept node and link members to it via BELONGS_TO."""
        concept = self._graph.make_node(
            text=name if not description else f"{name}: {description}",
            memory_type=MemoryType.CONCEPT,
            hierarchy_level=HierarchyLevel.CONCEPT,
            confidence=0.9,
            importance=0.9,
            source=source,
            epistemic_status=EpistemicStatus.DERIVED,
        )
        for mid in (member_node_ids or []):
            self._graph.make_edge(
                source_id=mid,
                target_id=concept.node_id,
                relation=EdgeRelation.BELONGS_TO,
                confidence=0.8,
                provenance=source,
            )
        return concept

    # ----------------------------------------------------------------
    # Knowledge gap operations
    # ----------------------------------------------------------------

    def add_gap(
        self,
        domain: str,
        uncertainty: float = 0.8,
        source: str = "curiosity_engine",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        """Record an identified knowledge gap."""
        return self._graph.make_node(
            text=f"Knowledge gap: {domain}",
            memory_type=MemoryType.GAP,
            hierarchy_level=HierarchyLevel.SESSION,
            confidence=1.0 - uncertainty,  # high uncertainty → low confidence
            importance=uncertainty,          # high uncertainty → high importance
            source=source,
            epistemic_status=EpistemicStatus.DERIVED,
            metadata={"domain": domain, "uncertainty": uncertainty, **(metadata or {})},
            tags=["gap", domain],
        )

    # ----------------------------------------------------------------
    # Summary / hierarchy
    # ----------------------------------------------------------------

    def add_summary(
        self,
        text: str,
        summarises_node_ids: list[str],
        hierarchy_level: HierarchyLevel = HierarchyLevel.DAILY,
        source: str = "hierarchy_manager",
    ) -> MemoryNode:
        """Add a summary node and link it to its source nodes."""
        summary = self._graph.make_node(
            text=text,
            memory_type=MemoryType.SUMMARY,
            hierarchy_level=hierarchy_level,
            confidence=0.85,
            importance=0.7,
            source=source,
            epistemic_status=EpistemicStatus.DERIVED,
        )
        for nid in summarises_node_ids:
            self._graph.make_edge(
                source_id=summary.node_id,
                target_id=nid,
                relation=EdgeRelation.SUMMARISES,
                confidence=0.9,
                provenance=source,
            )
        return summary

    # ----------------------------------------------------------------
    # Generic relationship
    # ----------------------------------------------------------------

    def link(
        self,
        source_id: str,
        target_id: str,
        relation: EdgeRelation,
        confidence: float = 0.7,
        provenance: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEdge:
        """Generic edge creation between two existing nodes."""
        return self._graph.make_edge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            confidence=confidence,
            weight=confidence * 0.8,
            provenance=provenance,
            metadata=metadata or {},
        )

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _get_or_create_node(
        self,
        text: str,
        memory_type: MemoryType,
        confidence: float,
        source: str,
    ) -> MemoryNode:
        existing = self._find_near_duplicate(text, memory_type)
        if existing:
            return existing
        return self._graph.make_node(
            text=text,
            memory_type=memory_type,
            confidence=confidence,
            source=source,
        )

    def _find_near_duplicate(
        self, text: str, memory_type: MemoryType,
    ) -> MemoryNode | None:
        """Very fast token-overlap duplicate check (exact match only)."""
        candidates = self._graph.search_by_text(text, limit=5)
        for node in candidates:
            if node.memory_type == memory_type and node.text.strip().lower() == text.strip().lower():
                return node
        return None
