"""
GraphStore — high-level graph CRUD built on top of HGSHMStore.

Handles:
  • MemoryNode and MemoryEdge persistence
  • In-memory adjacency index (source→edges, target→edges) for O(1) lookup
  • Node deduplication by text hash
  • Edge reinforcement (same pair + relation → update rather than duplicate)
  • History snapshots on node update
"""
from __future__ import annotations
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from memory.hybrid.models.memory_node import MemoryNode, MemoryType, HierarchyLevel, EpistemicStatus
from memory.hybrid.models.memory_edge import MemoryEdge, EdgeRelation
from memory.hybrid.storage.persistence import HGSHMStore

log = logging.getLogger(__name__)


class GraphStore:
    """
    Public interface for reading and writing nodes and edges in HGSHM.

    Parameters
    ----------
    memory_dir:
        Directory where hgshm.db is stored.
    """

    def __init__(self, memory_dir: Path) -> None:
        self._store = HGSHMStore(memory_dir)
        # In-memory adjacency index rebuilt on first access
        self._out_edges: dict[str, list[str]] = defaultdict(list)  # node_id → edge_ids
        self._in_edges:  dict[str, list[str]] = defaultdict(list)  # node_id → edge_ids
        self._edge_cache: dict[str, MemoryEdge] = {}
        self._node_cache: dict[str, MemoryNode] = {}
        self._index_loaded = False
        log.debug("GraphStore ready at %s", memory_dir)

    # ----------------------------------------------------------------
    # Index bootstrap
    # ----------------------------------------------------------------

    def _ensure_index(self) -> None:
        if self._index_loaded:
            return
        for ed in self._store.all_edges(limit=100_000):
            edge = MemoryEdge.from_dict(ed)
            self._edge_cache[edge.edge_id] = edge
            self._out_edges[edge.source_id].append(edge.edge_id)
            self._in_edges[edge.target_id].append(edge.edge_id)
        self._index_loaded = True
        log.debug("GraphStore: adjacency index loaded (%d edges)", len(self._edge_cache))

    def _invalidate_index(self) -> None:
        self._index_loaded = False
        self._out_edges.clear()
        self._in_edges.clear()
        self._edge_cache.clear()

    # ----------------------------------------------------------------
    # Node operations
    # ----------------------------------------------------------------

    def add_node(self, node: MemoryNode) -> MemoryNode:
        """Persist a new node. Returns the node (possibly merged with existing)."""
        self._store.save_node(node)
        self._node_cache[node.node_id] = node
        log.debug("GraphStore: node saved %s", node.node_id[:8])
        return node

    def get_node(self, node_id: str) -> MemoryNode | None:
        if node_id in self._node_cache:
            return self._node_cache[node_id]
        d = self._store.get_node(node_id)
        if d is None:
            return None
        node = MemoryNode.from_dict(d)
        self._node_cache[node_id] = node
        return node

    def update_node(self, node: MemoryNode, save_history: bool = True) -> None:
        """Update a node and optionally snapshot its previous state."""
        if save_history:
            self._store.save_history(node.node_id, node.version, node.to_dict())
        node.version += 1
        self._store.save_node(node)
        self._node_cache[node.node_id] = node

    def delete_node(self, node_id: str) -> bool:
        ok = self._store.delete_node(node_id)
        self._node_cache.pop(node_id, None)
        # Also delete dangling edges
        self._ensure_index()
        for eid in list(self._out_edges.get(node_id, [])):
            self._store.delete_edge(eid)
            self._edge_cache.pop(eid, None)
        for eid in list(self._in_edges.get(node_id, [])):
            self._store.delete_edge(eid)
            self._edge_cache.pop(eid, None)
        self._out_edges.pop(node_id, None)
        self._in_edges.pop(node_id, None)
        return ok

    def all_nodes(
        self,
        memory_type: MemoryType | None = None,
        hierarchy_level: HierarchyLevel | None = None,
        min_confidence: float = 0.0,
        min_importance: float = 0.0,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[MemoryNode]:
        dicts = self._store.all_nodes(
            memory_type=memory_type.value if memory_type else None,
            hierarchy_level=hierarchy_level.value if hierarchy_level is not None else None,
            min_confidence=min_confidence,
            min_importance=min_importance,
            limit=limit,
            offset=offset,
        )
        return [MemoryNode.from_dict(d) for d in dicts]

    def search_by_text(self, query: str, limit: int = 50) -> list[MemoryNode]:
        tokens = query.lower().split()
        dicts = self._store.search_nodes_by_text(tokens, limit=limit)
        return [MemoryNode.from_dict(d) for d in dicts]

    def count_nodes(self, memory_type: MemoryType | None = None) -> int:
        return self._store.count_nodes(
            memory_type.value if memory_type else None)

    def nodes_by_tags(
        self,
        required_tags: list[str],
        memory_type: MemoryType | None = None,
        limit: int = 500,
        order_by: str = "importance",
    ) -> list[MemoryNode]:
        """
        Return nodes containing ALL required_tags via json_each SQL query.
        O(k) DB round-trip, not O(n) full table scan.  (ISSUE-008)
        """
        dicts = self._store.nodes_by_tags(
            required_tags=required_tags,
            memory_type=memory_type.value if memory_type else None,
            limit=limit,
            order_by=order_by,
        )
        return [MemoryNode.from_dict(d) for d in dicts]

    def count_by_tag(self, tag: str,
                     memory_type: MemoryType | None = None) -> int:
        """Count nodes containing tag — O(index), not O(n).  (ISSUE-008)"""
        return self._store.count_by_tag(
            tag, memory_type.value if memory_type else None)

    def stats_by_tag(self, tag: str) -> dict[str, int]:
        """Return {memory_type: count} for nodes with tag.  (ISSUE-008)"""
        return self._store.stats_by_tag(tag)

    # ----------------------------------------------------------------
    # Edge operations
    # ----------------------------------------------------------------

    def add_edge(self, edge: MemoryEdge, reinforce_if_exists: bool = True) -> MemoryEdge:
        """
        Add an edge. If an edge with the same (source, target, relation)
        already exists and reinforce_if_exists=True, strengthen it instead.
        """
        self._ensure_index()
        existing_d = self._store.find_edge(
            edge.source_id, edge.target_id, edge.relation.value)
        if existing_d and reinforce_if_exists:
            existing = MemoryEdge.from_dict(existing_d)
            existing.reinforce(confidence_delta=0.05)
            self._store.save_edge(existing)
            self._edge_cache[existing.edge_id] = existing
            return existing

        self._store.save_edge(edge)
        self._edge_cache[edge.edge_id] = edge
        self._out_edges[edge.source_id].append(edge.edge_id)
        self._in_edges[edge.target_id].append(edge.edge_id)
        return edge

    def get_edge(self, edge_id: str) -> MemoryEdge | None:
        self._ensure_index()
        if edge_id in self._edge_cache:
            return self._edge_cache[edge_id]
        d = self._store.get_edge(edge_id)
        return MemoryEdge.from_dict(d) if d else None

    def delete_edge(self, edge_id: str) -> bool:
        self._ensure_index()
        edge = self._edge_cache.get(edge_id)
        ok = self._store.delete_edge(edge_id)
        if edge:
            self._edge_cache.pop(edge_id, None)
            if edge_id in self._out_edges.get(edge.source_id, []):
                self._out_edges[edge.source_id].remove(edge_id)
            if edge_id in self._in_edges.get(edge.target_id, []):
                self._in_edges[edge.target_id].remove(edge_id)
        return ok

    def outgoing_edges(self, node_id: str,
                       relation: EdgeRelation | None = None) -> list[MemoryEdge]:
        self._ensure_index()
        eids = self._out_edges.get(node_id, [])
        edges = [self._edge_cache[eid] for eid in eids if eid in self._edge_cache]
        if relation:
            edges = [e for e in edges if e.relation == relation]
        return edges

    def incoming_edges(self, node_id: str,
                       relation: EdgeRelation | None = None) -> list[MemoryEdge]:
        self._ensure_index()
        eids = self._in_edges.get(node_id, [])
        edges = [self._edge_cache[eid] for eid in eids if eid in self._edge_cache]
        if relation:
            edges = [e for e in edges if e.relation == relation]
        return edges

    def neighbours(self, node_id: str,
                   relation: EdgeRelation | None = None,
                   direction: str = "out") -> list[MemoryNode]:
        """Return neighbour nodes (out=successors, in=predecessors, both=all)."""
        edges: list[MemoryEdge] = []
        if direction in ("out", "both"):
            edges += self.outgoing_edges(node_id, relation)
        if direction in ("in", "both"):
            edges += self.incoming_edges(node_id, relation)
        node_ids = {
            e.target_id if e.source_id == node_id else e.source_id
            for e in edges
        }
        return [n for n in (self.get_node(nid) for nid in node_ids) if n]

    def count_edges(self, relation: EdgeRelation | None = None) -> int:
        return self._store.count_edges(
            relation.value if relation else None)

    def all_edges(self, relation: EdgeRelation | None = None,
                  limit: int = 5000) -> list[MemoryEdge]:
        self._ensure_index()
        dicts = self._store.all_edges(
            relation=relation.value if relation else None, limit=limit)
        return [MemoryEdge.from_dict(d) for d in dicts]

    # ----------------------------------------------------------------
    # Convenience constructors
    # ----------------------------------------------------------------

    def make_node(
        self,
        text: str,
        memory_type: MemoryType = MemoryType.RAW,
        confidence: float = 0.7,
        importance: float = 0.5,
        source: str = "system",
        epistemic_status: EpistemicStatus = EpistemicStatus.OBSERVED,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        hierarchy_level: HierarchyLevel = HierarchyLevel.RAW,
    ) -> MemoryNode:
        """Create and persist a MemoryNode."""
        node = MemoryNode(
            text=text,
            memory_type=memory_type,
            hierarchy_level=hierarchy_level,
            confidence=confidence,
            importance=importance,
            source=source,
            epistemic_status=epistemic_status,
            metadata=metadata or {},
            tags=tags or [],
        )
        return self.add_node(node)

    def make_edge(
        self,
        source_id: str,
        target_id: str,
        relation: EdgeRelation,
        confidence: float = 0.7,
        weight: float = 0.5,
        provenance: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEdge:
        """Create and persist a MemoryEdge."""
        edge = MemoryEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            confidence=confidence,
            weight=weight,
            provenance=provenance,
            metadata=metadata or {},
        )
        return self.add_edge(edge)

    # ----------------------------------------------------------------
    # Stats
    # ----------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            **self._store.stats(),
            "cached_nodes": len(self._node_cache),
            "cached_edges": len(self._edge_cache),
        }

    def close(self) -> None:
        self._store.close()
