"""
GraphIndex — secondary in-memory indices for fast HGSHM graph queries.

Maintains:
  • type_index:      memory_type  → set of node_ids
  • level_index:     hierarchy_level → set of node_ids
  • concept_index:   concept_id   → set of node_ids
  • tag_index:       tag          → set of node_ids
  • source_index:    source       → set of node_ids
  • relation_index:  relation     → set of edge_ids

The index is rebuilt from the GraphStore on first use (lazy) and
invalidated when nodes/edges are added or deleted.
"""
from __future__ import annotations
import logging
from collections import defaultdict

from memory.hybrid.models.memory_node import MemoryNode, MemoryType, HierarchyLevel
from memory.hybrid.models.memory_edge import MemoryEdge, EdgeRelation
from memory.hybrid.graph.graph_store import GraphStore

log = logging.getLogger(__name__)


class GraphIndex:
    """
    Secondary indices over a GraphStore.

    Usage
    -----
    index = GraphIndex(graph_store)
    belief_ids = index.by_type(MemoryType.BELIEF)
    gap_ids    = index.by_tag("gap")
    """

    def __init__(self, graph_store: GraphStore) -> None:
        self._graph = graph_store
        self._type_index:     dict[str, set[str]] = defaultdict(set)
        self._level_index:    dict[int,  set[str]] = defaultdict(set)
        self._concept_index:  dict[str, set[str]] = defaultdict(set)
        self._tag_index:      dict[str, set[str]] = defaultdict(set)
        self._source_index:   dict[str, set[str]] = defaultdict(set)
        self._relation_index: dict[str, set[str]] = defaultdict(set)
        self._built = False

    # ----------------------------------------------------------------
    # Build / refresh
    # ----------------------------------------------------------------

    def build(self, limit: int = 100_000) -> None:
        """Rebuild all indices from the GraphStore."""
        self._type_index.clear(); self._level_index.clear()
        self._concept_index.clear(); self._tag_index.clear()
        self._source_index.clear(); self._relation_index.clear()

        for node in self._graph.all_nodes(limit=limit):
            self._index_node(node)

        for edge in self._graph.all_edges(limit=limit):
            self._relation_index[edge.relation.value].add(edge.edge_id)

        self._built = True
        log.debug("GraphIndex: built (%d nodes indexed)", sum(len(v) for v in self._type_index.values()))

    def _ensure(self) -> None:
        if not self._built:
            self.build()

    def _index_node(self, node: MemoryNode) -> None:
        self._type_index[node.memory_type.value].add(node.node_id)
        self._level_index[node.hierarchy_level.value].add(node.node_id)
        if node.concept_id:
            self._concept_index[node.concept_id].add(node.node_id)
        for tag in node.tags:
            self._tag_index[tag].add(node.node_id)
        self._source_index[node.source].add(node.node_id)

    def register_node(self, node: MemoryNode) -> None:
        """Register a newly added node without rebuilding the full index."""
        self._index_node(node)

    def unregister_node(self, node_id: str) -> None:
        """Remove a deleted node from all indices."""
        for s in self._type_index.values():   s.discard(node_id)
        for s in self._level_index.values():  s.discard(node_id)
        for s in self._concept_index.values(): s.discard(node_id)
        for s in self._tag_index.values():    s.discard(node_id)
        for s in self._source_index.values(): s.discard(node_id)

    def register_edge(self, edge: MemoryEdge) -> None:
        self._relation_index[edge.relation.value].add(edge.edge_id)

    def unregister_edge(self, edge_id: str) -> None:
        for s in self._relation_index.values():
            s.discard(edge_id)

    def invalidate(self) -> None:
        self._built = False

    # ----------------------------------------------------------------
    # Query methods
    # ----------------------------------------------------------------

    def by_type(self, memory_type: MemoryType) -> set[str]:
        self._ensure()
        return set(self._type_index.get(memory_type.value, set()))

    def by_level(self, level: HierarchyLevel) -> set[str]:
        self._ensure()
        return set(self._level_index.get(level.value, set()))

    def by_concept(self, concept_id: str) -> set[str]:
        self._ensure()
        return set(self._concept_index.get(concept_id, set()))

    def by_tag(self, tag: str) -> set[str]:
        self._ensure()
        return set(self._tag_index.get(tag, set()))

    def by_source(self, source: str) -> set[str]:
        self._ensure()
        return set(self._source_index.get(source, set()))

    def by_relation(self, relation: EdgeRelation) -> set[str]:
        self._ensure()
        return set(self._relation_index.get(relation.value, set()))

    def nodes_at_or_above(self, min_level: HierarchyLevel) -> set[str]:
        self._ensure()
        result: set[str] = set()
        for lvl, ids in self._level_index.items():
            if lvl >= min_level.value:
                result |= ids
        return result

    def nodes_at_or_below(self, max_level: HierarchyLevel) -> set[str]:
        self._ensure()
        result: set[str] = set()
        for lvl, ids in self._level_index.items():
            if lvl <= max_level.value:
                result |= ids
        return result

    def stats(self) -> dict[str, int]:
        self._ensure()
        return {
            "types":     sum(len(v) for v in self._type_index.values()),
            "levels":    sum(len(v) for v in self._level_index.values()),
            "concepts":  sum(len(v) for v in self._concept_index.values()),
            "tags":      sum(len(v) for v in self._tag_index.values()),
            "relations": sum(len(v) for v in self._relation_index.values()),
        }
