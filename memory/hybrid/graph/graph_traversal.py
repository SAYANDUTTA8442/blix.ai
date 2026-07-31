"""
GraphTraversal — all graph search algorithms for HGSHM.

Implements
  • BFS              — breadth-first neighbourhood expansion
  • DFS              — depth-first neighbourhood exploration
  • Weighted search  — priority queue ordered by edge weight × node importance
  • Shortest path    — BFS-based unweighted shortest path
  • Concept expansion — follow SIMILAR_TO / BELONGS_TO / INSTANCE_OF edges
  • Temporal search  — filter by created_at / valid_from ranges
  • Importance-guided — greedy descent by node importance score
"""
from __future__ import annotations
import heapq
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from memory.hybrid.models.memory_node import MemoryNode
from memory.hybrid.models.memory_edge import MemoryEdge, EdgeRelation
from memory.hybrid.graph.graph_store import GraphStore

log = logging.getLogger(__name__)


@dataclass
class TraversalResult:
    """Result of a graph traversal."""
    start_node_id:  str
    visited_nodes:  list[MemoryNode]    = field(default_factory=list)
    visited_edges:  list[MemoryEdge]    = field(default_factory=list)
    paths:          list[list[str]]     = field(default_factory=list)   # paths as node_id lists
    depth_reached:  int                 = 0
    nodes_evaluated: int                = 0

    @property
    def node_ids(self) -> list[str]:
        return [n.node_id for n in self.visited_nodes]


class GraphTraversal:
    """
    All traversal algorithms over a GraphStore.

    Parameters
    ----------
    graph_store:
        The GraphStore to traverse.
    """

    def __init__(self, graph_store: GraphStore) -> None:
        self._graph = graph_store

    # ----------------------------------------------------------------
    # BFS
    # ----------------------------------------------------------------

    def bfs(
        self,
        start_id: str,
        max_depth: int = 3,
        max_nodes: int = 100,
        relations: list[EdgeRelation] | None = None,
        direction: str = "out",
        node_filter: Callable[[MemoryNode], bool] | None = None,
    ) -> TraversalResult:
        """
        Breadth-first search from start_id.

        Parameters
        ----------
        start_id:
            Root node.
        max_depth:
            Maximum hops from root.
        max_nodes:
            Stop after visiting this many nodes.
        relations:
            If set, only follow edges with these relation types.
        direction:
            "out" (follow outgoing), "in" (follow incoming), "both".
        node_filter:
            Optional predicate; skip nodes where filter(node) is False.
        """
        result = TraversalResult(start_node_id=start_id)
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])

        while queue and len(result.visited_nodes) < max_nodes:
            node_id, depth = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            result.nodes_evaluated += 1

            node = self._graph.get_node(node_id)
            if node is None:
                continue
            if node_filter and not node_filter(node):
                continue

            result.visited_nodes.append(node)
            result.depth_reached = max(result.depth_reached, depth)

            if depth >= max_depth:
                continue

            # Collect edges in desired direction
            edges: list[MemoryEdge] = []
            if direction in ("out", "both"):
                edges += self._graph.outgoing_edges(node_id)
            if direction in ("in", "both"):
                edges += self._graph.incoming_edges(node_id)

            for edge in edges:
                if relations and edge.relation not in relations:
                    continue
                next_id = edge.target_id if edge.source_id == node_id else edge.source_id
                if next_id not in visited:
                    result.visited_edges.append(edge)
                    queue.append((next_id, depth + 1))

        return result

    # ----------------------------------------------------------------
    # DFS
    # ----------------------------------------------------------------

    def dfs(
        self,
        start_id: str,
        max_depth: int = 5,
        max_nodes: int = 100,
        relations: list[EdgeRelation] | None = None,
        direction: str = "out",
    ) -> TraversalResult:
        """Depth-first search from start_id."""
        result = TraversalResult(start_node_id=start_id)
        visited: set[str] = set()
        self._dfs_recurse(start_id, 0, max_depth, max_nodes,
                          relations, direction, visited, result, [start_id])
        return result

    def _dfs_recurse(
        self,
        node_id: str,
        depth: int,
        max_depth: int,
        max_nodes: int,
        relations: list[EdgeRelation] | None,
        direction: str,
        visited: set[str],
        result: TraversalResult,
        current_path: list[str],
    ) -> None:
        if node_id in visited or len(result.visited_nodes) >= max_nodes:
            return
        visited.add(node_id)
        result.nodes_evaluated += 1

        node = self._graph.get_node(node_id)
        if node is None:
            return
        result.visited_nodes.append(node)
        result.depth_reached = max(result.depth_reached, depth)

        if depth >= max_depth:
            result.paths.append(list(current_path))
            return

        edges: list[MemoryEdge] = []
        if direction in ("out", "both"):
            edges += self._graph.outgoing_edges(node_id)
        if direction in ("in", "both"):
            edges += self._graph.incoming_edges(node_id)

        has_children = False
        for edge in edges:
            if relations and edge.relation not in relations:
                continue
            next_id = edge.target_id if edge.source_id == node_id else edge.source_id
            if next_id not in visited:
                has_children = True
                result.visited_edges.append(edge)
                self._dfs_recurse(
                    next_id, depth + 1, max_depth, max_nodes,
                    relations, direction, visited, result, current_path + [next_id])

        if not has_children:
            result.paths.append(list(current_path))

    # ----------------------------------------------------------------
    # Weighted search (priority queue by edge weight × node importance)
    # ----------------------------------------------------------------

    def weighted_search(
        self,
        start_id: str,
        max_nodes: int = 50,
        max_depth: int = 4,
        relations: list[EdgeRelation] | None = None,
        direction: str = "out",
    ) -> TraversalResult:
        """
        Best-first expansion ordered by (edge.weight × node.importance) descending.
        """
        result = TraversalResult(start_node_id=start_id)
        visited: set[str] = set()
        # heap: (-priority, node_id, depth)
        heap: list[tuple[float, str, int]] = [(0.0, start_id, 0)]

        while heap and len(result.visited_nodes) < max_nodes:
            neg_prio, node_id, depth = heapq.heappop(heap)
            if node_id in visited:
                continue
            visited.add(node_id)
            result.nodes_evaluated += 1

            node = self._graph.get_node(node_id)
            if node is None:
                continue
            result.visited_nodes.append(node)
            result.depth_reached = max(result.depth_reached, depth)

            if depth >= max_depth:
                continue

            edges: list[MemoryEdge] = []
            if direction in ("out", "both"):
                edges += self._graph.outgoing_edges(node_id)
            if direction in ("in", "both"):
                edges += self._graph.incoming_edges(node_id)

            for edge in edges:
                if relations and edge.relation not in relations:
                    continue
                next_id = edge.target_id if edge.source_id == node_id else edge.source_id
                if next_id not in visited:
                    neighbour = self._graph.get_node(next_id)
                    if neighbour:
                        priority = edge.weight * neighbour.importance
                        result.visited_edges.append(edge)
                        heapq.heappush(heap, (-priority, next_id, depth + 1))

        return result

    # ----------------------------------------------------------------
    # Shortest path (unweighted BFS)
    # ----------------------------------------------------------------

    def shortest_path(self, start_id: str, end_id: str,
                      max_depth: int = 8) -> list[MemoryNode] | None:
        """
        Find the shortest path from start_id to end_id.
        Returns ordered list of nodes, or None if unreachable.
        """
        if start_id == end_id:
            node = self._graph.get_node(start_id)
            return [node] if node else None

        visited: set[str] = {start_id}
        queue: deque[tuple[str, list[str]]] = deque([(start_id, [start_id])])

        while queue:
            node_id, path = queue.popleft()
            if len(path) > max_depth + 1:
                continue

            for edge in self._graph.outgoing_edges(node_id) + self._graph.incoming_edges(node_id):
                next_id = edge.target_id if edge.source_id == node_id else edge.source_id
                if next_id in visited:
                    continue
                new_path = path + [next_id]
                if next_id == end_id:
                    return [n for n in (self._graph.get_node(nid) for nid in new_path) if n]
                visited.add(next_id)
                queue.append((next_id, new_path))

        return None  # unreachable

    # ----------------------------------------------------------------
    # Neighbourhood expansion
    # ----------------------------------------------------------------

    def neighbourhood(self, node_id: str, radius: int = 1) -> TraversalResult:
        """Expand to all nodes within `radius` hops (all relations, both directions)."""
        return self.bfs(node_id, max_depth=radius, direction="both")

    # ----------------------------------------------------------------
    # Concept expansion
    # ----------------------------------------------------------------

    def concept_expansion(self, concept_node_id: str, max_depth: int = 2) -> TraversalResult:
        """Follow SIMILAR_TO, BELONGS_TO, INSTANCE_OF edges to expand a concept cluster."""
        return self.bfs(
            concept_node_id,
            max_depth=max_depth,
            relations=[EdgeRelation.SIMILAR_TO, EdgeRelation.BELONGS_TO,
                       EdgeRelation.INSTANCE_OF, EdgeRelation.PART_OF],
            direction="both",
        )

    # ----------------------------------------------------------------
    # Importance-guided search
    # ----------------------------------------------------------------

    def importance_guided(self, start_id: str, max_nodes: int = 30,
                          min_importance: float = 0.3) -> TraversalResult:
        """Greedy descent following the highest-importance neighbours."""
        result = TraversalResult(start_node_id=start_id)
        visited: set[str] = {start_id}
        current_id = start_id

        for _ in range(max_nodes):
            node = self._graph.get_node(current_id)
            if node is None:
                break
            result.visited_nodes.append(node)
            result.nodes_evaluated += 1

            neighbours = self._graph.neighbours(current_id, direction="out")
            candidates = [
                n for n in neighbours
                if n.node_id not in visited and n.importance >= min_importance
            ]
            if not candidates:
                break
            best = max(candidates, key=lambda n: n.importance)
            visited.add(best.node_id)
            current_id = best.node_id
            result.depth_reached += 1

        return result

    # ----------------------------------------------------------------
    # Temporal graph search
    # ----------------------------------------------------------------

    def temporal_search(
        self,
        start_id: str,
        after: datetime | None = None,
        before: datetime | None = None,
        max_nodes: int = 50,
    ) -> TraversalResult:
        """BFS that only visits nodes within a temporal window."""
        def time_filter(node: MemoryNode) -> bool:
            try:
                created = node.created_dt
                if after and created < after:
                    return False
                if before and created > before:
                    return False
                return True
            except Exception:
                return True

        return self.bfs(
            start_id,
            max_depth=5,
            max_nodes=max_nodes,
            direction="both",
            node_filter=time_filter,
        )
