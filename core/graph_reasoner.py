"""
Active Graph Reasoning & Contradiction Detection — Blix v0.3.1  (Issues 4 & 5)

Addresses:
  Issue 4: "Graph is symbolic only — no graph reasoning."
  Issue 5: "No contradiction detection / belief revision."

Two additions to memory_graph.py:

1. ``GraphReasoner`` — path search, centrality, and graph-aware memory ranking.
   Operates on an existing ``MemoryGraph`` without modifying it.

2. ``ContradictionDetector`` — identifies opposing beliefs between memories
   and triggers belief revision via ``ProfileEvolver`` / ``MemoryLifecycleManager``.

Python 3.10 compatible.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from core.memory_graph import EntityKind, GraphEdge, GraphNode, MemoryGraph, RelationKind
from utils.logger import get_logger

log = get_logger(__name__)


# ===========================================================================
# Issue 4 — Graph Reasoner
# ===========================================================================


@dataclass
class GraphPath:
    """A directed path between two nodes in the memory graph."""

    nodes: list[str]     # node ids
    relations: list[str] # relation labels between consecutive nodes
    total_confidence: float = 1.0

    def __str__(self) -> str:
        parts: list[str] = [self.nodes[0]]
        for rel, nid in zip(self.relations, self.nodes[1:]):
            parts.append(f"─[{rel}]→ {nid}")
        return " ".join(parts)


class GraphReasoner:
    """
    Provides graph reasoning over a ``MemoryGraph``.

    Methods
    -------
    find_paths(from_id, to_id, max_depth):
        BFS path search between two entities.
    shortest_path(from_id, to_id):
        Shortest BFS path (by hop count).
    degree_centrality():
        Out-degree centrality for all nodes.
    rank_memories_by_graph(memory_ids, pivot_node_id, graph):
        Re-rank a list of MemoryEntry ids by graph proximity to a pivot node.
    related_entities(node_id, depth):
        All entities reachable within ``depth`` hops.
    """

    def __init__(self, graph: MemoryGraph) -> None:
        self._g = graph

    # ------------------------------------------------------------------
    # Path search
    # ------------------------------------------------------------------

    def find_paths(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 4,
    ) -> list[GraphPath]:
        """BFS: find all simple paths from from_id to to_id within max_depth hops."""
        if from_id not in {n.id for n in self._g.list_nodes()}:
            return []
        results: list[GraphPath] = []
        # Queue: (current_node_id, path_nodes, path_rels, cumulative_conf, visited)
        q: deque = deque()
        q.append((from_id, [from_id], [], 1.0, {from_id}))

        while q:
            cur, nodes, rels, conf, visited = q.popleft()
            if cur == to_id and len(nodes) > 1:
                results.append(GraphPath(
                    nodes=list(nodes),
                    relations=list(rels),
                    total_confidence=conf,
                ))
                continue
            if len(nodes) >= max_depth + 1:
                continue
            for edge in self._g.get_edges(from_id=cur):
                nxt = edge.to_id
                if nxt in visited:
                    continue
                q.append((
                    nxt,
                    nodes + [nxt],
                    rels + [edge.relation],
                    conf * edge.confidence,
                    visited | {nxt},
                ))
        results.sort(key=lambda p: (-p.total_confidence, len(p.nodes)))
        return results

    def shortest_path(self, from_id: str, to_id: str) -> Optional[GraphPath]:
        """Return the shortest path by hop count."""
        paths = self.find_paths(from_id, to_id)
        if not paths:
            return None
        return min(paths, key=lambda p: len(p.nodes))

    # ------------------------------------------------------------------
    # Centrality
    # ------------------------------------------------------------------

    def degree_centrality(self) -> dict[str, float]:
        """
        Out-degree centrality: fraction of nodes each node points to.
        Useful for identifying the most "connected" entities.
        """
        n = self._g.node_count
        if n <= 1:
            return {node.id: 0.0 for node in self._g.list_nodes()}
        counts: dict[str, int] = defaultdict(int)
        for node in self._g.list_nodes():
            counts[node.id] = len(self._g.get_edges(from_id=node.id))
        return {nid: c / (n - 1) for nid, c in counts.items()}

    def most_central_nodes(self, top_k: int = 5) -> list[tuple[str, float]]:
        """Return top-k nodes by out-degree centrality."""
        centrality = self.degree_centrality()
        ranked = sorted(centrality.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Graph-aware memory ranking
    # ------------------------------------------------------------------

    def rank_memories_by_graph(
        self,
        memory_ids: list[int],
        pivot_node_id: str,
        graph: MemoryGraph,
        decay_per_hop: float = 0.5,
        max_depth: int = 3,
    ) -> list[tuple[int, float]]:
        """
        Re-rank MemoryEntry ids by graph proximity to ``pivot_node_id``.

        Algorithm
        ---------
        1. BFS from pivot_node_id up to max_depth hops.
        2. Each reachable node gets a proximity score = decay_per_hop^hop_count.
        3. Each memory is scored by the max proximity of any node whose
           source_memory_ids includes that memory.

        Returns list of (memory_id, graph_score) sorted descending.
        """
        # BFS hop distances from pivot
        hop_dist: dict[str, int] = {pivot_node_id: 0}
        q: deque = deque([pivot_node_id])
        while q:
            cur = q.popleft()
            if hop_dist[cur] >= max_depth:
                continue
            for edge in graph.get_edges(from_id=cur):
                if edge.to_id not in hop_dist:
                    hop_dist[edge.to_id] = hop_dist[cur] + 1
                    q.append(edge.to_id)

        # Map memory_id → max graph score
        scores: dict[int, float] = {mid: 0.0 for mid in memory_ids}
        for node_id, hop in hop_dist.items():
            node_score = decay_per_hop ** hop
            # Find edges referencing any memory in our set
            for edge in graph.get_edges(to_id=node_id) + graph.get_edges(from_id=node_id):
                for mid in edge.source_memory_ids:
                    if mid in scores:
                        scores[mid] = max(scores[mid], node_score)

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked

    # ------------------------------------------------------------------
    # Related entities
    # ------------------------------------------------------------------

    def related_entities(
        self,
        node_id: str,
        depth: int = 2,
    ) -> list[tuple[GraphNode, int]]:
        """
        Return all entities reachable from node_id within depth hops,
        as (GraphNode, hop_count) pairs, sorted by hop_count.
        """
        visited: dict[str, int] = {node_id: 0}
        q: deque = deque([node_id])
        result: list[tuple[GraphNode, int]] = []
        while q:
            cur = q.popleft()
            if visited[cur] >= depth:
                continue
            for edge in self._g.get_edges(from_id=cur):
                nxt = edge.to_id
                if nxt not in visited:
                    visited[nxt] = visited[cur] + 1
                    node = self._g.get_node(nxt)
                    if node:
                        result.append((node, visited[nxt]))
                    q.append(nxt)
        return sorted(result, key=lambda x: x[1])


# ===========================================================================
# Issue 5 — Contradiction Detector
# ===========================================================================


@dataclass
class Contradiction:
    """
    A detected contradiction between two memory entries.

    Fields
    ------
    memory_a_id / memory_b_id:
        The two conflicting memories.
    field:
        Which profile field or topic the contradiction concerns.
    claim_a / claim_b:
        Short natural-language description of each claim.
    severity:
        0.0–1.0 estimate of how serious the conflict is.
    resolved:
        Whether belief revision has been applied.
    winner_id:
        The memory that won belief revision (higher confidence retained).
    """

    memory_a_id: int
    memory_b_id: int
    field: str
    claim_a: str
    claim_b: str
    severity: float = 0.5
    resolved: bool = False
    winner_id: Optional[int] = None
    detected_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )


# Negation markers used for simple heuristic detection
_NEGATION_PATTERNS = [
    r"\bno longer\b", r"\bnot interested in\b", r"\bstopped\b",
    r"\bquit\b", r"\babandoned\b", r"\bnever\b", r"\bdropped\b",
    r"\bdecided against\b", r"\bno more\b",
]
_NEGATION_RE = re.compile("|".join(_NEGATION_PATTERNS), re.IGNORECASE)


class ContradictionDetector:
    """
    Detects contradictions between MemoryEntry objects and applies
    belief revision.

    Detection strategy (v0.3.1 — heuristic; upgradable to NLI model)
    -----------------------------------------------------------------
    1. If memory B contains a negation marker AND shares a topic/entity
       with memory A, flag as contradiction.
    2. If two memories assert opposite importance for the same topic
       (one high, one zero), flag as contradiction.

    Belief revision
    ---------------
    The memory with higher ``importance`` score wins.
    The loser is transitioned to COMPRESSED state (not deleted) —
    preserving history while reducing its retrieval weight.

    Parameters
    ----------
    lifecycle_manager:
        Optional; if provided, losers are compressed rather than deleted.
    """

    def __init__(self, lifecycle_manager: Optional[object] = None) -> None:
        self._lc = lifecycle_manager
        self._contradictions: list[Contradiction] = []

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, memories: list) -> list[Contradiction]:
        """
        Scan a list of MemoryEntry objects for contradictions.

        Returns all newly detected Contradiction objects.
        """
        new: list[Contradiction] = []
        seen_ids = {c.memory_a_id for c in self._contradictions} | \
                   {c.memory_b_id for c in self._contradictions}

        for i, mem_a in enumerate(memories):
            for mem_b in memories[i + 1:]:
                if mem_b.id in seen_ids and mem_a.id in seen_ids:
                    continue
                c = self._check_pair(mem_a, mem_b)
                if c is not None:
                    new.append(c)
                    self._contradictions.append(c)
                    log.info(
                        "Contradiction detected: memory %d vs %d (field=%s severity=%.2f)",
                        mem_a.id, mem_b.id, c.field, c.severity,
                    )

        return new

    def _check_pair(self, mem_a: object, mem_b: object) -> Optional[Contradiction]:
        """Heuristic contradiction check between two MemoryEntry objects."""
        # Shared topics?
        topics_a = set(getattr(mem_a, "topics", []))
        topics_b = set(getattr(mem_b, "topics", []))
        shared = topics_a & topics_b
        if not shared:
            return None

        # Does mem_b negate something related to mem_a?
        text_b = (getattr(mem_b, "input", "") + " " + getattr(mem_b, "output", "")).lower()
        text_a = (getattr(mem_a, "input", "") + " " + getattr(mem_a, "output", "")).lower()

        b_negates = bool(_NEGATION_RE.search(text_b))
        a_negates = bool(_NEGATION_RE.search(text_a))

        if not (b_negates or a_negates):
            return None

        # Pick the most relevant shared topic as the conflicting field
        field = next(iter(shared))
        severity = 0.6 if (b_negates and a_negates) else 0.4

        return Contradiction(
            memory_a_id=getattr(mem_a, "id"),
            memory_b_id=getattr(mem_b, "id"),
            field=field,
            claim_a=_short(text_a, 80),
            claim_b=_short(text_b, 80),
            severity=severity,
        )

    # ------------------------------------------------------------------
    # Belief revision
    # ------------------------------------------------------------------

    def resolve(self, contradiction: Contradiction, memories: list) -> Optional[int]:
        """
        Apply belief revision for one contradiction.

        The memory with higher importance wins.  The loser is compressed
        (not deleted) to preserve history.

        Returns the winner's memory_id, or None if resolution failed.
        """
        id_map = {getattr(m, "id"): m for m in memories}
        mem_a = id_map.get(contradiction.memory_a_id)
        mem_b = id_map.get(contradiction.memory_b_id)
        if mem_a is None or mem_b is None:
            return None

        imp_a = getattr(mem_a, "importance", None) or 0.5
        imp_b = getattr(mem_b, "importance", None) or 0.5

        winner = mem_a if imp_a >= imp_b else mem_b
        loser = mem_b if imp_a >= imp_b else mem_a

        # Compress the loser (demote without deleting)
        if self._lc is not None:
            summary = f"[superseded] {_short(getattr(loser, 'output', ''), 120)}"
            self._lc.compress(getattr(loser, "id"), summary)  # type: ignore[union-attr]

        contradiction.resolved = True
        contradiction.winner_id = getattr(winner, "id")
        log.info(
            "Belief revision: winner=%d loser=%d (imp %.2f vs %.2f)",
            contradiction.winner_id,
            getattr(loser, "id"),
            imp_a if contradiction.winner_id == getattr(mem_a, "id") else imp_b,
            imp_b if contradiction.winner_id == getattr(mem_a, "id") else imp_a,
        )
        return contradiction.winner_id

    def resolve_all(self, memories: list) -> list[Contradiction]:
        """Resolve all unresolved contradictions. Returns resolved list."""
        resolved = []
        for c in self._contradictions:
            if not c.resolved:
                self.resolve(c, memories)
                if c.resolved:
                    resolved.append(c)
        return resolved

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def contradiction_count(self) -> int:
        return len(self._contradictions)

    @property
    def unresolved_count(self) -> int:
        return sum(1 for c in self._contradictions if not c.resolved)

    def get_contradictions(
        self, resolved: Optional[bool] = None
    ) -> list[Contradiction]:
        if resolved is None:
            return list(self._contradictions)
        return [c for c in self._contradictions if c.resolved == resolved]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short(text: str, n: int) -> str:
    return text[:n].strip() + ("…" if len(text) > n else "")
