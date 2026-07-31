"""
Consolidation subsystem — duplicate detection, memory merging, importance scoring.
"""
from __future__ import annotations
import logging
import math
from typing import Any

from memory.hybrid.models.memory_node import MemoryNode, MemoryType
from memory.hybrid.models.memory_edge import MemoryEdge, EdgeRelation
from memory.hybrid.graph.graph_store import GraphStore
from memory.hybrid.vector.vector_index import VectorIndex
from memory.hybrid.vector.embedding_manager import EmbeddingManager

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Duplicate Detector
# ────────────────────────────────────────────────────────────────────

class DuplicateDetector:
    """
    Find semantically or textually duplicate MemoryNodes.

    Two nodes are considered duplicates if:
      - Their cosine similarity (via VectorIndex) exceeds `sim_threshold`, OR
      - Their token Jaccard similarity exceeds `jaccard_threshold`
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_index: VectorIndex,
        embedding_manager: EmbeddingManager,
        sim_threshold: float = 0.92,
        jaccard_threshold: float = 0.7,
    ) -> None:
        self._graph   = graph_store
        self._vector  = vector_index
        self._emb     = embedding_manager
        self._sim_t   = sim_threshold
        self._jacc_t  = jaccard_threshold

    def find_duplicates(
        self,
        node: MemoryNode,
        top_k: int = 10,
    ) -> list[tuple[MemoryNode, float]]:
        """
        Return [(candidate_node, similarity_score)] for nodes similar to `node`.
        Only considers nodes of the same memory_type.
        """
        query_vec = self._emb.embed(node.text)
        results   = self._vector.search(query_vec, top_k=top_k + 1, min_score=self._sim_t)
        duplicates = []
        for sr in results:
            if sr.node_id == node.node_id:
                continue
            candidate = self._graph.get_node(sr.node_id)
            if candidate is None:
                continue
            if candidate.memory_type != node.memory_type:
                continue
            # Double-check with Jaccard
            jacc = self._jaccard(node.text, candidate.text)
            score = max(sr.score, jacc)
            if score >= self._sim_t or jacc >= self._jacc_t:
                duplicates.append((candidate, score))
        return sorted(duplicates, key=lambda x: x[1], reverse=True)

    def scan_all_duplicates(
        self,
        memory_type: MemoryType | None = None,
        limit: int = 500,
    ) -> list[tuple[MemoryNode, MemoryNode, float]]:
        """
        Full scan for duplicate pairs. O(n²) on the sampled set.
        Returns [(node_a, node_b, score)].
        """
        nodes = self._graph.all_nodes(
            memory_type=memory_type, limit=limit)
        duplicates: list[tuple[MemoryNode, MemoryNode, float]] = []
        seen: set[frozenset[str]] = set()

        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1:]:
                pair = frozenset([node_a.node_id, node_b.node_id])
                if pair in seen:
                    continue
                seen.add(pair)
                jacc = self._jaccard(node_a.text, node_b.text)
                if jacc >= self._jacc_t:
                    duplicates.append((node_a, node_b, jacc))

        return sorted(duplicates, key=lambda x: x[2], reverse=True)

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta = set(a.lower().split()); tb = set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)


# ────────────────────────────────────────────────────────────────────
# Memory Merger
# ────────────────────────────────────────────────────────────────────

class MemoryMerger:
    """
    Merge two duplicate MemoryNodes into one canonical node.

    Strategy:
      - Keep the node with higher importance as the canonical.
      - Merge confidence (average, weighted by evidence).
      - Merge importance (max).
      - Transfer all edges from the merged node to the canonical.
      - Delete the merged node.
      - Record the merge in canonical's metadata.
    """

    def __init__(self, graph_store: GraphStore) -> None:
        self._graph = graph_store

    def merge(self, canonical_id: str, duplicate_id: str) -> MemoryNode | None:
        """
        Merge `duplicate_id` into `canonical_id`.
        Returns the updated canonical node, or None on failure.
        """
        canonical = self._graph.get_node(canonical_id)
        duplicate = self._graph.get_node(duplicate_id)
        if not canonical or not duplicate:
            log.warning("MemoryMerger: node not found (%s or %s)", canonical_id, duplicate_id)
            return None

        # Merge confidence (weighted average by access_count)
        total_evidence = canonical.access_count + duplicate.access_count + 2
        w_c = (canonical.access_count + 1) / total_evidence
        w_d = (duplicate.access_count  + 1) / total_evidence
        canonical.confidence = min(1.0, canonical.confidence * w_c + duplicate.confidence * w_d)
        canonical.importance  = max(canonical.importance, duplicate.importance)
        canonical.access_count += duplicate.access_count

        # Record merge in metadata
        merges = canonical.metadata.get("merged_from", [])
        merges.append({
            "duplicate_id": duplicate_id,
            "duplicate_text": duplicate.text[:80],
            "merged_at": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
        })
        canonical.metadata["merged_from"] = merges

        # Transfer edges
        for edge in self._graph.outgoing_edges(duplicate_id):
            new_edge = MemoryEdge(
                source_id=canonical_id,
                target_id=edge.target_id,
                relation=edge.relation,
                confidence=edge.confidence,
                weight=edge.weight,
                provenance=f"merge:{edge.provenance}",
            )
            self._graph.add_edge(new_edge)

        for edge in self._graph.incoming_edges(duplicate_id):
            new_edge = MemoryEdge(
                source_id=edge.source_id,
                target_id=canonical_id,
                relation=edge.relation,
                confidence=edge.confidence,
                weight=edge.weight,
                provenance=f"merge:{edge.provenance}",
            )
            self._graph.add_edge(new_edge)

        self._graph.update_node(canonical)
        self._graph.delete_node(duplicate_id)
        log.info("MemoryMerger: merged %s into %s", duplicate_id[:8], canonical_id[:8])
        return canonical

    def merge_pair(self, a: MemoryNode, b: MemoryNode) -> MemoryNode | None:
        """Merge the lower-importance node into the higher-importance one."""
        if a.importance >= b.importance:
            return self.merge(a.node_id, b.node_id)
        return self.merge(b.node_id, a.node_id)


# ────────────────────────────────────────────────────────────────────
# Consolidation Engine
# ────────────────────────────────────────────────────────────────────

class ConsolidationEngine:
    """
    Orchestrates full memory consolidation:
      1. Detect duplicates
      2. Merge duplicates
      3. Prune very low importance / confidence nodes
      4. Update importance scores after access patterns change

    Should be run periodically (e.g., end of session, daily).
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_index: VectorIndex,
        embedding_manager: EmbeddingManager,
        duplicate_detector: DuplicateDetector | None = None,
        memory_merger: MemoryMerger | None = None,
    ) -> None:
        self._graph   = graph_store
        self._vector  = vector_index
        self._emb     = embedding_manager
        self._detector = duplicate_detector or DuplicateDetector(
            graph_store, vector_index, embedding_manager)
        self._merger   = memory_merger or MemoryMerger(graph_store)

    def consolidate(
        self,
        max_scan: int = 500,
        prune_below_importance: float = 0.05,
        prune_below_confidence: float = 0.1,
        merge_duplicates: bool = True,
    ) -> dict[str, int]:
        """
        Run a full consolidation pass.

        Returns
        -------
        dict with keys: merged, pruned, updated
        """
        stats = {"merged": 0, "pruned": 0, "updated": 0}

        if merge_duplicates:
            pairs = self._detector.scan_all_duplicates(limit=max_scan)
            for node_a, node_b, score in pairs:
                result = self._merger.merge_pair(node_a, node_b)
                if result:
                    stats["merged"] += 1

        # Prune very low value nodes
        nodes = self._graph.all_nodes(
            min_confidence=0.0, min_importance=0.0, limit=max_scan)
        for node in nodes:
            if (node.importance < prune_below_importance and
                    node.confidence < prune_below_confidence and
                    node.access_count == 0):
                self._graph.delete_node(node.node_id)
                self._vector.delete_by_node(node.node_id)
                stats["pruned"] += 1

        log.info("ConsolidationEngine: %s", stats)
        return stats


# ────────────────────────────────────────────────────────────────────
# Importance Model
# ────────────────────────────────────────────────────────────────────

class ImportanceModel:
    """
    Dynamic importance scoring for MemoryNodes.

    Importance evolves based on:
      • Access frequency       — often retrieved → more important
      • Edge count             — many connections → more important
      • Confidence             — high confidence → more important
      • Hierarchy level        — higher level → generally more important
      • Causal centrality      — nodes that cause many things → more important
      • Age decay              — old unused nodes → less important
    """

    def __init__(self, graph_store: GraphStore) -> None:
        self._graph = graph_store

    def score(self, node: MemoryNode) -> float:
        """Compute a fresh importance score for a node [0, 1]."""
        out_degree = len(self._graph.outgoing_edges(node.node_id))
        in_degree  = len(self._graph.incoming_edges(node.node_id))

        access_factor     = min(1.0, node.access_count / 10.0)
        degree_factor     = min(1.0, (out_degree + in_degree) / 20.0)
        confidence_factor = node.confidence
        level_factor      = node.hierarchy_level.value / 11.0
        recency_factor    = node.recency_score
        causal_factor     = min(1.0, out_degree / 10.0)

        importance = (
            0.25 * access_factor +
            0.15 * degree_factor +
            0.20 * confidence_factor +
            0.15 * level_factor +
            0.10 * recency_factor +
            0.15 * causal_factor
        )
        return max(0.0, min(1.0, importance))

    def update_all(self, limit: int = 1000) -> int:
        """Recompute importance for all nodes and persist changes."""
        nodes = self._graph.all_nodes(limit=limit)
        updated = 0
        for node in nodes:
            new_imp = self.score(node)
            if abs(new_imp - node.importance) > 0.01:
                node.update_importance(new_imp)
                self._graph.update_node(node, save_history=False)
                updated += 1
        log.debug("ImportanceModel: updated %d nodes", updated)
        return updated
