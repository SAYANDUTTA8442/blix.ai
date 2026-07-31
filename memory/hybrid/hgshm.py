"""
HGSHM — Hybrid Graph-Based Semantic Hierarchical Memory

The single entry point for the v0.3.15 memory architecture.

Usage
-----
    hgshm = HGSHM(memory_dir=Path("memory/"))
    
    # Add memories
    node = hgshm.remember("The deployment pipeline was delayed due to timeout failures")
    
    # Retrieve context
    ctx = hgshm.recall("deployment pipeline status")
    for rm in ctx.primary_memories:
        print(rm.node.text, rm.final_score)
    
    # Add knowledge relationships
    hgshm.observe_cause("timeout_failure", "deployment_blocked")
    hgshm.add_principle("Always monitor timeout rates before deployment")
    
    # Subsystem-specific interfaces
    beliefs    = hgshm.beliefs       # BeliefInterface
    causal     = hgshm.causal        # CausalInterface
    hypotheses = hgshm.hypotheses    # HypothesisInterface (v0.3.14 compat)

Architecture
------------
    HGSHM
      ├── GraphStore         (node + edge persistence: hgshm.db)
      ├── VectorIndex        (sqlite-vec embeddings: vectors.db)
      ├── EmbeddingManager   (hash-projection default, pluggable)
      ├── GraphBuilder       (convenience factory)
      ├── GraphIndex         (secondary type/tag indices)
      ├── GraphTraversal     (BFS/DFS/weighted/shortest-path)
      ├── HybridRetriever    (11-factor ranked fusion)
      ├── HierarchyManager   (automatic compression)
      ├── ConsolidationEngine (duplicate detection + merging)
      ├── ImportanceModel    (dynamic importance scoring)
      └── ContextBuilder     (structured context assembly)
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

from memory.hybrid.models.memory_node import (
    MemoryNode, MemoryType, HierarchyLevel, EpistemicStatus
)
from memory.hybrid.models.memory_edge import MemoryEdge, EdgeRelation
from memory.hybrid.models.memory_cluster import MemoryCluster
from memory.hybrid.models.memory_context import MemoryContext, RetrievedMemory

from memory.hybrid.storage.persistence import HGSHMStore
from memory.hybrid.graph.graph_store import GraphStore
from memory.hybrid.graph.graph_builder import GraphBuilder
from memory.hybrid.graph.graph_index import GraphIndex
from memory.hybrid.graph.graph_traversal import GraphTraversal

from memory.hybrid.vector.embedding_manager import EmbeddingManager
from memory.hybrid.vector.vector_index import VectorIndex

from memory.hybrid.retrieval.hybrid_retriever import (
    HybridRetriever, HybridWeights,
    SemanticRetriever, GraphRetriever, TemporalRetriever,
)

from memory.hybrid.hierarchy.hierarchy_manager import (
    HierarchyManager, Summarizer, AbstractionEngine
)
from memory.hybrid.consolidation.consolidation_engine import (
    ConsolidationEngine, DuplicateDetector, MemoryMerger, ImportanceModel
)
from memory.hybrid.context.context_builder import ContextBuilder

log = logging.getLogger(__name__)


class HGSHM:
    """
    Hybrid Graph-Based Semantic Hierarchical Memory.

    The primary memory substrate for Blix v0.3.15+.
    Every cognitive subsystem accesses memory through this object.
    """

    def __init__(
        self,
        memory_dir: Path,
        embedding_dim: int = 256,
        retrieval_weights: HybridWeights | None = None,
        auto_embed: bool = True,
    ) -> None:
        self._memory_dir = memory_dir
        self._auto_embed = auto_embed

        # ── Core layers ────────────────────────────────────────────
        self.graph_store      = GraphStore(memory_dir)
        self.vector_index     = VectorIndex(memory_dir, dim=embedding_dim)
        self.embedding_manager = EmbeddingManager()

        # ── Graph utilities ─────────────────────────────────────────
        self.graph_builder    = GraphBuilder(self.graph_store)
        self.graph_index      = GraphIndex(self.graph_store)
        self.graph_traversal  = GraphTraversal(self.graph_store)

        # ── Retrieval ────────────────────────────────────────────────
        self.semantic_retriever = SemanticRetriever(
            self.graph_store, self.vector_index, self.embedding_manager)
        self.graph_retriever    = GraphRetriever(self.graph_store)
        self.temporal_retriever = TemporalRetriever(self.graph_store)
        self.hybrid_retriever   = HybridRetriever(
            self.graph_store, self.vector_index, self.embedding_manager,
            weights=retrieval_weights,
        )

        # ── Hierarchy ────────────────────────────────────────────────
        self.hierarchy_manager  = HierarchyManager(
            self.graph_store, self.graph_builder)

        # ── Consolidation ────────────────────────────────────────────
        self.importance_model     = ImportanceModel(self.graph_store)
        self.duplicate_detector   = DuplicateDetector(
            self.graph_store, self.vector_index, self.embedding_manager)
        self.memory_merger        = MemoryMerger(self.graph_store)
        self.consolidation_engine = ConsolidationEngine(
            self.graph_store, self.vector_index, self.embedding_manager,
            self.duplicate_detector, self.memory_merger,
        )

        # ── Context builder ─────────────────────────────────────────
        self.context_builder = ContextBuilder(
            self.hybrid_retriever, self.graph_store, self.graph_index)

        log.info("HGSHM initialised at %s", memory_dir)

    # ────────────────────────────────────────────────────────────────
    # Primary API — remember & recall
    # ────────────────────────────────────────────────────────────────

    def remember(
        self,
        text: str,
        memory_type: MemoryType = MemoryType.RAW,
        confidence: float = 0.7,
        importance: float = 0.5,
        source: str = "system",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        epistemic_status: EpistemicStatus = EpistemicStatus.OBSERVED,
        hierarchy_level: HierarchyLevel = HierarchyLevel.RAW,
    ) -> MemoryNode:
        """
        Store a new memory in HGSHM.

        Returns the persisted MemoryNode.
        """
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
        self.graph_store.add_node(node)

        if self._auto_embed:
            try:
                vec = self.embedding_manager.embed(text)
                eid = self.vector_index.add(
                    node_id=node.node_id,
                    vector=vec,
                    metadata={"memory_type": memory_type.value,
                              "confidence": confidence, "importance": importance},
                )
                node.embedding_id = eid
                self.graph_store.update_node(node, save_history=False)
            except Exception as exc:
                log.warning("HGSHM.remember: embedding failed: %s", exc)

        self.graph_index.register_node(node)
        return node

    def recall(
        self,
        query: str,
        top_k: int = 10,
        context_hint: str = "",
        memory_types: list[MemoryType] | None = None,
        min_confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryContext:
        """
        Retrieve a structured MemoryContext for the given query.

        This is the primary method used by all cognitive subsystems.
        """
        return self.context_builder.build(
            query=query,
            context_hint=context_hint,
            memory_types=memory_types,
            top_k=top_k,
            metadata=metadata,
        )

    def recall_nodes(
        self,
        query: str,
        top_k: int = 10,
        memory_types: list[MemoryType] | None = None,
        min_confidence: float = 0.0,
    ) -> list[MemoryNode]:
        """Convenience: recall and return just the node objects."""
        ctx = self.recall(query, top_k=top_k, memory_types=memory_types,
                          min_confidence=min_confidence)
        return [rm.node for rm in ctx.primary_memories]

    # ────────────────────────────────────────────────────────────────
    # Cognitive event shortcuts
    # ────────────────────────────────────────────────────────────────

    def believe(self, statement: str, confidence: float = 0.7,
                source: str = "system", importance: float = 0.5,
                tags: list[str] | None = None) -> MemoryNode:
        """Add a belief and auto-embed it."""
        node = self.graph_builder.add_belief(statement, confidence=confidence, source=source)
        if importance != 0.5:
            node.update_importance(importance)
            self.graph_store.update_node(node, save_history=False)
        if tags:
            node.tags = tags
            self.graph_store.update_node(node, save_history=False)
            self.graph_index.register_node(node)
        if self._auto_embed and node.embedding_id is None:
            self._embed_node(node)
        return node

    def hypothesise(self, statement: str, confidence: float = 0.3,
                    source: str = "system") -> MemoryNode:
        """Add a hypothesis node."""
        node = self.graph_builder.add_hypothesis(statement, confidence, source)
        if self._auto_embed:
            self._embed_node(node)
        return node

    def observe_cause(
        self,
        trigger: str,
        effect: str,
        relation: EdgeRelation = EdgeRelation.CAUSES,
        confidence: float = 0.6,
        source: str = "system",
    ) -> tuple[MemoryNode, MemoryNode, MemoryEdge]:
        """Record a causal observation and embed both nodes."""
        t_node, e_node, edge = self.graph_builder.add_causal_observation(
            trigger, effect, relation, confidence, source)
        if self._auto_embed:
            self._embed_node(t_node)
            self._embed_node(e_node)
        return t_node, e_node, edge

    def add_principle(self, statement: str, confidence: float = 0.8,
                      source: str = "system",
                      derived_from_ids: list[str] | None = None) -> MemoryNode:
        """Add a principle node."""
        node = self.graph_builder.add_principle(
            statement, confidence, source, derived_from_ids)
        if self._auto_embed:
            self._embed_node(node)
        return node

    def note_gap(self, domain: str, uncertainty: float = 0.8,
                 source: str = "curiosity_engine") -> MemoryNode:
        """Record a knowledge gap."""
        node = self.graph_builder.add_gap(domain, uncertainty, source)
        if self._auto_embed:
            self._embed_node(node)
        return node

    def link(
        self,
        source_id: str,
        target_id: str,
        relation: EdgeRelation,
        confidence: float = 0.7,
        provenance: str = "system",
    ) -> MemoryEdge:
        """Create an explicit edge between two nodes."""
        return self.graph_builder.link(source_id, target_id, relation,
                                        confidence, provenance)

    # ────────────────────────────────────────────────────────────────
    # Node access
    # ────────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> MemoryNode | None:
        return self.graph_store.get_node(node_id)

    def update_node(self, node: MemoryNode) -> None:
        self.graph_store.update_node(node)
        if self._auto_embed:
            self._embed_node(node)

    def delete_node(self, node_id: str) -> bool:
        self.vector_index.delete_by_node(node_id)
        self.graph_index.unregister_node(node_id)
        return self.graph_store.delete_node(node_id)

    def all_nodes(
        self,
        memory_type: MemoryType | None = None,
        min_confidence: float = 0.0,
        min_importance: float = 0.0,
        limit: int = 1000,
    ) -> list[MemoryNode]:
        return self.graph_store.all_nodes(
            memory_type=memory_type,
            min_confidence=min_confidence,
            min_importance=min_importance,
            limit=limit,
        )

    def nodes_by_tags(
        self,
        required_tags: list[str],
        memory_type: MemoryType | None = None,
        limit: int = 500,
        order_by: str = "importance",
    ) -> list[MemoryNode]:
        """
        Return nodes containing ALL required_tags.

        Uses json_each SQL — O(k) per required tag, not O(n) over all nodes.
        Replaces the all_nodes(limit=N)+Python-filter pattern. (ISSUE-008)
        """
        return self.graph_store.nodes_by_tags(
            required_tags=required_tags,
            memory_type=memory_type,
            limit=limit,
            order_by=order_by,
        )

    def count_by_tag(self, tag: str,
                     memory_type: MemoryType | None = None) -> int:
        """Count nodes with tag — single COUNT query.  (ISSUE-008)"""
        return self.graph_store.count_by_tag(tag, memory_type)

    def stats_by_tag(self, tag: str) -> dict[str, int]:
        """Return {memory_type: count} for a tag — single GROUP BY.  (ISSUE-008)"""
        return self.graph_store.stats_by_tag(tag)

    # ────────────────────────────────────────────────────────────────
    # Maintenance
    # ────────────────────────────────────────────────────────────────

    def consolidate(self, merge_duplicates: bool = True) -> dict[str, int]:
        """Run a full consolidation pass (dedup + prune + importance update)."""
        stats = self.consolidation_engine.consolidate(
            merge_duplicates=merge_duplicates)
        stats["importance_updated"] = self.importance_model.update_all()
        return stats

    def compress_hierarchy(self) -> dict[str, int]:
        """Compress lower-level memories upward through the hierarchy."""
        return self.hierarchy_manager.consolidate()

    def rebuild_vector_index(self) -> int:
        """Re-embed all nodes into the vector index (e.g., after backend swap)."""
        nodes = self.graph_store.all_nodes(limit=100_000)
        count = 0
        for node in nodes:
            if self._embed_node(node):
                count += 1
        log.info("HGSHM: rebuilt vector index for %d nodes", count)
        return count

    # ────────────────────────────────────────────────────────────────
    # Stats / diagnostics
    # ────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        gs = self.graph_store.stats()
        return {
            **gs,
            "vectors": self.vector_index.count(),
            "embedding_cache": self.embedding_manager.cache_size,
            "memory_dir": str(self._memory_dir),
        }

    def close(self) -> None:
        self.graph_store.close()
        self.vector_index.close()

    # ────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────

    def _embed_node(self, node: MemoryNode) -> bool:
        try:
            vec = self.embedding_manager.embed(node.text)
            eid = self.vector_index.add(
                node_id=node.node_id,
                vector=vec,
                metadata={"memory_type": node.memory_type.value,
                          "confidence": node.confidence},
                embedding_id=node.embedding_id,
            )
            if node.embedding_id != eid:
                node.embedding_id = eid
                self.graph_store.update_node(node, save_history=False)
            return True
        except Exception as exc:
            log.debug("HGSHM._embed_node: failed for %s: %s", node.node_id[:8], exc)
            return False
