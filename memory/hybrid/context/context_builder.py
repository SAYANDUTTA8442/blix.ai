"""
ContextBuilder — assembles a MemoryContext from the full HGSHM retrieval pipeline.

Pipeline
--------
Query
  → Semantic Search
  → Vector Search
  → Graph Expansion
  → Temporal Filter
  → Importance Ranking
  → Hierarchy Retrieval (concepts + principles)
  → Belief Validation
  → Contradiction Detection
  → Causal Chain Extraction
  → Gap Discovery
  → Context Assembly

Output: MemoryContext (typed object, not concatenated strings)
"""
from __future__ import annotations
import logging
import time
from typing import Any

from memory.hybrid.models.memory_node import MemoryNode, MemoryType, HierarchyLevel
from memory.hybrid.models.memory_edge import EdgeRelation
from memory.hybrid.models.memory_context import MemoryContext, RetrievedMemory
from memory.hybrid.graph.graph_store import GraphStore
from memory.hybrid.graph.graph_index import GraphIndex
from memory.hybrid.retrieval.hybrid_retriever import HybridRetriever, HybridWeights
from memory.hybrid.graph.graph_traversal import GraphTraversal

log = logging.getLogger(__name__)


class ContextBuilder:
    """
    Assembles a MemoryContext for a given query.

    Parameters
    ----------
    hybrid_retriever:
        The full hybrid retrieval engine.
    graph_store:
        For graph traversal and node lookup.
    graph_index:
        For fast type/tag lookup.
    top_k:
        How many primary memories to include.
    expand_depth:
        Graph expansion depth for supporting context.
    include_principles:
        Whether to include principle nodes.
    include_concepts:
        Whether to include concept nodes.
    min_confidence:
        Minimum node confidence for inclusion.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        graph_store: GraphStore,
        graph_index: GraphIndex | None = None,
        top_k: int = 10,
        expand_depth: int = 2,
        include_principles: bool = True,
        include_concepts: bool = True,
        min_confidence: float = 0.1,
    ) -> None:
        self._retriever     = hybrid_retriever
        self._graph         = graph_store
        self._index         = graph_index or GraphIndex(graph_store)
        self._traversal     = GraphTraversal(graph_store)
        self._top_k         = top_k
        self._expand_depth  = expand_depth
        self._inc_principles = include_principles
        self._inc_concepts   = include_concepts
        self._min_confidence = min_confidence

    def build(
        self,
        query: str,
        context_hint: str = "",
        memory_types: list[MemoryType] | None = None,
        top_k: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryContext:
        """
        Build a complete MemoryContext for the given query.

        Parameters
        ----------
        query:
            The natural-language query.
        context_hint:
            Optional extra context (e.g., current conversation topic).
        memory_types:
            Restrict primary retrieval to these types.
        top_k:
            Override default top_k.
        metadata:
            Extra metadata to attach to the context.
        """
        t0 = time.perf_counter()
        k = top_k or self._top_k

        ctx = MemoryContext(query=query, metadata=metadata or {})

        # ── Step 1: Primary retrieval (hybrid) ───────────────────────
        primary = self._retriever.retrieve(
            query=query,
            top_k=k,
            expand_graph=False,   # we do our own expansion below
            include_temporal=False,
            memory_types=memory_types,
            min_confidence=self._min_confidence,
            context_hint=context_hint,
        )
        ctx.primary_memories = primary
        ctx.total_nodes_searched = len(primary)

        primary_ids = [r.node.node_id for r in primary]

        # ── Step 2: Graph expansion (supporting context) ─────────────
        if primary_ids:
            supporting = self._expand_supporting(primary_ids, k)
            ctx.supporting_memories = supporting
            ctx.total_nodes_searched += len(supporting)

        # ── Step 3: Temporal memories (recent + frequent) ────────────
        ctx.temporal_memories = self._get_temporal(k // 2)

        # ── Step 4: Concepts ─────────────────────────────────────────
        if self._inc_concepts:
            ctx.concept_nodes = self._get_nodes_of_type(MemoryType.CONCEPT, k // 3)

        # ── Step 5: Principles ───────────────────────────────────────
        if self._inc_principles:
            ctx.principle_nodes = self._get_nodes_of_type(MemoryType.PRINCIPLE, k // 3)

        # ── Step 6: Beliefs ──────────────────────────────────────────
        belief_ids = list(self._index.by_type(MemoryType.BELIEF))
        all_ids = set(primary_ids)
        related_beliefs = [
            self._graph.get_node(bid)
            for bid in belief_ids
            if bid not in all_ids
        ]
        ctx.belief_nodes = [b for b in related_beliefs
                            if b is not None and b.confidence >= self._min_confidence][:k // 3]

        # ── Step 7: Contradictions ───────────────────────────────────
        all_retrieved_ids = (
            primary_ids +
            [r.node.node_id for r in ctx.supporting_memories] +
            [n.node_id for n in ctx.belief_nodes]
        )
        ctx.contradictions = self._detect_contradictions(all_retrieved_ids)

        # ── Step 8: Causal chains ─────────────────────────────────────
        ctx.causal_chains = self._extract_causal_chains(primary_ids)

        # ── Step 9: Knowledge gaps ────────────────────────────────────
        gap_ids = list(self._index.by_type(MemoryType.GAP))
        ctx.knowledge_gaps = [
            n for n in (self._graph.get_node(gid) for gid in gap_ids[:5])
            if n is not None
        ]

        # ── Step 10: Graph neighbourhood ─────────────────────────────
        ctx.graph_neighbourhood = self._get_neighbourhood_edges(primary_ids[:3])

        # ── Step 11: Touch accessed nodes ─────────────────────────────
        for rm in ctx.primary_memories:
            rm.node.touch()
            self._graph.update_node(rm.node, save_history=False)

        ctx.retrieval_latency_ms = (time.perf_counter() - t0) * 1000
        log.debug("ContextBuilder: built context in %.1fms (%d total memories)",
                  ctx.retrieval_latency_ms, ctx.total_memories)
        return ctx

    # ── Internal helpers ─────────────────────────────────────────────

    def _expand_supporting(
        self, seed_ids: list[str], max_nodes: int
    ) -> list[RetrievedMemory]:
        seen = set(seed_ids)
        results: list[RetrievedMemory] = []
        for seed_id in seed_ids[:3]:
            traversal = self._traversal.bfs(
                seed_id,
                max_depth=self._expand_depth,
                max_nodes=max_nodes,
                direction="both",
            )
            for node in traversal.visited_nodes:
                if node.node_id in seen:
                    continue
                seen.add(node.node_id)
                graph_score = node.importance * (0.7 ** traversal.depth_reached)
                results.append(RetrievedMemory(
                    node=node, graph_score=graph_score, final_score=graph_score))
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:max_nodes]

    def _get_temporal(self, top_k: int) -> list[RetrievedMemory]:
        from memory.hybrid.retrieval.hybrid_retriever import TemporalRetriever
        tr = TemporalRetriever(self._graph)
        return tr.recently_accessed_plus_frequent(top_k)

    def _get_nodes_of_type(self, memory_type: MemoryType, top_k: int) -> list[MemoryNode]:
        node_ids = list(self._index.by_type(memory_type))
        nodes = [self._graph.get_node(nid) for nid in node_ids[:top_k * 2]]
        nodes = [n for n in nodes if n is not None]
        nodes.sort(key=lambda n: n.importance, reverse=True)
        return nodes[:top_k]

    def _detect_contradictions(
        self, node_ids: list[str]
    ) -> list[tuple[MemoryNode, MemoryNode]]:
        id_set = set(node_ids)
        pairs: list[tuple[MemoryNode, MemoryNode]] = []
        seen: set[frozenset] = set()
        for nid in node_ids:
            for edge in self._graph.outgoing_edges(nid, EdgeRelation.CONTRADICTS):
                if edge.target_id in id_set:
                    pair_key = frozenset([nid, edge.target_id])
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)
                    a = self._graph.get_node(nid)
                    b = self._graph.get_node(edge.target_id)
                    if a and b:
                        pairs.append((a, b))
        return pairs

    def _extract_causal_chains(
        self, seed_ids: list[str], max_chain_depth: int = 3
    ) -> list[list[MemoryNode]]:
        chains: list[list[MemoryNode]] = []
        causal_relations = [EdgeRelation.CAUSES, EdgeRelation.ENABLES, EdgeRelation.BLOCKS]
        for seed_id in seed_ids[:3]:
            traversal = self._traversal.dfs(
                seed_id,
                max_depth=max_chain_depth,
                relations=causal_relations,
                direction="out",
            )
            for path in traversal.paths:
                if len(path) >= 2:
                    chain = [n for n in (self._graph.get_node(nid) for nid in path) if n]
                    if len(chain) >= 2:
                        chains.append(chain)
            if len(chains) >= 3:
                break
        return chains

    def _get_neighbourhood_edges(
        self, node_ids: list[str]
    ) -> list[Any]:
        from memory.hybrid.models.memory_edge import MemoryEdge
        edges: list[MemoryEdge] = []
        seen_ids: set[str] = set()
        for nid in node_ids:
            for edge in (self._graph.outgoing_edges(nid) +
                         self._graph.incoming_edges(nid)):
                if edge.edge_id not in seen_ids:
                    seen_ids.add(edge.edge_id)
                    edges.append(edge)
        return edges[:50]
