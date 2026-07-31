"""
HGSHM Retrieval Layer — four retrievers that compose into HybridRetriever.

SemanticRetriever  — cosine similarity over text tokens
GraphRetriever     — BFS/weighted expansion from seed nodes
TemporalRetriever  — recency and validity window filtering
HybridRetriever    — 11-factor ranked fusion of all three
"""
from __future__ import annotations
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from memory.hybrid.models.memory_node import MemoryNode, MemoryType
from memory.hybrid.models.memory_edge import EdgeRelation
from memory.hybrid.models.memory_context import RetrievedMemory
from memory.hybrid.graph.graph_store import GraphStore
from memory.hybrid.graph.graph_traversal import GraphTraversal
from memory.hybrid.vector.vector_index import VectorIndex
from memory.hybrid.vector.embedding_manager import EmbeddingManager

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Semantic Retriever
# ────────────────────────────────────────────────────────────────────

class SemanticRetriever:
    """
    Text-based semantic retrieval using token overlap + vector similarity.

    For each query:
      1. Compute query embedding via EmbeddingManager
      2. Search VectorIndex for nearest neighbours
      3. Fetch corresponding MemoryNodes from GraphStore
      4. Return as RetrievedMemory objects with scores
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_index: VectorIndex,
        embedding_manager: EmbeddingManager,
    ) -> None:
        self._graph = graph_store
        self._vector = vector_index
        self._emb = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        min_score: float = 0.0,
        memory_types: list[MemoryType] | None = None,
        filter_node_ids: list[str] | None = None,
    ) -> list[RetrievedMemory]:
        """
        Retrieve the most semantically similar memories.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Maximum memories to return.
        min_score:
            Minimum cosine similarity threshold.
        memory_types:
            If set, only return nodes of these types.
        filter_node_ids:
            If set, restrict search to these node_ids.
        """
        query_vec = self._emb.embed(query)
        search_results = self._vector.search(
            query_vec, top_k=top_k * 2, min_score=min_score,
            filter_node_ids=filter_node_ids)

        retrieved: list[RetrievedMemory] = []
        for sr in search_results:
            node = self._graph.get_node(sr.node_id)
            if node is None:
                continue
            if memory_types and node.memory_type not in memory_types:
                continue
            retrieved.append(RetrievedMemory(
                node=node,
                semantic_score=sr.score,
                vector_score=sr.score,
                final_score=sr.score,
            ))
            if len(retrieved) >= top_k:
                break

        # Fallback: token overlap search if vector store is sparse
        if len(retrieved) < min(3, top_k):
            fallback = self._token_search(query, top_k, memory_types)
            seen_ids = {r.node.node_id for r in retrieved}
            for rm in fallback:
                if rm.node.node_id not in seen_ids:
                    retrieved.append(rm)

        return retrieved[:top_k]

    def _token_search(
        self,
        query: str,
        top_k: int,
        memory_types: list[MemoryType] | None,
    ) -> list[RetrievedMemory]:
        """Token overlap fallback when vector store has few entries."""
        nodes = self._graph.search_by_text(query, limit=top_k * 2)
        results = []
        for node in nodes:
            if memory_types and node.memory_type not in memory_types:
                continue
            score = self._token_overlap(query, node.text)
            if score > 0:
                results.append(RetrievedMemory(
                    node=node, semantic_score=score, vector_score=0.0, final_score=score))
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _token_overlap(query: str, text: str) -> float:
        q_tokens = set(query.lower().split())
        t_tokens = set(text.lower().split())
        if not q_tokens or not t_tokens:
            return 0.0
        intersection = q_tokens & t_tokens
        return len(intersection) / math.sqrt(len(q_tokens) * len(t_tokens))

    def embed_node(self, node: MemoryNode) -> str | None:
        """Embed a node's text and upsert into vector index. Returns embedding_id."""
        try:
            vec = self._emb.embed(node.text)
            eid = self._vector.add(
                node_id=node.node_id,
                vector=vec,
                metadata={"memory_type": node.memory_type.value,
                          "confidence": node.confidence,
                          "importance": node.importance},
            )
            return eid
        except Exception as exc:
            log.warning("SemanticRetriever: embed_node failed for %s: %s",
                        node.node_id[:8], exc)
            return None


# ────────────────────────────────────────────────────────────────────
# Graph Retriever
# ────────────────────────────────────────────────────────────────────

class GraphRetriever:
    """
    Graph-based retrieval: expand from seed nodes via typed edges.

    Used to enrich a semantic search result with related context
    (causal chains, principles, supporting beliefs, etc.)
    """

    def __init__(self, graph_store: GraphStore) -> None:
        self._graph = graph_store
        self._traversal = GraphTraversal(graph_store)

    def expand(
        self,
        seed_node_ids: list[str],
        max_depth: int = 2,
        max_nodes: int = 50,
        relations: list[EdgeRelation] | None = None,
        direction: str = "both",
    ) -> list[RetrievedMemory]:
        """
        Expand from seed nodes and return enriching context nodes.

        The graph_score for expanded nodes decays with distance from seeds:
          score = base_importance * (0.8 ** depth)
        """
        visited: set[str] = set(seed_node_ids)
        result: list[RetrievedMemory] = []

        for seed_id in seed_node_ids:
            traversal = self._traversal.bfs(
                seed_id,
                max_depth=max_depth,
                max_nodes=max_nodes,
                relations=relations,
                direction=direction,
            )
            for node in traversal.visited_nodes:
                if node.node_id in visited:
                    continue
                visited.add(node.node_id)
                # Graph score: importance decayed by traversal depth
                graph_score = node.importance * (0.8 ** traversal.depth_reached)
                result.append(RetrievedMemory(
                    node=node,
                    graph_score=graph_score,
                    final_score=graph_score,
                ))
                if len(result) >= max_nodes:
                    break

        result.sort(key=lambda r: r.final_score, reverse=True)
        return result

    def causal_chain(self, node_id: str, max_depth: int = 4) -> list[list[MemoryNode]]:
        """Find causal chains passing through this node."""
        chains: list[list[MemoryNode]] = []
        # Forward: what does this cause?
        fwd = self._traversal.dfs(
            node_id, max_depth=max_depth,
            relations=[EdgeRelation.CAUSES, EdgeRelation.ENABLES, EdgeRelation.BLOCKS],
            direction="out",
        )
        if fwd.paths:
            for path in fwd.paths:
                chain = [n for n in (self._graph.get_node(nid) for nid in path) if n]
                if len(chain) >= 2:
                    chains.append(chain)
        return chains

    def find_contradictions(self, node_ids: list[str]) -> list[tuple[MemoryNode, MemoryNode]]:
        """Find CONTRADICTS edges within a set of nodes."""
        pairs: list[tuple[MemoryNode, MemoryNode]] = []
        id_set = set(node_ids)
        for nid in node_ids:
            for edge in self._graph.outgoing_edges(nid, EdgeRelation.CONTRADICTS):
                if edge.target_id in id_set:
                    a = self._graph.get_node(edge.source_id)
                    b = self._graph.get_node(edge.target_id)
                    if a and b:
                        pairs.append((a, b))
        return pairs


# ────────────────────────────────────────────────────────────────────
# Temporal Retriever
# ────────────────────────────────────────────────────────────────────

class TemporalRetriever:
    """
    Retrieval based on temporal properties: recency, validity windows,
    and historical access patterns.
    """

    def __init__(self, graph_store: GraphStore) -> None:
        self._graph = graph_store

    def recent(self, top_k: int = 20, max_age_hours: float = 24.0) -> list[RetrievedMemory]:
        """Return most recently created/updated nodes."""
        cutoff = datetime.now(timezone.utc).timestamp() - max_age_hours * 3600
        nodes = self._graph.all_nodes(limit=500)
        filtered = []
        for node in nodes:
            try:
                ts = node.created_dt.timestamp()
                if ts >= cutoff:
                    age_h = (datetime.now(timezone.utc).timestamp() - ts) / 3600
                    temporal_score = math.exp(-age_h / max_age_hours)
                    filtered.append(RetrievedMemory(
                        node=node,
                        temporal_score=temporal_score,
                        final_score=temporal_score,
                    ))
            except Exception:
                continue
        filtered.sort(key=lambda r: r.final_score, reverse=True)
        return filtered[:top_k]

    def valid_at(self, moment: datetime | None = None, top_k: int = 50) -> list[RetrievedMemory]:
        """Return nodes whose validity window covers `moment` (defaults to now)."""
        if moment is None:
            moment = datetime.now(timezone.utc)
        moment_str = moment.isoformat()
        nodes = self._graph.all_nodes(limit=1000)
        results = []
        for node in nodes:
            if node.valid_from and node.valid_from > moment_str:
                continue
            if node.valid_until and node.valid_until < moment_str:
                continue
            results.append(RetrievedMemory(
                node=node,
                temporal_score=node.confidence,
                final_score=node.confidence,
            ))
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]

    def frequently_accessed(self, top_k: int = 20) -> list[RetrievedMemory]:
        """Return most frequently accessed nodes."""
        nodes = self._graph.all_nodes(limit=500)
        nodes.sort(key=lambda n: n.access_count, reverse=True)
        return [
            RetrievedMemory(
                node=n,
                temporal_score=min(1.0, n.access_count / 10),
                final_score=min(1.0, n.access_count / 10),
            )
            for n in nodes[:top_k]
        ]

    def recently_accessed_plus_frequent(self, top_k: int = 10) -> list[RetrievedMemory]:
        """Blend recent and frequently-accessed for temporal context."""
        recent = self.recent(top_k=top_k)
        frequent = self.frequently_accessed(top_k=top_k)
        seen = {r.node.node_id for r in recent}
        combined = list(recent)
        for r in frequent:
            if r.node.node_id not in seen:
                combined.append(r)
        combined.sort(key=lambda r: r.final_score, reverse=True)
        return combined[:top_k]


# ────────────────────────────────────────────────────────────────────
# Hybrid Retriever weights
# ────────────────────────────────────────────────────────────────────

@dataclass
class HybridWeights:
    """
    Configurable weights for the 11-factor hybrid ranking score.

    All weights are relative; they are normalised internally so they
    don't need to sum to 1.0.
    """
    semantic:          float = 0.25
    vector:            float = 0.20
    graph_distance:    float = 0.10
    importance:        float = 0.15
    confidence:        float = 0.10
    recency:           float = 0.08
    hierarchy:         float = 0.04
    context_similarity: float = 0.03
    attention:         float = 0.02
    belief_confidence: float = 0.02
    planning_relevance: float = 0.01

    def normalised(self) -> "HybridWeights":
        total = sum([
            self.semantic, self.vector, self.graph_distance,
            self.importance, self.confidence, self.recency,
            self.hierarchy, self.context_similarity, self.attention,
            self.belief_confidence, self.planning_relevance,
        ])
        if total < 1e-9:
            return self
        factor = 1.0 / total
        return HybridWeights(
            semantic=           self.semantic           * factor,
            vector=             self.vector             * factor,
            graph_distance=     self.graph_distance     * factor,
            importance=         self.importance         * factor,
            confidence=         self.confidence         * factor,
            recency=            self.recency            * factor,
            hierarchy=          self.hierarchy          * factor,
            context_similarity= self.context_similarity * factor,
            attention=          self.attention          * factor,
            belief_confidence=  self.belief_confidence  * factor,
            planning_relevance= self.planning_relevance * factor,
        )


# ────────────────────────────────────────────────────────────────────
# Hybrid Retriever
# ────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    11-factor hybrid retrieval engine.

    Combines:
      1.  Semantic similarity  (embedding cosine)
      2.  Vector similarity    (sqlite-vec ANN)
      3.  Graph distance       (BFS hop count, normalised)
      4.  Importance           (node.importance)
      5.  Confidence           (node.confidence)
      6.  Recency              (exponential decay by age)
      7.  Hierarchy            (higher hierarchy → more abstract, may be preferred)
      8.  Context similarity   (query context overlap)
      9.  Attention            (recent access frequency)
      10. Belief confidence    (for BELIEF-type nodes)
      11. Planning relevance   (for PLAN/GOAL nodes)

    Weights are configurable via HybridWeights.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_index: VectorIndex,
        embedding_manager: EmbeddingManager,
        weights: HybridWeights | None = None,
    ) -> None:
        self._semantic  = SemanticRetriever(graph_store, vector_index, embedding_manager)
        self._graph_ret = GraphRetriever(graph_store)
        self._temporal  = TemporalRetriever(graph_store)
        self._graph     = graph_store
        self._emb       = embedding_manager
        self._weights   = (weights or HybridWeights()).normalised()

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        expand_graph: bool = True,
        include_temporal: bool = True,
        memory_types: list[MemoryType] | None = None,
        min_confidence: float = 0.0,
        context_hint: str = "",
    ) -> list[RetrievedMemory]:
        """
        Full hybrid retrieval pipeline.

        Parameters
        ----------
        query:
            Natural-language query.
        top_k:
            Final number of results to return.
        expand_graph:
            Whether to perform graph expansion from semantic hits.
        include_temporal:
            Whether to blend in recent/frequent memories.
        memory_types:
            Restrict to specific MemoryTypes.
        min_confidence:
            Filter out nodes below this confidence.
        context_hint:
            Additional context string for context_similarity scoring.
        """
        t0 = time.perf_counter()
        w = self._weights

        # --- 1. Semantic retrieval ---
        semantic_results = self._semantic.retrieve(
            query, top_k=top_k * 2, memory_types=memory_types)

        # --- 2. Graph expansion ---
        graph_results: list[RetrievedMemory] = []
        if expand_graph and semantic_results:
            seed_ids = [r.node.node_id for r in semantic_results[:5]]
            graph_results = self._graph_ret.expand(
                seed_ids, max_depth=2, max_nodes=top_k)

        # --- 3. Temporal results ---
        temporal_results: list[RetrievedMemory] = []
        if include_temporal:
            temporal_results = self._temporal.recent(top_k=10)

        # --- 4. Merge & score ---
        merged: dict[str, RetrievedMemory] = {}

        def _merge(rm: RetrievedMemory, sem: float = 0, vec: float = 0,
                   graph: float = 0, temporal: float = 0) -> None:
            nid = rm.node.node_id
            if nid in merged:
                existing = merged[nid]
                existing.semantic_score   = max(existing.semantic_score,   sem)
                existing.vector_score     = max(existing.vector_score,     vec)
                existing.graph_score      = max(existing.graph_score,      graph)
                existing.temporal_score   = max(existing.temporal_score,   temporal)
            else:
                rm.semantic_score  = sem
                rm.vector_score    = vec
                rm.graph_score     = graph
                rm.temporal_score  = temporal
                merged[nid] = rm

        for rm in semantic_results:
            _merge(rm, sem=rm.semantic_score, vec=rm.vector_score)
        for rm in graph_results:
            _merge(rm, graph=rm.graph_score)
        for rm in temporal_results:
            _merge(rm, temporal=rm.temporal_score)

        # --- 5. Apply 11-factor ranking ---
        query_vec = self._emb.embed(query)
        context_vec = self._emb.embed(context_hint) if context_hint else None
        final: list[RetrievedMemory] = []

        for rm in merged.values():
            node = rm.node
            if node.confidence < min_confidence:
                continue

            # Factors 4-11 (factors 1-3 already set above)
            importance_score    = node.importance
            confidence_score    = node.confidence
            recency_score       = node.recency_score
            hierarchy_score     = node.hierarchy_level.value / 11.0  # 0→1
            attention_score     = min(1.0, node.access_count / 10.0)
            belief_conf         = node.confidence if node.memory_type == MemoryType.BELIEF else 0.0
            planning_rel        = (1.0 if node.memory_type in
                                   (MemoryType.PLAN, MemoryType.GOAL) else 0.0)
            ctx_score           = 0.0
            if context_vec is not None:
                try:
                    node_vec = self._emb.embed(node.text)
                    ctx_score = max(0.0, self._emb.cosine_similarity(context_vec, node_vec))
                except Exception:
                    pass

            rm.importance_score = importance_score
            rm.final_score = (
                w.semantic           * rm.semantic_score +
                w.vector             * rm.vector_score +
                w.graph_distance     * rm.graph_score +
                w.importance         * importance_score +
                w.confidence         * confidence_score +
                w.recency            * recency_score +
                w.hierarchy          * hierarchy_score +
                w.context_similarity * ctx_score +
                w.attention          * attention_score +
                w.belief_confidence  * belief_conf +
                w.planning_relevance * planning_rel
            )
            final.append(rm)

        final.sort(key=lambda r: r.final_score, reverse=True)
        latency_ms = (time.perf_counter() - t0) * 1000
        log.debug("HybridRetriever: query=%r top=%d latency=%.1fms",
                  query[:40], top_k, latency_ms)
        return final[:top_k]
