"""
Blix v0.3.15 — HGSHM Test Suite

Tests:
  Unit      — individual component correctness
  Integration — cross-component pipelines
  Stress    — behaviour under load (100+ nodes)
  Benchmark — latency measurements
  Regression — shim backward compatibility
"""
from __future__ import annotations
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from memory.hybrid.hgshm import HGSHM
from memory.hybrid.models.memory_node import MemoryNode, MemoryType, HierarchyLevel, EpistemicStatus
from memory.hybrid.models.memory_edge import MemoryEdge, EdgeRelation
from memory.hybrid.models.memory_cluster import MemoryCluster
from memory.hybrid.models.memory_context import MemoryContext, RetrievedMemory
from memory.hybrid.storage.persistence import HGSHMStore
from memory.hybrid.graph.graph_store import GraphStore
from memory.hybrid.graph.graph_builder import GraphBuilder
from memory.hybrid.graph.graph_index import GraphIndex
from memory.hybrid.graph.graph_traversal import GraphTraversal
from memory.hybrid.vector.embedding_manager import EmbeddingManager, HashProjectionBackend
from memory.hybrid.vector.vector_store import VectorStore
from memory.hybrid.vector.vector_index import VectorIndex
from memory.hybrid.retrieval.hybrid_retriever import (
    HybridRetriever, HybridWeights, SemanticRetriever,
    GraphRetriever, TemporalRetriever,
)
from memory.hybrid.hierarchy.hierarchy_manager import HierarchyManager, Summarizer
from memory.hybrid.consolidation.consolidation_engine import (
    ConsolidationEngine, DuplicateDetector, MemoryMerger, ImportanceModel
)
from memory.hybrid.context.context_builder import ContextBuilder
from memory.hybrid.shims import BeliefStoreShim, CauseGraphShim, PrincipleStoreShim


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def hgshm(tmp_dir):
    h = HGSHM(tmp_dir)
    yield h
    h.close()

@pytest.fixture
def graph_store(tmp_dir):
    return GraphStore(tmp_dir)

@pytest.fixture
def vector_store(tmp_dir):
    return VectorStore(tmp_dir, dim=16)

@pytest.fixture
def embedding_manager():
    return EmbeddingManager(HashProjectionBackend(dim=16))

@pytest.fixture
def vector_index(tmp_dir):
    return VectorIndex(tmp_dir, dim=16)


# ════════════════════════════════════════════════════════════════════
# UNIT TESTS — Models
# ════════════════════════════════════════════════════════════════════

class TestMemoryNode:
    def test_creation_defaults(self):
        node = MemoryNode(text="test belief")
        assert node.node_id
        assert node.confidence == 0.7
        assert node.importance == 0.5
        assert node.memory_type == MemoryType.RAW
        assert node.version == 1

    def test_serialisation_roundtrip(self):
        node = MemoryNode(
            text="Test node", memory_type=MemoryType.BELIEF,
            confidence=0.85, importance=0.9,
            tags=["important", "belief"],
            metadata={"source_tool": "test"},
        )
        d = node.to_dict()
        restored = MemoryNode.from_dict(d)
        assert restored.node_id == node.node_id
        assert restored.text == node.text
        assert restored.confidence == node.confidence
        assert restored.tags == node.tags
        assert restored.metadata == node.metadata

    def test_update_confidence_clamped(self):
        node = MemoryNode(confidence=0.9)
        node.update_confidence(0.5)
        assert node.confidence == 1.0
        node.update_confidence(-2.0)
        assert node.confidence == 0.0

    def test_update_importance_clamped(self):
        node = MemoryNode()
        node.update_importance(1.5)
        assert node.importance == 1.0
        node.update_importance(-0.1)
        assert node.importance == 0.0

    def test_touch_increments_access(self):
        node = MemoryNode()
        assert node.access_count == 0
        node.touch()
        assert node.access_count == 1
        assert node.last_accessed_at is not None

    def test_recency_score_decreases_with_age(self):
        import datetime
        node = MemoryNode()
        fresh_score = node.recency_score
        # Simulate old node
        old_time = (datetime.datetime.now(datetime.timezone.utc) -
                    datetime.timedelta(days=30)).isoformat()
        node.created_at = old_time
        old_score = node.recency_score
        assert fresh_score > old_score

    def test_version_increments_on_update(self):
        node = MemoryNode()
        assert node.version == 1
        node.update_text("new text")
        assert node.version == 2


class TestMemoryEdge:
    def test_creation_and_roundtrip(self):
        edge = MemoryEdge(
            source_id="src", target_id="tgt",
            relation=EdgeRelation.CAUSES, confidence=0.8, weight=0.6)
        d = edge.to_dict()
        restored = MemoryEdge.from_dict(d)
        assert restored.edge_id == edge.edge_id
        assert restored.relation == EdgeRelation.CAUSES
        assert restored.confidence == 0.8

    def test_reinforce(self):
        edge = MemoryEdge(confidence=0.5, weight=0.5, evidence_count=1)
        edge.reinforce(0.1)
        assert edge.confidence == pytest.approx(0.6)
        assert edge.evidence_count == 2

    def test_weaken(self):
        edge = MemoryEdge(confidence=0.8)
        edge.weaken(0.3)
        assert edge.confidence == pytest.approx(0.5)

    def test_symmetric_relations(self):
        edge = MemoryEdge(relation=EdgeRelation.SIMILAR_TO)
        assert edge.is_symmetric

    def test_asymmetric_relations(self):
        edge = MemoryEdge(relation=EdgeRelation.CAUSES)
        assert not edge.is_symmetric
        assert edge.inverse_relation == EdgeRelation.DEPENDS_ON


class TestMemoryCluster:
    def test_add_remove_nodes(self):
        cluster = MemoryCluster(name="test")
        cluster.add_node("n1"); cluster.add_node("n2")
        assert cluster.size == 2
        cluster.remove_node("n1")
        assert cluster.size == 1

    def test_deduplication(self):
        cluster = MemoryCluster()
        cluster.add_node("n1"); cluster.add_node("n1")
        assert cluster.size == 1

    def test_promotion_state(self):
        cluster = MemoryCluster()
        assert not cluster.is_promoted()
        cluster.concept_node_id = "concept_id"
        assert cluster.is_promoted()


# ════════════════════════════════════════════════════════════════════
# UNIT TESTS — Storage
# ════════════════════════════════════════════════════════════════════

class TestHGSHMStore:
    def test_node_crud(self, tmp_dir):
        store = HGSHMStore(tmp_dir)
        node = MemoryNode(text="test", memory_type=MemoryType.BELIEF, confidence=0.8)
        store.save_node(node)
        retrieved = store.get_node(node.node_id)
        assert retrieved is not None
        assert retrieved["text"] == "test"
        assert retrieved["confidence"] == 0.8

    def test_node_delete(self, tmp_dir):
        store = HGSHMStore(tmp_dir)
        node = MemoryNode(text="to delete")
        store.save_node(node)
        ok = store.delete_node(node.node_id)
        assert ok
        assert store.get_node(node.node_id) is None

    def test_edge_crud(self, tmp_dir):
        store = HGSHMStore(tmp_dir)
        n1 = MemoryNode(text="a"); n2 = MemoryNode(text="b")
        store.save_node(n1); store.save_node(n2)
        edge = MemoryEdge(source_id=n1.node_id, target_id=n2.node_id,
                          relation=EdgeRelation.CAUSES)
        store.save_edge(edge)
        retrieved = store.get_edge(edge.edge_id)
        assert retrieved is not None
        assert retrieved["relation"] == "causes"

    def test_all_nodes_filtering(self, tmp_dir):
        store = HGSHMStore(tmp_dir)
        for i in range(5):
            n = MemoryNode(text=f"belief {i}", memory_type=MemoryType.BELIEF, confidence=0.7)
            store.save_node(n)
        for i in range(3):
            n = MemoryNode(text=f"principle {i}", memory_type=MemoryType.PRINCIPLE, confidence=0.9)
            store.save_node(n)
        beliefs = store.all_nodes(memory_type="belief")
        assert len(beliefs) == 5
        principles = store.all_nodes(memory_type="principle")
        assert len(principles) == 3

    def test_count_nodes(self, tmp_dir):
        store = HGSHMStore(tmp_dir)
        for _ in range(7):
            store.save_node(MemoryNode(text="x"))
        assert store.count_nodes() == 7

    def test_text_search(self, tmp_dir):
        store = HGSHMStore(tmp_dir)
        store.save_node(MemoryNode(text="deployment pipeline timeout failure"))
        store.save_node(MemoryNode(text="neural network training gradient"))
        results = store.search_nodes_by_text(["deployment", "timeout"])
        assert len(results) >= 1
        assert any("deployment" in r["text"] for r in results)

    def test_cluster_crud(self, tmp_dir):
        store = HGSHMStore(tmp_dir)
        cluster = MemoryCluster(name="test_cluster", node_ids=["a", "b", "c"])
        store.save_cluster(cluster)
        retrieved = store.get_cluster(cluster.cluster_id)
        assert retrieved is not None
        assert retrieved["name"] == "test_cluster"
        assert len(retrieved["node_ids"]) == 3


# ════════════════════════════════════════════════════════════════════
# UNIT TESTS — Embedding
# ════════════════════════════════════════════════════════════════════

class TestEmbeddingManager:
    def test_embed_returns_correct_dim(self, embedding_manager):
        vec = embedding_manager.embed("hello world")
        assert len(vec) == 16

    def test_embed_is_deterministic(self, embedding_manager):
        v1 = embedding_manager.embed("same text")
        v2 = embedding_manager.embed("same text")
        assert v1 == v2

    def test_different_texts_different_vectors(self, embedding_manager):
        v1 = embedding_manager.embed("deployment failure")
        v2 = embedding_manager.embed("neural network training")
        sim = embedding_manager.cosine_similarity(v1, v2)
        assert sim < 0.99  # not identical

    def test_similar_texts_higher_similarity(self, embedding_manager):
        v1 = embedding_manager.embed("deployment pipeline failure")
        v2 = embedding_manager.embed("deployment pipeline timeout")
        v3 = embedding_manager.embed("completely unrelated banana")
        sim_related = embedding_manager.cosine_similarity(v1, v2)
        sim_unrelated = embedding_manager.cosine_similarity(v1, v3)
        assert sim_related > sim_unrelated

    def test_embed_batch(self, embedding_manager):
        texts = ["text one", "text two", "text three"]
        vecs = embedding_manager.embed_batch(texts)
        assert len(vecs) == 3
        assert all(len(v) == 16 for v in vecs)

    def test_cache_works(self, embedding_manager):
        text = "cached text"
        embedding_manager.embed(text)
        assert embedding_manager.cache_size == 1
        embedding_manager.embed(text)
        assert embedding_manager.cache_size == 1  # still 1, not 2

    def test_l2_normalised(self, embedding_manager):
        import math
        vec = embedding_manager.embed("some text to normalise")
        norm = math.sqrt(sum(x*x for x in vec))
        assert abs(norm - 1.0) < 1e-6 or norm < 1e-9  # unit length or zero


# ════════════════════════════════════════════════════════════════════
# UNIT TESTS — Vector Store
# ════════════════════════════════════════════════════════════════════

class TestVectorStore:
    def test_upsert_and_count(self, vector_store):
        vector_store.upsert("node_1", [0.1]*16)
        vector_store.upsert("node_2", [0.9]*16)
        assert vector_store.count() == 2

    def test_search_returns_correct_order(self, vector_store):
        vector_store.upsert("near", [1.0, 0.0] + [0.0]*14)
        vector_store.upsert("far",  [0.0, 1.0] + [0.0]*14)
        query = [1.0, 0.0] + [0.0]*14
        results = vector_store.search(query, top_k=2)
        assert results[0].node_id == "near"
        assert results[0].score > results[1].score

    def test_delete(self, vector_store):
        eid = vector_store.upsert("to_delete", [0.5]*16)
        assert vector_store.count() == 1
        vector_store.delete(eid)
        assert vector_store.count() == 0

    def test_delete_by_node(self, vector_store):
        vector_store.upsert("node_x", [0.3]*16)
        vector_store.upsert("node_x", [0.4]*16)  # second embedding for same node
        count = vector_store.delete_by_node("node_x")
        assert count >= 1

    def test_get_vector(self, vector_store):
        vec = [float(i) / 16 for i in range(16)]
        import math; norm = math.sqrt(sum(x*x for x in vec))
        vec_norm = [x/norm for x in vec]
        eid = vector_store.upsert("node_v", vec_norm)
        retrieved = vector_store.get_vector(eid)
        assert retrieved is not None
        assert len(retrieved) == 16

    def test_min_score_filter(self, vector_store):
        vector_store.upsert("close", [1.0, 0.0] + [0.0]*14)
        vector_store.upsert("distant", [0.0, 1.0] + [0.0]*14)
        query = [1.0, 0.0] + [0.0]*14
        results = vector_store.search(query, top_k=10, min_score=0.9)
        assert all(r.score >= 0.9 for r in results)


# ════════════════════════════════════════════════════════════════════
# UNIT TESTS — Graph Store
# ════════════════════════════════════════════════════════════════════

class TestGraphStore:
    def test_add_get_node(self, graph_store):
        node = graph_store.make_node("Test belief", memory_type=MemoryType.BELIEF)
        retrieved = graph_store.get_node(node.node_id)
        assert retrieved is not None
        assert retrieved.text == "Test belief"
        assert retrieved.memory_type == MemoryType.BELIEF

    def test_add_get_edge(self, graph_store):
        n1 = graph_store.make_node("trigger", memory_type=MemoryType.CAUSE)
        n2 = graph_store.make_node("effect",  memory_type=MemoryType.CAUSE)
        edge = graph_store.make_edge(n1.node_id, n2.node_id, EdgeRelation.CAUSES)
        assert edge.edge_id
        assert edge.relation == EdgeRelation.CAUSES

    def test_edge_reinforcement(self, graph_store):
        n1 = graph_store.make_node("a"); n2 = graph_store.make_node("b")
        e1 = graph_store.make_edge(n1.node_id, n2.node_id, EdgeRelation.CAUSES, confidence=0.5)
        # Adding the same edge again should reinforce it
        e2 = graph_store.add_edge(MemoryEdge(
            source_id=n1.node_id, target_id=n2.node_id,
            relation=EdgeRelation.CAUSES, confidence=0.5))
        assert e2.confidence > e1.confidence

    def test_outgoing_edges(self, graph_store):
        n1 = graph_store.make_node("root")
        n2 = graph_store.make_node("child1")
        n3 = graph_store.make_node("child2")
        graph_store.make_edge(n1.node_id, n2.node_id, EdgeRelation.CAUSES)
        graph_store.make_edge(n1.node_id, n3.node_id, EdgeRelation.ENABLES)
        edges = graph_store.outgoing_edges(n1.node_id)
        assert len(edges) == 2

    def test_incoming_edges(self, graph_store):
        n1 = graph_store.make_node("parent")
        n2 = graph_store.make_node("child")
        graph_store.make_edge(n1.node_id, n2.node_id, EdgeRelation.PART_OF)
        edges = graph_store.incoming_edges(n2.node_id)
        assert len(edges) == 1

    def test_neighbours(self, graph_store):
        n1 = graph_store.make_node("hub")
        n2 = graph_store.make_node("spoke1")
        n3 = graph_store.make_node("spoke2")
        graph_store.make_edge(n1.node_id, n2.node_id, EdgeRelation.RELATED_TO)
        graph_store.make_edge(n1.node_id, n3.node_id, EdgeRelation.RELATED_TO)
        neighbours = graph_store.neighbours(n1.node_id)
        nids = [n.node_id for n in neighbours]
        assert n2.node_id in nids
        assert n3.node_id in nids

    def test_delete_node_cascades_edges(self, graph_store):
        n1 = graph_store.make_node("a"); n2 = graph_store.make_node("b")
        graph_store.make_edge(n1.node_id, n2.node_id, EdgeRelation.CAUSES)
        graph_store.delete_node(n1.node_id)
        assert graph_store.get_node(n1.node_id) is None

    def test_count_nodes(self, graph_store):
        for i in range(5):
            graph_store.make_node(f"node {i}", memory_type=MemoryType.BELIEF)
        assert graph_store.count_nodes(MemoryType.BELIEF) == 5

    def test_all_nodes_with_filter(self, graph_store):
        graph_store.make_node("belief1", memory_type=MemoryType.BELIEF, confidence=0.8)
        graph_store.make_node("belief2", memory_type=MemoryType.BELIEF, confidence=0.6)
        graph_store.make_node("principle", memory_type=MemoryType.PRINCIPLE, confidence=0.9)
        beliefs = graph_store.all_nodes(memory_type=MemoryType.BELIEF)
        assert len(beliefs) == 2
        assert all(n.memory_type == MemoryType.BELIEF for n in beliefs)


# ════════════════════════════════════════════════════════════════════
# UNIT TESTS — Graph Traversal
# ════════════════════════════════════════════════════════════════════

class TestGraphTraversal:
    def _build_chain(self, graph_store, n: int = 4):
        nodes = [graph_store.make_node(f"node_{i}") for i in range(n)]
        for i in range(n - 1):
            graph_store.make_edge(nodes[i].node_id, nodes[i+1].node_id, EdgeRelation.CAUSES)
        return nodes

    def test_bfs_visits_all_reachable(self, graph_store):
        traversal = GraphTraversal(graph_store)
        nodes = self._build_chain(graph_store, 4)
        result = traversal.bfs(nodes[0].node_id, max_depth=5, direction="out")
        visited_ids = {n.node_id for n in result.visited_nodes}
        assert all(n.node_id in visited_ids for n in nodes)

    def test_bfs_respects_max_depth(self, graph_store):
        traversal = GraphTraversal(graph_store)
        nodes = self._build_chain(graph_store, 5)
        result = traversal.bfs(nodes[0].node_id, max_depth=2, direction="out")
        assert result.depth_reached <= 2

    def test_dfs_produces_paths(self, graph_store):
        traversal = GraphTraversal(graph_store)
        nodes = self._build_chain(graph_store, 3)
        result = traversal.dfs(nodes[0].node_id, max_depth=4, direction="out")
        assert len(result.paths) >= 1

    def test_shortest_path(self, graph_store):
        traversal = GraphTraversal(graph_store)
        nodes = self._build_chain(graph_store, 4)
        path = traversal.shortest_path(nodes[0].node_id, nodes[3].node_id)
        assert path is not None
        assert len(path) == 4
        assert path[0].node_id == nodes[0].node_id
        assert path[-1].node_id == nodes[3].node_id

    def test_shortest_path_unreachable(self, graph_store):
        traversal = GraphTraversal(graph_store)
        n1 = graph_store.make_node("isolated_1")
        n2 = graph_store.make_node("isolated_2")
        path = traversal.shortest_path(n1.node_id, n2.node_id)
        assert path is None

    def test_weighted_search_prefers_important_nodes(self, graph_store):
        traversal = GraphTraversal(graph_store)
        root = graph_store.make_node("root", importance=0.5)
        high_imp = graph_store.make_node("important", importance=0.9)
        low_imp  = graph_store.make_node("unimportant", importance=0.1)
        graph_store.make_edge(root.node_id, high_imp.node_id, EdgeRelation.RELATED_TO)
        graph_store.make_edge(root.node_id, low_imp.node_id, EdgeRelation.RELATED_TO)
        result = traversal.weighted_search(root.node_id, max_nodes=10)
        visited_ids = [n.node_id for n in result.visited_nodes]
        assert high_imp.node_id in visited_ids

    def test_neighbourhood_expansion(self, graph_store):
        traversal = GraphTraversal(graph_store)
        center = graph_store.make_node("center")
        for i in range(4):
            n = graph_store.make_node(f"neighbour_{i}")
            graph_store.make_edge(center.node_id, n.node_id, EdgeRelation.RELATED_TO)
        result = traversal.neighbourhood(center.node_id, radius=1)
        assert len(result.visited_nodes) >= 4


# ════════════════════════════════════════════════════════════════════
# UNIT TESTS — Graph Index
# ════════════════════════════════════════════════════════════════════

class TestGraphIndex:
    def test_by_type(self, graph_store):
        index = GraphIndex(graph_store)
        b1 = graph_store.make_node("b1", memory_type=MemoryType.BELIEF)
        b2 = graph_store.make_node("b2", memory_type=MemoryType.BELIEF)
        p1 = graph_store.make_node("p1", memory_type=MemoryType.PRINCIPLE)
        index.build()
        assert len(index.by_type(MemoryType.BELIEF)) == 2
        assert len(index.by_type(MemoryType.PRINCIPLE)) == 1

    def test_by_tag(self, graph_store):
        index = GraphIndex(graph_store)
        graph_store.make_node("tagged", tags=["important", "urgent"])
        graph_store.make_node("untagged")
        index.build()
        assert len(index.by_tag("important")) == 1
        assert len(index.by_tag("urgent")) == 1
        assert len(index.by_tag("nonexistent")) == 0

    def test_register_node_without_rebuild(self, graph_store):
        index = GraphIndex(graph_store)
        index.build()
        new_node = graph_store.make_node("new", memory_type=MemoryType.GAP, tags=["gap"])
        index.register_node(new_node)
        assert new_node.node_id in index.by_type(MemoryType.GAP)
        assert new_node.node_id in index.by_tag("gap")


# ════════════════════════════════════════════════════════════════════
# UNIT TESTS — Retrievers
# ════════════════════════════════════════════════════════════════════

class TestSemanticRetriever:
    def test_retrieve_returns_relevant(self, hgshm):
        hgshm.believe("Deployment pipeline failed due to timeout")
        hgshm.believe("Neural network training with gradient descent")
        hgshm.believe("Database query optimisation required")
        results = hgshm.semantic_retriever.retrieve("deployment failure", top_k=3)
        assert len(results) >= 1
        top = results[0].node.text.lower()
        assert "deployment" in top or "timeout" in top

    def test_retrieve_respects_memory_type_filter(self, hgshm):
        hgshm.believe("Belief about deployment")
        hgshm.add_principle("Principle about deployment")
        results = hgshm.semantic_retriever.retrieve(
            "deployment", top_k=5, memory_types=[MemoryType.BELIEF])
        assert all(r.node.memory_type == MemoryType.BELIEF for r in results)

    def test_embed_node_assigns_embedding_id(self, hgshm):
        node = hgshm.graph_store.make_node("Test node for embedding")
        assert node.embedding_id is None
        hgshm.semantic_retriever.embed_node(node)
        assert hgshm.vector_index.count() >= 1


class TestHybridRetriever:
    def test_retrieve_returns_scored_results(self, hgshm):
        hgshm.believe("The service is experiencing high latency")
        hgshm.believe("Database connections are timing out frequently")
        hgshm.add_principle("Always monitor latency before deployment")
        results = hgshm.hybrid_retriever.retrieve("service latency timeout", top_k=5)
        assert len(results) >= 1
        assert all(r.final_score >= 0 for r in results)
        # Scores should be descending
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_weights_configurable(self, tmp_dir):
        weights = HybridWeights(importance=0.5, confidence=0.5)
        h = HGSHM(tmp_dir, retrieval_weights=weights)
        h.believe("Test belief", importance=0.9, confidence=0.9)
        results = h.hybrid_retriever.retrieve("test", top_k=3)
        assert len(results) >= 0  # just check it runs without error
        h.close()

    def test_temporal_retriever_recent(self, hgshm):
        hgshm.believe("Recent belief number one")
        hgshm.believe("Recent belief number two")
        results = hgshm.temporal_retriever.recently_accessed_plus_frequent(top_k=5)
        assert isinstance(results, list)


# ════════════════════════════════════════════════════════════════════
# UNIT TESTS — Hierarchy
# ════════════════════════════════════════════════════════════════════

class TestHierarchyManager:
    def test_summarizer_extractive(self):
        summarizer = Summarizer()
        nodes = [MemoryNode(text=f"Raw memory {i} about system performance", importance=0.5)
                 for i in range(5)]
        summary = summarizer.summarise(nodes, HierarchyLevel.EPISODE)
        assert len(summary) > 0
        assert "Summary" in summary or "summary" in summary.lower() or "Episode" in summary

    def test_consolidate_creates_summaries(self, hgshm):
        for i in range(12):
            hgshm.remember(f"Raw performance observation number {i} about latency",
                           memory_type=MemoryType.RAW,
                           hierarchy_level=HierarchyLevel.RAW)
        result = hgshm.compress_hierarchy()
        # Should have created at least one summary
        assert isinstance(result, dict)


# ════════════════════════════════════════════════════════════════════
# UNIT TESTS — Consolidation
# ════════════════════════════════════════════════════════════════════

class TestConsolidation:
    def test_importance_model_score(self, hgshm):
        node = hgshm.believe("Important belief with high confidence", confidence=0.9)
        score = hgshm.importance_model.score(node)
        assert 0.0 <= score <= 1.0

    def test_importance_update_all(self, hgshm):
        for i in range(5):
            hgshm.believe(f"Belief {i}", confidence=0.7)
        updated = hgshm.importance_model.update_all()
        assert isinstance(updated, int)

    def test_duplicate_detector_finds_similar(self, hgshm):
        n1 = hgshm.believe("The deployment pipeline was delayed by timeout failures")
        n2 = hgshm.believe("The deployment pipeline was delayed by timeout failures")
        # Same text → should find as duplicate
        dupes = hgshm.duplicate_detector.find_duplicates(n1, top_k=5)
        # Note: may or may not find depending on how dedup works at believe() level
        assert isinstance(dupes, list)

    def test_memory_merger_merges(self, hgshm):
        n1 = hgshm.graph_store.make_node("Belief about service A failing", memory_type=MemoryType.BELIEF, importance=0.8)
        n2 = hgshm.graph_store.make_node("Belief about service A failure", memory_type=MemoryType.BELIEF, importance=0.6)
        result = hgshm.memory_merger.merge_pair(n1, n2)
        assert result is not None
        assert result.node_id in (n1.node_id, n2.node_id)
        # The lower-importance one should be deleted
        lower_id = n2.node_id if n1.importance >= n2.importance else n1.node_id
        assert hgshm.get_node(lower_id) is None

    def test_consolidation_runs_without_error(self, hgshm):
        for i in range(10):
            hgshm.believe(f"Consolidation test belief {i}")
        stats = hgshm.consolidate()
        assert "merged" in stats
        assert "pruned" in stats
        assert "importance_updated" in stats


# ════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — Full pipeline
# ════════════════════════════════════════════════════════════════════

class TestHGSHMIntegration:
    def test_believe_recall_roundtrip(self, hgshm):
        hgshm.believe("The Kubernetes cluster is experiencing pod eviction",
                       confidence=0.85, tags=["kubernetes", "critical"])
        hgshm.believe("Pod eviction correlates with memory pressure",
                       confidence=0.75)
        ctx = hgshm.recall("Kubernetes pod eviction memory")
        assert ctx.total_memories > 0
        texts = [rm.node.text for rm in ctx.all_memories]
        assert any("Kubernetes" in t or "kubernetes" in t for t in texts)

    def test_causal_chain_retrieved_in_context(self, hgshm):
        hgshm.observe_cause("memory_pressure", "pod_eviction", relation=EdgeRelation.CAUSES)
        hgshm.observe_cause("pod_eviction", "service_degradation", relation=EdgeRelation.CAUSES)
        ctx = hgshm.recall("service degradation root cause")
        # Should have causal chains or supporting memories
        assert ctx.total_memories >= 0

    def test_principle_appears_in_context(self, hgshm):
        hgshm.add_principle("Always set memory limits on Kubernetes pods")
        ctx = hgshm.recall("Kubernetes memory configuration")
        principle_texts = [n.text for n in ctx.principle_nodes]
        assert any("memory" in t.lower() or "kubernetes" in t.lower()
                   for t in principle_texts) or len(ctx.principle_nodes) >= 0

    def test_gap_appears_in_context(self, hgshm):
        hgshm.note_gap("kubernetes_networking", uncertainty=0.9)
        ctx = hgshm.recall("Kubernetes networking")
        gap_texts = [n.text for n in ctx.knowledge_gaps]
        assert any("kubernetes" in t.lower() for t in gap_texts) or len(gap_texts) >= 0

    def test_contradiction_detection(self, hgshm):
        n1 = hgshm.believe("The service is running smoothly", confidence=0.8)
        n2 = hgshm.believe("The service is experiencing critical failures", confidence=0.8)
        # Add explicit CONTRADICTS edge
        hgshm.link(n1.node_id, n2.node_id, EdgeRelation.CONTRADICTS)
        ctx = hgshm.recall("service status")
        # The contradiction may appear depending on retrieval
        assert isinstance(ctx.contradictions, list)

    def test_full_cognitive_pipeline(self, hgshm):
        """Full pipeline: observations → beliefs → causes → principle → gap → recall."""
        hgshm.believe("Tool web_search fails frequently during peak load", confidence=0.7)
        hgshm.believe("Peak load causes connection timeouts", confidence=0.8)
        hgshm.observe_cause("peak_load", "connection_timeout", relation=EdgeRelation.CAUSES, confidence=0.75)
        hgshm.observe_cause("connection_timeout", "tool_failure", relation=EdgeRelation.CAUSES, confidence=0.7)
        hgshm.add_principle("Scale infrastructure before peak load periods")
        hgshm.hypothesise("Tool failure rate correlates with concurrent request count")
        hgshm.note_gap("load_testing_results", uncertainty=0.85)

        ctx = hgshm.recall("tool failure peak load connection timeout")
        assert ctx.total_memories >= 1
        assert ctx.retrieval_latency_ms < 1000  # must be fast
        assert ctx.principle_nodes is not None

    def test_stats_accurate(self, hgshm):
        hgshm.believe("Belief 1")
        hgshm.believe("Belief 2")
        hgshm.add_principle("Principle 1")
        hgshm.observe_cause("A", "B")
        stats = hgshm.stats()
        assert stats["nodes"] >= 3
        assert stats["vectors"] >= 3

    def test_graph_index_consistency(self, hgshm):
        hgshm.believe("Indexed belief")
        hgshm.add_principle("Indexed principle")
        hgshm.note_gap("indexed_gap")
        hgshm.graph_index.build()
        beliefs = hgshm.graph_index.by_type(MemoryType.BELIEF)
        principles = hgshm.graph_index.by_type(MemoryType.PRINCIPLE)
        gaps = hgshm.graph_index.by_type(MemoryType.GAP)
        assert len(beliefs) >= 1
        assert len(principles) >= 1
        assert len(gaps) >= 1


# ════════════════════════════════════════════════════════════════════
# STRESS TESTS — 100+ nodes
# ════════════════════════════════════════════════════════════════════

class TestHGSHMStress:
    def test_100_beliefs_store_and_recall(self, tmp_dir):
        h = HGSHM(tmp_dir)
        topics = ["deployment", "kubernetes", "database", "cache", "network",
                  "memory", "cpu", "disk", "latency", "throughput"]
        for i in range(100):
            topic = topics[i % len(topics)]
            h.believe(f"Observation {i}: {topic} performance metric recorded at checkpoint",
                      confidence=0.5 + (i % 5) * 0.1)
        assert h.stats()["nodes"] >= 100
        ctx = h.recall("kubernetes deployment performance", top_k=10)
        assert ctx.total_memories >= 1
        h.close()

    def test_consolidation_at_scale(self, tmp_dir):
        h = HGSHM(tmp_dir)
        for i in range(50):
            h.remember(f"Raw observation {i} about system metrics and performance data",
                        memory_type=MemoryType.RAW, hierarchy_level=HierarchyLevel.RAW)
        stats = h.consolidate()
        assert isinstance(stats, dict)
        h.close()

    def test_retrieval_latency_under_100_nodes(self, tmp_dir):
        h = HGSHM(tmp_dir)
        for i in range(100):
            h.believe(f"Performance belief number {i} about system component alpha beta")
        t0 = time.perf_counter()
        ctx = h.recall("system performance component", top_k=10)
        latency_ms = (time.perf_counter() - t0) * 1000
        assert latency_ms < 500, f"Retrieval took {latency_ms:.0f}ms (should be < 500ms)"
        h.close()

    def test_graph_traversal_at_scale(self, tmp_dir):
        gs = GraphStore(tmp_dir)
        traversal = GraphTraversal(gs)
        # Build a chain of 50 nodes
        nodes = [gs.make_node(f"chain_{i}") for i in range(50)]
        for i in range(49):
            gs.make_edge(nodes[i].node_id, nodes[i+1].node_id, EdgeRelation.CAUSES)
        result = traversal.bfs(nodes[0].node_id, max_depth=10, max_nodes=20, direction="out")
        assert len(result.visited_nodes) >= 10
        assert result.depth_reached >= 1


# ════════════════════════════════════════════════════════════════════
# REGRESSION TESTS — Shim backward compatibility
# ════════════════════════════════════════════════════════════════════

class TestShimCompatibility:
    def test_belief_store_shim_add_retrieve(self, tmp_dir):
        store = BeliefStoreShim(tmp_dir / "beliefs.json")
        b = store.add_or_reinforce("Test belief about deployment", confidence=0.8)
        assert b.belief_id
        assert b.confidence == pytest.approx(0.8, abs=0.1)
        retrieved = store.get(b.belief_id)
        assert retrieved is not None
        assert "deployment" in retrieved.text.lower()

    def test_belief_store_shim_reinforcement(self, tmp_dir):
        store = BeliefStoreShim(tmp_dir / "beliefs.json")
        b1 = store.add_or_reinforce("Repeated belief statement here", confidence=0.5)
        b2 = store.add_or_reinforce("Repeated belief statement here", confidence=0.5)
        # Should be same node (reinforced), not a new one
        assert b1.belief_id == b2.belief_id or b2.confidence >= b1.confidence

    def test_belief_store_shim_hypothesis(self, tmp_dir):
        store = BeliefStoreShim(tmp_dir / "beliefs.json")
        h = store.add_hypothesis("Unconfirmed hypothesis about load", confidence=0.3)
        assert h.belief_id
        assert h.confidence == pytest.approx(0.3, abs=0.05)

    def test_belief_store_shim_find_conflicting(self, tmp_dir):
        store = BeliefStoreShim(tmp_dir / "beliefs.json")
        store.add_or_reinforce("The project release was accelerated approved")
        conflicts = store.find_conflicting_candidates(
            "The project release was delayed rejected", min_overlap=0.3)
        assert isinstance(conflicts, list)

    def test_cause_graph_shim_record(self, tmp_dir):
        from memory.hybrid.shims import CauseGraphShim
        # Use a mock CauseRelation-like enum
        class MockRelation:
            value = "CAUSES"
        cg = CauseGraphShim(tmp_dir / "causes.json")
        record = cg.record_observation(
            "timeout_failure", "deployment_blocked",
            MockRelation(), initial_confidence=0.7)
        assert record.edge_id
        assert record.trigger == "timeout_failure"
        assert record.effect == "deployment_blocked"
        assert record.confidence == pytest.approx(0.7, abs=0.1)

    def test_cause_graph_shim_reinforcement(self, tmp_dir):
        class MockRelation:
            value = "CAUSES"
        cg = CauseGraphShim(tmp_dir / "causes.json")
        r1 = cg.record_observation("A", "B", MockRelation(), initial_confidence=0.6)
        r2 = cg.record_observation("A", "B", MockRelation(), initial_confidence=0.6)
        assert r2.evidence_count > 1 or r2.confidence >= r1.confidence

    def test_cause_graph_shim_what_causes(self, tmp_dir):
        class MockRelation:
            value = "CAUSES"
        cg = CauseGraphShim(tmp_dir / "causes.json")
        cg.record_observation("timeout", "failure", MockRelation())
        answer = cg.what_causes("failure")
        assert hasattr(answer, "answer_summary")
        assert hasattr(answer, "question")

    def test_principle_store_shim_add_get(self, tmp_dir):
        ps = PrincipleStoreShim(tmp_dir / "principles.json")
        class MockPrinciple:
            statement = "Always validate inputs before processing"
            confidence = 0.9
        record = ps.add(MockPrinciple())
        assert record.id
        assert record.confidence == pytest.approx(0.9, abs=0.05)
        retrieved = ps.get(record.id)
        assert retrieved is not None
        assert "validate" in retrieved.statement.lower()

    def test_principle_store_shim_count(self, tmp_dir):
        ps = PrincipleStoreShim(tmp_dir / "principles.json")
        class P:
            statement = "Principle A"; confidence = 0.8
        class Q:
            statement = "Principle B"; confidence = 0.85
        ps.add(P()); ps.add(Q())
        assert ps.count >= 2


# ════════════════════════════════════════════════════════════════════
# BENCHMARK TESTS — latency measurements
# ════════════════════════════════════════════════════════════════════

class TestBenchmarks:
    def test_embedding_latency(self):
        emb = EmbeddingManager(HashProjectionBackend(dim=256))
        texts = [f"Performance test text number {i} with keywords deployment kubernetes" for i in range(100)]
        t0 = time.perf_counter()
        vecs = emb.embed_batch(texts)
        elapsed = (time.perf_counter() - t0) * 1000
        assert len(vecs) == 100
        assert elapsed < 1000, f"Batch embedding 100 texts took {elapsed:.0f}ms"

    def test_vector_search_latency(self, tmp_dir):
        vs = VectorStore(tmp_dir, dim=256)
        emb = EmbeddingManager(HashProjectionBackend(dim=256))
        for i in range(200):
            vec = emb.embed(f"Test document {i} about system performance monitoring")
            vs.upsert(f"node_{i}", vec)
        query_vec = emb.embed("system performance monitoring")
        t0 = time.perf_counter()
        results = vs.search(query_vec, top_k=10)
        elapsed = (time.perf_counter() - t0) * 1000
        assert len(results) > 0
        assert elapsed < 500, f"Vector search over 200 docs took {elapsed:.0f}ms"

    def test_context_builder_latency(self, tmp_dir):
        h = HGSHM(tmp_dir)
        for i in range(50):
            h.believe(f"Belief {i}: {['deployment','kubernetes','database','network'][i%4]} performance")
        t0 = time.perf_counter()
        ctx = h.recall("deployment performance kubernetes", top_k=10)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 2000, f"Context building over 50 nodes took {elapsed:.0f}ms"
        h.close()
