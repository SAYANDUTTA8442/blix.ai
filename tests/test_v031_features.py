"""
Tests for Blix v0.3.1 modules — addressing the 14-issue review.

Covers:
1.  weight_learner          (Issue 1  — learnable retrieval weights)
2.  memory_lifecycle         (Issue 2  — forgetting mechanism)
3.  semantic_clusters        (Issue 3  — topic-based hierarchy)
4&5 graph_reasoner           (Issue 4  — graph reasoning, Issue 5 — contradictions)
6&7 fact_verifier            (Issue 6  — confidence propagation, Issue 7 — verification)
8&13 retrieval_postprocessors(Issue 8  — project bias, Issue 13 — MMR diversity)
9.  memory_types             (Issue 9  — episodic/semantic/procedural)
10&14 evaluation.research    (Issue 10 — extended metrics, Issue 14 — hypotheses)
11. background_processor     (Issue 11 — durable overflow, no dropped tasks)
12. user_namespace            (Issue 12 — multi-user namespacing)

Python 3.10 compatible — fully offline.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from core.background_processor import BackgroundProcessor, ProcessorJob
from core.fact_verifier import (
    ConfidencePropagator, FactVerifier, VerificationStatus, VerifiedFact,
)
from core.graph_reasoner import (
    Contradiction, ContradictionDetector, GraphPath, GraphReasoner,
)
from core.memory_graph import EntityKind, GraphEdge, GraphNode, MemoryGraph, RelationKind
from core.memory_lifecycle import (
    ForgettingPolicy, LifecycleState, MemoryLifecycleManager,
)
from core.memory_scorer import MemoryScorer, ScoringWeights
from core.memory_types import MemoryType, MemoryTypeClassifier, TypeAwareRetriever
from core.retrieval_postprocessors import MMRReranker, ProjectBiasedRetriever
from core.semantic_clusters import SemanticCluster, SemanticClusterIndex
from core.user_namespace import UserNamespace, UserRegistry, _slugify
from core.weight_learner import (
    BayesianWeightOptimizer, PairwiseWeightLearner, RetrievalFeedback,
)
from evaluation import EvalCase, EvalDataset, MemoryEvaluator
from evaluation.research import (
    ExtendedMemoryEvaluator, HypothesisRegistry, HypothesisStatus, ResearchHypothesis,
)
from schemas.memory_entry import MemoryEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(days_ago: float = 0.0) -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago)


def _entry(
    id: int,
    input: str = "q",
    output: str = "a",
    importance: Optional[float] = None,
    topics: Optional[list[str]] = None,
    facts: Optional[list[str]] = None,
    days_ago: float = 0.0,
) -> MemoryEntry:
    return MemoryEntry(
        id=id, input=input, output=output,
        timestamp=_ts(days_ago),
        importance=importance,
        topics=topics or [],
        extracted_facts=facts or [],
    )


# ===========================================================================
# Issue 1 — Learnable retrieval weights
# ===========================================================================


class TestWeightLearner:
    def test_random_weights_sum_to_one(self) -> None:
        from core.weight_learner import _random_weights
        w = _random_weights(seed=1)
        total = w.relevance + w.importance + w.recency + w.frequency
        assert abs(total - 1.0) < 1e-6

    def test_sample_simplex_respects_min(self) -> None:
        from core.weight_learner import _sample_simplex
        w = _sample_simplex(min_w=0.05, seed=3)
        for v in (w.relevance, w.importance, w.recency, w.frequency):
            assert v >= 0.05 - 1e-9

    def test_record_feedback(self, tmp_path: Path) -> None:
        learner = PairwiseWeightLearner(tmp_path / "weights.json")
        learner.record_feedback(RetrievalFeedback(query="x", winner_id=1, loser_id=2))
        assert len(learner._feedback) == 1

    def test_fit_insufficient_feedback_returns_current(self, tmp_path: Path) -> None:
        scorer = MemoryScorer()
        learner = PairwiseWeightLearner(tmp_path / "weights.json")
        learner.record_feedback(RetrievalFeedback(query="x", winner_id=1, loser_id=2))
        result = learner.fit(scorer, [])
        assert result is scorer._w

    def test_fit_with_enough_feedback(self, tmp_path: Path) -> None:
        scorer = MemoryScorer()
        learner = PairwiseWeightLearner(tmp_path / "weights.json", n_restarts=2)
        entries = [
            {"id": 1, "relevance": 0.9, "importance": 0.9, "timestamp": _ts(0)},
            {"id": 2, "relevance": 0.1, "importance": 0.1, "timestamp": _ts(60)},
            {"id": 3, "relevance": 0.5, "importance": 0.5, "timestamp": _ts(10)},
        ]
        for _ in range(4):
            learner.record_feedback(RetrievalFeedback(query="x", winner_id=1, loser_id=2))
        w = learner.fit(scorer, entries)
        assert isinstance(w, ScoringWeights)
        assert w.validate_sum()

    def test_training_log_persisted(self, tmp_path: Path) -> None:
        scorer = MemoryScorer()
        learner = PairwiseWeightLearner(tmp_path / "weights.json", n_restarts=1)
        entries = [
            {"id": 1, "relevance": 0.1, "importance": 0.1, "timestamp": _ts(60)},
            {"id": 2, "relevance": 0.9, "importance": 0.9, "timestamp": _ts(0)},
        ]
        # winner (1) is intentionally the *worse* entry, forcing nonzero loss
        for _ in range(3):
            learner.record_feedback(RetrievalFeedback(query="x", winner_id=1, loser_id=2))
        learner.fit(scorer, entries)
        assert (tmp_path / "weights.json").exists()
        with (tmp_path / "weights.json").open() as fh:
            data = json.load(fh)
        assert data["iterations"] == 1

    def test_bayesian_optimizer_improves_or_equal(self) -> None:
        scorer = MemoryScorer()
        opt = BayesianWeightOptimizer(n_trials=10)

        def precision_fn(w: ScoringWeights) -> float:
            # Reward weights that favour relevance
            return w.relevance

        best = opt.optimize(scorer, precision_fn)
        assert isinstance(best, ScoringWeights)
        assert best.relevance >= 0.05


# ===========================================================================
# Issue 2 — Memory lifecycle / forgetting
# ===========================================================================


class TestMemoryLifecycle:
    @pytest.fixture
    def lm(self, tmp_path: Path) -> MemoryLifecycleManager:
        return MemoryLifecycleManager(tmp_path / "lifecycle.json")

    def test_initial_state_active(self, lm: MemoryLifecycleManager) -> None:
        assert lm.get_state(1) == LifecycleState.ACTIVE

    def test_record_access(self, lm: MemoryLifecycleManager) -> None:
        lm.record_access(1)
        lm.record_access(1)
        assert lm.get_access_count(1) == 2

    def test_compress(self, lm: MemoryLifecycleManager) -> None:
        lm.compress(1, "summary text")
        assert lm.get_state(1) == LifecycleState.COMPRESSED

    def test_archive(self, lm: MemoryLifecycleManager) -> None:
        lm.compress(1, "s")
        lm.archive(1)
        assert lm.get_state(1) == LifecycleState.ARCHIVED

    def test_archive_from_active_directly(self, lm: MemoryLifecycleManager) -> None:
        lm.archive(1)
        assert lm.get_state(1) == LifecycleState.ARCHIVED

    def test_delete(self, lm: MemoryLifecycleManager) -> None:
        lm.delete(1)
        assert lm.get_state(1) == LifecycleState.DELETED

    def test_restore(self, lm: MemoryLifecycleManager) -> None:
        lm.archive(1)
        lm.restore(1)
        assert lm.get_state(1) == LifecycleState.ACTIVE

    def test_gc_compresses_old_unaccessed(self, tmp_path: Path) -> None:
        policy = ForgettingPolicy(compress_after_days=10.0, compress_min_access=3)
        lm = MemoryLifecycleManager(tmp_path / "lc.json", policy=policy)
        old_mem = _entry(1, days_ago=20, importance=0.3)
        report = lm.run_gc([old_mem])
        assert 1 in report["compressed"]
        assert lm.get_state(1) == LifecycleState.COMPRESSED

    def test_gc_skips_high_importance(self, tmp_path: Path) -> None:
        policy = ForgettingPolicy(compress_after_days=10.0, importance_protect_threshold=0.8)
        lm = MemoryLifecycleManager(tmp_path / "lc.json", policy=policy)
        important_mem = _entry(1, days_ago=20, importance=0.95)
        report = lm.run_gc([important_mem])
        assert 1 not in report["compressed"]
        assert lm.get_state(1) == LifecycleState.ACTIVE

    def test_gc_skips_recently_accessed(self, tmp_path: Path) -> None:
        policy = ForgettingPolicy(compress_after_days=10.0, compress_min_access=3)
        lm = MemoryLifecycleManager(tmp_path / "lc.json", policy=policy)
        for _ in range(5):
            lm.record_access(1)
        old_mem = _entry(1, days_ago=20, importance=0.3)
        report = lm.run_gc([old_mem])
        assert 1 not in report["compressed"]

    def test_gc_never_deletes_by_default(self, tmp_path: Path) -> None:
        policy = ForgettingPolicy(
            compress_after_days=1.0, archive_after_days=2.0, delete_after_days=None
        )
        lm = MemoryLifecycleManager(tmp_path / "lc.json", policy=policy)
        very_old = _entry(1, days_ago=1000, importance=0.1)
        lm.run_gc([very_old])
        lm.run_gc([very_old])  # second pass, now compressed→archived
        assert lm.get_state(1) in (LifecycleState.COMPRESSED, LifecycleState.ARCHIVED)
        assert lm.get_state(1) != LifecycleState.DELETED

    def test_gc_deletes_when_configured(self, tmp_path: Path) -> None:
        policy = ForgettingPolicy(
            compress_after_days=1.0, archive_after_days=2.0, delete_after_days=3.0,
            compress_min_access=999, archive_min_access=999,
        )
        lm = MemoryLifecycleManager(tmp_path / "lc.json", policy=policy)
        very_old = _entry(1, days_ago=1000, importance=0.0)
        # Run gc 3 times to walk through transitions
        for _ in range(3):
            lm.run_gc([very_old])
        assert lm.get_state(1) == LifecycleState.DELETED

    def test_filter_active(self, tmp_path: Path) -> None:
        lm = MemoryLifecycleManager(tmp_path / "lc.json")
        lm.archive(2)
        memories = [_entry(1), _entry(2), _entry(3)]
        active = lm.filter_active(memories)
        ids = {m.id for m in active}
        assert ids == {1, 3}

    def test_filter_archived(self, tmp_path: Path) -> None:
        lm = MemoryLifecycleManager(tmp_path / "lc.json")
        lm.archive(2)
        memories = [_entry(1), _entry(2), _entry(3)]
        archived = lm.filter_archived(memories)
        assert {m.id for m in archived} == {2}

    def test_state_counts(self, tmp_path: Path) -> None:
        lm = MemoryLifecycleManager(tmp_path / "lc.json")
        lm.compress(1, "s")
        lm.archive(2)
        lm.delete(3)
        counts = lm.state_counts()
        assert counts["compressed"] == 1
        assert counts["archived"] == 1
        assert counts["deleted"] == 1

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        lm1 = MemoryLifecycleManager(tmp_path / "lc.json")
        lm1.compress(1, "summary")
        lm1.record_access(1)
        lm2 = MemoryLifecycleManager(tmp_path / "lc.json")
        assert lm2.get_state(1) == LifecycleState.COMPRESSED
        assert lm2.get_access_count(1) == 1


# ===========================================================================
# Issue 3 — Semantic clustering
# ===========================================================================


class TestSemanticClusters:
    @pytest.fixture
    def index(self, tmp_path: Path) -> SemanticClusterIndex:
        return SemanticClusterIndex(tmp_path / "clusters.json", similarity_threshold=0.8)

    def test_first_memory_creates_cluster(self, index: SemanticClusterIndex) -> None:
        emb = np.array([1.0, 0.0, 0.0])
        cid = index.add_memory(1, emb, ["transformers"])
        assert cid == "cluster_0"
        assert index.cluster_count == 1

    def test_similar_memory_joins_cluster(self, index: SemanticClusterIndex) -> None:
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.99, 0.01, 0.0])
        index.add_memory(1, emb1, ["transformers"])
        cid2 = index.add_memory(2, emb2, ["attention"])
        assert cid2 == "cluster_0"
        assert index.cluster_count == 1

    def test_dissimilar_memory_creates_new_cluster(self, index: SemanticClusterIndex) -> None:
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        index.add_memory(1, emb1, ["transformers"])
        cid2 = index.add_memory(2, emb2, ["cooking"])
        assert cid2 == "cluster_1"
        assert index.cluster_count == 2

    def test_cluster_members(self, index: SemanticClusterIndex) -> None:
        emb = np.array([1.0, 0.0, 0.0])
        index.add_memory(1, emb, ["nlp"])
        index.add_memory(2, emb, ["nlp"])
        members = index.get_cluster_members("cluster_0")
        assert set(members) == {1, 2}

    def test_get_cluster_for_query(self, index: SemanticClusterIndex) -> None:
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        index.add_memory(1, emb1, ["nlp"])
        index.add_memory(2, emb2, ["cooking"])
        query = np.array([0.9, 0.1, 0.0])
        results = index.get_cluster_for_query(query, top_k=1)
        assert len(results) == 1
        cluster, sim = results[0]
        assert "nlp" in cluster.dominant_topics

    def test_label_auto_generated(self, index: SemanticClusterIndex) -> None:
        emb = np.array([1.0, 0.0, 0.0])
        index.add_memory(1, emb, ["transformers", "attention"])
        cluster = index.get_cluster("cluster_0")
        assert cluster is not None
        assert "transformers" in cluster.label

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        idx1 = SemanticClusterIndex(tmp_path / "c.json")
        idx1.add_memory(1, np.array([1.0, 0.0]), ["topic"])
        idx2 = SemanticClusterIndex(tmp_path / "c.json")
        assert idx2.cluster_count == 1
        assert idx2.get_cluster_members("cluster_0") == [1]

    def test_summary_string(self, index: SemanticClusterIndex) -> None:
        assert "No semantic clusters" in index.summary()
        index.add_memory(1, np.array([1.0, 0.0, 0.0]), ["nlp"])
        assert "cluster" in index.summary()

    def test_list_clusters_sorted_by_size(self, tmp_path: Path) -> None:
        idx = SemanticClusterIndex(tmp_path / "c.json", similarity_threshold=0.99)
        idx.add_memory(1, np.array([1.0, 0.0, 0.0]), ["a"])
        idx.add_memory(2, np.array([1.0, 0.0, 0.0]), ["a"])
        idx.add_memory(3, np.array([0.0, 1.0, 0.0]), ["b"])
        clusters = idx.list_clusters()
        assert len(clusters[0].member_ids) >= len(clusters[-1].member_ids)


# ===========================================================================
# Issue 4 — Graph reasoner
# ===========================================================================


class TestGraphReasoner:
    @pytest.fixture
    def graph(self, tmp_path: Path) -> MemoryGraph:
        g = MemoryGraph(tmp_path / "graph.json")
        g.upsert_relation("Sayan", EntityKind.PERSON, RelationKind.WORKS_ON, "Blix", EntityKind.PROJECT)
        g.upsert_relation("Blix", EntityKind.PROJECT, RelationKind.USES, "Semantic Retrieval", EntityKind.SKILL)
        g.upsert_relation("Sayan", EntityKind.PERSON, RelationKind.STUDIES_AT, "IIT Patna", EntityKind.ORGANIZATION)
        return g

    def test_find_paths(self, graph: MemoryGraph) -> None:
        reasoner = GraphReasoner(graph)
        paths = reasoner.find_paths("sayan", "semantic_retrieval", max_depth=3)
        assert len(paths) >= 1
        assert paths[0].nodes[0] == "sayan"
        assert paths[0].nodes[-1] == "semantic_retrieval"

    def test_find_paths_no_path(self, graph: MemoryGraph) -> None:
        reasoner = GraphReasoner(graph)
        paths = reasoner.find_paths("iit_patna", "blix", max_depth=2)
        assert paths == []

    def test_shortest_path(self, graph: MemoryGraph) -> None:
        reasoner = GraphReasoner(graph)
        path = reasoner.shortest_path("sayan", "semantic_retrieval")
        assert path is not None
        assert len(path.nodes) == 3  # sayan -> blix -> semantic_retrieval

    def test_shortest_path_unreachable(self, graph: MemoryGraph) -> None:
        reasoner = GraphReasoner(graph)
        assert reasoner.shortest_path("blix", "iit_patna") is None

    def test_degree_centrality(self, graph: MemoryGraph) -> None:
        reasoner = GraphReasoner(graph)
        centrality = reasoner.degree_centrality()
        assert centrality["sayan"] > centrality["semantic_retrieval"]

    def test_most_central_nodes(self, graph: MemoryGraph) -> None:
        reasoner = GraphReasoner(graph)
        top = reasoner.most_central_nodes(top_k=1)
        assert top[0][0] == "sayan"

    def test_related_entities(self, graph: MemoryGraph) -> None:
        reasoner = GraphReasoner(graph)
        related = reasoner.related_entities("sayan", depth=2)
        ids = {n.id for n, _ in related}
        assert "blix" in ids
        assert "semantic_retrieval" in ids  # 2 hops away

    def test_related_entities_depth_limit(self, graph: MemoryGraph) -> None:
        reasoner = GraphReasoner(graph)
        related = reasoner.related_entities("sayan", depth=1)
        ids = {n.id for n, _ in related}
        assert "blix" in ids
        assert "semantic_retrieval" not in ids

    def test_rank_memories_by_graph(self, tmp_path: Path) -> None:
        g = MemoryGraph(tmp_path / "g2.json")
        g.add_node(GraphNode(id="blix", kind=EntityKind.PROJECT, label="Blix"))
        g.add_node(GraphNode(id="nlp", kind=EntityKind.TOPIC, label="NLP"))
        g.add_edge(GraphEdge(
            from_id="blix", relation=RelationKind.USES, to_id="nlp",
            source_memory_ids=[42],
        ))
        reasoner = GraphReasoner(g)
        ranked = reasoner.rank_memories_by_graph([42, 99], "blix", g)
        ranked_dict = dict(ranked)
        assert ranked_dict[42] > ranked_dict[99]

    def test_graph_path_str(self, graph: MemoryGraph) -> None:
        reasoner = GraphReasoner(graph)
        path = reasoner.shortest_path("sayan", "blix")
        assert "works_on" in str(path) or "WORKS_ON" in str(path).upper()


# ===========================================================================
# Issue 5 — Contradiction detection & belief revision
# ===========================================================================


class TestContradictionDetector:
    def test_no_contradiction_without_shared_topic(self) -> None:
        detector = ContradictionDetector()
        m1 = _entry(1, output="I love NLP", topics=["nlp"])
        m2 = _entry(2, output="I love cooking", topics=["cooking"])
        result = detector.detect([m1, m2])
        assert result == []

    def test_detects_negation_with_shared_topic(self) -> None:
        detector = ContradictionDetector()
        m1 = _entry(1, output="I am interested in NLP", topics=["nlp"], importance=0.7)
        m2 = _entry(2, output="I am no longer interested in NLP", topics=["nlp"], importance=0.5)
        result = detector.detect([m1, m2])
        assert len(result) == 1
        assert result[0].field == "nlp"

    def test_resolve_picks_higher_importance(self) -> None:
        detector = ContradictionDetector()
        m1 = _entry(1, output="interested in NLP", topics=["nlp"], importance=0.9)
        m2 = _entry(2, output="no longer interested in NLP", topics=["nlp"], importance=0.3)
        contradictions = detector.detect([m1, m2])
        winner = detector.resolve(contradictions[0], [m1, m2])
        assert winner == 1

    def test_resolve_with_lifecycle_manager_compresses_loser(self, tmp_path: Path) -> None:
        lm = MemoryLifecycleManager(tmp_path / "lc.json")
        detector = ContradictionDetector(lifecycle_manager=lm)
        m1 = _entry(1, output="interested in NLP", topics=["nlp"], importance=0.9)
        m2 = _entry(2, output="no longer interested in NLP", topics=["nlp"], importance=0.3)
        contradictions = detector.detect([m1, m2])
        detector.resolve(contradictions[0], [m1, m2])
        assert lm.get_state(2) == LifecycleState.COMPRESSED

    def test_resolve_all(self) -> None:
        detector = ContradictionDetector()
        m1 = _entry(1, output="interested in NLP", topics=["nlp"], importance=0.9)
        m2 = _entry(2, output="no longer interested in NLP", topics=["nlp"], importance=0.3)
        detector.detect([m1, m2])
        resolved = detector.resolve_all([m1, m2])
        assert len(resolved) == 1
        assert resolved[0].resolved

    def test_contradiction_count_tracking(self) -> None:
        detector = ContradictionDetector()
        m1 = _entry(1, output="interested in NLP", topics=["nlp"], importance=0.9)
        m2 = _entry(2, output="no longer interested in NLP", topics=["nlp"], importance=0.3)
        detector.detect([m1, m2])
        assert detector.contradiction_count == 1
        assert detector.unresolved_count == 1
        detector.resolve_all([m1, m2])
        assert detector.unresolved_count == 0

    def test_get_contradictions_filter(self) -> None:
        detector = ContradictionDetector()
        m1 = _entry(1, output="interested in NLP", topics=["nlp"], importance=0.9)
        m2 = _entry(2, output="no longer interested in NLP", topics=["nlp"], importance=0.3)
        detector.detect([m1, m2])
        assert len(detector.get_contradictions(resolved=False)) == 1
        assert len(detector.get_contradictions(resolved=True)) == 0


# ===========================================================================
# Issue 6 — Confidence propagation
# ===========================================================================


class TestConfidencePropagator:
    def test_profile_confidence_decay(self) -> None:
        prop = ConfidencePropagator(base_confidence=1.0, topic_decay=0.9)
        assert prop.profile_confidence() == pytest.approx(0.9)

    def test_profile_confidence_with_specificity(self) -> None:
        prop = ConfidencePropagator(base_confidence=1.0, topic_decay=0.9)
        c = prop.profile_confidence(topic_specificity=0.5)
        assert c == pytest.approx(0.45)

    def test_graph_confidence_further_decays(self) -> None:
        prop = ConfidencePropagator(base_confidence=1.0, topic_decay=0.9, graph_decay=0.8)
        pc = prop.profile_confidence()
        gc = prop.graph_confidence(pc)
        assert gc < pc

    def test_low_extraction_confidence_propagates(self) -> None:
        prop = ConfidencePropagator(base_confidence=0.2)
        assert prop.profile_confidence() < 0.3
        assert prop.graph_confidence() < 0.3

    def test_from_extraction_result(self) -> None:
        prop = ConfidencePropagator.from_extraction_result(importance=0.0)
        assert prop._base >= 0.1  # floors at 0.1

    def test_fact_belief_score(self) -> None:
        prop = ConfidencePropagator(base_confidence=0.8)
        vf = VerifiedFact(text="x", belief_score=0.9)
        score = prop.fact_belief_score(vf)
        assert score == pytest.approx(0.72)


# ===========================================================================
# Issue 7 — Fact verification
# ===========================================================================


class TestFactVerifier:
    def test_verify_basic(self) -> None:
        verifier = FactVerifier()
        result = verifier.verify(["Sayan likes NLP"], 0.7, [])
        assert len(result) == 1
        assert result[0].belief_score == 0.7
        assert result[0].verification_status == VerificationStatus.UNVERIFIED

    def test_corroboration_boosts_belief(self) -> None:
        verifier = FactVerifier(min_overlap_words=1)
        existing = [
            _entry(1, facts=["Sayan likes natural language processing"]),
            _entry(2, facts=["Sayan enjoys natural language processing work"]),
        ]
        result = verifier.verify(["Sayan likes natural language processing"], 0.5, existing)
        assert result[0].belief_score > 0.5
        assert result[0].verification_status == VerificationStatus.CONFIRMED
        assert result[0].source_count > 1

    def test_profile_contradiction_penalizes(self) -> None:
        verifier = FactVerifier()
        profile = {"interests": ["nlp", "machine learning"]}
        result = verifier.verify(
            ["User is no longer interested in nlp"], 0.8, [], profile_dict=profile
        )
        assert result[0].belief_score < 0.8
        assert result[0].verification_status == VerificationStatus.UNCERTAIN

    def test_source_memory_ids_tracked(self) -> None:
        verifier = FactVerifier()
        result = verifier.verify(["fact"], 0.6, [], source_memory_id=42)
        assert 42 in result[0].source_memory_ids

    def test_verified_fact_to_dict(self) -> None:
        vf = VerifiedFact(text="x", belief_score=0.5)
        d = vf.to_dict()
        assert d["text"] == "x"
        assert d["verification_status"] == "unverified"


# ===========================================================================
# Issue 8 — Project-biased retrieval
# ===========================================================================


class TestProjectBiasedRetriever:
    def test_no_project_returns_score_order(self) -> None:
        retriever = ProjectBiasedRetriever()
        memories = [_entry(1), _entry(2)]
        scores = {1: 0.3, 2: 0.7}
        result = retriever.rerank(memories, scores, None, None, None)
        assert [m.id for m in result] == [2, 1]

    def test_direct_link_boosts_score(self) -> None:
        class FakeProjectManager:
            def get(self, name):
                class P:
                    related_session_ids = ["session-1"]
                return P()

        class FakeMem:
            def __init__(self, id, session_id):
                self.id = id
                self.session_id = session_id

        m1 = FakeMem(1, "session-1")
        m2 = FakeMem(2, "session-2")

        retriever = ProjectBiasedRetriever(project_bias=0.5)
        scores = {1: 0.3, 2: 0.4}
        result = retriever.rerank([m1, m2], scores, "Blix", FakeProjectManager(), None)
        assert result[0].id == 1  # 0.3+0.5=0.8 > 0.4


# ===========================================================================
# Issue 13 — MMR diversity reranking
# ===========================================================================


class TestMMRReranker:
    def test_invalid_lambda_raises(self) -> None:
        with pytest.raises(ValueError):
            MMRReranker(lambda_mmr=1.5)

    def test_no_embeddings_falls_back_to_relevance(self) -> None:
        mmr = MMRReranker(lambda_mmr=0.5, top_k=2)
        memories = [_entry(1), _entry(2), _entry(3)]
        scores = {1: 0.9, 2: 0.5, 3: 0.1}
        result = mmr.rerank(memories, scores)
        assert [m.id for m in result] == [1, 2]

    def test_diversification_with_embeddings(self) -> None:
        mmr = MMRReranker(lambda_mmr=0.5, top_k=2)
        memories = [_entry(1), _entry(2), _entry(3)]
        scores = {1: 0.9, 2: 0.85, 3: 0.5}
        # 1 and 2 are nearly identical; 3 is different
        embeddings = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.99, 0.01]),
            3: np.array([0.0, 1.0]),
        }
        result = mmr.rerank(memories, scores, embeddings)
        ids = {m.id for m in result}
        # Should prefer diversity: 1 and 3 (not 1 and 2, which are redundant)
        assert 1 in ids
        assert 3 in ids

    def test_empty_input(self) -> None:
        mmr = MMRReranker()
        assert mmr.rerank([], {}) == []

    def test_set_lambda(self) -> None:
        mmr = MMRReranker(lambda_mmr=0.5)
        mmr.set_lambda(0.8)
        assert mmr.lambda_mmr == 0.8
        with pytest.raises(ValueError):
            mmr.set_lambda(-0.1)


# ===========================================================================
# Issue 9 — Memory type separation
# ===========================================================================


class TestMemoryTypeClassifier:
    def test_episodic_detection(self) -> None:
        clf = MemoryTypeClassifier()
        result = clf.classify_text("Yesterday I debugged the embedding store and fixed a bug.")
        assert result == MemoryType.EPISODIC

    def test_semantic_detection(self) -> None:
        clf = MemoryTypeClassifier()
        result = clf.classify_text("Cosine similarity is defined as the cosine of the angle between two vectors.")
        assert result == MemoryType.SEMANTIC

    def test_procedural_detection(self) -> None:
        clf = MemoryTypeClassifier()
        result = clf.classify_text("To run the evaluation, use: python -m blix.evaluation.cli --dataset x.json")
        assert result == MemoryType.PROCEDURAL

    def test_unknown_for_unclassifiable(self) -> None:
        clf = MemoryTypeClassifier()
        result = clf.classify_text("ok")
        assert result == MemoryType.UNKNOWN

    def test_type_weight_default(self) -> None:
        clf = MemoryTypeClassifier()
        assert clf.type_weight(MemoryType.SEMANTIC) == 1.0
        assert clf.type_weight(MemoryType.EPISODIC) < 1.0

    def test_set_weight(self) -> None:
        clf = MemoryTypeClassifier()
        clf.set_weight(MemoryType.EPISODIC, 1.5)
        assert clf.type_weight(MemoryType.EPISODIC) == 1.5

    def test_classify_memory_entry(self) -> None:
        clf = MemoryTypeClassifier()
        m = _entry(1, input="how to install blix", output="To install: pip install blix")
        assert clf.classify(m) == MemoryType.PROCEDURAL


class TestTypeAwareRetriever:
    def test_detect_query_type(self) -> None:
        retriever = TypeAwareRetriever()
        qt = retriever.detect_query_type("How to run the eval CLI")
        assert qt == MemoryType.PROCEDURAL

    def test_rerank_boosts_matching_type(self) -> None:
        retriever = TypeAwareRetriever()
        m1 = _entry(1, output="To run tests: pytest -q")  # procedural
        m2 = _entry(2, output="Yesterday I had a meeting")  # episodic
        scores = {1: 0.5, 2: 0.5}
        result = retriever.rerank([m1, m2], scores, query="How to run tests")
        assert result[0].id == 1

    def test_filter_by_type(self) -> None:
        retriever = TypeAwareRetriever()
        m1 = _entry(1, output="To run tests: pytest -q")
        m2 = _entry(2, output="Yesterday I worked on the project")
        filtered = retriever.filter_by_type([m1, m2], MemoryType.PROCEDURAL)
        assert all(m.id == 1 for m in filtered)

    def test_annotate(self) -> None:
        retriever = TypeAwareRetriever()
        m1 = _entry(1, output="To run tests: pytest -q")
        annotated = retriever.annotate([m1])
        assert annotated[0][1] == MemoryType.PROCEDURAL


# ===========================================================================
# Issue 10 — Extended research metrics
# ===========================================================================


class TestExtendedMetrics:
    def test_retention_over_time(self) -> None:
        cases = [
            EvalCase(case_id="c1", query="q1", relevant_memory_ids=[1, 2]),
        ]

        def retrieval_fn(query: str, age: float) -> list[int]:
            return [1] if age < 30 else [1, 2]

        result = ExtendedMemoryEvaluator.retention_over_time(retrieval_fn, cases, [7, 60])
        assert result[7] < result[60] or result[7] == result[60]
        assert set(result.keys()) == {7, 60}

    def test_forgetting_curve(self) -> None:
        cases = [EvalCase(case_id="c1", query="q1", relevant_memory_ids=[1])]

        def retrieval_fn(query: str, age: float) -> list[int]:
            return [1]

        curve = ExtendedMemoryEvaluator.forgetting_curve(retrieval_fn, cases, [7, 30, 90])
        assert curve == [(7, 1.0), (30, 1.0), (90, 1.0)]

    def test_contradiction_rate(self) -> None:
        detector = ContradictionDetector()
        m1 = _entry(1, output="interested in NLP", topics=["nlp"], importance=0.9)
        m2 = _entry(2, output="no longer interested in NLP", topics=["nlp"], importance=0.3)
        rate = ExtendedMemoryEvaluator.contradiction_rate(detector, [m1, m2])
        assert rate == 1.0  # 1 contradiction / 1 pair

    def test_contradiction_rate_no_pairs(self) -> None:
        detector = ContradictionDetector()
        assert ExtendedMemoryEvaluator.contradiction_rate(detector, [_entry(1)]) == 0.0

    def test_memory_drift_no_drift(self) -> None:
        emb = [0.0] * 5
        emb[0] = 1.0
        data = [(0.0, list(emb)), (10.0, list(emb))]
        drift = ExtendedMemoryEvaluator.memory_drift(data)
        assert drift == pytest.approx(0.0, abs=1e-5)

    def test_memory_drift_full_rotation(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        data = [(0.0, a), (10.0, b)]
        drift = ExtendedMemoryEvaluator.memory_drift(data)
        assert drift == pytest.approx(1.0, abs=1e-5)

    def test_memory_drift_insufficient_data(self) -> None:
        assert ExtendedMemoryEvaluator.memory_drift([(0.0, [1.0, 0.0])]) == 0.0

    def test_profile_drift(self) -> None:
        from core.profile_evolver import ProfileAuditEntry
        entries = [
            ProfileAuditEntry(field="name", new_value="x", timestamp=_ts(1)),
            ProfileAuditEntry(field="interests", new_value="y", timestamp=_ts(2)),
        ]
        drift = ExtendedMemoryEvaluator.profile_drift(entries, time_window_days=30.0)
        assert drift == pytest.approx(2 / 30)

    def test_profile_drift_empty(self) -> None:
        assert ExtendedMemoryEvaluator.profile_drift([], 30.0) == 0.0

    def test_temporal_consistency(self) -> None:
        m1, m2, m3, m4 = _entry(1), _entry(2), _entry(3), _entry(4)
        retrieved = [m1, m2, m3, m4]
        newer_ids = {1, 2}
        result = ExtendedMemoryEvaluator.temporal_consistency(retrieved, newer_ids)
        assert result == 1.0  # top half (1,2) are both "newer"

    def test_temporal_consistency_empty(self) -> None:
        assert ExtendedMemoryEvaluator.temporal_consistency([], set()) == 0.0

    def test_evaluate_extended_combines_metrics(self) -> None:
        ev = ExtendedMemoryEvaluator()
        detector = ContradictionDetector()
        m1 = _entry(1, output="interested in NLP", topics=["nlp"], importance=0.9)
        m2 = _entry(2, output="no longer interested in NLP", topics=["nlp"], importance=0.3)
        results = ev.evaluate_extended([m1, m2], contradiction_detector=detector)
        assert "contradiction_rate" in results


# ===========================================================================
# Issue 14 — Research hypothesis framework
# ===========================================================================


class TestResearchHypothesis:
    def test_built_in_hypotheses_loaded(self) -> None:
        registry = HypothesisRegistry()
        ids = {h.id for h in registry.list_all()}
        assert {"H1", "H2", "H3", "H4"}.issubset(ids)

    def test_hypothesis_untested_by_default(self) -> None:
        registry = HypothesisRegistry()
        h1 = registry.get("H1")
        assert h1 is not None
        assert h1.status == HypothesisStatus.UNTESTED

    def test_evaluate_support_higher_direction(self) -> None:
        h = ResearchHypothesis(
            id="TEST1", statement="x", independent_variable="a",
            dependent_variable="b", metric_name="m",
            baseline_condition="base", treatment_condition="treat",
            expected_direction="higher",
        )
        h.record_result("baseline", 0.5)
        h.record_result("treatment", 0.7)
        status = h.evaluate_support()
        assert status == HypothesisStatus.SUPPORTED

    def test_evaluate_support_lower_direction(self) -> None:
        h = ResearchHypothesis(
            id="TEST2", statement="x", independent_variable="a",
            dependent_variable="b", metric_name="m",
            baseline_condition="base", treatment_condition="treat",
            expected_direction="lower",
        )
        h.record_result("baseline", 0.5)
        h.record_result("treatment", 0.3)
        status = h.evaluate_support()
        assert status == HypothesisStatus.SUPPORTED

    def test_evaluate_support_refuted(self) -> None:
        h = ResearchHypothesis(
            id="TEST3", statement="x", independent_variable="a",
            dependent_variable="b", metric_name="m",
            baseline_condition="base", treatment_condition="treat",
            expected_direction="higher",
        )
        h.record_result("baseline", 0.7)
        h.record_result("treatment", 0.5)
        status = h.evaluate_support()
        assert status == HypothesisStatus.REFUTED

    def test_evaluate_support_inconclusive_with_small_delta(self) -> None:
        h = ResearchHypothesis(
            id="TEST4", statement="x", independent_variable="a",
            dependent_variable="b", metric_name="m",
            baseline_condition="base", treatment_condition="treat",
            expected_direction="higher",
        )
        h.record_result("baseline", 0.500)
        h.record_result("treatment", 0.501)
        status = h.evaluate_support()
        assert status == HypothesisStatus.INCONCLUSIVE

    def test_evaluate_support_untested_without_results(self) -> None:
        h = ResearchHypothesis(
            id="TEST5", statement="x", independent_variable="a",
            dependent_variable="b", metric_name="m",
            baseline_condition="base", treatment_condition="treat",
        )
        assert h.evaluate_support() == HypothesisStatus.UNTESTED

    def test_registry_persistence(self, tmp_path: Path) -> None:
        registry_file = tmp_path / "hypotheses.json"
        registry1 = HypothesisRegistry(registry_file)
        custom = ResearchHypothesis(
            id="H5", statement="custom", independent_variable="a",
            dependent_variable="b", metric_name="m",
            baseline_condition="base", treatment_condition="treat",
        )
        registry1.add(custom)

        registry2 = HypothesisRegistry(registry_file)
        assert registry2.get("H5") is not None

    def test_evaluate_all(self) -> None:
        registry = HypothesisRegistry()
        h1 = registry.get("H1")
        h1.record_result("baseline", 0.5)
        h1.record_result("treatment", 0.7)
        statuses = registry.evaluate_all()
        assert statuses["H1"] == HypothesisStatus.SUPPORTED

    def test_to_dict_roundtrip(self) -> None:
        h = ResearchHypothesis(
            id="H6", statement="x", independent_variable="a",
            dependent_variable="b", metric_name="m",
            baseline_condition="base", treatment_condition="treat",
        )
        d = h.to_dict()
        assert d["id"] == "H6"
        assert d["status"] == "untested"


# ===========================================================================
# Issue 11 — Durable background queue (no dropped tasks)
# ===========================================================================


class TestDurableBackgroundProcessor:
    def test_overflow_to_disk_when_queue_full(self, tmp_path: Path) -> None:
        overflow = tmp_path / "overflow.jsonl"
        bp = BackgroundProcessor(max_queue_size=1, overflow_file=overflow)
        # Fill the queue (worker not started, so it stays full)
        bp._queue.put_nowait(__import__("core.background_processor", fromlist=["MemoryTask"]).MemoryTask(
            job=ProcessorJob.EXTRACT_AND_UPDATE, payload={"x": 0}
        ))
        # This submit should overflow to disk
        ok = bp.submit(ProcessorJob.EXTRACT_AND_UPDATE, {"x": 1})
        assert ok is True
        assert overflow.exists()
        assert bp.overflow_pending == 1

    def test_drop_without_overflow_file(self, tmp_path: Path) -> None:
        bp = BackgroundProcessor(max_queue_size=1, overflow_file=None)
        bp._queue.put_nowait(__import__("core.background_processor", fromlist=["MemoryTask"]).MemoryTask(
            job=ProcessorJob.EXTRACT_AND_UPDATE, payload={"x": 0}
        ))
        ok = bp.submit(ProcessorJob.EXTRACT_AND_UPDATE, {"x": 1})
        assert ok is False

    def test_drain_overflow_requeues_tasks(self, tmp_path: Path) -> None:
        overflow = tmp_path / "overflow.jsonl"
        results: list[int] = []

        bp = BackgroundProcessor(max_queue_size=5, overflow_file=overflow)
        bp.register(ProcessorJob.EXTRACT_AND_UPDATE, lambda p: results.append(p["x"]))

        # Manually write an overflow entry
        overflow.parent.mkdir(parents=True, exist_ok=True)
        with overflow.open("w") as fh:
            fh.write(json.dumps({
                "job": "EXTRACT_AND_UPDATE", "payload": {"x": 99}, "attempt": 0, "max_attempts": 3
            }) + "\n")

        drained = bp.drain_overflow()
        assert drained == 1
        assert not overflow.exists()  # fully drained → file removed

        bp.start()
        time.sleep(0.2)
        bp.stop()
        assert 99 in results

    def test_overflow_pending_zero_when_no_file(self, tmp_path: Path) -> None:
        bp = BackgroundProcessor(overflow_file=tmp_path / "nope.jsonl")
        assert bp.overflow_pending == 0

    def test_stats_includes_overflow_fields(self) -> None:
        bp = BackgroundProcessor()
        stats = bp.stats
        assert "overflowed" in stats
        assert "overflow_pending" in stats


# ===========================================================================
# Issue 12 — Multi-user namespacing
# ===========================================================================


class TestUserNamespace:
    def test_default_namespace_uses_base_dir(self, tmp_path: Path) -> None:
        ns = UserNamespace(tmp_path, user_id=None)
        assert ns.root == tmp_path
        assert ns.user_id == "default"

    def test_named_user_gets_subdirectory(self, tmp_path: Path) -> None:
        ns = UserNamespace(tmp_path, user_id="Sayan")
        assert ns.root == tmp_path / "users" / "sayan"

    def test_path_creates_parent_dirs(self, tmp_path: Path) -> None:
        ns = UserNamespace(tmp_path, user_id="alice")
        p = ns.path("hierarchy/sessions.json")
        assert p.parent.exists()
        assert p == tmp_path / "users" / "alice" / "hierarchy" / "sessions.json"

    def test_slugify(self) -> None:
        assert _slugify("Sayan D!") == "sayan_d"
        assert _slugify("") == "default"

    def test_two_users_get_isolated_paths(self, tmp_path: Path) -> None:
        ns1 = UserNamespace(tmp_path, "sayan")
        ns2 = UserNamespace(tmp_path, "alice")
        assert ns1.path("graph.json") != ns2.path("graph.json")


class TestUserRegistry:
    def test_register_new_user(self, tmp_path: Path) -> None:
        registry = UserRegistry(tmp_path)
        ns = registry.register("Sayan")
        assert registry.user_count == 1
        assert ns.slug == "sayan"

    def test_register_idempotent(self, tmp_path: Path) -> None:
        registry = UserRegistry(tmp_path)
        registry.register("Sayan")
        registry.register("Sayan")
        assert registry.user_count == 1

    def test_get_user(self, tmp_path: Path) -> None:
        registry = UserRegistry(tmp_path)
        registry.register("Sayan", display_name="Sayan D")
        rec = registry.get("Sayan")
        assert rec is not None
        assert rec.display_name == "Sayan D"

    def test_touch_updates_last_active(self, tmp_path: Path) -> None:
        registry = UserRegistry(tmp_path)
        registry.register("Sayan")
        registry.touch("Sayan")
        rec = registry.get("Sayan")
        assert rec.last_active is not None

    def test_namespace_for_lazily_registers(self, tmp_path: Path) -> None:
        registry = UserRegistry(tmp_path)
        ns = registry.namespace_for("newuser")
        assert registry.get("newuser") is not None
        assert ns.slug == "newuser"

    def test_namespace_for_none_is_default(self, tmp_path: Path) -> None:
        registry = UserRegistry(tmp_path)
        ns = registry.namespace_for(None)
        assert ns.user_id == "default"
        assert ns.root == tmp_path

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        registry1 = UserRegistry(tmp_path)
        registry1.register("Sayan")
        registry1.register("Alice")
        registry2 = UserRegistry(tmp_path)
        assert registry2.user_count == 2

    def test_list_users(self, tmp_path: Path) -> None:
        registry = UserRegistry(tmp_path)
        registry.register("Sayan")
        registry.register("Alice")
        slugs = {u.slug for u in registry.list_users()}
        assert slugs == {"sayan", "alice"}
