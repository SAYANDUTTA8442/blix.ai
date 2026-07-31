"""
Tests for Blix v0.3 new modules.

Covers:
- MemoryScorer (Feature 2)
- MemoryGraph (Feature 3)
- ProfileEvolver (Feature 4)
- ProjectManager (Feature 5)
- BackgroundProcessor (Feature 6)
- HierarchyManager (Feature 7 / Feature 1)
- MemoryEvaluator (Feature 7)

No network, GPU, or LLM required — all tests are offline.
Python 3.10 compatible.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from core.memory_scorer import MemoryScorer, ScoringWeights, MemoryScore
from core.memory_graph import MemoryGraph, GraphNode, GraphEdge, EntityKind, RelationKind
from core.profile_evolver import ProfileEvolver, VersionedProfile
from core.project_manager import ProjectManager
from core.background_processor import BackgroundProcessor, ProcessorJob, MemoryTask
from core.hierarchy_manager import HierarchyManager, _date_to_week
from evaluation import MemoryEvaluator, EvalCase, EvalDataset
from schemas.memory_entry import MemoryEntry
from schemas.memory_layers import (
    SessionSummary, DailySummary, WeeklySummary, ProjectSummary, MemoryLayerKind
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ts(days_ago: float = 0.0) -> datetime:
    from datetime import timedelta
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago)


def _entry(id: int, input: str = "q", output: str = "a",
           importance: Optional[float] = None,
           topics: Optional[list[str]] = None) -> MemoryEntry:
    return MemoryEntry(
        id=id, input=input, output=output,
        timestamp=_ts(),
        importance=importance,
        topics=topics or [],
    )


# ===========================================================================
# Feature 2 — MemoryScorer
# ===========================================================================


class TestScoringWeights:
    def test_default_sum_to_one(self) -> None:
        w = ScoringWeights()
        assert abs(w.relevance + w.importance + w.recency + w.frequency - 1.0) < 1e-6

    def test_validate_sum_true(self) -> None:
        assert ScoringWeights().validate_sum()

    def test_custom_weights(self) -> None:
        w = ScoringWeights(relevance=0.5, importance=0.3, recency=0.1, frequency=0.1)
        assert w.validate_sum()


class TestMemoryScorer:
    def _scorer(self) -> MemoryScorer:
        return MemoryScorer()

    def test_score_returns_memory_score(self) -> None:
        s = self._scorer().score(1, relevance=0.8, importance=0.7, timestamp=_ts())
        assert isinstance(s, MemoryScore)
        assert s.memory_id == 1

    def test_score_in_range(self) -> None:
        s = self._scorer().score(1, relevance=1.0, importance=1.0, timestamp=_ts())
        assert 0.0 <= s.final_score <= 1.0

    def test_recency_decays_with_age(self) -> None:
        scorer = MemoryScorer(recency_half_life_days=10.0)
        fresh = scorer.score(1, relevance=0.5, importance=0.5, timestamp=_ts(0))
        old = scorer.score(1, relevance=0.5, importance=0.5, timestamp=_ts(20))
        assert fresh.final_score > old.final_score

    def test_explanation_keys(self) -> None:
        s = self._scorer().score(1, relevance=0.5, importance=0.5, timestamp=_ts())
        assert set(s.explanation.keys()) == {"relevance", "importance", "recency", "frequency"}

    def test_debug_str_is_string(self) -> None:
        s = self._scorer().score(1, relevance=0.5, importance=0.5, timestamp=_ts())
        assert isinstance(s.debug_str(), str)

    def test_batch_sorted_descending(self) -> None:
        scorer = MemoryScorer()
        entries = [
            {"id": 1, "relevance": 0.1, "importance": 0.1, "timestamp": _ts(30)},
            {"id": 2, "relevance": 0.9, "importance": 0.9, "timestamp": _ts(0)},
            {"id": 3, "relevance": 0.5, "importance": 0.5, "timestamp": _ts(5)},
        ]
        scores = scorer.score_batch(entries)
        values = [s.final_score for s in scores]
        assert values == sorted(values, reverse=True)

    def test_custom_weights_applied(self) -> None:
        # With all weight on relevance, two memories with same relevance should tie
        w = ScoringWeights(relevance=1.0, importance=0.0, recency=0.0, frequency=0.0)
        scorer = MemoryScorer(weights=w)
        s1 = scorer.score(1, relevance=0.8, importance=0.0, timestamp=_ts(100))
        s2 = scorer.score(2, relevance=0.8, importance=1.0, timestamp=_ts(0))
        assert abs(s1.final_score - s2.final_score) < 0.01  # importance/recency ignored


# ===========================================================================
# Feature 3 — MemoryGraph
# ===========================================================================


class TestMemoryGraph:
    @pytest.fixture
    def graph(self, tmp_path: Path) -> MemoryGraph:
        return MemoryGraph(tmp_path / "graph.json")

    def test_add_node(self, graph: MemoryGraph) -> None:
        n = GraphNode(id="sayan", kind=EntityKind.PERSON, label="Sayan")
        graph.add_node(n)
        assert graph.node_count == 1

    def test_get_node(self, graph: MemoryGraph) -> None:
        n = GraphNode(id="blix", kind=EntityKind.PROJECT, label="Blix")
        graph.add_node(n)
        assert graph.get_node("blix") is not None

    def test_get_node_missing(self, graph: MemoryGraph) -> None:
        assert graph.get_node("nonexistent") is None

    def test_merge_aliases(self, graph: MemoryGraph) -> None:
        n1 = GraphNode(id="sayan", kind=EntityKind.PERSON, label="Sayan", aliases=["S"])
        n2 = GraphNode(id="sayan", kind=EntityKind.PERSON, label="Sayan", aliases=["Sayan D"])
        graph.add_node(n1)
        graph.add_node(n2)
        assert graph.node_count == 1
        assert "Sayan D" in graph.get_node("sayan").aliases

    def test_add_edge(self, graph: MemoryGraph) -> None:
        graph.add_node(GraphNode(id="sayan", kind=EntityKind.PERSON, label="Sayan"))
        graph.add_node(GraphNode(id="blix", kind=EntityKind.PROJECT, label="Blix"))
        e = GraphEdge(from_id="sayan", relation=RelationKind.WORKS_ON, to_id="blix")
        graph.add_edge(e)
        assert graph.edge_count == 1

    def test_get_edges_filter(self, graph: MemoryGraph) -> None:
        graph.add_node(GraphNode(id="sayan", kind=EntityKind.PERSON, label="Sayan"))
        graph.add_node(GraphNode(id="blix", kind=EntityKind.PROJECT, label="Blix"))
        graph.add_node(GraphNode(id="nlp", kind=EntityKind.TOPIC, label="NLP"))
        graph.add_edge(GraphEdge(from_id="sayan", relation=RelationKind.WORKS_ON, to_id="blix"))
        graph.add_edge(GraphEdge(from_id="sayan", relation=RelationKind.INTERESTED_IN, to_id="nlp"))
        edges = graph.get_edges(from_id="sayan", relation=RelationKind.WORKS_ON)
        assert len(edges) == 1
        assert edges[0].to_id == "blix"

    def test_neighbours(self, graph: MemoryGraph) -> None:
        graph.add_node(GraphNode(id="sayan", kind=EntityKind.PERSON, label="Sayan"))
        graph.add_node(GraphNode(id="blix", kind=EntityKind.PROJECT, label="Blix"))
        graph.add_edge(GraphEdge(from_id="sayan", relation=RelationKind.WORKS_ON, to_id="blix"))
        n = graph.neighbours("sayan")
        assert len(n) == 1
        rel, node = n[0]
        assert rel == RelationKind.WORKS_ON
        assert node.id == "blix"

    def test_upsert_relation_creates_nodes(self, graph: MemoryGraph) -> None:
        graph.upsert_relation(
            from_label="Sayan", from_kind=EntityKind.PERSON,
            relation=RelationKind.STUDIES_AT,
            to_label="IIT Patna", to_kind=EntityKind.ORGANIZATION,
        )
        assert graph.node_count == 2
        assert graph.edge_count == 1

    def test_find_node_by_label_case_insensitive(self, graph: MemoryGraph) -> None:
        graph.add_node(GraphNode(id="sayan", kind=EntityKind.PERSON, label="Sayan"))
        assert graph.find_node_by_label("sayan") is not None
        assert graph.find_node_by_label("SAYAN") is not None

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        g1 = MemoryGraph(tmp_path / "g.json")
        g1.upsert_relation(
            from_label="Alice", from_kind=EntityKind.PERSON,
            relation=RelationKind.WORKS_ON,
            to_label="Blix", to_kind=EntityKind.PROJECT,
        )
        g2 = MemoryGraph(tmp_path / "g.json")
        assert g2.node_count == 2
        assert g2.edge_count == 1

    def test_edge_confidence_merge(self, graph: MemoryGraph) -> None:
        graph.add_node(GraphNode(id="a", kind=EntityKind.PERSON, label="A"))
        graph.add_node(GraphNode(id="b", kind=EntityKind.PROJECT, label="B"))
        e1 = GraphEdge(from_id="a", relation=RelationKind.WORKS_ON, to_id="b", confidence=0.5)
        e2 = GraphEdge(from_id="a", relation=RelationKind.WORKS_ON, to_id="b", confidence=0.9)
        graph.add_edge(e1)
        graph.add_edge(e2)
        assert graph.edge_count == 1
        edges = graph.get_edges(from_id="a")
        assert edges[0].confidence == 0.9


# ===========================================================================
# Feature 4 — ProfileEvolver
# ===========================================================================


class TestProfileEvolver:
    @pytest.fixture
    def evolver(self, tmp_path: Path) -> ProfileEvolver:
        return ProfileEvolver(tmp_path / "vp.json")

    def test_initial_profile_empty(self, evolver: ProfileEvolver) -> None:
        assert evolver.profile.is_empty()

    def test_initial_version_is_one(self, evolver: ProfileEvolver) -> None:
        assert evolver.versioned.version == 1

    def test_update_name(self, evolver: ProfileEvolver) -> None:
        changed = evolver.update(name="Sayan", confidence=0.9)
        assert changed
        assert evolver.profile.name == "Sayan"
        assert evolver.versioned.version == 2

    def test_name_not_overwritten_by_lower_confidence(self, evolver: ProfileEvolver) -> None:
        evolver.update(name="Sayan", confidence=0.9)
        evolver.update(name="Bob", confidence=0.3)
        assert evolver.profile.name == "Sayan"

    def test_name_overwritten_by_higher_confidence(self, evolver: ProfileEvolver) -> None:
        evolver.update(name="Sayan", confidence=0.5)
        evolver.update(name="Sayan D", confidence=0.9)
        assert evolver.profile.name == "Sayan D"

    def test_interests_extended(self, evolver: ProfileEvolver) -> None:
        evolver.update(new_interests=["NLP"])
        evolver.update(new_interests=["LLMs", "NLP"])
        assert "LLMs" in evolver.profile.interests
        assert evolver.profile.interests.count("NLP") == 1

    def test_audit_trail_records_changes(self, evolver: ProfileEvolver) -> None:
        evolver.update(name="Sayan")
        evolver.update(new_interests=["NLP"])
        audit = evolver.get_audit()
        assert len(audit) >= 2

    def test_audit_filter_by_field(self, evolver: ProfileEvolver) -> None:
        evolver.update(name="Sayan")
        evolver.update(education="IIT Patna")
        name_audit = evolver.get_audit("name")
        assert all(e.field == "name" for e in name_audit)

    def test_no_change_returns_false(self, evolver: ProfileEvolver) -> None:
        assert not evolver.update()

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        e1 = ProfileEvolver(tmp_path / "vp.json")
        e1.update(name="Sayan", education="IIT Patna", new_interests=["NLP"])
        e2 = ProfileEvolver(tmp_path / "vp.json")
        assert e2.profile.name == "Sayan"
        assert e2.versioned.version == e1.versioned.version


# ===========================================================================
# Feature 5 — ProjectManager
# ===========================================================================


class TestProjectManager:
    @pytest.fixture
    def pm(self, tmp_path: Path) -> ProjectManager:
        return ProjectManager(tmp_path / "projects.json")

    def test_get_or_create(self, pm: ProjectManager) -> None:
        p = pm.get_or_create("Blix")
        assert p.project_name == "Blix"
        assert pm.count == 1

    def test_get_or_create_idempotent(self, pm: ProjectManager) -> None:
        pm.get_or_create("Blix")
        pm.get_or_create("Blix")
        assert pm.count == 1

    def test_get_missing_returns_none(self, pm: ProjectManager) -> None:
        assert pm.get("nonexistent") is None

    def test_list_all(self, pm: ProjectManager) -> None:
        pm.get_or_create("Blix")
        pm.get_or_create("ECOT")
        assert len(pm.list_all()) == 2

    def test_update_goals(self, pm: ProjectManager) -> None:
        pm.get_or_create("Blix")
        pm.update("Blix", goals=["Long-term memory architecture"])
        p = pm.get("Blix")
        assert "Long-term memory architecture" in p.goals

    def test_update_extends_list(self, pm: ProjectManager) -> None:
        pm.get_or_create("Blix")
        pm.update("Blix", goals=["Goal A"])
        pm.update("Blix", goals=["Goal B"])
        p = pm.get("Blix")
        assert "Goal A" in p.goals
        assert "Goal B" in p.goals

    def test_link_session(self, pm: ProjectManager) -> None:
        pm.get_or_create("Blix")
        pm.link_session("Blix", "session-1")
        p = pm.get("Blix")
        assert "session-1" in p.related_session_ids

    def test_record_progress(self, pm: ProjectManager) -> None:
        pm.get_or_create("Blix")
        pm.record_progress("Blix", completed=["Implemented scorer"], next_actions=["Write tests"])
        p = pm.get("Blix")
        assert "Implemented scorer" in p.completed_work
        assert "Write tests" in p.next_actions

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        pm1 = ProjectManager(tmp_path / "p.json")
        pm1.get_or_create("Blix")
        pm1.update("Blix", goals=["v0.3"])
        pm2 = ProjectManager(tmp_path / "p.json")
        assert pm2.count == 1
        assert "v0.3" in pm2.get("Blix").goals

    def test_list_by_status(self, pm: ProjectManager) -> None:
        pm.get_or_create("Blix")
        pm.get_or_create("ECOT")
        pm.update("ECOT", current_status="paused", _replace=True)
        active = pm.list_all(status="active")
        paused = pm.list_all(status="paused")
        assert any(p.project_name == "Blix" for p in active)
        assert any(p.project_name == "ECOT" for p in paused)


# ===========================================================================
# Feature 6 — BackgroundProcessor
# ===========================================================================


class TestBackgroundProcessor:
    def test_start_stop(self) -> None:
        bp = BackgroundProcessor()
        bp.start()
        assert bp.stats["running"]
        bp.stop()
        assert not bp.stats["running"]

    def test_submit_and_process(self) -> None:
        results: list[str] = []

        bp = BackgroundProcessor()
        bp.register(ProcessorJob.EXTRACT_AND_UPDATE, lambda p: results.append(p["val"]))
        bp.start()
        bp.submit(ProcessorJob.EXTRACT_AND_UPDATE, {"val": "done"})
        time.sleep(0.2)
        bp.stop()
        assert "done" in results

    def test_multiple_jobs(self) -> None:
        count: list[int] = [0]

        def handler(p: dict) -> None:
            count[0] += 1

        bp = BackgroundProcessor()
        bp.register(ProcessorJob.EXTRACT_AND_UPDATE, handler)
        bp.start()
        for i in range(5):
            bp.submit(ProcessorJob.EXTRACT_AND_UPDATE, {"i": i})
        time.sleep(0.5)
        bp.stop()
        assert count[0] == 5

    def test_stats_processed(self) -> None:
        bp = BackgroundProcessor()
        bp.register(ProcessorJob.EXTRACT_AND_UPDATE, lambda p: None)
        bp.start()
        bp.submit(ProcessorJob.EXTRACT_AND_UPDATE, {})
        time.sleep(0.2)
        bp.stop()
        assert bp.stats["processed"] >= 1

    def test_retry_on_failure(self) -> None:
        attempts: list[int] = [0]

        def flaky(p: dict) -> None:
            attempts[0] += 1
            if attempts[0] < 3:
                raise RuntimeError("temporary failure")

        bp = BackgroundProcessor()
        bp.register(ProcessorJob.EXTRACT_AND_UPDATE, flaky)
        bp.start()
        bp.submit(ProcessorJob.EXTRACT_AND_UPDATE, {})
        time.sleep(1.0)  # allow retries
        bp.stop()
        assert attempts[0] >= 3

    def test_no_handler_drops_silently(self) -> None:
        bp = BackgroundProcessor()
        bp.start()
        bp.submit(ProcessorJob.REBUILD_INDEX, {})  # no handler registered
        time.sleep(0.1)
        bp.stop()  # must not raise


# ===========================================================================
# Feature 1 — HierarchyManager
# ===========================================================================


class TestHierarchyManager:
    @pytest.fixture
    def hm(self, tmp_path: Path) -> HierarchyManager:
        return HierarchyManager(tmp_path / "hierarchy")

    def _make_memories(self, n: int) -> list[MemoryEntry]:
        return [
            MemoryEntry(
                id=i, input=f"q{i}", output=f"a{i}",
                timestamp=_ts(),
                topics=["NLP"],
            )
            for i in range(1, n + 1)
        ]

    def test_create_session_summary(self, hm: HierarchyManager) -> None:
        memories = self._make_memories(3)
        ss = hm.create_session_summary(1, memories)
        assert isinstance(ss, SessionSummary)
        assert ss.turn_count == 3
        assert ss.session_index == 1
        assert hm.session_count == 1

    def test_session_raw_ids_correct(self, hm: HierarchyManager) -> None:
        memories = self._make_memories(3)
        ss = hm.create_session_summary(1, memories)
        assert set(ss.raw_memory_ids) == {1, 2, 3}

    def test_get_latest_sessions(self, hm: HierarchyManager) -> None:
        for i in range(1, 4):
            hm.create_session_summary(i, self._make_memories(2))
        latest = hm.get_latest_sessions(2)
        assert len(latest) == 2
        assert latest[0].session_index > latest[1].session_index

    def test_roll_up_daily(self, hm: HierarchyManager) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        memories = self._make_memories(3)
        hm.create_session_summary(1, memories)
        ds = hm.roll_up_daily(today)
        assert ds is not None
        assert ds.session_count == 1
        assert hm.daily_count == 1

    def test_roll_up_weekly(self, hm: HierarchyManager) -> None:
        today = datetime.now(timezone.utc)
        date_str = today.strftime("%Y-%m-%d")
        memories = self._make_memories(2)
        hm.create_session_summary(1, memories)
        hm.roll_up_daily(date_str)
        week_label = _date_to_week(date_str)
        ws = hm.roll_up_weekly(week_label)
        assert ws is not None
        assert hm.weekly_count == 1

    def test_get_hierarchy_context_returns_string(self, hm: HierarchyManager) -> None:
        ctx = hm.get_hierarchy_context()
        assert isinstance(ctx, str)

    def test_persistence_sessions(self, tmp_path: Path) -> None:
        hm1 = HierarchyManager(tmp_path / "h")
        hm1.create_session_summary(1, self._make_memories(2))
        hm2 = HierarchyManager(tmp_path / "h")
        assert hm2.session_count == 1


# ===========================================================================
# Feature 7 — MemoryEvaluator
# ===========================================================================


class TestMemoryEvaluator:
    def _dataset(self) -> EvalDataset:
        return EvalDataset(
            name="test",
            cases=[
                EvalCase(
                    case_id="c1",
                    query="gradient descent",
                    relevant_memory_ids=[1, 2],
                    ground_truth_facts=["Gradient descent minimises loss."],
                    ground_truth_profile={"name": "Sayan"},
                    ground_truth_edges=[("sayan", "works_on", "blix")],
                    reference_summary="Discussed gradient descent optimisation.",
                ),
                EvalCase(
                    case_id="c2",
                    query="transformers attention",
                    relevant_memory_ids=[3],
                    ground_truth_facts=["Attention assigns weights to tokens."],
                    reference_summary="Explained the attention mechanism.",
                ),
            ],
        )

    def test_precision_at_k(self) -> None:
        ev = MemoryEvaluator()
        assert ev.precision_at_k([1, 2, 5], [1, 2, 3]) == pytest.approx(2 / 3)

    def test_recall_at_k(self) -> None:
        ev = MemoryEvaluator()
        assert ev.recall_at_k([1, 2], [1, 2, 3]) == pytest.approx(2 / 3)

    def test_f1_perfect(self) -> None:
        ev = MemoryEvaluator()
        assert ev.f1(1.0, 1.0) == 1.0

    def test_f1_zero(self) -> None:
        ev = MemoryEvaluator()
        assert ev.f1(0.0, 0.0) == 0.0

    def test_fact_accuracy_all_correct(self) -> None:
        ev = MemoryEvaluator()
        assert ev.fact_accuracy(["gradient descent"], ["gradient descent minimises loss"]) == 1.0

    def test_hallucination_rate_none(self) -> None:
        ev = MemoryEvaluator()
        assert ev.hallucination_rate([], ["any fact"]) == 0.0

    def test_profile_accuracy_perfect(self) -> None:
        ev = MemoryEvaluator()
        assert ev.profile_accuracy({"name": "sayan"}, {"name": "sayan"}) == 1.0

    def test_profile_accuracy_list_check(self) -> None:
        ev = MemoryEvaluator()
        result = ev.profile_accuracy(
            {"interests": ["NLP", "LLMs", "RL"]},
            {"interests": ["NLP", "LLMs"]},
        )
        assert result == 1.0

    def test_graph_consistency_perfect(self) -> None:
        ev = MemoryEvaluator()
        actual = [("sayan", "works_on", "blix")]
        gt = [("sayan", "works_on", "blix")]
        assert ev.graph_consistency(actual, gt) == 1.0

    def test_graph_consistency_mismatch(self) -> None:
        ev = MemoryEvaluator()
        actual = [("sayan", "works_on", "blix"), ("sayan", "works_on", "ghost")]
        gt = [("sayan", "works_on", "blix")]
        assert ev.graph_consistency(actual, gt) == 0.5

    def test_summary_quality_exact(self) -> None:
        ev = MemoryEvaluator()
        assert ev.summary_quality("hello world", "hello world") == 1.0

    def test_summary_quality_empty_gen(self) -> None:
        ev = MemoryEvaluator()
        assert ev.summary_quality("", "hello world") == 0.0

    def test_evaluate_no_callables(self) -> None:
        ev = MemoryEvaluator()
        report = ev.evaluate(self._dataset())
        assert report.dataset_name == "test"
        assert len(report.metrics) == 8  # all 8 metrics present (may be NaN)

    def test_evaluate_with_retrieval_fn(self) -> None:
        ev = MemoryEvaluator()
        report = ev.evaluate(
            self._dataset(),
            retrieval_fn=lambda q: [1, 2],
        )
        prec = next(m for m in report.metrics if m.name == "retrieval_precision")
        assert prec.value == prec.value  # not NaN

    def test_save_and_load_report(self, tmp_path: Path) -> None:
        ev = MemoryEvaluator()
        report = ev.evaluate(self._dataset())
        out = tmp_path / "report.json"
        ev.save_report(report, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["dataset"] == "test"

    def test_per_case_populated(self) -> None:
        ev = MemoryEvaluator()
        report = ev.evaluate(self._dataset(), retrieval_fn=lambda q: [1])
        assert len(report.per_case) == 2


# ===========================================================================
# Schemas — memory_layers
# ===========================================================================


class TestMemoryLayerSchemas:
    def test_session_summary_creation(self) -> None:
        ss = SessionSummary(
            id="session-1",
            session_index=1,
            summary="Worked on retrieval.",
            raw_memory_ids=[1, 2, 3],
            turn_count=3,
        )
        assert ss.kind == MemoryLayerKind.SESSION
        assert ss.turn_count == 3

    def test_daily_summary_creation(self) -> None:
        ds = DailySummary(
            id="daily-2025-07-15",
            date="2025-07-15",
            summary="Focused on retrieval subsystem.",
            session_count=2,
        )
        assert ds.kind == MemoryLayerKind.DAILY

    def test_weekly_summary_creation(self) -> None:
        ws = WeeklySummary(
            id="weekly-2025-W29",
            week_label="2025-W29",
            summary="Memory architecture evolved.",
            daily_count=5,
        )
        assert ws.kind == MemoryLayerKind.WEEKLY

    def test_project_summary_creation(self) -> None:
        ps = ProjectSummary(
            id="project-blix",
            project_name="Blix",
            summary="Long-term memory system.",
            goals=["Scalable memory"],
            current_status="active",
        )
        assert ps.kind == MemoryLayerKind.PROJECT
        assert "Scalable memory" in ps.goals

    def test_roundtrip_session(self) -> None:
        ss = SessionSummary(
            id="session-5",
            session_index=5,
            summary="NLP session.",
            raw_memory_ids=[10, 11],
            turn_count=2,
        )
        restored = SessionSummary.model_validate(ss.model_dump())
        assert restored.turn_count == 2
        assert restored.raw_memory_ids == [10, 11]
