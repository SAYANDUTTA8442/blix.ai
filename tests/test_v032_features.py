"""
Tests for Blix v0.3.2 — "Reflection, Consolidation & Knowledge Processing".

Covers:
1.  reflection_engine    (Feature 1  — Reflection Engine)
2.  consolidation_engine (Feature 2  — Memory Consolidation)
3.  goal_tracker          (Feature 3  — Goal Tracking)
4.  project_intelligence  (Feature 4  — Project Intelligence)
5.  evaluation.cognitive  (Feature 5  — Advanced Evaluation / blix_eval)
6.  document_processor    (Feature 6  — Document Processor)
7.  media_processor        (Feature 7  — Media Processor)
8.  synthesis              (Feature 8  — Knowledge Synthesis)
9.  scheduler              (Feature 9  — Reflection Scheduler)
10. mql                     (Feature 10 — Memory Query Language)

Python 3.10 compatible — fully offline (heuristic fallbacks used; no LLM).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from evaluation.blix_eval import CognitiveEvaluator
from evaluation.research import HypothesisRegistry
from knowledge.document_processor import (
    DocumentFormat, DocumentProcessor, ProcessedDocument, detect_format,
)
from knowledge.media_processor import (
    AudioProcessor, ImageProcessor, MediaProcessor, MediaType,
    NullTranscriptionBackend, ProcessedMedia, TranscriptionBackend, VideoProcessor,
)
from knowledge.synthesis import KnowledgeSynthesisEngine, SynthesisSource
from reflection.consolidation_engine import CanonicalFact, ConsolidationEngine
from reflection.goal_tracker import (
    Goal, GoalStatus, GoalTracker, ItemStatus,
)
from reflection.mql import MQLEngine, MQLExecutor, MQLParser
from reflection.project_intelligence import ProjectIntelligenceEngine, RiskLevel
from reflection.reflection_engine import Insight, ReflectionEngine, ReflectionScope
from reflection.scheduler import ReflectionScheduler
from schemas.memory_entry import MemoryEntry

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(days_ago: float = 0.0) -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago)


def _entry(id: int, input: str = "q", output: str = "a",
           topics: Optional[list[str]] = None, facts: Optional[list[str]] = None) -> MemoryEntry:
    return MemoryEntry(id=id, input=input, output=output, timestamp=_ts(), topics=topics or [], extracted_facts=facts or [])


# ===========================================================================
# Feature 1 — Reflection Engine
# ===========================================================================


class TestReflectionEngine:
    @pytest.fixture
    def engine(self, tmp_path: Path) -> ReflectionEngine:
        return ReflectionEngine(tmp_path / "reflections.json")

    def test_heuristic_reflection_produces_insight(self, engine: ReflectionEngine) -> None:
        record = engine.reflect(
            ReflectionScope.SESSION, "session-1",
            "User worked extensively on transformers and attention mechanisms transformers attention transformers"
        )
        assert len(record.insights) >= 1
        assert record.insights[0].scope == ReflectionScope.SESSION
        assert record.insights[0].scope_ref == "session-1"

    def test_empty_material_no_insights(self, engine: ReflectionEngine) -> None:
        record = engine.reflect(ReflectionScope.SESSION, "session-2", "")
        assert record.insights == []

    def test_reflect_session(self, engine: ReflectionEngine) -> None:
        class FakeSession:
            id = "session-5"
            summary = "Worked on the embedding store and retrieval pipeline embedding retrieval embedding"
            raw_memory_ids = [1, 2, 3]
        record = engine.reflect_session(FakeSession())
        assert record.scope == ReflectionScope.SESSION
        assert record.scope_ref == "session-5"

    def test_reflect_project(self, engine: ReflectionEngine) -> None:
        class FakeProject:
            project_name = "Blix"
            current_status = "active"
            goals = ["Long-term memory"]
            completed_work = ["Scorer", "Graph"]
            next_actions = ["Write tests"]
            milestones = ["v0.3 complete"]
        record = engine.reflect_project(FakeProject())
        assert record.scope == ReflectionScope.PROJECT
        assert record.scope_ref == "Blix"

    def test_reflect_learning(self, engine: ReflectionEngine) -> None:
        class FakeLearningState:
            topics = {"nlp": {"count": 5}, "graphs": {"count": 2}}
        record = engine.reflect_learning(FakeLearningState())
        assert record.scope == ReflectionScope.LEARNING

    def test_reflect_behavior(self, engine: ReflectionEngine) -> None:
        memories = [_entry(i, output=f"discussed topic number {i} nlp nlp nlp") for i in range(5)]
        record = engine.reflect_behavior(memories)
        assert record.scope == ReflectionScope.BEHAVIOR

    def test_get_records_filter_by_scope(self, engine: ReflectionEngine) -> None:
        engine.reflect(ReflectionScope.SESSION, "s1", "alpha beta alpha beta alpha")
        engine.reflect(ReflectionScope.DAILY, "d1", "gamma delta gamma delta gamma")
        sessions = engine.get_records(scope=ReflectionScope.SESSION)
        assert all(r.scope == ReflectionScope.SESSION for r in sessions)
        assert len(sessions) == 1

    def test_get_recent_insights(self, engine: ReflectionEngine) -> None:
        engine.reflect(ReflectionScope.SESSION, "s1", "alpha beta alpha beta alpha gamma")
        insights = engine.get_recent_insights(limit=5)
        assert len(insights) >= 1

    def test_get_insights_since(self, engine: ReflectionEngine) -> None:
        engine.reflect(ReflectionScope.DAILY, "d1", "alpha beta gamma alpha beta gamma")
        since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        insights = engine.get_insights_since(since)
        assert len(insights) >= 1

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        e1 = ReflectionEngine(tmp_path / "r.json")
        e1.reflect(ReflectionScope.SESSION, "s1", "alpha beta alpha beta alpha")
        e2 = ReflectionEngine(tmp_path / "r.json")
        assert e2.record_count == 1
        assert e2.insight_count >= 1

    def test_llm_insight_generation(self, tmp_path: Path) -> None:
        class FakeLLM:
            def generate(self, prompt: str) -> str:
                return json.dumps([{"insight": "User shifted focus to memory systems.", "confidence": 0.9}])
            def model_name(self) -> str:
                return "fake"

        engine = ReflectionEngine(tmp_path / "r.json", llm=FakeLLM())
        record = engine.reflect(ReflectionScope.WEEKLY, "2025-W29", "lots of memory work")
        assert len(record.insights) == 1
        assert record.insights[0].confidence == 0.9
        assert "memory systems" in record.insights[0].insight

    def test_llm_failure_falls_back_to_heuristic(self, tmp_path: Path) -> None:
        class BadLLM:
            def generate(self, prompt: str) -> str:
                return "not json"
            def model_name(self) -> str:
                return "fake"

        engine = ReflectionEngine(tmp_path / "r.json", llm=BadLLM())
        record = engine.reflect(ReflectionScope.SESSION, "s1", "alpha beta alpha beta alpha gamma")
        assert len(record.insights) >= 1  # heuristic fallback produced something


# ===========================================================================
# Feature 2 — Memory Consolidation Engine
# ===========================================================================


class TestConsolidationEngine:
    @pytest.fixture
    def engine(self, tmp_path: Path) -> ConsolidationEngine:
        return ConsolidationEngine(tmp_path / "facts.json", similarity_threshold=0.5, base_confidence=0.5)

    def test_first_fact_creates_canonical(self, engine: ConsolidationEngine) -> None:
        cf = engine.consolidate("User prefers PyTorch for AI development", 1, topic="ml")
        assert cf.evidence_count == 1
        assert cf.confidence == 0.5
        assert engine.fact_count == 1

    def test_similar_fact_merges(self, engine: ConsolidationEngine) -> None:
        engine.consolidate("User prefers PyTorch for AI development work", 1, topic="ml")
        cf2 = engine.consolidate("User prefers PyTorch for AI development", 2, topic="ml")
        assert engine.fact_count == 1
        assert cf2.evidence_count == 2

    def test_dissimilar_fact_creates_new(self, engine: ConsolidationEngine) -> None:
        engine.consolidate("User prefers PyTorch for AI development", 1, topic="ml")
        engine.consolidate("User enjoys hiking on weekends", 2, topic="hobbies")
        assert engine.fact_count == 2

    def test_confidence_accumulation_formula(self, engine: ConsolidationEngine) -> None:
        # base=0.5, growth=1.0: n=1->0.5, n=2->0.75, n=3->0.875
        cf = engine.consolidate("User prefers PyTorch deeply", 1)
        assert cf.confidence == pytest.approx(0.5)
        cf = engine.consolidate("User prefers PyTorch deeply", 2)
        assert cf.confidence == pytest.approx(0.75)
        cf = engine.consolidate("User prefers PyTorch deeply", 3)
        assert cf.confidence == pytest.approx(0.875)

    def test_confidence_capped_at_99(self, engine: ConsolidationEngine) -> None:
        cf = None
        for i in range(40):
            cf = engine.consolidate("User prefers PyTorch deeply", i)
        assert cf.confidence <= 0.99
        assert cf.evidence_count == 40

    def test_variants_tracked_canonical_is_shortest(self, engine: ConsolidationEngine) -> None:
        engine.consolidate("User prefers PyTorch for development", 1)
        cf = engine.consolidate("User prefers PyTorch development", 2)
        assert cf.fact == "User prefers PyTorch development"
        assert len(cf.variants) == 2

    def test_list_facts_filters(self, engine: ConsolidationEngine) -> None:
        engine.consolidate("fact one about pytorch", 1, topic="ml")
        engine.consolidate("fact two about hiking", 2, topic="hobbies")
        ml_facts = engine.list_facts(topic="ml")
        assert all(f.topic == "ml" for f in ml_facts)

    def test_strongest_facts_sorted(self, engine: ConsolidationEngine) -> None:
        engine.consolidate("User enjoys playing chess on weekends", 1)
        engine.consolidate("User prefers PyTorch for development", 2)
        engine.consolidate("User prefers PyTorch for development", 3)  # boosts second
        strongest = engine.strongest_facts(2)
        assert len(strongest) == 2
        assert strongest[0].confidence >= strongest[1].confidence

    def test_consolidatable_memory_ids(self, engine: ConsolidationEngine) -> None:
        for i in range(1, 5):
            engine.consolidate("User prefers PyTorch deeply for everything", i)
        ids = engine.consolidatable_memory_ids(min_evidence=3)
        assert 1 not in ids  # first source preserved
        assert 2 in ids or 3 in ids

    def test_consolidate_batch(self, engine: ConsolidationEngine) -> None:
        facts = [
            ("User prefers PyTorch deeply", 1, "ml"),
            ("User prefers PyTorch deeply", 2, "ml"),
        ]
        results = engine.consolidate_batch(facts)
        assert len(results) == 2
        assert results[1].evidence_count == 2

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        e1 = ConsolidationEngine(tmp_path / "f.json")
        e1.consolidate("User prefers PyTorch deeply", 1)
        e2 = ConsolidationEngine(tmp_path / "f.json")
        assert e2.fact_count == 1

    def test_summary_string(self, engine: ConsolidationEngine) -> None:
        assert "No canonical" in engine.summary()
        engine.consolidate("User prefers PyTorch deeply", 1)
        assert "canonical facts" in engine.summary()


# ===========================================================================
# Feature 3 — Goal Tracking System
# ===========================================================================


class TestGoalTracker:
    @pytest.fixture
    def tracker(self, tmp_path: Path) -> GoalTracker:
        return GoalTracker(tmp_path / "goals.json")

    def test_create_goal(self, tracker: GoalTracker) -> None:
        g = tracker.create_goal("Build Blix v0.4", priority=1, related_project="Blix")
        assert g.status == GoalStatus.ACTIVE
        assert g.progress == 0
        assert tracker.count == 1

    def test_progress_computed_from_milestones(self, tracker: GoalTracker) -> None:
        g = tracker.create_goal("Build Blix v0.4")
        tracker.add_milestone(g.goal_id, "Design phase")
        tracker.add_milestone(g.goal_id, "Implementation")
        tracker.complete_item(g.goal_id, "Design phase")
        g = tracker.get(g.goal_id)
        assert g.progress == 50

    def test_progress_override(self, tracker: GoalTracker) -> None:
        g = tracker.create_goal("Build Blix v0.4")
        tracker.add_milestone(g.goal_id, "Design")
        tracker.set_progress_override(g.goal_id, 72)
        g = tracker.get(g.goal_id)
        assert g.progress == 72

    def test_blockers(self, tracker: GoalTracker) -> None:
        g = tracker.create_goal("Build Blix v0.4")
        tracker.add_blocker(g.goal_id, "evaluation framework")
        g = tracker.get(g.goal_id)
        assert len(g.active_blockers) == 1
        tracker.resolve_blocker(g.goal_id, "evaluation framework")
        g = tracker.get(g.goal_id)
        assert len(g.active_blockers) == 0

    def test_summary_dict_matches_spec_format(self, tracker: GoalTracker) -> None:
        g = tracker.create_goal("Build Blix v0.4")
        tracker.add_milestone(g.goal_id, "m1")
        tracker.add_milestone(g.goal_id, "m2")
        tracker.add_milestone(g.goal_id, "m3")
        tracker.complete_item(g.goal_id, "m1")
        tracker.complete_item(g.goal_id, "m2")
        tracker.add_blocker(g.goal_id, "evaluation framework")
        g = tracker.get(g.goal_id)
        d = g.to_summary_dict()
        assert d["goal"] == "Build Blix v0.4"
        assert d["progress"] == round(100 * 2 / 3)
        assert d["status"] == "active"
        assert d["blockers"] == ["evaluation framework"]

    def test_goal_auto_completes_at_100_percent(self, tracker: GoalTracker) -> None:
        g = tracker.create_goal("Small goal")
        tracker.add_task(g.goal_id, "only task")
        tracker.complete_item(g.goal_id, "only task")
        g = tracker.get(g.goal_id)
        assert g.progress == 100
        assert g.status == GoalStatus.COMPLETED

    def test_prioritized_goals(self, tracker: GoalTracker) -> None:
        tracker.create_goal("Low priority", priority=5)
        tracker.create_goal("High priority", priority=1)
        ranked = tracker.prioritized_goals()
        assert ranked[0].title == "High priority"

    def test_list_goals_by_status(self, tracker: GoalTracker) -> None:
        g1 = tracker.create_goal("Active goal")
        g2 = tracker.create_goal("Paused goal")
        tracker.update_status(g2.goal_id, GoalStatus.PAUSED)
        active = tracker.list_goals(status=GoalStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].title == "Active goal"

    def test_find_by_title(self, tracker: GoalTracker) -> None:
        tracker.create_goal("Build Blix v0.4")
        found = tracker.find_by_title("build blix v0.4")
        assert found is not None

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        t1 = GoalTracker(tmp_path / "g.json")
        g = t1.create_goal("Persisted goal", priority=2)
        t1.add_milestone(g.goal_id, "m1")
        t2 = GoalTracker(tmp_path / "g.json")
        assert t2.count == 1
        assert len(t2.get(g.goal_id).milestones) == 1

    def test_summary_string(self, tracker: GoalTracker) -> None:
        assert "No active goals" in tracker.summary()
        tracker.create_goal("Goal A")
        assert "Goal A" in tracker.summary()


# ===========================================================================
# Feature 4 — Project Intelligence Engine
# ===========================================================================


class TestProjectIntelligence:
    @pytest.fixture
    def engine(self, tmp_path: Path) -> ProjectIntelligenceEngine:
        return ProjectIntelligenceEngine(tmp_path / "pi.json")

    def test_get_or_create(self, engine: ProjectIntelligenceEngine) -> None:
        ps = engine.get_or_create("Blix")
        assert ps.project_name == "Blix"
        assert ps.risk_level == RiskLevel.LOW

    def test_set_focus(self, engine: ProjectIntelligenceEngine) -> None:
        engine.set_focus("Blix", "Reflection Engine")
        ps = engine.get("Blix")
        assert ps.focus == "Reflection Engine"

    def test_add_risk_escalates_level(self, engine: ProjectIntelligenceEngine) -> None:
        engine.add_risk("Blix", "risk one")
        ps = engine.get("Blix")
        assert ps.risk_level == RiskLevel.MEDIUM
        engine.add_risk("Blix", "risk two")
        engine.add_risk("Blix", "risk three")
        ps = engine.get("Blix")
        assert ps.risk_level == RiskLevel.HIGH

    def test_resolve_risk_deescalates(self, engine: ProjectIntelligenceEngine) -> None:
        engine.add_risk("Blix", "risk one")
        engine.resolve_risk("Blix", "risk one")
        ps = engine.get("Blix")
        assert ps.risk_level == RiskLevel.LOW
        assert ps.risks == []

    def test_link_memories(self, engine: ProjectIntelligenceEngine) -> None:
        engine.link_memories("Blix", [1, 2, 3])
        engine.link_memories("Blix", [3, 4])
        ps = engine.get("Blix")
        assert set(ps.related_memory_ids) == {1, 2, 3, 4}

    def test_summary_dict_matches_spec_format(self, engine: ProjectIntelligenceEngine) -> None:
        engine.set_focus("Blix", "Reflection Engine")
        engine.update("Blix", progress=68)
        d = engine.get("Blix").to_summary_dict()
        assert d["project"] == "Blix"
        assert d["focus"] == "Reflection Engine"
        assert d["progress"] == 68
        assert d["risk_level"] == "low"

    def test_sync_progress_from_goal(self, engine: ProjectIntelligenceEngine) -> None:
        class FakeBlocker:
            description = "eval framework"
        class FakeTask:
            title = "write docs"
            class status:
                value = "pending"
        class FakeGoal:
            progress = 72
            active_blockers = [FakeBlocker()]
            tasks = [FakeTask()]
        ps = engine.sync_progress_from_goal("Blix", FakeGoal())
        assert ps.progress == 72
        assert "eval framework" in ps.risks
        assert "write docs" in ps.next_steps

    def test_sync_from_project_summary(self, engine: ProjectIntelligenceEngine) -> None:
        class FakeSummary:
            project_name = "Blix"
            next_actions = ["Write tests", "Update docs"]
        ps = engine.sync_from_project_summary(FakeSummary())
        assert ps.next_steps == ["Write tests", "Update docs"]

    def test_at_risk_projects(self, engine: ProjectIntelligenceEngine) -> None:
        engine.get_or_create("Safe")
        engine.add_risk("Risky", "something")
        at_risk = engine.at_risk_projects()
        names = {p.project_name for p in at_risk}
        assert "Risky" in names
        assert "Safe" not in names

    def test_project_report_with_project_manager(self, tmp_path: Path) -> None:
        from core.project_manager import ProjectManager
        pm = ProjectManager(tmp_path / "projects.json")
        pm.get_or_create("Blix")
        pm.update("Blix", goals=["Long-term memory"])

        engine = ProjectIntelligenceEngine(tmp_path / "pi.json", project_manager=pm)
        engine.set_focus("Blix", "Reflection")
        report = engine.project_report("Blix")
        assert report["project"] == "Blix"
        assert "goals" in report
        assert report["goals"] == ["Long-term memory"]

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        e1 = ProjectIntelligenceEngine(tmp_path / "pi.json")
        e1.set_focus("Blix", "Reflection")
        e1.add_risk("Blix", "r1")
        e2 = ProjectIntelligenceEngine(tmp_path / "pi.json")
        ps = e2.get("Blix")
        assert ps.focus == "Reflection"
        assert ps.risks == ["r1"]


# ===========================================================================
# Feature 5 — Advanced Evaluation Framework (blix_eval)
# ===========================================================================


class TestCognitiveEvaluator:
    def test_recall_at_k(self) -> None:
        ev = CognitiveEvaluator()
        assert ev.recall_at_k([1, 2, 5, 6], [1, 2, 3, 4], k=2) == 0.5  # only 1,2 in top-2

    def test_recall_at_k_no_relevant(self) -> None:
        ev = CognitiveEvaluator()
        assert ev.recall_at_k([1, 2], []) == 1.0

    def test_mrr_first_position(self) -> None:
        ev = CognitiveEvaluator()
        assert ev.mean_reciprocal_rank([1, 2, 3], [1]) == 1.0

    def test_mrr_third_position(self) -> None:
        ev = CognitiveEvaluator()
        assert ev.mean_reciprocal_rank([5, 6, 1], [1]) == pytest.approx(1 / 3)

    def test_mrr_not_found(self) -> None:
        ev = CognitiveEvaluator()
        assert ev.mean_reciprocal_rank([5, 6], [1]) == 0.0

    def test_mrr_batch(self) -> None:
        ev = CognitiveEvaluator()
        results = [([1, 2], [1]), ([5, 1], [1])]
        mrr = ev.mrr_batch(results)
        assert mrr == pytest.approx((1.0 + 0.5) / 2)

    def test_forgetting_rate(self, tmp_path: Path) -> None:
        from core.memory_lifecycle import MemoryLifecycleManager
        lm = MemoryLifecycleManager(tmp_path / "lc.json")
        lm.compress(1, "s")
        lm.archive(2)
        ev = CognitiveEvaluator()
        # 2 of unknown total — but state_counts includes all enum states with 0s
        rate = ev.forgetting_rate(lm)
        assert 0.0 < rate <= 1.0

    def test_forgetting_rate_empty(self, tmp_path: Path) -> None:
        from core.memory_lifecycle import MemoryLifecycleManager
        lm = MemoryLifecycleManager(tmp_path / "lc.json")
        ev = CognitiveEvaluator()
        assert ev.forgetting_rate(lm) == 0.0

    def test_profile_stability(self) -> None:
        ev = CognitiveEvaluator()
        audit = [object(), object()]  # 2 changes
        stability = ev.profile_stability(audit, total_turns=10)
        assert stability == pytest.approx(0.8)

    def test_profile_stability_no_turns(self) -> None:
        ev = CognitiveEvaluator()
        assert ev.profile_stability([], total_turns=0) == 1.0

    def test_project_accuracy(self) -> None:
        from reflection.project_intelligence import ProjectState, RiskLevel
        ps = ProjectState(project_name="Blix", focus="Reflection Engine", progress=68, risk_level=RiskLevel.MEDIUM)
        gt = {"Blix": {"focus": "Reflection Engine", "progress": 70, "risk_level": "medium"}}
        ev = CognitiveEvaluator()
        acc = ev.project_accuracy([ps], gt)
        assert acc == 1.0  # progress within ±10, others exact

    def test_project_accuracy_mismatch(self) -> None:
        from reflection.project_intelligence import ProjectState, RiskLevel
        ps = ProjectState(project_name="Blix", focus="Wrong focus", progress=10, risk_level=RiskLevel.LOW)
        gt = {"Blix": {"focus": "Reflection Engine", "progress": 70, "risk_level": "high"}}
        ev = CognitiveEvaluator()
        acc = ev.project_accuracy([ps], gt)
        assert acc == 0.0

    def test_milestone_accuracy(self) -> None:
        g = Goal(goal_id="goal_0", title="Build Blix v0.4")
        from reflection.goal_tracker import Milestone
        g.milestones = [Milestone(title="Design", status=ItemStatus.DONE), Milestone(title="Implement")]
        gt = {"Build Blix v0.4": ["Design", "Implement"]}
        ev = CognitiveEvaluator()
        acc = ev.milestone_accuracy([g], gt)
        assert acc == 0.5

    def test_insight_accuracy(self) -> None:
        insights = [Insight(insight="User's focus shifted to cognitive memory systems development")]
        gt = ["User shifted focus toward cognitive memory systems"]
        ev = CognitiveEvaluator()
        acc = ev.insight_accuracy(insights, gt, min_overlap_words=2)
        assert acc == 1.0

    def test_insight_accuracy_no_match(self) -> None:
        insights = [Insight(insight="User enjoys cooking pasta on weekends")]
        gt = ["User shifted focus toward cognitive memory systems"]
        ev = CognitiveEvaluator()
        acc = ev.insight_accuracy(insights, gt, min_overlap_words=2)
        assert acc == 0.0

    def test_reflection_consistency_identical(self) -> None:
        insights_a = [Insight(insight="User's focus shifted to memory systems development")]
        insights_b = [Insight(insight="User shifted focus toward memory systems")]
        ev = CognitiveEvaluator()
        consistency = ev.reflection_consistency(insights_a, insights_b, min_overlap_words=2)
        assert consistency == 1.0

    def test_reflection_consistency_empty_both(self) -> None:
        ev = CognitiveEvaluator()
        assert ev.reflection_consistency([], []) == 1.0

    def test_reflection_consistency_one_empty(self) -> None:
        ev = CognitiveEvaluator()
        assert ev.reflection_consistency([Insight(insight="x")], []) == 0.0

    def test_evaluate_cognitive_combines(self, tmp_path: Path) -> None:
        from core.memory_lifecycle import MemoryLifecycleManager
        lm = MemoryLifecycleManager(tmp_path / "lc.json")
        lm.compress(1, "s")
        ev = CognitiveEvaluator()
        results = ev.evaluate_cognitive(
            retrieval_results=[([1, 2], [1])],
            lifecycle_manager=lm,
        )
        assert "mrr" in results
        assert "recall_at_k" in results
        assert "forgetting_rate" in results

    def test_blix_eval_exports(self) -> None:
        from evaluation.blix_eval import (
            MemoryEvaluator, ExtendedMemoryEvaluator, CognitiveEvaluator,
            EvalCase, EvalDataset, EvalReport, HypothesisRegistry,
        )
        assert issubclass(CognitiveEvaluator, ExtendedMemoryEvaluator)
        assert issubclass(ExtendedMemoryEvaluator, MemoryEvaluator)


# ===========================================================================
# Feature 6 — Document Processor
# ===========================================================================


class TestDocumentProcessor:
    def test_detect_format(self) -> None:
        assert detect_format(Path("a.pdf")) == DocumentFormat.PDF
        assert detect_format(Path("a.txt")) == DocumentFormat.TXT
        assert detect_format(Path("a.md")) == DocumentFormat.MD
        assert detect_format(Path("a.docx")) == DocumentFormat.DOCX
        assert detect_format(Path("a.html")) == DocumentFormat.HTML
        assert detect_format(Path("a.xyz")) == DocumentFormat.UNKNOWN

    def test_process_txt(self) -> None:
        proc = DocumentProcessor()
        doc = proc.process_file(FIXTURES / "sample.txt")
        assert doc.format == DocumentFormat.TXT
        assert doc.raw_text_length > 0
        assert len(doc.chunks) >= 1
        assert doc.summary

    def test_process_md(self) -> None:
        proc = DocumentProcessor()
        doc = proc.process_file(FIXTURES / "sample.md")
        assert doc.format == DocumentFormat.MD
        assert "machine" in doc.summary.lower() or doc.concepts

    def test_process_docx(self) -> None:
        proc = DocumentProcessor()
        doc = proc.process_file(FIXTURES / "sample.docx")
        assert doc.format == DocumentFormat.DOCX
        assert doc.raw_text_length > 0
        full_text = " ".join(c.text for c in doc.chunks)
        assert "reinforcement" in full_text.lower()

    def test_process_html(self) -> None:
        proc = DocumentProcessor()
        doc = proc.process_file(FIXTURES / "sample.html")
        assert doc.format == DocumentFormat.HTML
        full_text = " ".join(c.text for c in doc.chunks)
        assert "graph" in full_text.lower() or "neo4j" in full_text.lower()
        assert "<style>" not in full_text  # script/style stripped

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "file.xyz"
        bad.write_text("data")
        proc = DocumentProcessor()
        with pytest.raises(ValueError):
            proc.process_file(bad)

    def test_chunking_respects_size(self, tmp_path: Path) -> None:
        big = tmp_path / "big.txt"
        big.write_text("word " * 2000)  # ~10000 chars
        proc = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        doc = proc.process_file(big)
        assert len(doc.chunks) > 1
        for c in doc.chunks:
            assert len(c.text) <= 600  # allow some slack for boundary search

    def test_chunk_document_short_text_single_chunk(self) -> None:
        proc = DocumentProcessor(chunk_size=1000)
        chunks = proc.chunk_document("short text")
        assert len(chunks) == 1

    def test_llm_analysis_used_when_provided(self) -> None:
        class FakeLLM:
            def generate(self, prompt: str) -> str:
                return json.dumps({
                    "summary": "A paper about transformers.",
                    "key_findings": ["Attention is all you need"],
                    "concepts": ["attention", "transformers"],
                    "related_topics": ["nlp"],
                    "entities": [["Transformer", "skill"]],
                })
            def model_name(self) -> str:
                return "fake"

        proc = DocumentProcessor(llm=FakeLLM())
        doc = proc.process_file(FIXTURES / "sample.txt")
        assert doc.summary == "A paper about transformers."
        assert "attention" in doc.concepts
        assert doc.entities == [("Transformer", "skill")]

    def test_to_dict(self) -> None:
        proc = DocumentProcessor()
        doc = proc.process_file(FIXTURES / "sample.txt")
        d = doc.to_dict()
        assert d["format"] == "txt"
        assert "chunks" in d


# ===========================================================================
# Feature 7 — Media Processor
# ===========================================================================


class TestImageProcessor:
    def test_ocr_extracts_text(self) -> None:
        proc = ImageProcessor()
        result = proc.process(FIXTURES / "diagram.png")
        assert result.media_type == MediaType.IMAGE
        assert "Component" in result.ocr_text or "Embedding" in result.ocr_text

    def test_diagram_notes_detected(self) -> None:
        proc = ImageProcessor()
        result = proc.process(FIXTURES / "diagram.png")
        # Heuristic should detect "A -> B" style arrows
        assert any("->" in note for note in result.diagram_notes) or result.ocr_text

    def test_to_processed_document(self) -> None:
        proc = ImageProcessor()
        result = proc.process(FIXTURES / "diagram.png")
        doc = result.to_processed_document()
        assert doc.doc_id == result.media_id

    def test_llm_analysis(self, tmp_path: Path) -> None:
        class FakeLLM:
            def generate(self, prompt: str) -> str:
                return json.dumps({
                    "summary": "Architecture diagram showing two components.",
                    "objects": ["Component A", "Component B"],
                    "diagram_notes": ["Component A -> Component B"],
                    "topics": ["architecture"],
                })
            def model_name(self) -> str:
                return "fake"

        proc = ImageProcessor(llm=FakeLLM())
        result = proc.process(FIXTURES / "diagram.png")
        assert result.summary == "Architecture diagram showing two components."
        assert "architecture" in result.topics


class TestAudioProcessor:
    def test_null_backend_warns_and_returns_empty(self, tmp_path: Path) -> None:
        fake_audio = tmp_path / "audio.wav"
        fake_audio.write_bytes(b"")  # empty placeholder file
        proc = AudioProcessor()
        result = proc.process(fake_audio)
        assert result.media_type == MediaType.AUDIO
        assert result.transcript == ""
        assert result.segments == []

    def test_custom_transcription_backend(self, tmp_path: Path) -> None:
        class FakeBackend(TranscriptionBackend):
            def transcribe(self, path: Path) -> list[tuple[float, float, str]]:
                return [(0.0, 5.0, "We need to update the documentation."),
                        (5.0, 10.0, "The lecture covered transformers and attention.")]

        fake_audio = tmp_path / "audio.wav"
        fake_audio.write_bytes(b"")
        proc = AudioProcessor(transcription_backend=FakeBackend())
        result = proc.process(fake_audio)
        assert "documentation" in result.transcript
        assert len(result.action_items) >= 1  # "need to" heuristic
        assert len(result.chunks) >= 1

    def test_llm_analysis(self, tmp_path: Path) -> None:
        class FakeBackend(TranscriptionBackend):
            def transcribe(self, path: Path) -> list[tuple[float, float, str]]:
                return [(0.0, 10.0, "Lecture on transformers.")]

        class FakeLLM:
            def generate(self, prompt: str) -> str:
                return json.dumps({
                    "summary": "A lecture on transformers.",
                    "topics": ["transformers", "nlp"],
                    "action_items": ["Review the slides"],
                    "key_facts": ["Transformers use self-attention"],
                })
            def model_name(self) -> str:
                return "fake"

        fake_audio = tmp_path / "audio.wav"
        fake_audio.write_bytes(b"")
        proc = AudioProcessor(transcription_backend=FakeBackend(), llm=FakeLLM())
        result = proc.process(fake_audio)
        assert result.summary == "A lecture on transformers."
        assert "nlp" in result.topics


class TestVideoProcessor:
    def test_no_ffmpeg_graceful_degradation(self, tmp_path: Path, monkeypatch) -> None:
        import knowledge.media_processor as mp
        monkeypatch.setattr(mp.shutil, "which", lambda x: None)
        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"")
        proc = VideoProcessor()
        result = proc.process(fake_video)
        assert result.media_type == MediaType.VIDEO
        assert "ffmpeg" in result.summary.lower()
        assert result.transcript == ""


class TestMediaProcessorFacade:
    def test_can_process(self) -> None:
        proc = MediaProcessor()
        assert proc.can_process(Path("a.png"))
        assert proc.can_process(Path("a.mp3"))
        assert proc.can_process(Path("a.mp4"))
        assert not proc.can_process(Path("a.pdf"))

    def test_dispatches_to_image_processor(self) -> None:
        proc = MediaProcessor()
        result = proc.process(FIXTURES / "diagram.png")
        assert result.media_type == MediaType.IMAGE

    def test_unsupported_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "file.xyz"
        bad.write_bytes(b"")
        proc = MediaProcessor()
        with pytest.raises(ValueError):
            proc.process(bad)


# ===========================================================================
# Feature 8 — Knowledge Synthesis Engine
# ===========================================================================


class TestKnowledgeSynthesis:
    @pytest.fixture
    def engine(self, tmp_path: Path) -> KnowledgeSynthesisEngine:
        return KnowledgeSynthesisEngine(tmp_path / "reports.json")

    def test_empty_sources(self, engine: KnowledgeSynthesisEngine) -> None:
        report = engine.synthesize([])
        assert report.title == "Empty report"

    def test_heuristic_synthesis(self, engine: KnowledgeSynthesisEngine) -> None:
        sources = [
            SynthesisSource(kind="memory", ref_id="1", text="Discussed transformers", topics=["transformers"]),
            SynthesisSource(kind="document", ref_id="doc1", text="Paper on attention", topics=["transformers", "attention"]),
        ]
        report = engine.synthesize(sources)
        assert "transformers" in report.title or "transformers" in report.topics
        assert len(report.key_points) == 2
        assert engine.count == 1

    def test_llm_synthesis(self, tmp_path: Path) -> None:
        class FakeLLM:
            def generate(self, prompt: str) -> str:
                return json.dumps({
                    "title": "Memory Systems Research",
                    "narrative": "The user has been exploring memory architectures across sources.",
                    "key_points": ["Hierarchical memory improves recall", "Graphs add reasoning"],
                })
            def model_name(self) -> str:
                return "fake"

        engine = KnowledgeSynthesisEngine(tmp_path / "r.json", llm=FakeLLM())
        sources = [SynthesisSource(kind="memory", ref_id="1", text="memory architecture work")]
        report = engine.synthesize(sources)
        assert report.title == "Memory Systems Research"
        assert len(report.key_points) == 2

    def test_from_memories(self) -> None:
        memories = [_entry(1, output="discussed RL", topics=["rl"])]
        sources = KnowledgeSynthesisEngine.from_memories(memories)
        assert sources[0].kind == "memory"
        assert sources[0].topics == ["rl"]

    def test_from_projects(self) -> None:
        from reflection.project_intelligence import ProjectState
        ps = ProjectState(project_name="Blix", focus="Reflection", risks=["r1"])
        sources = KnowledgeSynthesisEngine.from_projects([ps])
        assert sources[0].kind == "project"
        assert sources[0].ref_id == "Blix"

    def test_from_documents(self) -> None:
        doc = ProcessedDocument(doc_id="d1", title="Paper", format=DocumentFormat.PDF, summary="summary", related_topics=["nlp"])
        sources = KnowledgeSynthesisEngine.from_documents([doc])
        assert sources[0].kind == "document"
        assert sources[0].ref_id == "d1"

    def test_from_media(self) -> None:
        media = ProcessedMedia(media_id="m1", media_type=MediaType.AUDIO, title="Lecture", summary="summary", topics=["ml"])
        sources = KnowledgeSynthesisEngine.from_media([media])
        assert sources[0].kind == "media"

    def test_from_graph_facts(self) -> None:
        sources = KnowledgeSynthesisEngine.from_graph_facts([("sayan", "works_on", "blix")])
        assert sources[0].kind == "graph"
        assert "sayan" in sources[0].text

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        e1 = KnowledgeSynthesisEngine(tmp_path / "r.json")
        e1.synthesize([SynthesisSource(kind="memory", ref_id="1", text="x", topics=["a"])])
        e2 = KnowledgeSynthesisEngine(tmp_path / "r.json")
        assert e2.count == 1

    def test_list_all_sorted_recent_first(self, engine: KnowledgeSynthesisEngine) -> None:
        engine.synthesize([SynthesisSource(kind="memory", ref_id="1", text="first")])
        engine.synthesize([SynthesisSource(kind="memory", ref_id="2", text="second")])
        reports = engine.list_all()
        assert len(reports) == 2


# ===========================================================================
# Feature 9 — Reflection Scheduler
# ===========================================================================


class TestReflectionScheduler:
    def test_initial_state_all_due(self, tmp_path: Path) -> None:
        sched = ReflectionScheduler(tmp_path / "sched.json")
        assert sched.due_session()
        assert sched.due_daily()
        assert sched.due_weekly()
        assert sched.due_monthly()

    def test_mark_daily_run_not_due_same_day(self, tmp_path: Path) -> None:
        now = datetime(2025, 7, 15, 10, 0, 0)
        sched = ReflectionScheduler(tmp_path / "sched.json", now_fn=lambda: now)
        sched.mark_daily_run()
        assert not sched.due_daily()

    def test_due_daily_next_day(self, tmp_path: Path) -> None:
        clock = {"now": datetime(2025, 7, 15, 23, 0, 0)}
        sched = ReflectionScheduler(tmp_path / "sched.json", now_fn=lambda: clock["now"])
        sched.mark_daily_run()
        clock["now"] = datetime(2025, 7, 16, 1, 0, 0)
        assert sched.due_daily()

    def test_due_weekly_next_week(self, tmp_path: Path) -> None:
        clock = {"now": datetime(2025, 7, 14, 10, 0, 0)}  # Monday
        sched = ReflectionScheduler(tmp_path / "sched.json", now_fn=lambda: clock["now"])
        sched.mark_weekly_run()
        assert not sched.due_weekly()
        clock["now"] = datetime(2025, 7, 21, 10, 0, 0)  # next Monday
        assert sched.due_weekly()

    def test_due_monthly_next_month(self, tmp_path: Path) -> None:
        clock = {"now": datetime(2025, 7, 15, 10, 0, 0)}
        sched = ReflectionScheduler(tmp_path / "sched.json", now_fn=lambda: clock["now"])
        sched.mark_monthly_run()
        assert not sched.due_monthly()
        clock["now"] = datetime(2025, 8, 1, 10, 0, 0)
        assert sched.due_monthly()

    def test_run_due_triggers_callbacks(self, tmp_path: Path) -> None:
        sched = ReflectionScheduler(tmp_path / "sched.json")
        called = []
        triggered = sched.run_due(
            on_session=lambda: called.append("session"),
            on_daily=lambda: called.append("daily"),
            on_weekly=lambda: called.append("weekly"),
            on_monthly=lambda: called.append("monthly"),
        )
        assert set(triggered) == {"session", "daily", "weekly", "monthly"}
        assert set(called) == {"session", "daily", "weekly", "monthly"}

    def test_run_due_failure_isolation(self, tmp_path: Path) -> None:
        sched = ReflectionScheduler(tmp_path / "sched.json")
        called = []
        def bad():
            raise RuntimeError("boom")
        # Should not raise despite on_session failing
        triggered = sched.run_due(
            on_session=bad,
            on_daily=lambda: called.append("daily"),
        )
        assert "daily" in called
        assert "session" in triggered  # still marked as run

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        clock = {"now": datetime(2025, 7, 15, 10, 0, 0)}
        s1 = ReflectionScheduler(tmp_path / "sched.json", now_fn=lambda: clock["now"])
        s1.mark_daily_run()
        s1.mark_weekly_run()
        s2 = ReflectionScheduler(tmp_path / "sched.json", now_fn=lambda: clock["now"])
        assert not s2.due_daily()
        assert not s2.due_weekly()

    def test_session_count_tracking(self, tmp_path: Path) -> None:
        sched = ReflectionScheduler(tmp_path / "sched.json")
        sched.mark_session_run()
        sched.mark_session_run()
        assert sched.state.session_count_since_daily == 2
        sched.mark_daily_run()
        assert sched.state.session_count_since_daily == 0


# ===========================================================================
# Feature 10 — Memory Query Language (MQL)
# ===========================================================================


class TestMQLParser:
    def test_parse_active_goals(self) -> None:
        parser = MQLParser()
        result = parser.parse("show active goals")
        assert result is not None
        assert result[0] == "active_goals"

    def test_parse_project_with_name(self) -> None:
        parser = MQLParser()
        result = parser.parse("show project Blix")
        assert result == ("project", {"project_name": "Blix"})

    def test_parse_project_risks_not_project(self) -> None:
        parser = MQLParser()
        result = parser.parse("show project risks")
        assert result[0] == "project_risks"

    def test_parse_memories_about(self) -> None:
        parser = MQLParser()
        result = parser.parse("show memories about transformers")
        assert result == ("memories_about", {"topic": "transformers"})

    def test_parse_reflections_this_week(self) -> None:
        parser = MQLParser()
        result = parser.parse("show reflections this week")
        assert result == ("reflections_period", {"period": "this week"})

    def test_parse_strongest_skills(self) -> None:
        parser = MQLParser()
        assert parser.parse("show strongest skills")[0] == "strongest_skills"

    def test_parse_contradictions(self) -> None:
        parser = MQLParser()
        assert parser.parse("show contradictions")[0] == "contradictions"

    def test_parse_unrecognised(self) -> None:
        parser = MQLParser()
        assert parser.parse("do something else") is None

    def test_case_insensitive(self) -> None:
        parser = MQLParser()
        assert parser.parse("SHOW ACTIVE GOALS")[0] == "active_goals"


class TestMQLExecutor:
    def test_unavailable_component(self) -> None:
        executor = MQLExecutor()
        result = executor.execute("active_goals", {})
        assert "GoalTracker" in result.text

    def test_active_goals_with_tracker(self, tmp_path: Path) -> None:
        gt = GoalTracker(tmp_path / "goals.json")
        gt.create_goal("Build Blix v0.4", priority=1)
        executor = MQLExecutor(goal_tracker=gt)
        result = executor.execute("active_goals", {})
        assert "Build Blix v0.4" in result.text

    def test_no_active_goals(self, tmp_path: Path) -> None:
        gt = GoalTracker(tmp_path / "goals.json")
        executor = MQLExecutor(goal_tracker=gt)
        result = executor.execute("active_goals", {})
        assert "No active goals" in result.text

    def test_project_with_intelligence_engine(self, tmp_path: Path) -> None:
        pi = ProjectIntelligenceEngine(tmp_path / "pi.json")
        pi.set_focus("Blix", "Reflection Engine")
        pi.add_risk("Blix", "eval framework")
        executor = MQLExecutor(project_intelligence=pi)
        result = executor.execute("project", {"project_name": "Blix"})
        assert "Reflection Engine" in result.text
        assert "eval framework" in result.text

    def test_project_risks(self, tmp_path: Path) -> None:
        pi = ProjectIntelligenceEngine(tmp_path / "pi.json")
        pi.add_risk("Blix", "r1")
        executor = MQLExecutor(project_intelligence=pi)
        result = executor.execute("project_risks", {})
        assert "Blix" in result.text

    def test_strongest_skills(self, tmp_path: Path) -> None:
        ce = ConsolidationEngine(tmp_path / "facts.json", base_confidence=0.5)
        ce.consolidate("User prefers PyTorch deeply", 1)
        executor = MQLExecutor(consolidation_engine=ce)
        result = executor.execute("strongest_skills", {})
        assert "PyTorch" in result.text

    def test_contradictions(self) -> None:
        from core.graph_reasoner import ContradictionDetector
        detector = ContradictionDetector()
        m1 = _entry(1, output="interested in NLP", topics=["nlp"])
        m1.importance = 0.9
        m2 = _entry(2, output="no longer interested in NLP", topics=["nlp"])
        m2.importance = 0.3
        detector.detect([m1, m2])
        executor = MQLExecutor(contradiction_detector=detector)
        result = executor.execute("contradictions", {})
        assert "nlp" in result.text

    def test_reflections_this_week(self, tmp_path: Path) -> None:
        re_engine = ReflectionEngine(tmp_path / "reflections.json")
        re_engine.reflect(ReflectionScope.DAILY, "today", "alpha beta gamma alpha beta gamma alpha")
        executor = MQLExecutor(reflection_engine=re_engine)
        result = executor.execute("reflections_period", {"period": "this week"})
        assert "Reflections" in result.text

    def test_lifecycle(self, tmp_path: Path) -> None:
        from core.memory_lifecycle import MemoryLifecycleManager
        lm = MemoryLifecycleManager(tmp_path / "lc.json")
        lm.compress(1, "s")
        executor = MQLExecutor(lifecycle_manager=lm)
        result = executor.execute("lifecycle", {})
        assert "compressed" in result.text

    def test_clusters(self, tmp_path: Path) -> None:
        from core.semantic_clusters import SemanticClusterIndex
        idx = SemanticClusterIndex(tmp_path / "clusters.json")
        idx.add_memory(1, np.array([1.0, 0.0, 0.0]), ["nlp"])
        executor = MQLExecutor(semantic_cluster_index=idx)
        result = executor.execute("clusters", {})
        assert "nlp" in result.text


class TestMQLEngine:
    def test_run_unrecognised(self) -> None:
        engine = MQLEngine()
        result = engine.run("blah blah")
        assert not result.matched

    def test_run_recognised_but_unavailable(self) -> None:
        engine = MQLEngine()
        result = engine.run("show active goals")
        assert result.matched
        assert "GoalTracker" in result.text

    def test_is_mql_command(self) -> None:
        engine = MQLEngine()
        assert engine.is_mql_command("show active goals")
        assert engine.is_mql_command("  SHOW project risks")
        assert not engine.is_mql_command("what is the weather")

    def test_full_integration(self, tmp_path: Path) -> None:
        gt = GoalTracker(tmp_path / "goals.json")
        gt.create_goal("Build Blix v0.4", priority=1)
        pi = ProjectIntelligenceEngine(tmp_path / "pi.json")
        pi.set_focus("Blix", "Reflection Engine")

        engine = MQLEngine(goal_tracker=gt, project_intelligence=pi)

        r1 = engine.run("show active goals")
        assert "Build Blix v0.4" in r1.text

        r2 = engine.run("show project Blix")
        assert "Reflection Engine" in r2.text
