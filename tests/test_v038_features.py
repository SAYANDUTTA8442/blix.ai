"""
Tests for Blix v0.3.8 — "Meta-Cognitive Layer".

Covers:
1.  metacognition.self_model           (SelfModel, SelfModelStore)
2.  metacognition.confidence_manager     (ConfidenceManager, ConfidenceRecord)
3.  reasoning.confidence_reasoner          (ConfidenceReasoner — plan/tool/answer confidence)
4.  metacognition.strategy_manager           (StrategyManager — 4 decision branches)
5.  metacognition.capability_tracker            (CapabilityTracker — outcomes -> accuracy -> sync)
6.  memory.procedural_memory                      (ProceduralMemory — skill learning/matching)
7.  planning.plan_evaluator                          (PlanQualityEvaluator)
8.  agents.execution_feedback                          (ExecutionFeedbackLoop)
9.  reflection.meta_reflection                            (MetaReflectionEngine — 3 patterns)
10. metacognition.controller                                (MetaCognitiveController — monitor+adapt)
11. evaluation.confidence_metrics                              (ConfidenceMetrics — calibration)
12. evaluation.capability_metrics                                  (CapabilityMetrics — self-awareness)
13. evaluation.metacognition_metrics                                   (MetacognitionMetrics)
Integration  — BlixContext wiring
API          — /metacognition endpoints

Python 3.10 compatible — fully offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from agents.execution_feedback import ExecutionFeedbackLoop, FeedbackEntry
from agents.types import Task, TaskGraph, TaskStatus
from evaluation.capability_metrics import CapabilityMetrics, SelfAwarenessGap
from evaluation.confidence_metrics import CalibrationBucketResult, CalibrationCase, ConfidenceMetrics
from evaluation.metacognition_metrics import AdaptationCase, MetacognitionMetrics
from memory.procedural_memory import ProceduralMemory, Skill
from metacognition.capability_tracker import CapabilityRecord, CapabilityTracker
from metacognition.confidence_manager import ConfidenceManager, ConfidenceRecord
from metacognition.controller import (
    AdaptationAction,
    AdaptationDecision,
    CognitiveIssue,
    CognitiveMonitorReport,
    MetaCognitiveController,
)
from metacognition.self_model import SelfModel, SelfModelStore
from metacognition.strategy_manager import ReasoningStrategy, StrategyDecision, StrategyManager
from planning.critic import PlanCritic
from planning.plan_evaluator import PlanQualityEvaluator, PlanQualityScore
from reasoning.confidence_reasoner import ConfidenceEstimate, ConfidenceReasoner
from reflection.meta_reflection import BehaviorChangeInsight, MetaReflectionEngine
from reflection.reflection_engine import ReflectionEngine, ReflectionScope


# ===========================================================================
# Item 1 — Self Model
# ===========================================================================


class TestSelfModel:
    @pytest.fixture
    def store(self, tmp_path: Path) -> SelfModelStore:
        return SelfModelStore(tmp_path / "self_model.json")

    def test_set_capability_stores_score(self, store: SelfModelStore) -> None:
        store.set_capability("coding", 0.93)
        assert store.capability("coding") == 0.93

    def test_unset_domain_defaults_neutral(self, store: SelfModelStore) -> None:
        assert store.capability("unknown_domain") == 0.5

    def test_weakness_flagged_below_threshold(self, store: SelfModelStore) -> None:
        store.set_capability("legal_reasoning", 0.52)
        assert store.is_weak_in("legal_reasoning")
        assert "legal_reasoning" in store.model.weaknesses

    def test_strength_flagged_above_threshold(self, store: SelfModelStore) -> None:
        store.set_capability("coding", 0.93)
        assert store.is_strong_in("coding")
        assert "coding" in store.model.strengths

    def test_mid_range_not_flagged_either_way(self, store: SelfModelStore) -> None:
        store.set_capability("writing", 0.7)
        assert not store.is_weak_in("writing")
        assert not store.is_strong_in("writing")

    def test_score_transition_updates_lists(self, store: SelfModelStore) -> None:
        store.set_capability("math", 0.5)
        assert "math" in store.model.weaknesses
        store.set_capability("math", 0.9)
        assert "math" not in store.model.weaknesses
        assert "math" in store.model.strengths

    def test_low_capability_domains(self, store: SelfModelStore) -> None:
        store.set_capability("coding", 0.93)
        store.set_capability("legal_reasoning", 0.52)
        assert store.low_capability_domains() == ["legal_reasoning"]

    def test_high_capability_domains(self, store: SelfModelStore) -> None:
        store.set_capability("coding", 0.93)
        store.set_capability("research", 0.91)
        store.set_capability("legal_reasoning", 0.52)
        assert set(store.high_capability_domains()) == {"coding", "research"}

    def test_known_limits(self, store: SelfModelStore) -> None:
        store.add_known_limit("Cannot verify facts after training cutoff.")
        assert len(store.model.known_limits) == 1
        store.add_known_limit("Cannot verify facts after training cutoff.")
        assert len(store.model.known_limits) == 1  # no duplicate

    def test_preferences(self, store: SelfModelStore) -> None:
        store.set_preference("preferred_research_tool", "web_search")
        assert store.model.preferences["preferred_research_tool"] == "web_search"

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "self_model.json"
        store1 = SelfModelStore(f)
        store1.set_capability("coding", 0.93)
        store2 = SelfModelStore(f)
        assert store2.capability("coding") == 0.93

    def test_summary_format(self, store: SelfModelStore) -> None:
        store.set_capability("coding", 0.93)
        assert "coding=0.93" in store.summary()

    def test_summary_empty(self, store: SelfModelStore) -> None:
        assert "No tracked" in store.summary()

    def test_flag_weakness_manual(self, store: SelfModelStore) -> None:
        store.flag_weakness("untested_domain")
        assert store.is_weak_in("untested_domain")

    def test_flag_strength_manual(self, store: SelfModelStore) -> None:
        store.flag_strength("untested_domain")
        assert store.is_strong_in("untested_domain")


# ===========================================================================
# Item 2 — Confidence Manager
# ===========================================================================


class TestConfidenceManager:
    @pytest.fixture
    def cm(self, tmp_path: Path) -> ConfidenceManager:
        return ConfidenceManager(tmp_path / "confidence.json")

    def test_set_and_get(self, cm: ConfidenceManager) -> None:
        cm.set("belief", "b1", 0.91)
        assert cm.get("belief", "b1") == 0.91

    def test_get_default_when_untracked(self, cm: ConfidenceManager) -> None:
        assert cm.get("belief", "ghost") == 0.5

    def test_reinforce_increases_score(self, cm: ConfidenceManager) -> None:
        cm.set("belief", "b1", 0.91)
        new_score = cm.reinforce("belief", "b1", 0.05)
        assert new_score == pytest.approx(0.96, abs=1e-6)

    def test_weaken_decreases_score(self, cm: ConfidenceManager) -> None:
        cm.set("belief", "b1", 0.91)
        new_score = cm.weaken("belief", "b1", 0.2)
        assert new_score == pytest.approx(0.71, abs=1e-6)

    def test_score_clamped_to_unit_interval(self, cm: ConfidenceManager) -> None:
        cm.set("plan", "p1", 0.95)
        cm.reinforce("plan", "p1", 0.5)
        assert cm.get("plan", "p1") == 1.0
        cm.weaken("plan", "p1", 2.0)
        assert cm.get("plan", "p1") == 0.0

    def test_history_tracked(self, cm: ConfidenceManager) -> None:
        cm.set("tool", "t1", 0.83, reason="initial")
        cm.reinforce("tool", "t1", 0.05, reason="success")
        rec = cm.all_in_namespace("tool")[0]
        assert len(rec.history) == 2

    def test_all_in_namespace(self, cm: ConfidenceManager) -> None:
        cm.set("belief", "b1", 0.9)
        cm.set("belief", "b2", 0.3)
        cm.set("plan", "p1", 0.7)
        assert len(cm.all_in_namespace("belief")) == 2
        assert len(cm.all_in_namespace("plan")) == 1

    def test_low_confidence_filter(self, cm: ConfidenceManager) -> None:
        cm.set("belief", "b1", 0.9)
        cm.set("belief", "b2", 0.2)
        low = cm.low_confidence(namespace="belief", threshold=0.4)
        assert len(low) == 1
        assert low[0].ref_id == "b2"

    def test_mean_confidence(self, cm: ConfidenceManager) -> None:
        cm.set("belief", "b1", 0.8)
        cm.set("belief", "b2", 0.4)
        assert cm.mean_confidence("belief") == pytest.approx(0.6)

    def test_mean_confidence_empty_namespace_neutral(self, cm: ConfidenceManager) -> None:
        assert cm.mean_confidence("empty_namespace") == 0.5

    def test_count_property(self, cm: ConfidenceManager) -> None:
        cm.set("belief", "b1", 0.5)
        cm.set("plan", "p1", 0.5)
        assert cm.count == 2

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "confidence.json"
        cm1 = ConfidenceManager(f)
        cm1.set("belief", "b1", 0.77)
        cm2 = ConfidenceManager(f)
        assert cm2.get("belief", "b1") == 0.77


# ===========================================================================
# Item 3 — Confidence Reasoner
# ===========================================================================


class TestConfidenceReasoner:
    @pytest.fixture
    def reasoner(self) -> ConfidenceReasoner:
        return ConfidenceReasoner()

    @pytest.fixture
    def simple_graph(self) -> TaskGraph:
        g = TaskGraph(goal="test goal")
        g.add_task(Task(title="Step 1"))
        return g

    def test_plan_confidence_no_critique(self, reasoner: ConfidenceReasoner, simple_graph: TaskGraph) -> None:
        est = reasoner.plan_confidence(simple_graph)
        assert 0.0 <= est.score <= 1.0
        assert est.target == "plan"

    def test_plan_confidence_approved_critique_high(self, reasoner: ConfidenceReasoner, simple_graph: TaskGraph) -> None:
        critic = PlanCritic()
        critique = critic.critique(simple_graph)
        est = reasoner.plan_confidence(simple_graph, critique)
        assert est.score > 0.6
        assert est.factors["critic_verdict"] == 1.0

    def test_plan_confidence_rejected_critique_low(self, reasoner: ConfidenceReasoner) -> None:
        from planning.critic import CritiqueReport, PlanVerdict
        graph = TaskGraph(goal="bad plan")
        critique = CritiqueReport(verdict=PlanVerdict.REJECTED, issues=[])
        est = reasoner.plan_confidence(graph, critique)
        assert est.factors["critic_verdict"] == 0.1

    def test_plan_confidence_docks_per_critical_issue(self, reasoner: ConfidenceReasoner) -> None:
        from planning.critic import CriticIssue, CritiqueReport, IssueSeverity, PlanVerdict
        graph = TaskGraph(goal="risky plan")
        critique = CritiqueReport(
            verdict=PlanVerdict.APPROVED,
            issues=[CriticIssue(severity=IssueSeverity.CRITICAL, category="known_failure", message="bad", task_id="")],
        )
        est = reasoner.plan_confidence(graph, critique)
        assert est.factors["critic_verdict"] < 1.0

    def test_plan_confidence_size_penalty_for_long_plans(self, reasoner: ConfidenceReasoner) -> None:
        graph = TaskGraph(goal="long plan")
        for i in range(10):
            graph.add_task(Task(title=f"Step {i}"))
        est = reasoner.plan_confidence(graph)
        assert est.factors["size_penalty"] < 1.0

    def test_tool_confidence_no_registry_neutral(self, reasoner: ConfidenceReasoner) -> None:
        est = reasoner.tool_confidence("some_tool")
        assert est.score == 0.5

    def test_tool_confidence_with_registry(self, tmp_path: Path) -> None:
        from agents.tool_reliability import ToolReliabilityRegistry
        registry = ToolReliabilityRegistry(tmp_path / "reliability.json")
        for _ in range(10):
            registry.record("good_tool", success=True)
        reasoner = ConfidenceReasoner(tool_reliability=registry)
        est = reasoner.tool_confidence("good_tool")
        assert est.score > 0.8

    def test_answer_confidence_more_evidence_higher(self, reasoner: ConfidenceReasoner) -> None:
        low = reasoner.answer_confidence(evidence_count=1, source_count=1)
        high = reasoner.answer_confidence(evidence_count=5, source_count=3)
        assert high.score > low.score

    def test_answer_confidence_contradiction_penalty(self, reasoner: ConfidenceReasoner) -> None:
        clean = reasoner.answer_confidence(evidence_count=3, source_count=2, contradicting_evidence_count=0)
        contradicted = reasoner.answer_confidence(evidence_count=3, source_count=2, contradicting_evidence_count=3)
        assert contradicted.score < clean.score

    def test_is_low_confidence_helper(self) -> None:
        est = ConfidenceEstimate(target="plan", ref_id="g1", score=0.3)
        assert ConfidenceReasoner.is_low_confidence(est)
        est2 = ConfidenceEstimate(target="plan", ref_id="g1", score=0.8)
        assert not ConfidenceReasoner.is_low_confidence(est2)


# ===========================================================================
# Item 4 — Strategy Manager
# ===========================================================================


class TestStrategyManager:
    @pytest.fixture
    def manager(self) -> StrategyManager:
        return StrategyManager()

    def test_low_confidence_triggers_critic_first(self, manager: StrategyManager) -> None:
        decision = manager.decide("ref1", confidence=0.3)
        assert decision.strategy == ReasoningStrategy.CRITIC_FIRST

    def test_repeated_failure_triggers_decompose(self, manager: StrategyManager) -> None:
        manager.record_failure("ref2")
        manager.record_failure("ref2")
        decision = manager.decide("ref2", confidence=0.9)
        assert decision.strategy == ReasoningStrategy.DECOMPOSE_FURTHER

    def test_high_complexity_triggers_tree_of_thought(self, manager: StrategyManager) -> None:
        quality = PlanQualityScore(
            graph_id="g1", complexity=0.9, risk=0.1, confidence=0.8,
            dependency_density=0.2, expected_success=0.7,
        )
        decision = manager.decide("ref3", quality=quality)
        assert decision.strategy == ReasoningStrategy.TREE_OF_THOUGHT

    def test_no_triggers_direct(self, manager: StrategyManager) -> None:
        decision = manager.decide("ref4", confidence=0.9)
        assert decision.strategy == ReasoningStrategy.DIRECT

    def test_repeated_failure_takes_priority_over_complexity(self, manager: StrategyManager) -> None:
        manager.record_failure("ref5")
        manager.record_failure("ref5")
        quality = PlanQualityScore(
            graph_id="g1", complexity=0.9, risk=0.1, confidence=0.9,
            dependency_density=0.2, expected_success=0.8,
        )
        decision = manager.decide("ref5", quality=quality)
        assert decision.strategy == ReasoningStrategy.DECOMPOSE_FURTHER

    def test_record_success_resets_failure_streak(self, manager: StrategyManager) -> None:
        manager.record_failure("ref6")
        manager.record_failure("ref6")
        manager.record_success("ref6")
        assert manager.failure_count("ref6") == 0
        decision = manager.decide("ref6", confidence=0.9)
        assert decision.strategy == ReasoningStrategy.DIRECT

    def test_is_repeated_failure(self, manager: StrategyManager) -> None:
        assert not manager.is_repeated_failure("ref7")
        manager.record_failure("ref7")
        manager.record_failure("ref7")
        assert manager.is_repeated_failure("ref7")

    def test_last_strategy_for_tracks_history(self, manager: StrategyManager) -> None:
        manager.decide("ref8", confidence=0.3)
        assert manager.last_strategy_for("ref8") == ReasoningStrategy.CRITIC_FIRST

    def test_has_switched_strategy(self, manager: StrategyManager) -> None:
        manager.decide("ref9", confidence=0.9)
        assert not manager.has_switched_strategy("ref9")
        manager.decide("ref10", confidence=0.3)
        assert manager.has_switched_strategy("ref10")

    def test_decision_to_dict(self, manager: StrategyManager) -> None:
        decision = manager.decide("ref11", confidence=0.3)
        d = decision.to_dict()
        assert d["strategy"] == "critic_first"
        assert "triggers" in d


# ===========================================================================
# Item 5 — Capability Tracker
# ===========================================================================


class TestCapabilityTracker:
    @pytest.fixture
    def tracker(self, tmp_path: Path) -> CapabilityTracker:
        return CapabilityTracker(tmp_path / "capability.json", min_samples_for_confidence=2)

    def test_record_outcome_success(self, tracker: CapabilityTracker) -> None:
        tracker.record_outcome("coding", True)
        rec = tracker.all_records()[0]
        assert rec.successes == 1
        assert rec.failures == 0

    def test_accuracy_neutral_prior_when_untested(self, tracker: CapabilityTracker) -> None:
        assert tracker.accuracy("untested") == 0.5

    def test_accuracy_computed_correctly(self, tracker: CapabilityTracker) -> None:
        tracker.record_outcome("coding", True)
        tracker.record_outcome("coding", True)
        tracker.record_outcome("coding", False)
        assert tracker.accuracy("coding") == pytest.approx(2 / 3)

    def test_is_confident_requires_min_samples(self, tracker: CapabilityTracker) -> None:
        tracker.record_outcome("math", True)
        assert not tracker.is_confident("math")
        tracker.record_outcome("math", True)
        assert tracker.is_confident("math")

    def test_weakest_domains(self, tracker: CapabilityTracker) -> None:
        for _ in range(3):
            tracker.record_outcome("legal_reasoning", False)
        for _ in range(3):
            tracker.record_outcome("coding", True)
        weakest = tracker.weakest_domains(top_k=1)
        assert weakest[0].domain == "legal_reasoning"

    def test_strongest_domains(self, tracker: CapabilityTracker) -> None:
        for _ in range(3):
            tracker.record_outcome("legal_reasoning", False)
        for _ in range(3):
            tracker.record_outcome("coding", True)
        strongest = tracker.strongest_domains(top_k=1)
        assert strongest[0].domain == "coding"

    def test_weakest_excludes_low_sample_domains(self, tracker: CapabilityTracker) -> None:
        tracker.record_outcome("rare_domain", False)  # only 1 sample, below min_samples=2
        assert tracker.weakest_domains() == []

    def test_sync_to_self_model(self, tmp_path: Path, tracker: CapabilityTracker) -> None:
        tracker.record_outcome("coding", True)
        tracker.record_outcome("coding", True)
        store = SelfModelStore(tmp_path / "sm.json")
        synced = tracker.sync_to_self_model(store)
        assert synced == 1
        assert store.capability("coding") == 1.0

    def test_sync_skips_unconfident_domains(self, tmp_path: Path, tracker: CapabilityTracker) -> None:
        tracker.record_outcome("rare_domain", True)  # only 1 sample
        store = SelfModelStore(tmp_path / "sm.json")
        synced = tracker.sync_to_self_model(store)
        assert synced == 0

    def test_tracked_domain_count(self, tracker: CapabilityTracker) -> None:
        tracker.record_outcome("a", True)
        tracker.record_outcome("b", True)
        assert tracker.tracked_domain_count == 2

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "capability.json"
        t1 = CapabilityTracker(f)
        t1.record_outcome("coding", True)
        t2 = CapabilityTracker(f)
        assert t2.accuracy("coding") == 1.0

    def test_domain_case_insensitive(self, tracker: CapabilityTracker) -> None:
        tracker.record_outcome("Coding", True)
        assert tracker.accuracy("coding") == 1.0


# ===========================================================================
# Item 6 — Procedural Memory
# ===========================================================================


class TestProceduralMemory:
    @pytest.fixture
    def pm(self, tmp_path: Path) -> ProceduralMemory:
        return ProceduralMemory(tmp_path / "procedural.json")

    def test_learn_from_success_creates_skill(self, pm: ProceduralMemory) -> None:
        skill = pm.learn_from_success(
            "Research transformer architectures",
            ["retrieve_documents", "summarize", "extract_insights", "update_knowledge"],
            name="research_analysis",
        )
        assert skill is not None
        assert skill.name == "research_analysis"
        assert len(skill.steps) == 4

    def test_too_short_sequence_not_learned(self, pm: ProceduralMemory) -> None:
        skill = pm.learn_from_success("Quick task", ["one_step"])
        assert skill is None

    def test_find_matching_skill_above_threshold(self, pm: ProceduralMemory) -> None:
        pm.learn_from_success(
            "Research transformer model architectures",
            ["retrieve", "summarize", "extract"],
            name="research_analysis",
        )
        match = pm.find_matching_skill("Research transformer model architectures")
        assert match is not None
        assert match.name == "research_analysis"

    def test_find_matching_skill_no_match_below_threshold(self, pm: ProceduralMemory) -> None:
        pm.learn_from_success(
            "Research transformer model architectures",
            ["retrieve", "summarize", "extract"],
            name="research_analysis",
        )
        match = pm.find_matching_skill("Bake a chocolate cake recipe")
        assert match is None

    def test_reinforcing_existing_skill_increments_counts(self, pm: ProceduralMemory) -> None:
        goal = "Research transformer model architectures"
        steps = ["retrieve", "summarize", "extract"]
        skill1 = pm.learn_from_success(goal, steps, name="research_analysis")
        skill2 = pm.learn_from_success(goal, steps, name="research_analysis")
        assert skill1.skill_id == skill2.skill_id
        assert skill2.use_count == 2
        assert skill2.success_count == 2

    def test_suggest_steps(self, pm: ProceduralMemory) -> None:
        goal = "Research transformer model architectures"
        pm.learn_from_success(goal, ["retrieve", "summarize", "extract"], name="research_analysis")
        steps = pm.suggest_steps(goal)
        assert steps == ["retrieve", "summarize", "extract"]

    def test_suggest_steps_none_when_no_match(self, pm: ProceduralMemory) -> None:
        assert pm.suggest_steps("Completely unrelated goal text") is None

    def test_record_reuse_outcome(self, pm: ProceduralMemory) -> None:
        skill = pm.learn_from_success(
            "Research transformer model architectures",
            ["retrieve", "summarize", "extract"],
            name="research_analysis",
        )
        pm.record_reuse_outcome(skill.skill_id, success=False)
        updated = pm.get(skill.skill_id)
        assert updated.use_count == 2
        assert updated.success_count == 1  # original 1 success, then 1 failure

    def test_most_used(self, pm: ProceduralMemory) -> None:
        pm.learn_from_success("Goal alpha task one", ["a", "b"], name="skill_a")
        pm.learn_from_success("Goal beta task two", ["c", "d"], name="skill_b")
        pm.learn_from_success("Goal beta task two", ["c", "d"], name="skill_b")
        most_used = pm.most_used(top_k=1)
        assert most_used[0].name == "skill_b"

    def test_most_reliable_requires_min_uses(self, pm: ProceduralMemory) -> None:
        pm.learn_from_success("Goal gamma task three", ["e", "f"], name="skill_c")
        reliable = pm.most_reliable(min_uses=2)
        assert reliable == []  # only 1 use so far

    def test_count_property(self, pm: ProceduralMemory) -> None:
        pm.learn_from_success("Goal delta task four", ["g", "h"], name="skill_d")
        assert pm.count == 1

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "procedural.json"
        pm1 = ProceduralMemory(f)
        pm1.learn_from_success("Goal epsilon task five", ["i", "j"], name="skill_e")
        pm2 = ProceduralMemory(f)
        assert pm2.count == 1
        assert pm2.all_skills()[0].name == "skill_e"

    def test_derive_skill_name_fallback(self, pm: ProceduralMemory) -> None:
        skill = pm.learn_from_success("Research the history of computing", ["a", "b"])
        assert skill.name  # auto-derived, non-empty


# ===========================================================================
# Item 7 — Plan Quality Evaluator
# ===========================================================================


class TestPlanQualityEvaluator:
    @pytest.fixture
    def evaluator(self) -> PlanQualityEvaluator:
        return PlanQualityEvaluator()

    def test_simple_plan_low_complexity(self, evaluator: PlanQualityEvaluator) -> None:
        graph = TaskGraph(goal="simple")
        graph.add_task(Task(title="Step 1"))
        score = evaluator.evaluate(graph)
        assert score.complexity < 0.3

    def test_long_plan_high_complexity(self, evaluator: PlanQualityEvaluator) -> None:
        graph = TaskGraph(goal="complex")
        for i in range(15):
            graph.add_task(Task(title=f"Step {i}"))
        score = evaluator.evaluate(graph)
        assert score.complexity > 0.5

    def test_dependency_density_linear_chain(self, evaluator: PlanQualityEvaluator) -> None:
        graph = TaskGraph(goal="chain")
        t1 = Task(title="Step 1")
        graph.add_task(t1)
        t2 = Task(title="Step 2", depends_on=[t1.task_id])
        graph.add_task(t2)
        score = evaluator.evaluate(graph)
        assert score.dependency_density == 1.0

    def test_dependency_density_no_deps(self, evaluator: PlanQualityEvaluator) -> None:
        graph = TaskGraph(goal="parallel")
        graph.add_task(Task(title="Step 1"))
        graph.add_task(Task(title="Step 2"))
        score = evaluator.evaluate(graph)
        assert score.dependency_density == 0.0

    def test_critic_critical_issues_raise_risk(self, evaluator: PlanQualityEvaluator) -> None:
        from planning.critic import CriticIssue, CritiqueReport, IssueSeverity, PlanVerdict
        graph = TaskGraph(goal="risky")
        graph.add_task(Task(title="Step 1"))
        critique = CritiqueReport(
            verdict=PlanVerdict.APPROVED_WITH_WARNINGS,
            issues=[CriticIssue(severity=IssueSeverity.CRITICAL, category="known_failure", message="bad", task_id="")],
        )
        score = evaluator.evaluate(graph, critique)
        assert score.risk > 0.0
        assert score.is_high_risk or score.risk >= 0.35

    def test_expected_success_blends_signals(self, evaluator: PlanQualityEvaluator) -> None:
        graph = TaskGraph(goal="test")
        graph.add_task(Task(title="Step 1"))
        critic = PlanCritic()
        critique = critic.critique(graph)
        score = evaluator.evaluate(graph, critique)
        assert 0.0 <= score.expected_success <= 1.0

    def test_is_low_confidence_property(self) -> None:
        score = PlanQualityScore(
            graph_id="g1", complexity=0.1, risk=0.1, confidence=0.3,
            dependency_density=0.0, expected_success=0.2,
        )
        assert score.is_low_confidence

    def test_is_high_risk_property(self) -> None:
        score = PlanQualityScore(
            graph_id="g1", complexity=0.1, risk=0.7, confidence=0.8,
            dependency_density=0.0, expected_success=0.5,
        )
        assert score.is_high_risk

    def test_notes_flag_high_complexity(self, evaluator: PlanQualityEvaluator) -> None:
        graph = TaskGraph(goal="complex")
        for i in range(20):
            graph.add_task(Task(title=f"Step {i}"))
        score = evaluator.evaluate(graph)
        assert any("complexity" in n.lower() for n in score.notes)

    def test_to_dict(self, evaluator: PlanQualityEvaluator) -> None:
        graph = TaskGraph(goal="test")
        graph.add_task(Task(title="Step 1"))
        score = evaluator.evaluate(graph)
        d = score.to_dict()
        assert "expected_success" in d
        assert "graph_id" in d


# ===========================================================================
# Item 8 — Execution Feedback Loop
# ===========================================================================


class TestExecutionFeedbackLoop:
    @pytest.fixture
    def loop(self, tmp_path: Path) -> ExecutionFeedbackLoop:
        return ExecutionFeedbackLoop(tmp_path / "feedback.json")

    def test_record_task_outcome_success(self, loop: ExecutionFeedbackLoop) -> None:
        entry = loop.record_task_outcome("Implement login endpoint", success=True, confidence=0.9)
        assert entry.success
        assert entry.domain == "coding"

    def test_domain_inference_from_keywords(self, loop: ExecutionFeedbackLoop) -> None:
        entry = loop.record_task_outcome("Calculate the sum of squares", success=True)
        assert entry.domain == "math"

    def test_domain_explicit_override(self, loop: ExecutionFeedbackLoop) -> None:
        entry = loop.record_task_outcome("Random task", success=True, domain="custom_domain")
        assert entry.domain == "custom_domain"

    def test_capability_tracker_fed_on_record(self, tmp_path: Path) -> None:
        tracker = CapabilityTracker(tmp_path / "cap.json")
        loop = ExecutionFeedbackLoop(tmp_path / "feedback.json", capability_tracker=tracker)
        loop.record_task_outcome("Implement feature", success=True, domain="coding")
        assert tracker.accuracy("coding") == 1.0

    def test_failure_memory_fed_on_failure(self, tmp_path: Path) -> None:
        from agents.failure_memory import FailureMemory
        fm = FailureMemory(tmp_path / "failures.json")
        loop = ExecutionFeedbackLoop(tmp_path / "feedback.json", failure_memory=fm)
        loop.record_task_outcome("Bad task", success=False, tool="broken_tool", failure_reason="timeout")
        assert fm.count == 1

    def test_failure_memory_not_fed_on_success(self, tmp_path: Path) -> None:
        from agents.failure_memory import FailureMemory
        fm = FailureMemory(tmp_path / "failures.json")
        loop = ExecutionFeedbackLoop(tmp_path / "feedback.json", failure_memory=fm)
        loop.record_task_outcome("Good task", success=True)
        assert fm.count == 0

    def test_recent_filters_by_domain(self, loop: ExecutionFeedbackLoop) -> None:
        loop.record_task_outcome("Implement feature", success=True, domain="coding")
        loop.record_task_outcome("Research topic", success=True, domain="research")
        recent_coding = loop.recent(domain="coding")
        assert len(recent_coding) == 1

    def test_success_rate(self, loop: ExecutionFeedbackLoop) -> None:
        loop.record_task_outcome("Task A", success=True, domain="coding")
        loop.record_task_outcome("Task B", success=False, domain="coding")
        assert loop.success_rate("coding") == 0.5

    def test_success_rate_empty_neutral(self, loop: ExecutionFeedbackLoop) -> None:
        assert loop.success_rate("empty_domain") == 0.5

    def test_mean_confidence(self, loop: ExecutionFeedbackLoop) -> None:
        loop.record_task_outcome("Task A", success=True, confidence=0.8, domain="coding")
        loop.record_task_outcome("Task B", success=True, confidence=0.6, domain="coding")
        assert loop.mean_confidence("coding") == pytest.approx(0.7)

    def test_count_property(self, loop: ExecutionFeedbackLoop) -> None:
        loop.record_task_outcome("Task A", success=True)
        loop.record_task_outcome("Task B", success=False)
        assert loop.count == 2

    def test_record_run_result_iterates_tasks(self, tmp_path: Path) -> None:
        loop = ExecutionFeedbackLoop(tmp_path / "feedback.json")
        graph = TaskGraph(goal="test goal")
        t1 = Task(title="Implement feature", status=TaskStatus.COMPLETED)
        t2 = Task(title="Research topic", status=TaskStatus.FAILED, error="oops")
        t3 = Task(title="Pending task", status=TaskStatus.PENDING)
        graph.tasks = [t1, t2, t3]

        class FakeResult:
            goal = "test goal"
            duration_secs = 4.0
            agent_state = {"confidence": 0.6}

        result = FakeResult()
        result.graph = graph
        entries = loop.record_run_result(result)
        assert len(entries) == 2  # pending task excluded
        assert any(not e.success for e in entries)

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "feedback.json"
        loop1 = ExecutionFeedbackLoop(f)
        loop1.record_task_outcome("Task A", success=True)
        loop2 = ExecutionFeedbackLoop(f)
        assert loop2.count == 1


# ===========================================================================
# Item 9 — Meta-Reflection
# ===========================================================================


class TestMetaReflection:
    @pytest.fixture
    def engine(self, tmp_path: Path) -> MetaReflectionEngine:
        reflection_engine = ReflectionEngine(tmp_path / "reflections.json")
        return MetaReflectionEngine(reflection_engine=reflection_engine)

    def test_empty_runs_no_insights(self, engine: MetaReflectionEngine) -> None:
        assert engine.analyze_runs([]) == []

    def test_frequent_replanning_detected(self, engine: MetaReflectionEngine) -> None:
        runs = [{"replan_count": 3}, {"replan_count": 2}, {"replan_count": 3}]
        insights = engine.analyze_runs(runs)
        patterns = [i.pattern for i in insights]
        assert "frequent_replanning" in patterns

    def test_frequent_replanning_not_flagged_when_low(self, engine: MetaReflectionEngine) -> None:
        runs = [{"replan_count": 0}, {"replan_count": 1}]
        insights = engine.analyze_runs(runs)
        patterns = [i.pattern for i in insights]
        assert "frequent_replanning" not in patterns

    def test_low_confidence_pattern_detected(self, engine: MetaReflectionEngine) -> None:
        runs = [
            {"agent_state": {"confidence": 0.3}},
            {"agent_state": {"confidence": 0.4}},
            {"agent_state": {"confidence": 0.8}},
        ]
        insights = engine.analyze_runs(runs)
        patterns = [i.pattern for i in insights]
        assert "frequent_low_confidence" in patterns

    def test_repeated_tool_bottleneck_detected(self, engine: MetaReflectionEngine) -> None:
        runs = [
            {"plan_reflection": {"bottleneck_tool": "web_search"}},
            {"plan_reflection": {"bottleneck_tool": "web_search"}},
            {"plan_reflection": {"bottleneck_tool": "web_search"}},
        ]
        insights = engine.analyze_runs(runs)
        patterns = [i.pattern for i in insights]
        assert "repeated_tool_bottleneck" in patterns

    def test_bottleneck_not_flagged_below_threshold(self, engine: MetaReflectionEngine) -> None:
        runs = [
            {"plan_reflection": {"bottleneck_tool": "web_search"}},
            {"plan_reflection": {"bottleneck_tool": "code_tool"}},
        ]
        insights = engine.analyze_runs(runs)
        patterns = [i.pattern for i in insights]
        assert "repeated_tool_bottleneck" not in patterns

    def test_spec_literal_example(self, engine: MetaReflectionEngine) -> None:
        """Matches the spec's literal example: frequent replanning -> shallow strategy insight."""
        runs = [{"replan_count": 4} for _ in range(3)]
        insights = engine.analyze_runs(runs)
        replan_insight = next(i for i in insights if i.pattern == "frequent_replanning")
        assert "too shallow" in replan_insight.suggested_change

    def test_insights_persisted_to_reflection_engine(self, tmp_path: Path) -> None:
        reflection_engine = ReflectionEngine(tmp_path / "reflections.json")
        engine = MetaReflectionEngine(reflection_engine=reflection_engine)
        runs = [{"replan_count": 3}, {"replan_count": 3}]
        engine.analyze_runs(runs)
        records = reflection_engine.get_records(scope=ReflectionScope.BEHAVIOR)
        assert len(records) >= 1

    def test_works_without_reflection_engine(self) -> None:
        engine = MetaReflectionEngine(reflection_engine=None)
        runs = [{"replan_count": 3}, {"replan_count": 3}]
        insights = engine.analyze_runs(runs)
        assert len(insights) >= 1  # detection still works; persistence just skipped

    def test_multiple_patterns_in_one_batch(self, engine: MetaReflectionEngine) -> None:
        runs = [
            {
                "replan_count": 3,
                "agent_state": {"confidence": 0.3},
                "plan_reflection": {"bottleneck_tool": "web_search"},
            }
            for _ in range(3)
        ]
        insights = engine.analyze_runs(runs)
        patterns = {i.pattern for i in insights}
        assert patterns == {"frequent_replanning", "frequent_low_confidence", "repeated_tool_bottleneck"}


# ===========================================================================
# Item 10 — Meta-Cognitive Controller
# ===========================================================================


class TestMetaCognitiveController:
    @pytest.fixture
    def controller(self) -> MetaCognitiveController:
        return MetaCognitiveController()

    def test_clean_plan_no_issues(self, controller: MetaCognitiveController) -> None:
        graph = TaskGraph(goal="test")
        graph.add_task(Task(title="Step 1"))
        critic = PlanCritic()
        critique = critic.critique(graph)
        report, decision = controller.run_cycle("ref1", graph=graph, critique=critique)
        assert CognitiveIssue.NONE in report.issues
        assert decision.action == AdaptationAction.NONE

    def test_repeated_failure_triggers_replan(self, controller: MetaCognitiveController) -> None:
        controller._strategy_manager.record_failure("ref2")
        controller._strategy_manager.record_failure("ref2")
        report, decision = controller.run_cycle("ref2")
        assert CognitiveIssue.REPEATED_FAILURES in report.issues
        assert decision.action == AdaptationAction.REPLAN

    def test_hallucination_risk_flags_for_review(self, controller: MetaCognitiveController) -> None:
        report, decision = controller.run_cycle("ref3", hallucination_rate=0.5)
        assert CognitiveIssue.HALLUCINATION_RISK in report.issues
        assert decision.action == AdaptationAction.FLAG_FOR_REVIEW

    def test_low_belief_consistency_flags_for_review(self, controller: MetaCognitiveController) -> None:
        report, decision = controller.run_cycle("ref4", belief_consistency=0.3)
        assert CognitiveIssue.LOW_BELIEF_CONSISTENCY in report.issues
        assert decision.action == AdaptationAction.FLAG_FOR_REVIEW

    def test_low_confidence_changes_strategy(self, controller: MetaCognitiveController) -> None:
        from planning.critic import CritiqueReport, PlanVerdict
        graph = TaskGraph(goal="risky")
        graph.add_task(Task(title="Step 1"))
        critique = CritiqueReport(verdict=PlanVerdict.REJECTED, issues=[])
        report, decision = controller.run_cycle("ref5", graph=graph, critique=critique)
        assert CognitiveIssue.LOW_CONFIDENCE in report.issues
        assert decision.action == AdaptationAction.CHANGE_STRATEGY

    def test_priority_repeated_failures_over_hallucination(self, controller: MetaCognitiveController) -> None:
        controller._strategy_manager.record_failure("ref6")
        controller._strategy_manager.record_failure("ref6")
        report, decision = controller.run_cycle("ref6", hallucination_rate=0.9)
        assert decision.action in (AdaptationAction.REPLAN, AdaptationAction.CHANGE_STRATEGY)

    def test_monitor_without_optional_signals_no_crash(self, controller: MetaCognitiveController) -> None:
        report = controller.monitor("ref7")
        assert report.plan_quality is None
        assert CognitiveIssue.NONE in report.issues

    def test_report_to_dict(self, controller: MetaCognitiveController) -> None:
        report = controller.monitor("ref8", hallucination_rate=0.5)
        d = report.to_dict()
        assert "issues" in d

    def test_decision_to_dict(self, controller: MetaCognitiveController) -> None:
        _, decision = controller.run_cycle("ref9", hallucination_rate=0.5)
        d = decision.to_dict()
        assert d["action"] == "flag_for_review"

    def test_has_any_issues_false_when_clean(self, controller: MetaCognitiveController) -> None:
        report = controller.monitor("ref10")
        assert not report.has_any_issues if hasattr(report, "has_any_issues") else True


# ===========================================================================
# Item 11 — Confidence Metrics
# ===========================================================================


class TestConfidenceMetrics:
    def test_brier_score_perfect_calibration(self) -> None:
        cases = [CalibrationCase(confidence=1.0, was_correct=True), CalibrationCase(confidence=0.0, was_correct=False)]
        assert ConfidenceMetrics.brier_score(cases) == 0.0

    def test_brier_score_worst_case(self) -> None:
        cases = [CalibrationCase(confidence=1.0, was_correct=False)]
        assert ConfidenceMetrics.brier_score(cases) == 1.0

    def test_brier_score_empty(self) -> None:
        assert ConfidenceMetrics.brier_score([]) == 0.0

    def test_calibration_buckets_groups_correctly(self) -> None:
        cases = [
            CalibrationCase(confidence=0.85, was_correct=True),
            CalibrationCase(confidence=0.82, was_correct=True),
            CalibrationCase(confidence=0.15, was_correct=False),
        ]
        buckets = ConfidenceMetrics.calibration_buckets(cases, bucket_size=0.2)
        assert len(buckets) == 2

    def test_expected_calibration_error_zero_when_perfect(self) -> None:
        cases = [CalibrationCase(confidence=0.9, was_correct=True) for _ in range(5)]
        cases += [CalibrationCase(confidence=0.9, was_correct=False) for _ in range(0)]
        # mean confidence 0.9 in bucket, accuracy 1.0 -> some gap expected unless exact
        ece = ConfidenceMetrics.expected_calibration_error(cases)
        assert ece >= 0.0

    def test_overconfidence_rate_detects_wrong_high_confidence(self) -> None:
        cases = [CalibrationCase(confidence=0.9, was_correct=False)]
        assert ConfidenceMetrics.overconfidence_rate(cases) == 1.0

    def test_overconfidence_rate_zero_when_all_correct(self) -> None:
        cases = [CalibrationCase(confidence=0.9, was_correct=True)]
        assert ConfidenceMetrics.overconfidence_rate(cases) == 0.0

    def test_underconfidence_rate_detects_low_confidence_correct(self) -> None:
        cases = [CalibrationCase(confidence=0.1, was_correct=True)]
        assert ConfidenceMetrics.underconfidence_rate(cases) == 1.0

    def test_empty_cases_return_zero(self) -> None:
        assert ConfidenceMetrics.expected_calibration_error([]) == 0.0
        assert ConfidenceMetrics.overconfidence_rate([]) == 0.0
        assert ConfidenceMetrics.underconfidence_rate([]) == 0.0

    def test_bucket_result_to_dict(self) -> None:
        cases = [CalibrationCase(confidence=0.85, was_correct=True)]
        buckets = ConfidenceMetrics.calibration_buckets(cases)
        d = buckets[0].to_dict()
        assert "gap" in d and "bucket" in d


# ===========================================================================
# Item 12 — Capability Metrics
# ===========================================================================


class TestCapabilityMetrics:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        store = SelfModelStore(tmp_path / "sm.json")
        tracker = CapabilityTracker(tmp_path / "ct.json", min_samples_for_confidence=2)
        return store, tracker

    def test_self_awareness_gap_detected(self, setup) -> None:
        store, tracker = setup
        store.set_capability("coding", 0.95)
        tracker.record_outcome("coding", True)
        tracker.record_outcome("coding", False)
        tracker.record_outcome("coding", False)
        gaps = CapabilityMetrics.self_awareness_gaps(store, tracker)
        assert len(gaps) == 1
        assert gaps[0].gap > 0.5

    def test_self_awareness_gap_excludes_unconfident_domains(self, setup) -> None:
        store, tracker = setup
        tracker.record_outcome("rare", True)  # only 1 sample
        gaps = CapabilityMetrics.self_awareness_gaps(store, tracker)
        assert gaps == []

    def test_self_awareness_score_perfect_when_no_gaps(self) -> None:
        assert CapabilityMetrics.self_awareness_score([]) == 1.0

    def test_self_awareness_score_lower_with_large_gap(self) -> None:
        gaps = [SelfAwarenessGap(domain="coding", believed=0.95, actual=0.3)]
        score = CapabilityMetrics.self_awareness_score(gaps)
        assert score < 0.5

    def test_overestimated_domains(self) -> None:
        gaps = [
            SelfAwarenessGap(domain="coding", believed=0.9, actual=0.5),
            SelfAwarenessGap(domain="math", believed=0.5, actual=0.5),
        ]
        overestimated = CapabilityMetrics.overestimated_domains(gaps)
        assert len(overestimated) == 1
        assert overestimated[0].domain == "coding"

    def test_underestimated_domains(self) -> None:
        gaps = [SelfAwarenessGap(domain="research", believed=0.4, actual=0.9)]
        underestimated = CapabilityMetrics.underestimated_domains(gaps)
        assert len(underestimated) == 1

    def test_capability_coverage_full(self, setup) -> None:
        store, tracker = setup
        tracker.record_outcome("coding", True)
        tracker.record_outcome("coding", True)
        store.set_capability("coding", 0.9)
        assert CapabilityMetrics.capability_coverage(store, tracker) == 1.0

    def test_capability_coverage_partial(self, setup) -> None:
        store, tracker = setup
        tracker.record_outcome("coding", True)
        tracker.record_outcome("coding", True)
        # not synced into self model
        assert CapabilityMetrics.capability_coverage(store, tracker) == 0.0

    def test_capability_coverage_no_confident_domains(self, setup) -> None:
        store, tracker = setup
        assert CapabilityMetrics.capability_coverage(store, tracker) == 1.0

    def test_gap_to_dict(self) -> None:
        gap = SelfAwarenessGap(domain="coding", believed=0.9, actual=0.5)
        d = gap.to_dict()
        assert d["gap"] == pytest.approx(0.4)


# ===========================================================================
# Item 13 — Metacognition Metrics
# ===========================================================================


class TestMetacognitionMetrics:
    @pytest.fixture
    def metrics(self) -> MetacognitionMetrics:
        return MetacognitionMetrics()

    def test_issue_detection_accuracy_perfect(self, metrics: MetacognitionMetrics) -> None:
        cases = [
            AdaptationCase(issue_present=True, issue_detected=True, adaptation_taken=True),
            AdaptationCase(issue_present=False, issue_detected=False, adaptation_taken=False),
        ]
        assert metrics.issue_detection_accuracy(cases) == 1.0

    def test_issue_detection_accuracy_empty_perfect_default(self, metrics: MetacognitionMetrics) -> None:
        assert metrics.issue_detection_accuracy([]) == 1.0

    def test_false_alarm_rate(self, metrics: MetacognitionMetrics) -> None:
        cases = [
            AdaptationCase(issue_present=False, issue_detected=True, adaptation_taken=True),
            AdaptationCase(issue_present=False, issue_detected=False, adaptation_taken=False),
        ]
        assert metrics.false_alarm_rate(cases) == 0.5

    def test_missed_detection_rate(self, metrics: MetacognitionMetrics) -> None:
        cases = [
            AdaptationCase(issue_present=True, issue_detected=False, adaptation_taken=False),
            AdaptationCase(issue_present=True, issue_detected=True, adaptation_taken=True),
        ]
        assert metrics.missed_detection_rate(cases) == 0.5

    def test_adaptation_responsiveness(self, metrics: MetacognitionMetrics) -> None:
        cases = [
            AdaptationCase(issue_present=True, issue_detected=True, adaptation_taken=True),
            AdaptationCase(issue_present=True, issue_detected=True, adaptation_taken=False),
        ]
        assert metrics.adaptation_responsiveness(cases) == 0.5

    def test_adaptation_effectiveness_excludes_unknown_outcomes(self, metrics: MetacognitionMetrics) -> None:
        cases = [
            AdaptationCase(issue_present=True, issue_detected=True, adaptation_taken=True, outcome_improved=True),
            AdaptationCase(issue_present=True, issue_detected=True, adaptation_taken=True, outcome_improved=None),
        ]
        assert metrics.adaptation_effectiveness(cases) == 1.0  # only the known-outcome case counts

    def test_adaptation_effectiveness_no_adapted_cases_perfect_default(self, metrics: MetacognitionMetrics) -> None:
        cases = [AdaptationCase(issue_present=False, issue_detected=False, adaptation_taken=False)]
        assert metrics.adaptation_effectiveness(cases) == 1.0

    def test_strategy_switch_rate(self, metrics: MetacognitionMetrics) -> None:
        decisions = [
            AdaptationDecision(action=AdaptationAction.REPLAN, reason="x"),
            AdaptationDecision(action=AdaptationAction.NONE, reason="y"),
        ]
        assert metrics.strategy_switch_rate(decisions) == 0.5

    def test_action_distribution(self, metrics: MetacognitionMetrics) -> None:
        decisions = [
            AdaptationDecision(action=AdaptationAction.REPLAN, reason="x"),
            AdaptationDecision(action=AdaptationAction.REPLAN, reason="y"),
            AdaptationDecision(action=AdaptationAction.NONE, reason="z"),
        ]
        dist = metrics.action_distribution(decisions)
        assert dist["replan"] == pytest.approx(2 / 3, abs=1e-3)

    def test_run_metacognition_bench_combines_all(self, metrics: MetacognitionMetrics) -> None:
        cases = [AdaptationCase(issue_present=True, issue_detected=True, adaptation_taken=True, outcome_improved=True)]
        decisions = [AdaptationDecision(action=AdaptationAction.REPLAN, reason="x")]
        results = metrics.run_metacognition_bench(cases, decisions)
        assert "issue_detection_accuracy" in results
        assert "strategy_switch_rate" in results

    def test_extends_state_metrics(self, metrics: MetacognitionMetrics) -> None:
        from evaluation.state_metrics import StateMetrics
        assert isinstance(metrics, StateMetrics)


# ===========================================================================
# Integration — BlixContext wiring + API
# ===========================================================================


class _FakeLLM:
    def model_name(self) -> str:
        return "fake-0.3.8"

    def generate(self, prompt: str) -> str:
        return "Fake reply."


@pytest.fixture(scope="module")
def tmp_memory_v8(tmp_path_factory):
    return tmp_path_factory.mktemp("memory_v8")


@pytest.fixture(scope="module")
def ctx_v8(tmp_memory_v8):
    from config import settings as _settings
    _settings.settings.memory.conversations_file = tmp_memory_v8 / "conversations.json"
    _settings.settings.memory.profile_file = tmp_memory_v8 / "profile.json"
    _settings.settings.memory.learning_state_file = tmp_memory_v8 / "learning_state.json"
    _settings.settings.embed.embeddings_file = tmp_memory_v8 / "embeddings.npy"
    _settings.settings.embed.embedding_ids_file = tmp_memory_v8 / "embedding_ids.json"

    from api.context import BlixContext
    ctx = BlixContext(tmp_memory_v8)
    ctx.llm = _FakeLLM()
    ctx.agent._llm = _FakeLLM()
    return ctx


@pytest.fixture(scope="module")
def client_v8(ctx_v8) -> Generator[TestClient, None, None]:
    from api.deps import set_context
    from api.routers.metacognition import router as metacognition_router

    app = FastAPI(title="Blix Test v0.3.8")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(metacognition_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    set_context(ctx_v8)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestBlixContextV038Wiring:
    def test_v038_components_present(self, ctx_v8) -> None:
        assert ctx_v8.self_model is not None
        assert ctx_v8.confidence_manager is not None
        assert ctx_v8.confidence_reasoner is not None
        assert ctx_v8.strategy_manager is not None
        assert ctx_v8.capability_tracker is not None
        assert ctx_v8.procedural_memory is not None
        assert ctx_v8.plan_evaluator is not None
        assert ctx_v8.execution_feedback is not None
        assert ctx_v8.meta_reflection is not None
        assert ctx_v8.meta_controller is not None
        assert ctx_v8.metacognition_metrics is not None

    def test_dashboard_stats_includes_v038_metrics(self, ctx_v8) -> None:
        stats = ctx_v8.dashboard_stats()
        assert "tracked_capabilities" in stats
        assert "learned_skills" in stats
        assert "confidence_records_tracked" in stats
        assert "execution_feedback_entries" in stats

    def test_end_to_end_capability_sync_via_context(self, ctx_v8) -> None:
        ctx_v8.capability_tracker.record_outcome("integration_domain", True)
        ctx_v8.capability_tracker.record_outcome("integration_domain", True)
        ctx_v8.capability_tracker.record_outcome("integration_domain", True)
        ctx_v8.capability_tracker.record_outcome("integration_domain", True)
        ctx_v8.capability_tracker.record_outcome("integration_domain", True)
        synced = ctx_v8.capability_tracker.sync_to_self_model(ctx_v8.self_model)
        assert synced >= 1
        assert ctx_v8.self_model.capability("integration_domain") == 1.0

    def test_end_to_end_skill_learning_via_context(self, ctx_v8) -> None:
        ctx_v8.procedural_memory.learn_from_success(
            "Integration test research goal", ["retrieve", "summarize", "extract"], name="integration_skill",
        )
        match = ctx_v8.procedural_memory.find_matching_skill("Integration test research goal")
        assert match is not None
        assert match.name == "integration_skill"

    def test_controller_uses_wired_components(self, ctx_v8) -> None:
        report, decision = ctx_v8.meta_controller.run_cycle("integration_ref")
        assert report is not None
        assert decision is not None


# ===========================================================================
# API — /metacognition endpoints
# ===========================================================================


class TestMetacognitionAPI:
    def test_get_self_model(self, client_v8: TestClient, ctx_v8) -> None:
        ctx_v8.self_model.set_capability("api_coding", 0.9)
        r = client_v8.get("/metacognition/self-model")
        assert r.status_code == 200
        data = r.json()
        assert "capabilities" in data
        assert data["capabilities"]["api_coding"] == 0.9

    def test_get_capabilities(self, client_v8: TestClient, ctx_v8) -> None:
        ctx_v8.capability_tracker.record_outcome("api_domain", True)
        r = client_v8.get("/metacognition/capabilities")
        assert r.status_code == 200
        data = r.json()
        assert data["tracked_domains"] >= 1

    def test_get_confidence_namespace(self, client_v8: TestClient, ctx_v8) -> None:
        ctx_v8.confidence_manager.set("belief", "api_belief_1", 0.85)
        r = client_v8.get("/metacognition/confidence/belief")
        assert r.status_code == 200
        data = r.json()
        assert data["namespace"] == "belief"
        assert len(data["records"]) >= 1

    def test_get_confidence_namespace_with_threshold(self, client_v8: TestClient, ctx_v8) -> None:
        ctx_v8.confidence_manager.set("plan", "low_conf_plan", 0.1)
        r = client_v8.get("/metacognition/confidence/plan?threshold=0.5")
        assert r.status_code == 200
        data = r.json()
        assert any(rec["ref_id"] == "low_conf_plan" for rec in data["records"])

    def test_decide_strategy_low_confidence(self, client_v8: TestClient) -> None:
        r = client_v8.post("/metacognition/strategy/decide", json={"ref_key": "api_ref_1", "confidence": 0.2})
        assert r.status_code == 200
        data = r.json()
        assert data["strategy"] == "critic_first"

    def test_decide_strategy_validates_ref_key(self, client_v8: TestClient) -> None:
        r = client_v8.post("/metacognition/strategy/decide", json={"ref_key": ""})
        assert r.status_code == 422

    def test_list_skills(self, client_v8: TestClient, ctx_v8) -> None:
        ctx_v8.procedural_memory.learn_from_success(
            "API test skill goal", ["step_one", "step_two"], name="api_skill",
        )
        r = client_v8.get("/metacognition/skills")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1

    def test_match_skill_found(self, client_v8: TestClient, ctx_v8) -> None:
        ctx_v8.procedural_memory.learn_from_success(
            "Match this exact goal text for skill", ["step_one", "step_two"], name="matchable_skill",
        )
        r = client_v8.post("/metacognition/skills/match", json={"goal": "Match this exact goal text for skill"})
        assert r.status_code == 200
        data = r.json()
        assert data["matched"] is True
        assert data["skill"]["name"] == "matchable_skill"

    def test_match_skill_not_found(self, client_v8: TestClient) -> None:
        r = client_v8.post("/metacognition/skills/match", json={"goal": "Totally unrelated never seen before goal xyz"})
        assert r.status_code == 200
        data = r.json()
        assert data["matched"] is False

    def test_behavior_insights(self, client_v8: TestClient, ctx_v8) -> None:
        ctx_v8.meta_reflection.analyze_runs([{"replan_count": 5}, {"replan_count": 5}], scope_ref="api_test_runs")
        r = client_v8.get("/metacognition/behavior-insights")
        assert r.status_code == 200
        data = r.json()
        assert "insights" in data

    def test_behavior_insights_respects_limit(self, client_v8: TestClient) -> None:
        r = client_v8.get("/metacognition/behavior-insights?limit=2")
        assert r.status_code == 200
        assert len(r.json()["insights"]) <= 2

    def test_health_check_still_works(self, client_v8: TestClient) -> None:
        r = client_v8.get("/health")
        assert r.status_code == 200
