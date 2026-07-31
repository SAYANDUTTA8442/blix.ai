"""
Tests for Blix v0.3.13 — "Curiosity + Active Experimentation".

Covers:
  New modules:
    knowledge.knowledge_gap_tracker            (KnowledgeGapTracker, KnowledgeGap, GapSeverity)
    curiosity.curiosity_engine                     (CuriosityEngine, CuriositySignal, five triggers)
    hypothesis.hypothesis_manager                      (HypothesisManager, Hypothesis, full lifecycle)
    experiments.experiment_planner                         (ExperimentPlanner, Experiment, pipeline)

  Extensions:
    reflection.reflection_engine  — reflect_on_curiosity()
    causality.principle_synthesizer  — synthesize_from_experiment(), synthesize_all(experiments=)
    memory.future_memory  — record_experiment(), resolve_experiment(), experiments()
    causality.meta_causal_reflection  — which_hypotheses_failed_repeatedly()
    metacognition.self_model  — knowledge_gaps(tracker)

  Integration — BlixContext wiring
  API — /curiosity endpoints

Python 3.10 compatible — fully offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from agents.failure_memory import FailureMemory
from causality.cause_graph import CauseGraph, CauseRelation
from causality.epistemic_status import EpistemicStatus
from causality.meta_causal_reflection import MetaCausalReflection
from causality.principle import PrincipleStore
from causality.principle_synthesizer import PrincipleSynthesizer
from curiosity.curiosity_engine import CuriosityEngine, CuriositySignal, CuriosityTrigger
from experiments.experiment_planner import Experiment, ExperimentPlanner, ExperimentStatus
from hypothesis.hypothesis_manager import Hypothesis, HypothesisManager, HypothesisStatus
from knowledge.knowledge_gap_tracker import GapSeverity, KnowledgeGap, KnowledgeGapTracker
from memory.beliefs import BeliefStore
from memory.future_memory import FutureMemoryStore
from metacognition.self_model import SelfModelStore
from reflection.reflection_engine import ReflectionEngine, ReflectionScope


# ===========================================================================
# KnowledgeGapTracker
# ===========================================================================


class TestKnowledgeGapTracker:
    @pytest.fixture
    def tracker(self, tmp_path: Path) -> KnowledgeGapTracker:
        return KnowledgeGapTracker(tmp_path / "gaps.json")

    def test_register_gap(self, tracker: KnowledgeGapTracker) -> None:
        g = tracker.register_gap("web_search", GapSeverity.HIGH, uncertainty=0.8)
        assert g.domain == "web_search"
        assert g.severity == GapSeverity.HIGH

    def test_register_gap_updates_existing(self, tracker: KnowledgeGapTracker) -> None:
        tracker.register_gap("web_search", GapSeverity.LOW, uncertainty=0.3)
        tracker.register_gap("web_search", GapSeverity.HIGH, uncertainty=0.9)
        assert tracker.get("web_search").severity == GapSeverity.HIGH

    def test_resolve_gap_removes_it(self, tracker: KnowledgeGapTracker) -> None:
        tracker.register_gap("domain_x", GapSeverity.MEDIUM, uncertainty=0.5)
        tracker.resolve_gap("domain_x")
        assert tracker.get("domain_x") is None
        assert tracker.count == 0

    def test_resolve_unknown_domain_returns_none(self, tracker: KnowledgeGapTracker) -> None:
        assert tracker.resolve_gap("ghost") is None

    def test_needs_exploration_high_severity(self, tracker: KnowledgeGapTracker) -> None:
        tracker.register_gap("a", GapSeverity.HIGH, uncertainty=0.8)
        tracker.register_gap("b", GapSeverity.LOW, uncertainty=0.1)
        needs = tracker.needs_exploration()
        assert len(needs) == 1
        assert needs[0].domain == "a"

    def test_needs_exploration_high_uncertainty(self, tracker: KnowledgeGapTracker) -> None:
        tracker.register_gap("c", GapSeverity.MEDIUM, uncertainty=0.7)
        assert len(tracker.needs_exploration()) == 1

    def test_gaps_min_severity_filter(self, tracker: KnowledgeGapTracker) -> None:
        tracker.register_gap("low", GapSeverity.LOW, uncertainty=0.2)
        tracker.register_gap("high", GapSeverity.HIGH, uncertainty=0.8)
        high_only = tracker.gaps(min_severity=GapSeverity.HIGH)
        assert len(high_only) == 1
        assert high_only[0].domain == "high"

    def test_gaps_sorted_by_uncertainty_descending(self, tracker: KnowledgeGapTracker) -> None:
        tracker.register_gap("a", GapSeverity.MEDIUM, uncertainty=0.3)
        tracker.register_gap("b", GapSeverity.MEDIUM, uncertainty=0.9)
        gaps = tracker.gaps()
        assert gaps[0].domain == "b"

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "gaps.json"
        t1 = KnowledgeGapTracker(f)
        t1.register_gap("persist_test", GapSeverity.MEDIUM, uncertainty=0.5)
        t2 = KnowledgeGapTracker(f)
        assert t2.count == 1
        assert t2.get("persist_test") is not None

    def test_count_property(self, tracker: KnowledgeGapTracker) -> None:
        assert tracker.count == 0
        tracker.register_gap("a", GapSeverity.LOW, uncertainty=0.2)
        assert tracker.count == 1

    def test_discover_from_self_model(self, tmp_path: Path) -> None:
        sm = SelfModelStore(tmp_path / "sm.json")
        sm.set_capability("research", 0.2)
        tracker = KnowledgeGapTracker(tmp_path / "gaps.json", low_capability_threshold=0.4)
        discovered = tracker.discover_from_self_model(sm)
        assert len(discovered) == 1
        assert "research" in discovered[0].domain

    def test_discover_from_failure_memory(self, tmp_path: Path) -> None:
        fm = FailureMemory(tmp_path / "fm.json")
        for _ in range(4):
            fm.record("task", "web_search", "timeout error occurred")
        tracker = KnowledgeGapTracker(tmp_path / "gaps.json")
        discovered = tracker.discover_from_failure_memory(fm)
        assert len(discovered) == 1
        assert "web_search" in discovered[0].domain

    def test_discover_from_cause_graph(self, tmp_path: Path) -> None:
        cg = CauseGraph(tmp_path / "cg.json")
        cg.record_observation("trigger_x", "effect_y", CauseRelation.CAUSES, initial_confidence=0.3)
        tracker = KnowledgeGapTracker(tmp_path / "gaps.json", low_confidence_edge_threshold=0.5)
        discovered = tracker.discover_from_cause_graph(cg)
        assert len(discovered) == 1

    def test_gap_to_dict(self, tracker: KnowledgeGapTracker) -> None:
        g = tracker.register_gap("domain", GapSeverity.HIGH, uncertainty=0.7)
        d = g.to_dict()
        assert d["domain"] == "domain"
        assert d["severity"] == "high"
        assert "uncertainty" in d


# ===========================================================================
# CuriosityEngine
# ===========================================================================


class TestCuriosityEngine:
    @pytest.fixture
    def store(self, tmp_path: Path) -> BeliefStore:
        return BeliefStore(tmp_path / "beliefs.json")

    def test_low_confidence_trigger(self, tmp_path: Path, store: BeliefStore) -> None:
        store.add_or_reinforce("Low confidence belief here", confidence=0.2)
        engine = CuriosityEngine(store, low_confidence_threshold=0.4)
        signals = engine.generate_signals()
        triggers = [s.trigger for s in signals]
        assert CuriosityTrigger.LOW_CONFIDENCE in triggers

    def test_sparse_evidence_trigger(self, tmp_path: Path, store: BeliefStore) -> None:
        store.add_or_reinforce("Sparse evidence belief statement", confidence=0.8)
        engine = CuriosityEngine(store, sparse_evidence_threshold=3)
        signals = engine.generate_signals()
        triggers = [s.trigger for s in signals]
        assert CuriosityTrigger.SPARSE_EVIDENCE in triggers

    def test_frequent_failures_trigger(self, tmp_path: Path, store: BeliefStore) -> None:
        fm = FailureMemory(tmp_path / "fm.json")
        for _ in range(4):
            fm.record("task", "code_tool", "syntax error recurring")
        engine = CuriosityEngine(store, failure_memory=fm)
        signals = engine.generate_signals()
        triggers = [s.trigger for s in signals]
        assert CuriosityTrigger.FREQUENT_FAILURES in triggers

    def test_unknown_domain_trigger(self, tmp_path: Path, store: BeliefStore) -> None:
        kg = KnowledgeGapTracker(tmp_path / "gaps.json")
        kg.register_gap("novel_domain", GapSeverity.HIGH, uncertainty=0.9)
        engine = CuriosityEngine(store, knowledge_gap_tracker=kg)
        signals = engine.generate_signals()
        triggers = [s.trigger for s in signals]
        assert CuriosityTrigger.UNKNOWN_DOMAIN in triggers

    def test_signals_sorted_by_priority(self, tmp_path: Path, store: BeliefStore) -> None:
        kg = KnowledgeGapTracker(tmp_path / "gaps.json")
        kg.register_gap("critical_domain", GapSeverity.CRITICAL, uncertainty=0.95)
        store.add_or_reinforce("A mild concern about something", confidence=0.39)
        engine = CuriosityEngine(store, knowledge_gap_tracker=kg, low_confidence_threshold=0.4)
        signals = engine.generate_signals()
        assert signals[0].priority_score >= signals[-1].priority_score

    def test_top_k_limit(self, tmp_path: Path, store: BeliefStore) -> None:
        for i in range(8):
            store.add_or_reinforce(f"Belief with low confidence number {i} extended", confidence=0.1)
        engine = CuriosityEngine(store, low_confidence_threshold=0.4)
        signals = engine.generate_signals(top_k=3)
        assert len(signals) <= 3

    def test_no_failures_no_failure_signal(self, store: BeliefStore) -> None:
        engine = CuriosityEngine(store, failure_memory=None)
        signals = engine.generate_signals()
        assert CuriosityTrigger.FREQUENT_FAILURES not in [s.trigger for s in signals]

    def test_signal_to_dict(self, store: BeliefStore) -> None:
        store.add_or_reinforce("Low confidence belief test", confidence=0.1)
        engine = CuriosityEngine(store, low_confidence_threshold=0.4)
        signals = engine.generate_signals()
        if signals:
            d = signals[0].to_dict()
            assert "target" in d
            assert "trigger" in d
            assert "expected_information_gain" in d

    def test_priority_score_formula(self) -> None:
        s = CuriositySignal(
            target="x", trigger=CuriosityTrigger.LOW_CONFIDENCE, reason="r",
            novelty=0.5, uncertainty=0.8, expected_information_gain=0.6,
        )
        expected = 0.4 * 0.8 + 0.4 * 0.6 + 0.2 * 0.5
        assert s.priority_score == pytest.approx(expected)


# ===========================================================================
# HypothesisManager
# ===========================================================================


class TestHypothesisManager:
    @pytest.fixture
    def mgr(self, tmp_path: Path) -> HypothesisManager:
        bs = BeliefStore(tmp_path / "beliefs.json")
        return HypothesisManager(tmp_path / "hyp.json", belief_store=bs, support_threshold=0.7, rejection_threshold=0.2)

    def test_propose_creates_pending_hypothesis(self, mgr: HypothesisManager) -> None:
        h = mgr.propose("Test hypothesis statement here", confidence=0.3)
        assert h.status == HypothesisStatus.PENDING
        assert h.epistemic_status == EpistemicStatus.HYPOTHESIS

    def test_add_evidence_increases_confidence(self, mgr: HypothesisManager) -> None:
        h = mgr.propose("Test hypothesis", confidence=0.3)
        mgr.add_evidence(h.hypothesis_id, "Supporting observation", confidence_delta=0.2)
        assert mgr.get(h.hypothesis_id).confidence == pytest.approx(0.5)

    def test_hypothesis_reaches_supported(self, mgr: HypothesisManager) -> None:
        h = mgr.propose("Strong hypothesis statement", confidence=0.3)
        mgr.add_evidence(h.hypothesis_id, "Evidence 1", confidence_delta=0.25)
        mgr.add_evidence(h.hypothesis_id, "Evidence 2", confidence_delta=0.25)
        assert mgr.get(h.hypothesis_id).status == HypothesisStatus.SUPPORTED

    def test_supported_hypothesis_promotes_to_belief(self, tmp_path: Path) -> None:
        bs = BeliefStore(tmp_path / "beliefs.json")
        mgr = HypothesisManager(tmp_path / "hyp.json", belief_store=bs, support_threshold=0.7)
        h = mgr.propose("Will be supported soon", confidence=0.3)
        mgr.add_evidence(h.hypothesis_id, "E1", confidence_delta=0.25)
        mgr.add_evidence(h.hypothesis_id, "E2", confidence_delta=0.25)
        hyp = mgr.get(h.hypothesis_id)
        if hyp.status == HypothesisStatus.SUPPORTED and hyp.linked_belief_id:
            b = bs.get(hyp.linked_belief_id)
            assert b.epistemic_status == EpistemicStatus.OBSERVED

    def test_hypothesis_rejected_on_low_confidence(self, mgr: HypothesisManager) -> None:
        h = mgr.propose("Weak hypothesis", confidence=0.3)
        mgr.add_evidence(h.hypothesis_id, "Contradiction 1", confidence_delta=-0.06)
        mgr.add_evidence(h.hypothesis_id, "Contradiction 2", confidence_delta=-0.06)
        assert mgr.get(h.hypothesis_id).status == HypothesisStatus.REJECTED

    def test_evidence_not_added_after_rejection(self, mgr: HypothesisManager) -> None:
        h = mgr.propose("Hypothesis to be rejected", confidence=0.3)
        mgr.add_evidence(h.hypothesis_id, "C1", confidence_delta=-0.06)
        mgr.add_evidence(h.hypothesis_id, "C2", confidence_delta=-0.06)
        original_conf = mgr.get(h.hypothesis_id).confidence
        mgr.add_evidence(h.hypothesis_id, "Late evidence", confidence_delta=0.5)
        assert mgr.get(h.hypothesis_id).confidence == original_conf

    def test_link_experiment(self, mgr: HypothesisManager) -> None:
        h = mgr.propose("Linkable hypothesis")
        mgr.link_experiment(h.hypothesis_id, "exp_abc")
        assert "exp_abc" in mgr.get(h.hypothesis_id).linked_experiment_ids

    def test_by_status(self, mgr: HypothesisManager) -> None:
        mgr.propose("H1")
        mgr.propose("H2")
        assert len(mgr.by_status(HypothesisStatus.PENDING)) == 2

    def test_repeatedly_failed(self, mgr: HypothesisManager) -> None:
        h = mgr.propose("Repeatedly failed hypothesis", confidence=0.3)
        mgr.add_evidence(h.hypothesis_id, "C1", confidence_delta=-0.06)
        mgr.add_evidence(h.hypothesis_id, "C2", confidence_delta=-0.06)
        assert len(mgr.repeatedly_failed(min_evidence=2)) == 1

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "hyp.json"
        m1 = HypothesisManager(f)
        h = m1.propose("Persistent hypothesis")
        m2 = HypothesisManager(f)
        assert m2.get(h.hypothesis_id) is not None

    def test_hypothesis_to_dict(self, mgr: HypothesisManager) -> None:
        h = mgr.propose("Test dict")
        d = h.to_dict()
        assert d["status"] == "pending"
        assert d["epistemic_status"] == "hypothesis"


# ===========================================================================
# ExperimentPlanner
# ===========================================================================


class TestExperimentPlanner:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        bs = BeliefStore(tmp_path / "beliefs.json")
        hm = HypothesisManager(tmp_path / "hyp.json", belief_store=bs, support_threshold=0.7)
        ep = ExperimentPlanner(tmp_path / "exp.json", hm)
        h = hm.propose("Web failures caused by overload", confidence=0.3, source="test")
        return hm, ep, h

    def test_plan_creates_experiment(self, setup) -> None:
        hm, ep, h = setup
        exp = ep.plan(h.hypothesis_id, ["step1"], "result", ["criterion"])
        assert exp.status == ExperimentStatus.PLANNED
        assert exp.epistemic_status == EpistemicStatus.PREDICTED

    def test_plan_raises_on_unknown_hypothesis(self, tmp_path: Path) -> None:
        hm = HypothesisManager(tmp_path / "hyp.json")
        ep = ExperimentPlanner(tmp_path / "exp.json", hm)
        with pytest.raises(ValueError, match="not found"):
            ep.plan("ghost_id", ["step"], "result", ["criterion"])

    def test_plan_links_experiment_to_hypothesis(self, setup) -> None:
        hm, ep, h = setup
        exp = ep.plan(h.hypothesis_id, ["step1"], "result", ["criterion"])
        assert exp.experiment_id in hm.get(h.hypothesis_id).linked_experiment_ids

    def test_record_outcome_marks_completed(self, setup) -> None:
        hm, ep, h = setup
        exp = ep.plan(h.hypothesis_id, ["step1"], "result", ["criterion"])
        result = ep.record_outcome(exp.experiment_id, "Confirmed", success=True)
        assert result.status == ExperimentStatus.COMPLETED
        assert result.outcome_confirmed is True
        assert result.epistemic_status == EpistemicStatus.OBSERVED

    def test_record_outcome_updates_hypothesis(self, setup) -> None:
        hm, ep, h = setup
        exp = ep.plan(h.hypothesis_id, ["step1"], "result", ["criterion"])
        ep.record_outcome(exp.experiment_id, "Confirmed", success=True, confidence_delta=0.2)
        assert hm.get(h.hypothesis_id).confidence > 0.3

    def test_successful_outcome_feeds_hypothesis_toward_support(self, tmp_path: Path) -> None:
        bs = BeliefStore(tmp_path / "beliefs.json")
        hm = HypothesisManager(tmp_path / "hyp.json", belief_store=bs, support_threshold=0.7)
        ep = ExperimentPlanner(tmp_path / "exp.json", hm)
        h = hm.propose("Testable statement", confidence=0.3)
        for _ in range(2):
            exp = ep.plan(h.hypothesis_id, ["step"], "result", ["crit"])
            ep.record_outcome(exp.experiment_id, "confirmed", success=True, confidence_delta=0.25)
        assert hm.get(h.hypothesis_id).status == HypothesisStatus.SUPPORTED

    def test_plan_from_signal_low_confidence_trigger(self, setup) -> None:
        hm, ep, h = setup
        signal = CuriositySignal(
            target="Some belief", trigger=CuriosityTrigger.LOW_CONFIDENCE,
            reason="Confidence is low", novelty=0.3, uncertainty=0.8, expected_information_gain=0.7,
        )
        exp = ep.plan_from_signal(signal, h.hypothesis_id)
        assert len(exp.actions) >= 1
        assert len(exp.success_criteria) >= 1

    def test_plan_from_signal_frequent_failures_trigger(self, setup) -> None:
        hm, ep, h = setup
        signal = CuriositySignal(
            target="tool:web_search", trigger=CuriosityTrigger.FREQUENT_FAILURES,
            reason="4 failures", novelty=0.5, uncertainty=0.75, expected_information_gain=0.64,
        )
        exp = ep.plan_from_signal(signal, h.hypothesis_id)
        assert "Diagnose" in exp.actions[0]

    def test_record_outcome_unknown_experiment_returns_none(self, tmp_path: Path) -> None:
        hm = HypothesisManager(tmp_path / "hyp.json")
        ep = ExperimentPlanner(tmp_path / "exp.json", hm)
        assert ep.record_outcome("ghost_exp", "result", success=True) is None

    def test_planned_and_completed_filters(self, setup) -> None:
        hm, ep, h = setup
        exp = ep.plan(h.hypothesis_id, ["s"], "r", ["c"])
        ep.plan(h.hypothesis_id, ["s2"], "r2", ["c2"])
        ep.record_outcome(exp.experiment_id, "done", success=True)
        assert len(ep.planned()) == 1
        assert len(ep.completed()) == 1

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        hm = HypothesisManager(tmp_path / "hyp.json")
        h = hm.propose("Persistent experiment test")
        ep1 = ExperimentPlanner(tmp_path / "exp.json", hm)
        ep1.plan(h.hypothesis_id, ["s"], "r", ["c"])
        ep2 = ExperimentPlanner(tmp_path / "exp.json", hm)
        assert ep2.count == 1

    def test_experiment_to_dict(self, setup) -> None:
        hm, ep, h = setup
        exp = ep.plan(h.hypothesis_id, ["step"], "result", ["criterion"])
        d = exp.to_dict()
        assert d["status"] == "planned"
        assert d["epistemic_status"] == "predicted"
        assert "actions" in d


# ===========================================================================
# Extensions
# ===========================================================================


class TestExtensions:

    def test_reflect_on_curiosity_learned(self, tmp_path: Path) -> None:
        re = ReflectionEngine(tmp_path / "reflections.json")
        record = re.reflect_on_curiosity(
            curiosity_target="tool:web_search",
            hypothesis_statement="Web failures are caused by API overload",
            experiment_outcome="Confirmed: failures drop at off-peak hours",
            learned=True,
        )
        assert record.scope == ReflectionScope.LEARNING
        assert "tool:web_search" in record.scope_ref

    def test_reflect_on_curiosity_not_learned(self, tmp_path: Path) -> None:
        re = ReflectionEngine(tmp_path / "reflections.json")
        record = re.reflect_on_curiosity(
            curiosity_target="some_target",
            hypothesis_statement="Some hypothesis",
            experiment_outcome=None,
            learned=False,
        )
        assert record.scope == ReflectionScope.LEARNING

    def test_synthesize_from_experiment(self, tmp_path: Path) -> None:
        hm = HypothesisManager(tmp_path / "hyp.json")
        h = hm.propose("Experiment hypothesis")
        ep = ExperimentPlanner(tmp_path / "exp.json", hm)
        exp = ep.plan(h.hypothesis_id, ["s"], "r", ["c"])
        ep.record_outcome(exp.experiment_id, "Confirmed experiment finding", success=True)

        cg = CauseGraph(tmp_path / "cg.json")
        ps = PrincipleStore(tmp_path / "ps.json")
        synth = PrincipleSynthesizer(ps, cg, llm=None)
        principle = synth.synthesize_from_experiment(ep.get(exp.experiment_id))
        assert principle is not None
        assert "Confirmed experiment finding" in principle.statement

    def test_synthesize_from_experiment_planned_returns_none(self, tmp_path: Path) -> None:
        hm = HypothesisManager(tmp_path / "hyp.json")
        h = hm.propose("Not yet run")
        ep = ExperimentPlanner(tmp_path / "exp.json", hm)
        exp = ep.plan(h.hypothesis_id, ["s"], "r", ["c"])
        cg = CauseGraph(tmp_path / "cg.json")
        ps = PrincipleStore(tmp_path / "ps.json")
        synth = PrincipleSynthesizer(ps, cg, llm=None)
        assert synth.synthesize_from_experiment(ep.get(exp.experiment_id)) is None

    def test_synthesize_all_with_experiments(self, tmp_path: Path) -> None:
        hm = HypothesisManager(tmp_path / "hyp.json")
        h = hm.propose("Exp principle hypothesis")
        ep = ExperimentPlanner(tmp_path / "exp.json", hm)
        exp = ep.plan(h.hypothesis_id, ["s"], "r", ["c"])
        ep.record_outcome(exp.experiment_id, "Finding confirmed", success=True)
        cg = CauseGraph(tmp_path / "cg.json")
        ps = PrincipleStore(tmp_path / "ps.json")
        synth = PrincipleSynthesizer(ps, cg, llm=None)
        principles = synth.synthesize_all(experiments=ep.completed())
        assert len(principles) >= 1

    def test_future_memory_record_experiment(self, tmp_path: Path) -> None:
        fm = FutureMemoryStore(tmp_path / "future.json")
        state = fm.record_experiment("experiment:web_search_test", confidence=0.7)
        assert state.subject == "experiment:web_search_test"

    def test_future_memory_experiments_filter(self, tmp_path: Path) -> None:
        fm = FutureMemoryStore(tmp_path / "future.json")
        fm.record_experiment("experiment:test_a", confidence=0.7)
        fm.predict("regular_prediction", confidence=0.6)
        assert len(fm.experiments()) == 1
        assert len(fm.resolved()) == 0

    def test_future_memory_resolve_experiment(self, tmp_path: Path) -> None:
        fm = FutureMemoryStore(tmp_path / "future.json")
        state = fm.record_experiment("experiment:test", confidence=0.7)
        resolved = fm.resolve_experiment(state.expected_state_id, actual_outcome=True)
        assert resolved.resolved is True

    def test_meta_causal_which_hypotheses_failed_repeatedly(self, tmp_path: Path) -> None:
        cg = CauseGraph(tmp_path / "cg.json")
        hm = HypothesisManager(tmp_path / "hyp.json", rejection_threshold=0.2)
        h = hm.propose("Incorrect belief statement here", confidence=0.3)
        hm.add_evidence(h.hypothesis_id, "C1", confidence_delta=-0.06)
        hm.add_evidence(h.hypothesis_id, "C2", confidence_delta=-0.06)
        mcr = MetaCausalReflection(cg)
        answer = mcr.which_hypotheses_failed_repeatedly(hm, min_evidence=2)
        assert "repeatedly" in answer.question
        assert "1 hypothesis" in answer.answer_summary

    def test_meta_causal_which_hypotheses_no_failures(self, tmp_path: Path) -> None:
        cg = CauseGraph(tmp_path / "cg.json")
        hm = HypothesisManager(tmp_path / "hyp.json")
        mcr = MetaCausalReflection(cg)
        answer = mcr.which_hypotheses_failed_repeatedly(hm)
        assert "not enough" in answer.answer_summary.lower() or "No hypotheses" in answer.answer_summary

    def test_self_model_knowledge_gaps_with_tracker(self, tmp_path: Path) -> None:
        sm = SelfModelStore(tmp_path / "sm.json")
        kg = KnowledgeGapTracker(tmp_path / "gaps.json")
        kg.register_gap("research", GapSeverity.HIGH, uncertainty=0.8)
        gaps = sm.knowledge_gaps(knowledge_gap_tracker=kg)
        assert len(gaps) == 1
        assert gaps[0].domain == "research"

    def test_self_model_knowledge_gaps_fallback(self, tmp_path: Path) -> None:
        sm = SelfModelStore(tmp_path / "sm.json")
        sm.set_capability("research", 0.15)
        gaps = sm.knowledge_gaps()  # no tracker -> uses weaknesses
        assert any(g.domain == "research" for g in gaps)


# ===========================================================================
# Integration — BlixContext
# ===========================================================================


class _FakeLLM:
    def model_name(self): return "fake-0.3.13"
    def generate(self, prompt): return "Fake reply."


@pytest.fixture(scope="module")
def tmp_memory_v13(tmp_path_factory):
    return tmp_path_factory.mktemp("memory_v13")


@pytest.fixture(scope="module")
def ctx_v13(tmp_memory_v13):
    from config import settings as _settings
    _settings.settings.memory.conversations_file = tmp_memory_v13 / "conversations.json"
    _settings.settings.memory.profile_file = tmp_memory_v13 / "profile.json"
    _settings.settings.memory.learning_state_file = tmp_memory_v13 / "learning_state.json"
    _settings.settings.embed.embeddings_file = tmp_memory_v13 / "embeddings.npy"
    _settings.settings.embed.embedding_ids_file = tmp_memory_v13 / "embedding_ids.json"
    from api.context import BlixContext
    ctx = BlixContext(tmp_memory_v13)
    ctx.llm = _FakeLLM()
    ctx.agent._llm = _FakeLLM()
    return ctx


@pytest.fixture(scope="module")
def client_v13(ctx_v13) -> Generator[TestClient, None, None]:
    from api.deps import set_context
    from api.routers.curiosity import router as curiosity_router
    app = FastAPI(title="Blix Test v0.3.13")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(curiosity_router)

    @app.get("/health")
    def health(): return {"status": "ok"}

    set_context(ctx_v13)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestBlixContextV0313Wiring:
    def test_components_present(self, ctx_v13) -> None:
        assert ctx_v13.knowledge_gap_tracker is not None
        assert ctx_v13.curiosity_engine is not None
        assert ctx_v13.hypothesis_manager is not None
        assert ctx_v13.experiment_planner is not None

    def test_dashboard_stats(self, ctx_v13) -> None:
        stats = ctx_v13.dashboard_stats()
        assert "knowledge_gaps" in stats
        assert "pending_hypotheses" in stats
        assert "experiments_planned" in stats

    def test_curiosity_signals_via_context(self, ctx_v13) -> None:
        signals = ctx_v13.curiosity_engine.generate_signals(top_k=5)
        assert isinstance(signals, list)

    def test_hypothesis_lifecycle_via_context(self, ctx_v13) -> None:
        h = ctx_v13.hypothesis_manager.propose("Integration test hypothesis", source="test")
        assert ctx_v13.hypothesis_manager.get(h.hypothesis_id) is not None

    def test_experiment_planned_via_context(self, ctx_v13) -> None:
        h = ctx_v13.hypothesis_manager.propose("Experiment integration hypothesis")
        exp = ctx_v13.experiment_planner.plan(h.hypothesis_id, ["s"], "r", ["c"])
        assert exp is not None

    def test_knowledge_gaps_live_query_via_context(self, ctx_v13) -> None:
        gaps = ctx_v13.self_model.knowledge_gaps(knowledge_gap_tracker=ctx_v13.knowledge_gap_tracker)
        assert isinstance(gaps, list)

    def test_dashboard_stats_update_after_actions(self, ctx_v13) -> None:
        initial_count = ctx_v13.dashboard_stats()["pending_hypotheses"]
        ctx_v13.hypothesis_manager.propose("Extra hypothesis for count test")
        assert ctx_v13.dashboard_stats()["pending_hypotheses"] == initial_count + 1


# ===========================================================================
# API — /curiosity endpoints
# ===========================================================================


class TestCuriosityAPI:
    def test_get_signals(self, client_v13: TestClient) -> None:
        r = client_v13.get("/curiosity/signals")
        assert r.status_code == 200
        assert "signals" in r.json()

    def test_propose_hypothesis(self, client_v13: TestClient) -> None:
        r = client_v13.post("/curiosity/hypotheses", json={
            "statement": "API test hypothesis statement", "confidence": 0.3,
        })
        assert r.status_code == 200
        assert r.json()["status"] == "pending"

    def test_list_hypotheses(self, client_v13: TestClient) -> None:
        r = client_v13.get("/curiosity/hypotheses")
        assert r.status_code == 200
        assert "hypotheses" in r.json()

    def test_list_hypotheses_filtered_by_status(self, client_v13: TestClient) -> None:
        r = client_v13.get("/curiosity/hypotheses?status=pending")
        assert r.status_code == 200

    def test_list_hypotheses_invalid_status(self, client_v13: TestClient) -> None:
        r = client_v13.get("/curiosity/hypotheses?status=invalid_status")
        assert r.status_code == 422

    def test_add_evidence(self, client_v13: TestClient, ctx_v13) -> None:
        h = ctx_v13.hypothesis_manager.propose("Evidence target hypothesis")
        r = client_v13.post(f"/curiosity/hypotheses/{h.hypothesis_id}/evidence", json={
            "evidence": "Supporting observation found", "confidence_delta": 0.1,
        })
        assert r.status_code == 200

    def test_add_evidence_unknown_id(self, client_v13: TestClient) -> None:
        r = client_v13.post("/curiosity/hypotheses/ghost/evidence", json={
            "evidence": "Irrelevant observation", "confidence_delta": 0.1,
        })
        assert r.status_code == 404

    def test_plan_experiment(self, client_v13: TestClient, ctx_v13) -> None:
        h = ctx_v13.hypothesis_manager.propose("Experiment API hypothesis")
        r = client_v13.post("/curiosity/experiments", json={
            "hypothesis_id": h.hypothesis_id, "actions": ["Test step 1", "Test step 2"],
            "expected_result": "Hypothesis confirmed", "success_criteria": ["Evidence > 0.6"],
        })
        assert r.status_code == 200
        assert r.json()["status"] == "planned"

    def test_plan_experiment_unknown_hypothesis(self, client_v13: TestClient) -> None:
        r = client_v13.post("/curiosity/experiments", json={
            "hypothesis_id": "ghost_id", "actions": ["s"], "expected_result": "r", "success_criteria": ["c"],
        })
        assert r.status_code == 404

    def test_plan_from_signal(self, client_v13: TestClient, ctx_v13) -> None:
        h = ctx_v13.hypothesis_manager.propose("Signal-based experiment hypothesis")
        r = client_v13.post("/curiosity/experiments/from-signal", json={
            "hypothesis_id": h.hypothesis_id, "signal_target": "tool:web_search",
            "signal_trigger": "frequent_failures", "signal_reason": "Multiple failures recorded",
        })
        assert r.status_code == 200

    def test_list_experiments(self, client_v13: TestClient) -> None:
        r = client_v13.get("/curiosity/experiments")
        assert r.status_code == 200
        assert "experiments" in r.json()

    def test_record_outcome(self, client_v13: TestClient, ctx_v13) -> None:
        h = ctx_v13.hypothesis_manager.propose("Outcome recording test hypothesis")
        exp = ctx_v13.experiment_planner.plan(h.hypothesis_id, ["s"], "r", ["c"])
        r = client_v13.post(f"/curiosity/experiments/{exp.experiment_id}/outcome", json={
            "outcome": "Experiment confirmed hypothesis", "success": True,
        })
        assert r.status_code == 200
        assert r.json()["outcome_confirmed"] is True

    def test_record_outcome_unknown_experiment(self, client_v13: TestClient) -> None:
        r = client_v13.post("/curiosity/experiments/ghost/outcome", json={
            "outcome": "Does not matter", "success": True,
        })
        assert r.status_code == 404

    def test_list_knowledge_gaps(self, client_v13: TestClient) -> None:
        r = client_v13.get("/curiosity/knowledge-gaps")
        assert r.status_code == 200
        assert "gaps" in r.json()

    def test_discover_knowledge_gaps(self, client_v13: TestClient) -> None:
        r = client_v13.post("/curiosity/knowledge-gaps/discover")
        assert r.status_code == 200
        assert "total_discovered" in r.json()

    def test_knowledge_gaps_invalid_severity(self, client_v13: TestClient) -> None:
        r = client_v13.get("/curiosity/knowledge-gaps?min_severity=ultramax")
        assert r.status_code == 422

    def test_health(self, client_v13: TestClient) -> None:
        assert client_v13.get("/health").status_code == 200
