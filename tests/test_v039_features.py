"""
Tests for Blix v0.3.9 — "Global Workspace".

Covers:
1.  events.event_types / event_bus / event_store          (Cognitive Event Bus)
2.  workspace.attention_manager                              (Attention System)
3.  workspace.broadcast_bus                                    (Broadcast Mechanism)
4.  workspace.workspace_memory                                   (Workspace working stage)
5.  workspace.global_workspace                                     (Global Workspace orchestration)
6.  specialists.*                                                    (Internal Specialists + Consensus)
7.  retrieval.active_attention_retriever                                (Active Attention Retrieval)
8.  workspace.snapshot                                                    (Workspace Snapshotting)
9.  workspace.inner_dialogue                                                (Internal Dialogue)
10. evaluation.workspace_metrics / attention_metrics / coordination_metrics  (Coordination Metrics)
Integration  — BlixContext wiring
API          — /workspace endpoints

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
from agents.types import ExecutionResult, ExecutionStatus, Task, TaskGraph
from evaluation.attention_metrics import AttentionGroundTruthCase, AttentionMetrics
from evaluation.coordination_metrics import CoordinationMetrics, SubsystemParticipation
from evaluation.workspace_metrics import WorkspaceCycleStats, WorkspaceMetrics
from events.event_bus import EventBus
from events.event_store import EventStore
from events.event_types import (
    CognitiveEvent,
    EventType,
    failure_event,
    task_completed_event,
)
from memory.beliefs import BeliefStore
from planning.critic import PlanCritic
from planning.plan_evaluator import PlanQualityEvaluator
from retrieval.active_attention_retriever import ActiveAttentionRetriever, AttentionRetrievalWeights
from schemas.memory_entry import MemoryEntry
from specialists.base import BaseSpecialist, SpecialistOpinion
from specialists.consensus import ConsensusResult, SpecialistConsensus
from specialists.memory_specialist import MemorySpecialist
from specialists.planning_specialist import PlanningSpecialist
from specialists.reflection_specialist import ReflectionSpecialist
from specialists.verification_specialist import VerificationSpecialist
from verification.verifier import VerificationEngine
from workspace.attention_manager import AttentionCandidate, AttentionManager, AttentionScore, AttentionWeights
from workspace.broadcast_bus import BroadcastBus, BroadcastRecord
from workspace.global_workspace import GlobalWorkspace, WorkspaceCycleResult
from workspace.inner_dialogue import (
    DialogueTranscript,
    InnerDialogue,
    critic_voice,
    planner_voice,
    reflection_voice,
    self_model_voice,
)
from workspace.snapshot import WorkspaceSnapshot, WorkspaceSnapshotStore
from workspace.workspace_memory import WorkspaceItem, WorkspaceMemory
from metacognition.self_model import SelfModelStore
from metacognition.strategy_manager import StrategyManager
from datetime import datetime, timezone


# ===========================================================================
# Item 4 — Cognitive Event Bus (event_types, event_bus, event_store)
# ===========================================================================


class TestEventTypes:
    def test_task_completed_event_payload(self) -> None:
        evt = task_completed_event("executor", "Step 1", True, domain="coding")
        assert evt.event_type == EventType.TASK_COMPLETED
        assert evt.payload["success"] is True

    def test_failure_event_payload(self) -> None:
        evt = failure_event("planner", "Step 1", "timeout", tool="web_search")
        assert evt.event_type == EventType.FAILURE
        assert evt.payload["reason"] == "timeout"

    def test_event_to_dict_and_back(self) -> None:
        evt = task_completed_event("executor", "Step 1", True)
        d = evt.to_dict()
        restored = CognitiveEvent.from_dict(d)
        assert restored.event_type == evt.event_type
        assert restored.payload == evt.payload

    def test_event_has_unique_id(self) -> None:
        evt1 = task_completed_event("a", "x", True)
        evt2 = task_completed_event("a", "x", True)
        assert evt1.event_id != evt2.event_id


class TestEventBus:
    @pytest.fixture
    def bus(self) -> EventBus:
        return EventBus()

    def test_subscribe_and_publish(self, bus: EventBus) -> None:
        received = []
        bus.subscribe(EventType.FAILURE, lambda e: received.append(e))
        bus.publish(failure_event("planner", "Step 1", "oops"))
        assert len(received) == 1

    def test_unrelated_event_type_not_received(self, bus: EventBus) -> None:
        received = []
        bus.subscribe(EventType.FAILURE, lambda e: received.append(e))
        bus.publish(task_completed_event("executor", "Step 1", True))
        assert len(received) == 0

    def test_wildcard_subscriber_receives_everything(self, bus: EventBus) -> None:
        received = []
        bus.subscribe_all(lambda e: received.append(e))
        bus.publish(failure_event("planner", "Step 1", "oops"))
        bus.publish(task_completed_event("executor", "Step 1", True))
        assert len(received) == 2

    def test_multiple_subscribers_all_invoked(self, bus: EventBus) -> None:
        counter = {"n": 0}
        bus.subscribe(EventType.FAILURE, lambda e: counter.__setitem__("n", counter["n"] + 1))
        bus.subscribe(EventType.FAILURE, lambda e: counter.__setitem__("n", counter["n"] + 1))
        bus.publish(failure_event("planner", "Step 1", "oops"))
        assert counter["n"] == 2

    def test_unsubscribe_removes_handler(self, bus: EventBus) -> None:
        received = []
        handler = lambda e: received.append(e)
        bus.subscribe(EventType.FAILURE, handler)
        assert bus.unsubscribe(EventType.FAILURE, handler) is True
        bus.publish(failure_event("planner", "Step 1", "oops"))
        assert len(received) == 0

    def test_broken_subscriber_does_not_crash_publish(self, bus: EventBus) -> None:
        def bad_handler(e):
            raise ValueError("boom")
        good_received = []
        bus.subscribe(EventType.FAILURE, bad_handler)
        bus.subscribe(EventType.FAILURE, lambda e: good_received.append(e))
        bus.publish(failure_event("planner", "Step 1", "oops"))
        assert len(good_received) == 1  # second subscriber still ran

    def test_publish_persists_to_event_store(self, tmp_path: Path) -> None:
        store = EventStore(tmp_path / "events.json")
        bus = EventBus(event_store=store)
        bus.publish(failure_event("planner", "Step 1", "oops"))
        assert store.count == 1

    def test_subscriber_count(self, bus: EventBus) -> None:
        bus.subscribe(EventType.FAILURE, lambda e: None)
        bus.subscribe(EventType.FAILURE, lambda e: None)
        assert bus.subscriber_count(EventType.FAILURE) == 2

    def test_publish_count_tracked(self, bus: EventBus) -> None:
        bus.publish(failure_event("planner", "Step 1", "oops"))
        bus.publish(task_completed_event("executor", "Step 1", True))
        assert bus.publish_count == 2


class TestEventStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> EventStore:
        return EventStore(tmp_path / "events.json")

    def test_append_and_count(self, store: EventStore) -> None:
        store.append(failure_event("planner", "Step 1", "oops"))
        assert store.count == 1

    def test_recent_returns_latest_n(self, store: EventStore) -> None:
        for i in range(5):
            store.append(task_completed_event("executor", f"Step {i}", True))
        recent = store.recent(limit=2)
        assert len(recent) == 2

    def test_recent_filters_by_type(self, store: EventStore) -> None:
        store.append(failure_event("planner", "Step 1", "oops"))
        store.append(task_completed_event("executor", "Step 2", True))
        failures = store.recent(event_type=EventType.FAILURE)
        assert len(failures) == 1

    def test_by_source(self, store: EventStore) -> None:
        store.append(failure_event("planner", "Step 1", "oops"))
        store.append(failure_event("executor", "Step 2", "oops2"))
        planner_events = store.by_source("planner")
        assert len(planner_events) == 1

    def test_count_by_type(self, store: EventStore) -> None:
        store.append(failure_event("planner", "Step 1", "oops"))
        store.append(failure_event("planner", "Step 2", "oops2"))
        store.append(task_completed_event("executor", "Step 3", True))
        counts = store.count_by_type()
        assert counts["failure"] == 2
        assert counts["task_completed"] == 1

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "events.json"
        s1 = EventStore(f)
        s1.append(failure_event("planner", "Step 1", "oops"))
        s2 = EventStore(f)
        assert s2.count == 1


# ===========================================================================
# Item 2 — Attention System
# ===========================================================================


class TestAttentionManager:
    @pytest.fixture
    def manager(self) -> AttentionManager:
        return AttentionManager()

    def test_score_formula_weights(self, manager: AttentionManager) -> None:
        candidate = AttentionCandidate(
            ref_id="c1", source="planner", content_summary="x",
            relevance=1.0, urgency=1.0, novelty=1.0, confidence=1.0,
        )
        scored = manager.score(candidate)
        assert scored.score == pytest.approx(1.0)

    def test_score_zero_when_all_zero(self, manager: AttentionManager) -> None:
        candidate = AttentionCandidate(ref_id="c1", source="x", content_summary="x", relevance=0, urgency=0, novelty=0, confidence=0)
        scored = manager.score(candidate)
        assert scored.score == 0.0

    def test_relevance_weighted_highest(self) -> None:
        manager = AttentionManager()
        high_relevance = AttentionCandidate(ref_id="a", source="x", content_summary="x", relevance=1.0, urgency=0, novelty=0, confidence=0)
        high_confidence = AttentionCandidate(ref_id="b", source="x", content_summary="x", relevance=0, urgency=0, novelty=0, confidence=1.0)
        assert manager.score(high_relevance).score > manager.score(high_confidence).score

    def test_score_many_sorted_descending(self, manager: AttentionManager) -> None:
        candidates = [
            AttentionCandidate(ref_id="low", source="x", content_summary="x", relevance=0.1, urgency=0.1, novelty=0.1, confidence=0.1),
            AttentionCandidate(ref_id="high", source="x", content_summary="x", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9),
        ]
        scored = manager.score_many(candidates)
        assert scored[0].candidate.ref_id == "high"

    def test_select_for_workspace_threshold_filters(self) -> None:
        manager = AttentionManager(entry_threshold=0.5)
        candidates = [
            AttentionCandidate(ref_id="low", source="x", content_summary="x", relevance=0.1, urgency=0.1, novelty=0.1, confidence=0.1),
            AttentionCandidate(ref_id="high", source="x", content_summary="x", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9),
        ]
        selected = manager.select_for_workspace(candidates)
        assert len(selected) == 1
        assert selected[0].candidate.ref_id == "high"

    def test_select_for_workspace_capacity_limit(self) -> None:
        manager = AttentionManager(entry_threshold=0.0, capacity=2)
        candidates = [
            AttentionCandidate(ref_id=f"c{i}", source="x", content_summary="x", relevance=0.5, urgency=0.5, novelty=0.5, confidence=0.5)
            for i in range(5)
        ]
        selected = manager.select_for_workspace(candidates)
        assert len(selected) == 2

    def test_novelty_for_unseen_item(self, manager: AttentionManager) -> None:
        assert manager.novelty_for("never_seen") == 1.0

    def test_novelty_decays_after_mark_seen(self, manager: AttentionManager) -> None:
        manager.mark_seen("ref1")
        assert manager.novelty_for("ref1") == 0.2

    def test_custom_weights(self) -> None:
        weights = AttentionWeights(relevance=1.0, urgency=0.0, novelty=0.0, confidence=0.0)
        manager = AttentionManager(weights=weights)
        candidate = AttentionCandidate(ref_id="a", source="x", content_summary="x", relevance=1.0, urgency=0, novelty=0, confidence=0)
        assert manager.score(candidate).score == pytest.approx(1.0)

    def test_threshold_and_capacity_properties(self) -> None:
        manager = AttentionManager(entry_threshold=0.6, capacity=3)
        assert manager.threshold == 0.6
        assert manager.capacity == 3


# ===========================================================================
# Item 3 — Broadcast Mechanism
# ===========================================================================


class TestBroadcastBus:
    @pytest.fixture
    def bus(self) -> BroadcastBus:
        return BroadcastBus()

    def test_register_subsystem_and_broadcast(self, bus: BroadcastBus) -> None:
        received = []
        bus.register_subsystem("reflection", EventType.FAILURE, lambda e: received.append(e))
        bus.broadcast(failure_event("planner", "Step 1", "oops"))
        assert len(received) == 1

    def test_broadcast_record_logged(self, bus: BroadcastBus) -> None:
        bus.register_subsystem("reflection", EventType.FAILURE, lambda e: None)
        record = bus.broadcast(failure_event("planner", "Step 1", "oops"))
        assert record.listener_count == 1
        assert bus.broadcast_count == 1

    def test_zero_listener_broadcast_recorded(self, bus: BroadcastBus) -> None:
        bus.broadcast(failure_event("planner", "Step 1", "oops"))
        assert len(bus.broadcasts_with_zero_listeners()) == 1

    def test_registered_subsystems_listed(self, bus: BroadcastBus) -> None:
        bus.register_subsystem("reflection", EventType.FAILURE, lambda e: None)
        bus.register_subsystem("self_model", EventType.TASK_COMPLETED, lambda e: None)
        assert set(bus.registered_subsystems()) == {"reflection", "self_model"}

    def test_unregister_subsystem(self, bus: BroadcastBus) -> None:
        bus.register_subsystem("reflection", EventType.FAILURE, lambda e: None)
        removed = bus.unregister_subsystem("reflection")
        assert removed == 1
        assert "reflection" not in bus.registered_subsystems()

    def test_mean_listener_count(self, bus: BroadcastBus) -> None:
        bus.register_subsystem("reflection", EventType.FAILURE, lambda e: None)
        bus.broadcast(failure_event("planner", "Step 1", "oops"))
        bus.broadcast(task_completed_event("executor", "Step 2", True))  # no listeners for this type
        assert bus.mean_listener_count() == pytest.approx(0.5)

    def test_multiple_subsystems_same_event_type(self, bus: BroadcastBus) -> None:
        received_a, received_b = [], []
        bus.register_subsystem("a", EventType.FAILURE, lambda e: received_a.append(e))
        bus.register_subsystem("b", EventType.FAILURE, lambda e: received_b.append(e))
        bus.broadcast(failure_event("planner", "Step 1", "oops"))
        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_recent_broadcasts(self, bus: BroadcastBus) -> None:
        for i in range(3):
            bus.broadcast(task_completed_event("executor", f"Step {i}", True))
        assert len(bus.recent_broadcasts(limit=2)) == 2


# ===========================================================================
# Workspace Memory
# ===========================================================================


class TestWorkspaceMemory:
    @pytest.fixture
    def wm(self) -> WorkspaceMemory:
        return WorkspaceMemory()

    def test_set_items_replaces_contents(self, wm: WorkspaceMemory) -> None:
        items = [WorkspaceItem(ref_id="a", source="x", content_summary="x", attention_score=0.9)]
        wm.set_items(items)
        assert wm.count == 1

    def test_set_items_sets_attention_focus_to_first(self, wm: WorkspaceMemory) -> None:
        items = [
            WorkspaceItem(ref_id="a", source="x", content_summary="x", attention_score=0.9),
            WorkspaceItem(ref_id="b", source="x", content_summary="x", attention_score=0.5),
        ]
        wm.set_items(items)
        assert wm.attention_focus == "a"

    def test_add_item_updates_focus_if_higher_score(self, wm: WorkspaceMemory) -> None:
        wm.add_item(WorkspaceItem(ref_id="a", source="x", content_summary="x", attention_score=0.5))
        wm.add_item(WorkspaceItem(ref_id="b", source="x", content_summary="x", attention_score=0.9))
        assert wm.attention_focus == "b"

    def test_remove_item(self, wm: WorkspaceMemory) -> None:
        wm.add_item(WorkspaceItem(ref_id="a", source="x", content_summary="x", attention_score=0.5))
        assert wm.remove_item("a") is True
        assert wm.count == 0

    def test_remove_nonexistent_returns_false(self, wm: WorkspaceMemory) -> None:
        assert wm.remove_item("ghost") is False

    def test_clear(self, wm: WorkspaceMemory) -> None:
        wm.add_item(WorkspaceItem(ref_id="a", source="x", content_summary="x", attention_score=0.5))
        wm.clear()
        assert wm.count == 0
        assert wm.attention_focus is None

    def test_active_goal(self, wm: WorkspaceMemory) -> None:
        wm.set_active_goal("fix the bug")
        assert wm.active_goal == "fix the bug"

    def test_items_from_source(self, wm: WorkspaceMemory) -> None:
        wm.add_item(WorkspaceItem(ref_id="a", source="planner", content_summary="x", attention_score=0.5))
        wm.add_item(WorkspaceItem(ref_id="b", source="memory", content_summary="x", attention_score=0.5))
        assert len(wm.items_from_source("planner")) == 1

    def test_to_dict(self, wm: WorkspaceMemory) -> None:
        wm.set_active_goal("goal")
        wm.add_item(WorkspaceItem(ref_id="a", source="x", content_summary="x", attention_score=0.5))
        d = wm.to_dict()
        assert d["active_goal"] == "goal"
        assert len(d["items"]) == 1


# ===========================================================================
# Item 1 — Global Workspace
# ===========================================================================


class TestGlobalWorkspace:
    @pytest.fixture
    def gw(self) -> GlobalWorkspace:
        return GlobalWorkspace()

    def test_submit_and_cycle(self, gw: GlobalWorkspace) -> None:
        gw.submit_candidate(AttentionCandidate(ref_id="a", source="planner", content_summary="important", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9))
        result = gw.run_cycle()
        assert len(result.entered) == 1

    def test_low_attention_rejected(self, gw: GlobalWorkspace) -> None:
        gw.submit_candidate(AttentionCandidate(ref_id="trivial", source="memory", content_summary="x", relevance=0.05, urgency=0.05, novelty=0.05, confidence=0.05))
        result = gw.run_cycle()
        assert len(result.entered) == 0
        assert result.rejected_count == 1

    def test_pending_cleared_after_cycle(self, gw: GlobalWorkspace) -> None:
        gw.submit_candidate(AttentionCandidate(ref_id="a", source="x", content_summary="x", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9))
        gw.run_cycle()
        assert gw.pending_count == 0

    def test_active_goal_set_on_cycle(self, gw: GlobalWorkspace) -> None:
        gw.run_cycle(active_goal="ship the feature")
        assert gw.memory.active_goal == "ship the feature"

    def test_broadcast_sent_for_entered_items(self, gw: GlobalWorkspace) -> None:
        received = []
        gw.register_subsystem("listener", EventType.WORKSPACE_BROADCAST, lambda e: received.append(e))
        gw.submit_candidate(AttentionCandidate(ref_id="a", source="x", content_summary="x", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9))
        gw.run_cycle()
        assert len(received) == 1

    def test_cycle_count_increments(self, gw: GlobalWorkspace) -> None:
        gw.run_cycle()
        gw.run_cycle()
        assert gw.cycle_count == 2

    def test_submit_many(self, gw: GlobalWorkspace) -> None:
        gw.submit_many([
            AttentionCandidate(ref_id="a", source="x", content_summary="x", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9),
            AttentionCandidate(ref_id="b", source="x", content_summary="x", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9),
        ])
        assert gw.pending_count == 2

    def test_workspace_accessors(self, gw: GlobalWorkspace) -> None:
        assert gw.memory is not None
        assert gw.attention is not None
        assert gw.broadcast_bus is not None

    def test_result_to_dict(self, gw: GlobalWorkspace) -> None:
        gw.submit_candidate(AttentionCandidate(ref_id="a", source="x", content_summary="x", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9))
        result = gw.run_cycle()
        d = result.to_dict()
        assert "entered" in d
        assert "broadcasts_sent" in d


# ===========================================================================
# Item 5 — Internal Specialists
# ===========================================================================


class TestMemorySpecialist:
    def test_no_lookup_fn_no_opinion(self) -> None:
        spec = MemorySpecialist(lookup_fn=None)
        opinion = spec.consult("topic")
        assert opinion.verdict == "no_opinion"

    def test_empty_results_no_opinion(self) -> None:
        spec = MemorySpecialist(lookup_fn=lambda t: [])
        opinion = spec.consult("topic")
        assert opinion.verdict == "no_opinion"

    def test_high_confidence_results_supports(self) -> None:
        class FakeBelief:
            confidence = 0.9
        spec = MemorySpecialist(lookup_fn=lambda t: [FakeBelief()])
        opinion = spec.consult("topic")
        assert opinion.verdict == "supports"

    def test_low_confidence_results_uncertain(self) -> None:
        class FakeBelief:
            confidence = 0.3
        spec = MemorySpecialist(lookup_fn=lambda t: [FakeBelief()])
        opinion = spec.consult("topic")
        assert opinion.verdict == "uncertain"


class TestPlanningSpecialist:
    def test_no_graph_no_opinion(self) -> None:
        spec = PlanningSpecialist(PlanQualityEvaluator())
        opinion = spec.consult("topic")
        assert opinion.verdict == "no_opinion"

    def test_good_plan_supports(self) -> None:
        spec = PlanningSpecialist(PlanQualityEvaluator())
        graph = TaskGraph(goal="test")
        graph.add_task(Task(title="Step 1"))
        critique = PlanCritic().critique(graph)
        opinion = spec.consult("topic", graph=graph, critique=critique)
        assert opinion.verdict == "supports"

    def test_bad_plan_opposes_or_uncertain(self) -> None:
        from planning.critic import CritiqueReport, PlanVerdict
        spec = PlanningSpecialist(PlanQualityEvaluator())
        graph = TaskGraph(goal="risky")
        graph.add_task(Task(title="Step 1"))
        critique = CritiqueReport(verdict=PlanVerdict.REJECTED, issues=[])
        opinion = spec.consult("topic", graph=graph, critique=critique)
        assert opinion.verdict in ("opposes", "uncertain")


class TestReflectionSpecialist:
    @pytest.fixture
    def fm(self, tmp_path: Path) -> FailureMemory:
        return FailureMemory(tmp_path / "fm.json")

    def test_no_failure_memory_no_opinion(self) -> None:
        spec = ReflectionSpecialist(failure_memory=None)
        assert spec.consult("Step 1").verdict == "no_opinion"

    def test_no_known_failure_supports(self, fm: FailureMemory) -> None:
        spec = ReflectionSpecialist(failure_memory=fm)
        opinion = spec.consult("Brand new task")
        assert opinion.verdict == "supports"

    def test_known_failure_opposes(self, fm: FailureMemory) -> None:
        fm.record("Risky task", "web_search", "timeout error")
        spec = ReflectionSpecialist(failure_memory=fm)
        opinion = spec.consult("Risky task")
        assert opinion.verdict == "opposes"


class TestVerificationSpecialist:
    def test_no_task_no_opinion(self) -> None:
        spec = VerificationSpecialist(VerificationEngine())
        opinion = spec.consult("topic")
        assert opinion.verdict == "no_opinion"

    def test_passing_result_supports(self) -> None:
        spec = VerificationSpecialist(VerificationEngine())
        task = Task(title="Step 1")
        result = ExecutionResult(task_id=task.task_id, tool_name="x", status=ExecutionStatus.SUCCESS, output="some real output text")
        opinion = spec.consult("topic", task=task, result=result)
        assert opinion.verdict == "supports"

    def test_failing_result_opposes(self) -> None:
        spec = VerificationSpecialist(VerificationEngine())
        task = Task(title="Step 1")
        result = ExecutionResult(task_id=task.task_id, tool_name="x", status=ExecutionStatus.SUCCESS, output="")
        opinion = spec.consult("topic", task=task, result=result)
        assert opinion.verdict == "opposes"


class TestSpecialistConsensus:
    def test_register_and_consult_all(self) -> None:
        class AlwaysSupports(BaseSpecialist):
            name = "always_supports"
            def consult(self, topic, **context):
                return SpecialistOpinion(specialist=self.name, verdict="supports", confidence=0.9)

        consensus = SpecialistConsensus([AlwaysSupports()])
        opinions = consensus.consult_all("topic")
        assert len(opinions) == 1

    def test_decide_majority_verdict(self) -> None:
        class Supports(BaseSpecialist):
            name = "s1"
            def consult(self, topic, **context):
                return SpecialistOpinion(specialist=self.name, verdict="supports", confidence=0.8)
        class AlsoSupports(BaseSpecialist):
            name = "s2"
            def consult(self, topic, **context):
                return SpecialistOpinion(specialist=self.name, verdict="supports", confidence=0.7)

        consensus = SpecialistConsensus([Supports(), AlsoSupports()])
        result = consensus.decide("topic")
        assert result.majority_verdict == "supports"
        assert result.agreement_ratio == 1.0

    def test_decide_contested(self) -> None:
        class Supports(BaseSpecialist):
            name = "s1"
            def consult(self, topic, **context):
                return SpecialistOpinion(specialist=self.name, verdict="supports", confidence=0.8)
        class Opposes(BaseSpecialist):
            name = "s2"
            def consult(self, topic, **context):
                return SpecialistOpinion(specialist=self.name, verdict="opposes", confidence=0.7)

        consensus = SpecialistConsensus([Supports(), Opposes()])
        result = consensus.decide("topic")
        assert result.is_contested

    def test_no_opinion_specialists_excluded_from_majority(self) -> None:
        class NoOpinion(BaseSpecialist):
            name = "s1"
            def consult(self, topic, **context):
                return SpecialistOpinion(specialist=self.name, verdict="no_opinion", confidence=0.0)
        class Supports(BaseSpecialist):
            name = "s2"
            def consult(self, topic, **context):
                return SpecialistOpinion(specialist=self.name, verdict="supports", confidence=0.8)

        consensus = SpecialistConsensus([NoOpinion(), Supports()])
        result = consensus.decide("topic")
        assert result.majority_verdict == "supports"
        assert result.agreement_ratio == 1.0  # only counts opinionated specialists
        assert len(result.opinions) == 2  # but both preserved in full opinion list

    def test_all_no_opinion(self) -> None:
        class NoOpinion(BaseSpecialist):
            name = "s1"
            def consult(self, topic, **context):
                return SpecialistOpinion(specialist=self.name, verdict="no_opinion", confidence=0.0)

        consensus = SpecialistConsensus([NoOpinion()])
        result = consensus.decide("topic")
        assert result.majority_verdict == "no_opinion"

    def test_registered_names(self) -> None:
        class S(BaseSpecialist):
            name = "my_specialist"
            def consult(self, topic, **context):
                return SpecialistOpinion(specialist=self.name, verdict="no_opinion", confidence=0.0)
        consensus = SpecialistConsensus()
        consensus.register(S())
        assert consensus.registered_names() == ["my_specialist"]


# ===========================================================================
# Item 6 — Active Attention Retrieval
# ===========================================================================


class TestActiveAttentionRetriever:
    def _mem(self, id_, input_, output_):
        return MemoryEntry(id=id_, input=input_, output=output_, timestamp=datetime.now(timezone.utc), importance=0.5)

    def test_no_context_matches_base_ordering(self) -> None:
        retriever = ActiveAttentionRetriever()
        memories = [self._mem(1, "a", "a"), self._mem(2, "b", "b")]
        result_with_none = retriever.retrieve(memories, "query")
        from core.memory_retriever import MemoryRetriever
        base_result = MemoryRetriever().retrieve(memories, "query")
        assert [m.id for m in result_with_none] == [m.id for m in base_result]

    def test_empty_memories_returns_empty(self) -> None:
        retriever = ActiveAttentionRetriever()
        assert retriever.retrieve([], "query") == []

    def test_workspace_alignment_reorders_results(self) -> None:
        retriever = ActiveAttentionRetriever()
        mem_unrelated = self._mem(1, "something about the weather today", "It is sunny outside.")
        mem_aligned = self._mem(2, "machine learning pipeline training optimization", "We tuned hyperparameters.")
        mem_other = self._mem(3, "a random unrelated note", "Just a note.")

        ws = WorkspaceMemory()
        ws.set_active_goal("machine learning pipeline training optimization")
        ws.add_item(WorkspaceItem(ref_id="w1", source="planner", content_summary="machine learning pipeline training optimization", attention_score=0.9))

        result = retriever.retrieve(
            [mem_other, mem_unrelated, mem_aligned], "general query",
            current_workspace=ws, active_goal=ws.active_goal,
        )
        assert result[0].id == 2  # the ML-aligned memory should now rank first

    def test_active_goal_alone_influences_ranking(self) -> None:
        retriever = ActiveAttentionRetriever()
        mem_aligned = self._mem(1, "budget planning for next quarter", "Reviewed budget numbers.")
        mem_other = self._mem(2, "completely different topic note", "Nothing related.")
        result = retriever.retrieve([mem_other, mem_aligned], "general query", active_goal="budget planning for next quarter")
        assert result[0].id == 1

    def test_custom_weights(self) -> None:
        weights = AttentionRetrievalWeights(base_retrieval=1.0, goal_alignment=0.0, workspace_alignment=0.0)
        retriever = ActiveAttentionRetriever(weights=weights)
        memories = [self._mem(1, "a", "a"), self._mem(2, "b", "b")]
        from core.memory_retriever import MemoryRetriever
        base_result = MemoryRetriever().retrieve(memories, "query")
        result = retriever.retrieve(memories, "query", active_goal="totally unrelated goal text xyz")
        assert [m.id for m in result] == [m.id for m in base_result]


# ===========================================================================
# Item 7 — Workspace Snapshotting
# ===========================================================================


class TestWorkspaceSnapshot:
    @pytest.fixture
    def store(self, tmp_path: Path) -> WorkspaceSnapshotStore:
        return WorkspaceSnapshotStore(tmp_path / "snapshots.json")

    @pytest.fixture
    def workspace_with_item(self) -> GlobalWorkspace:
        gw = GlobalWorkspace()
        gw.submit_candidate(AttentionCandidate(ref_id="a", source="planner", content_summary="x", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9))
        gw.run_cycle(active_goal="ship the feature")
        return gw

    def test_capture_basic(self, store: WorkspaceSnapshotStore, workspace_with_item: GlobalWorkspace) -> None:
        snap = store.capture(workspace_with_item)
        assert snap.active_goal == "ship the feature"
        assert snap.attention_focus == "a"

    def test_capture_with_full_context(self, store: WorkspaceSnapshotStore, workspace_with_item: GlobalWorkspace) -> None:
        snap = store.capture(
            workspace_with_item, important_beliefs=["b1", "b2"],
            current_plan_graph_id="g1", current_plan_summary="2-step plan",
            current_failures=["timeout"],
        )
        assert snap.important_beliefs == ["b1", "b2"]
        assert snap.current_plan_graph_id == "g1"
        assert snap.current_failures == ["timeout"]

    def test_restore_sets_active_goal(self, store: WorkspaceSnapshotStore, workspace_with_item: GlobalWorkspace) -> None:
        snap = store.capture(workspace_with_item)
        fresh_workspace = GlobalWorkspace()
        restored = store.restore(snap.snapshot_id, fresh_workspace)
        assert restored is not None
        assert fresh_workspace.memory.active_goal == "ship the feature"

    def test_restore_unknown_id_returns_none(self, store: WorkspaceSnapshotStore, workspace_with_item: GlobalWorkspace) -> None:
        assert store.restore("ghost", workspace_with_item) is None

    def test_get(self, store: WorkspaceSnapshotStore, workspace_with_item: GlobalWorkspace) -> None:
        snap = store.capture(workspace_with_item)
        assert store.get(snap.snapshot_id) is not None

    def test_all_snapshots(self, store: WorkspaceSnapshotStore, workspace_with_item: GlobalWorkspace) -> None:
        store.capture(workspace_with_item)
        store.capture(workspace_with_item)
        assert len(store.all_snapshots()) == 2

    def test_most_recent(self, store: WorkspaceSnapshotStore, workspace_with_item: GlobalWorkspace) -> None:
        store.capture(workspace_with_item)
        latest = store.capture(workspace_with_item)
        assert store.most_recent().snapshot_id == latest.snapshot_id

    def test_delete(self, store: WorkspaceSnapshotStore, workspace_with_item: GlobalWorkspace) -> None:
        snap = store.capture(workspace_with_item)
        assert store.delete(snap.snapshot_id) is True
        assert store.get(snap.snapshot_id) is None

    def test_persistence_round_trip(self, tmp_path: Path, workspace_with_item: GlobalWorkspace) -> None:
        f = tmp_path / "snapshots.json"
        store1 = WorkspaceSnapshotStore(f)
        snap = store1.capture(workspace_with_item)
        store2 = WorkspaceSnapshotStore(f)
        assert store2.get(snap.snapshot_id) is not None

    def test_count_property(self, store: WorkspaceSnapshotStore, workspace_with_item: GlobalWorkspace) -> None:
        store.capture(workspace_with_item)
        assert store.count == 1


# ===========================================================================
# Item 8 — Inner Dialogue
# ===========================================================================


class TestInnerDialogue:
    def test_run_collects_turns_in_order(self) -> None:
        dialogue = InnerDialogue()
        dialogue.register_voice("First", lambda t: "first remark")
        dialogue.register_voice("Second", lambda t: "second remark")
        transcript = dialogue.run("topic")
        assert [t.voice for t in transcript.turns] == ["First", "Second"]

    def test_none_remark_skipped(self) -> None:
        dialogue = InnerDialogue()
        dialogue.register_voice("Silent", lambda t: None)
        dialogue.register_voice("Vocal", lambda t: "something")
        transcript = dialogue.run("topic")
        assert len(transcript.turns) == 1
        assert transcript.turns[0].voice == "Vocal"

    def test_voice_exception_treated_as_no_remark(self) -> None:
        def broken(t):
            raise ValueError("boom")
        dialogue = InnerDialogue()
        dialogue.register_voice("Broken", broken)
        transcript = dialogue.run("topic")
        assert len(transcript.turns) == 0

    def test_as_text_format(self) -> None:
        dialogue = InnerDialogue()
        dialogue.register_voice("Planner", lambda t: "Need strategy.")
        transcript = dialogue.run("topic")
        assert "Planner:" in transcript.as_text()
        assert "Need strategy." in transcript.as_text()

    def test_planner_voice_repeated_failure(self) -> None:
        sm = StrategyManager()
        sm.record_failure("ref1")
        sm.record_failure("ref1")
        voice = planner_voice(sm, "ref1")
        remark = voice("topic")
        assert "Need strategy" in remark

    def test_planner_voice_no_failures(self) -> None:
        sm = StrategyManager()
        voice = planner_voice(sm, "ref2")
        remark = voice("topic")
        assert "Need strategy" not in remark

    def test_critic_voice_none_score(self) -> None:
        voice = critic_voice(None)
        assert voice("topic") is None

    def test_critic_voice_low_confidence(self) -> None:
        from planning.plan_evaluator import PlanQualityScore
        score = PlanQualityScore(graph_id="g1", complexity=0.1, risk=0.1, confidence=0.3, dependency_density=0.0, expected_success=0.2)
        voice = critic_voice(score)
        assert "low" in voice("topic").lower()

    def test_self_model_voice(self, tmp_path: Path) -> None:
        store = SelfModelStore(tmp_path / "sm.json")
        store.set_capability("math", 0.9)
        voice = self_model_voice(store, "math")
        remark = voice("topic")
        assert "high" in remark.lower()

    def test_reflection_voice_no_failures(self, tmp_path: Path) -> None:
        fm = FailureMemory(tmp_path / "fm.json")
        voice = reflection_voice(fm)
        assert voice("topic") is None

    def test_reflection_voice_with_failures(self, tmp_path: Path) -> None:
        fm = FailureMemory(tmp_path / "fm.json")
        fm.record("Step 1", "web_search", "timeout error")
        voice = reflection_voice(fm)
        remark = voice("topic")
        assert "timeout error" in remark

    def test_spec_literal_example_all_four_voices(self, tmp_path: Path) -> None:
        sm = StrategyManager()
        sm.record_failure("ref1")
        sm.record_failure("ref1")
        store = SelfModelStore(tmp_path / "sm.json")
        store.set_capability("math", 0.9)
        fm = FailureMemory(tmp_path / "fm.json")
        fm.record("Step 1", "web_search", "previous failure X")

        dialogue = InnerDialogue()
        dialogue.register_voice("Planner", planner_voice(sm, "ref1"))
        dialogue.register_voice("Critic", critic_voice(None))
        dialogue.register_voice("Self Model", self_model_voice(store, "math"))
        dialogue.register_voice("Reflection", reflection_voice(fm))

        transcript = dialogue.run("how to proceed")
        voices = [t.voice for t in transcript.turns]
        assert voices == ["Planner", "Self Model", "Reflection"]  # Critic skipped (None score)


# ===========================================================================
# Item 9 — Cognitive Coordination Metrics
# ===========================================================================


class TestWorkspaceMetrics:
    def test_cycle_stats_empty(self) -> None:
        stats = WorkspaceMetrics.cycle_stats([])
        assert stats.total_cycles == 0

    def test_cycle_stats_aggregates(self) -> None:
        gw = GlobalWorkspace()
        gw.submit_candidate(AttentionCandidate(ref_id="a", source="x", content_summary="x", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9))
        gw.submit_candidate(AttentionCandidate(ref_id="b", source="x", content_summary="x", relevance=0.05, urgency=0.05, novelty=0.05, confidence=0.05))
        result1 = gw.run_cycle()
        stats = WorkspaceMetrics.cycle_stats([result1])
        assert stats.total_entered == 1
        assert stats.total_rejected == 1
        assert stats.entry_rate == pytest.approx(0.5)

    def test_broadcast_quality_no_broadcasts(self) -> None:
        bus = BroadcastBus()
        quality = WorkspaceMetrics.broadcast_quality(bus)
        assert quality["total_broadcasts"] == 0

    def test_broadcast_quality_with_listeners(self) -> None:
        bus = BroadcastBus()
        bus.register_subsystem("reflection", EventType.FAILURE, lambda e: None)
        bus.broadcast(failure_event("planner", "Step 1", "oops"))
        quality = WorkspaceMetrics.broadcast_quality(bus)
        assert quality["zero_listener_rate"] == 0.0

    def test_utilization(self) -> None:
        assert WorkspaceMetrics.utilization(5, 10) == 0.5
        assert WorkspaceMetrics.utilization(15, 10) == 1.0
        assert WorkspaceMetrics.utilization(5, 0) == 0.0


class TestAttentionMetrics:
    def test_selection_accuracy_perfect(self) -> None:
        manager = AttentionManager()
        candidates = [
            AttentionCandidate(ref_id="a", source="x", content_summary="x", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9),
            AttentionCandidate(ref_id="b", source="x", content_summary="x", relevance=0.1, urgency=0.1, novelty=0.1, confidence=0.1),
        ]
        scored = manager.score_many(candidates)
        gt = [AttentionGroundTruthCase("a", True), AttentionGroundTruthCase("b", False)]
        assert AttentionMetrics.selection_accuracy(scored, gt, threshold=0.5) == 1.0

    def test_selection_accuracy_empty_ground_truth(self) -> None:
        assert AttentionMetrics.selection_accuracy([], [], threshold=0.5) == 1.0

    def test_mean_score(self) -> None:
        manager = AttentionManager()
        candidates = [AttentionCandidate(ref_id="a", source="x", content_summary="x", relevance=1.0, urgency=1.0, novelty=1.0, confidence=1.0)]
        scored = manager.score_many(candidates)
        assert AttentionMetrics.mean_score(scored) == pytest.approx(1.0)

    def test_score_variance_zero_when_uniform(self) -> None:
        manager = AttentionManager()
        candidates = [
            AttentionCandidate(ref_id="a", source="x", content_summary="x", relevance=0.5, urgency=0.5, novelty=0.5, confidence=0.5),
            AttentionCandidate(ref_id="b", source="x", content_summary="x", relevance=0.5, urgency=0.5, novelty=0.5, confidence=0.5),
        ]
        scored = manager.score_many(candidates)
        assert AttentionMetrics.score_variance(scored) == pytest.approx(0.0)

    def test_false_admission_rate(self) -> None:
        manager = AttentionManager()
        candidates = [AttentionCandidate(ref_id="bad", source="x", content_summary="x", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9)]
        scored = manager.score_many(candidates)
        gt = [AttentionGroundTruthCase("bad", False)]
        assert AttentionMetrics.false_admission_rate(scored, gt, threshold=0.5) == 1.0

    def test_false_rejection_rate(self) -> None:
        manager = AttentionManager()
        candidates = [AttentionCandidate(ref_id="good", source="x", content_summary="x", relevance=0.1, urgency=0.1, novelty=0.1, confidence=0.1)]
        scored = manager.score_many(candidates)
        gt = [AttentionGroundTruthCase("good", True)]
        assert AttentionMetrics.false_rejection_rate(scored, gt, threshold=0.5) == 1.0


class TestCoordinationMetrics:
    @pytest.fixture
    def metrics(self) -> CoordinationMetrics:
        return CoordinationMetrics()

    def test_consensus_convergence_rate(self, metrics: CoordinationMetrics) -> None:
        converged = ConsensusResult(topic="t1", opinions=[], majority_verdict="supports", agreement_ratio=1.0, mean_confidence=0.8)
        contested = ConsensusResult(topic="t2", opinions=[], majority_verdict="supports", agreement_ratio=0.5, mean_confidence=0.5)
        rate = metrics.consensus_convergence_rate([converged, contested])
        assert rate == 0.5

    def test_consensus_convergence_empty(self, metrics: CoordinationMetrics) -> None:
        assert metrics.consensus_convergence_rate([]) == 1.0

    def test_mean_agreement_ratio(self, metrics: CoordinationMetrics) -> None:
        r1 = ConsensusResult(topic="t1", opinions=[], majority_verdict="supports", agreement_ratio=1.0, mean_confidence=0.8)
        r2 = ConsensusResult(topic="t2", opinions=[], majority_verdict="supports", agreement_ratio=0.5, mean_confidence=0.5)
        assert metrics.mean_agreement_ratio([r1, r2]) == pytest.approx(0.75)

    def test_no_opinion_rate(self, metrics: CoordinationMetrics) -> None:
        opinions = [
            SpecialistOpinion(specialist="a", verdict="no_opinion", confidence=0.0),
            SpecialistOpinion(specialist="b", verdict="supports", confidence=0.8),
        ]
        r1 = ConsensusResult(topic="t1", opinions=opinions, majority_verdict="supports", agreement_ratio=1.0, mean_confidence=0.8)
        assert metrics.no_opinion_rate([r1]) == 0.5

    def test_subsystem_participation_rate_none_registered(self, metrics: CoordinationMetrics) -> None:
        bus = BroadcastBus()
        assert metrics.subsystem_participation_rate(bus) == 0.0

    def test_subsystem_participation_rate_full(self, metrics: CoordinationMetrics) -> None:
        bus = BroadcastBus()
        bus.register_subsystem("reflection", EventType.FAILURE, lambda e: None)
        bus.broadcast(failure_event("planner", "Step 1", "oops"))
        assert metrics.subsystem_participation_rate(bus) == 1.0

    def test_isolation_rate(self, metrics: CoordinationMetrics) -> None:
        bus = BroadcastBus()
        bus.broadcast(failure_event("planner", "Step 1", "oops"))
        assert metrics.isolation_rate(bus) == 1.0

    def test_run_coordination_bench(self, metrics: CoordinationMetrics) -> None:
        r1 = ConsensusResult(topic="t1", opinions=[], majority_verdict="supports", agreement_ratio=1.0, mean_confidence=0.8)
        bus = BroadcastBus()
        results = metrics.run_coordination_bench([r1], bus)
        assert "consensus_convergence_rate" in results
        assert "subsystem_participation_rate" in results

    def test_extends_metacognition_metrics(self, metrics: CoordinationMetrics) -> None:
        from evaluation.metacognition_metrics import MetacognitionMetrics
        assert isinstance(metrics, MetacognitionMetrics)


# ===========================================================================
# Integration — BlixContext wiring + API
# ===========================================================================


class _FakeLLM:
    def model_name(self) -> str:
        return "fake-0.3.9"

    def generate(self, prompt: str) -> str:
        return "Fake reply."


@pytest.fixture(scope="module")
def tmp_memory_v9(tmp_path_factory):
    return tmp_path_factory.mktemp("memory_v9")


@pytest.fixture(scope="module")
def ctx_v9(tmp_memory_v9):
    from config import settings as _settings
    _settings.settings.memory.conversations_file = tmp_memory_v9 / "conversations.json"
    _settings.settings.memory.profile_file = tmp_memory_v9 / "profile.json"
    _settings.settings.memory.learning_state_file = tmp_memory_v9 / "learning_state.json"
    _settings.settings.embed.embeddings_file = tmp_memory_v9 / "embeddings.npy"
    _settings.settings.embed.embedding_ids_file = tmp_memory_v9 / "embedding_ids.json"

    from api.context import BlixContext
    ctx = BlixContext(tmp_memory_v9)
    ctx.llm = _FakeLLM()
    ctx.agent._llm = _FakeLLM()
    return ctx


@pytest.fixture(scope="module")
def client_v9(ctx_v9) -> Generator[TestClient, None, None]:
    from api.deps import set_context
    from api.routers.workspace import router as workspace_router

    app = FastAPI(title="Blix Test v0.3.9")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(workspace_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    set_context(ctx_v9)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestBlixContextV039Wiring:
    def test_v039_components_present(self, ctx_v9) -> None:
        assert ctx_v9.event_bus is not None
        assert ctx_v9.event_store is not None
        assert ctx_v9.attention_manager is not None
        assert ctx_v9.workspace_memory is not None
        assert ctx_v9.broadcast_bus is not None
        assert ctx_v9.global_workspace is not None
        assert ctx_v9.workspace_snapshots is not None
        assert ctx_v9.active_attention_retriever is not None
        assert ctx_v9.memory_specialist is not None
        assert ctx_v9.planning_specialist is not None
        assert ctx_v9.reflection_specialist is not None
        assert ctx_v9.verification_specialist is not None
        assert ctx_v9.specialist_consensus is not None
        assert ctx_v9.inner_dialogue is not None
        assert ctx_v9.workspace_metrics is not None
        assert ctx_v9.attention_metrics is not None
        assert ctx_v9.coordination_metrics is not None

    def test_dashboard_stats_includes_v039_metrics(self, ctx_v9) -> None:
        stats = ctx_v9.dashboard_stats()
        assert "workspace_cycle_count" in stats
        assert "workspace_snapshots_stored" in stats
        assert "cognitive_events_logged" in stats
        assert "broadcasts_sent" in stats

    def test_end_to_end_workspace_cycle_via_context(self, ctx_v9) -> None:
        ctx_v9.global_workspace.submit_candidate(
            AttentionCandidate(ref_id="integration1", source="planner", content_summary="important integration event", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9)
        )
        result = ctx_v9.global_workspace.run_cycle(active_goal="integration test goal")
        assert len(result.entered) == 1
        assert ctx_v9.global_workspace.memory.active_goal == "integration test goal"

    def test_specialist_consensus_via_context(self, ctx_v9) -> None:
        graph = TaskGraph(goal="integration test")
        graph.add_task(Task(title="Step 1"))
        result = ctx_v9.specialist_consensus.decide("Step 1", graph=graph)
        assert result.topic == "Step 1"

    def test_inner_dialogue_via_context(self, ctx_v9) -> None:
        transcript = ctx_v9.inner_dialogue.run("integration topic")
        assert transcript.topic == "integration topic"

    def test_event_logged_to_store_via_broadcast(self, ctx_v9) -> None:
        initial_count = ctx_v9.event_store.count
        ctx_v9.global_workspace.submit_candidate(
            AttentionCandidate(ref_id="evt_test", source="planner", content_summary="event log test", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9)
        )
        ctx_v9.global_workspace.run_cycle()
        assert ctx_v9.event_store.count > initial_count


# ===========================================================================
# API — /workspace endpoints
# ===========================================================================


class TestWorkspaceAPI:
    def test_get_workspace_state(self, client_v9: TestClient) -> None:
        r = client_v9.get("/workspace/state")
        assert r.status_code == 200
        assert "items" in r.json()

    def test_submit_candidate(self, client_v9: TestClient) -> None:
        r = client_v9.post("/workspace/submit", json={
            "ref_id": "api_candidate_1", "source": "api_test", "content_summary": "test candidate",
            "relevance": 0.8, "urgency": 0.7, "novelty": 0.6, "confidence": 0.7,
        })
        assert r.status_code == 200
        assert r.json()["submitted"] is True

    def test_run_cycle(self, client_v9: TestClient) -> None:
        client_v9.post("/workspace/submit", json={
            "ref_id": "api_candidate_2", "source": "api_test", "content_summary": "cycle test",
            "relevance": 0.9, "urgency": 0.9, "novelty": 0.9, "confidence": 0.9,
        })
        r = client_v9.post("/workspace/cycle", json={"active_goal": "api test goal"})
        assert r.status_code == 200
        data = r.json()
        assert "entered" in data

    def test_capture_and_list_snapshots(self, client_v9: TestClient) -> None:
        r = client_v9.post("/workspace/snapshots", json={
            "important_beliefs": ["b1"], "current_plan_summary": "test plan",
        })
        assert r.status_code == 200
        snapshot_id = r.json()["snapshot_id"]

        r2 = client_v9.get("/workspace/snapshots")
        assert r2.status_code == 200
        assert any(s["snapshot_id"] == snapshot_id for s in r2.json()["snapshots"])

    def test_restore_snapshot(self, client_v9: TestClient) -> None:
        r = client_v9.post("/workspace/snapshots", json={})
        snapshot_id = r.json()["snapshot_id"]
        r2 = client_v9.post(f"/workspace/snapshots/{snapshot_id}/restore")
        assert r2.status_code == 200
        assert r2.json()["restored"] is True

    def test_restore_unknown_snapshot_404(self, client_v9: TestClient) -> None:
        r = client_v9.post("/workspace/snapshots/ghost_id/restore")
        assert r.status_code == 404

    def test_consult_specialists(self, client_v9: TestClient) -> None:
        r = client_v9.post("/workspace/specialists/consult", json={"topic": "test topic for specialists"})
        assert r.status_code == 200
        data = r.json()
        assert "majority_verdict" in data
        assert "opinions" in data

    def test_consult_specialists_validates_topic(self, client_v9: TestClient) -> None:
        r = client_v9.post("/workspace/specialists/consult", json={"topic": ""})
        assert r.status_code == 422

    def test_run_inner_dialogue(self, client_v9: TestClient) -> None:
        r = client_v9.post("/workspace/inner-dialogue", json={"topic": "how should we proceed"})
        assert r.status_code == 200
        data = r.json()
        assert "turns" in data

    def test_health_check_still_works(self, client_v9: TestClient) -> None:
        r = client_v9.get("/health")
        assert r.status_code == 200
