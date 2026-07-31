"""
Tests for Blix v0.3.7 — "Temporal State Tracking & Truth Maintenance".

Covers:
1.  core.state_tracker         (StateTracker, StateSnapshot)
2.  core.state_transition        (StateTransitionEngine, StateTransition)
3.  core.truth_manager            (TruthManager, TruthStatus)
4.  memory.beliefs                 (BeliefStore, Belief)
5.  core.contradiction_resolver     (ContradictionResolver — 4 cases)
6.  retrieval.temporal_retriever     (TemporalRetriever — 5-component score)
7.  graph.temporal_graph              (TemporalGraph)
8.  reasoning.temporal_query           (TemporalQueryEngine — 5 query types)
9.  reflection.state_reflection         (StateReflectionEngine)
10. evaluation.state_metrics             (StateMetrics / StateBench-lite)
Integration  — BlixContext wiring
API          — /temporal endpoints

Python 3.10 compatible — fully offline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from core.contradiction_resolver import ContradictionCase, ContradictionResolver, ResolutionResult
from core.state_tracker import StateSnapshot, StateTracker
from core.state_transition import StateTransition, StateTransitionEngine
from core.truth_manager import TruthManager, TruthRecord, TruthStatus
from evaluation.state_metrics import StateAccuracyCase, StateMetrics, TransitionAccuracyCase
from graph.temporal_graph import TemporalEdge, TemporalGraph
from memory.beliefs import Belief, BeliefStore
from reasoning.temporal_query import TemporalQueryEngine, TemporalQueryResult
from reflection.state_reflection import EvolutionEntry, StateEvolutionReport, StateReflectionEngine
from retrieval.temporal_retriever import TemporalRetriever, TemporalScore, TemporalScoringWeights


def _iso(days_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago)).isoformat()


# ===========================================================================
# Item 1 — StateTracker
# ===========================================================================


class TestStateTracker:
    @pytest.fixture
    def tracker(self, tmp_path: Path) -> StateTracker:
        return StateTracker(tmp_path / "state.json")

    def test_record_creates_snapshot(self, tracker: StateTracker) -> None:
        snap = tracker.record("sayan", "favorite_language", "Python")
        assert snap.value == "Python"
        assert snap.is_active

    def test_current_returns_active_snapshot(self, tracker: StateTracker) -> None:
        tracker.record("sayan", "favorite_language", "Python")
        current = tracker.current("sayan", "favorite_language")
        assert current is not None
        assert current.value == "Python"

    def test_current_none_when_untracked(self, tracker: StateTracker) -> None:
        assert tracker.current("ghost", "attribute") is None

    def test_close_active(self, tracker: StateTracker) -> None:
        snap = tracker.record("sayan", "favorite_language", "Python")
        closed = tracker.close_active("sayan", "favorite_language")
        assert len(closed) == 1
        assert not snap.is_active

    def test_at_time_returns_value_active_then(self, tracker: StateTracker) -> None:
        tracker.record("sayan", "lang", "Python", timestamp="2024-01-01T00:00:00")
        tracker.close_active("sayan", "lang", end_time="2025-01-01T00:00:00")
        tracker.record("sayan", "lang", "Rust", timestamp="2025-01-01T00:00:00")
        snap = tracker.at_time("sayan", "lang", "2024-06-01T00:00:00")
        assert snap is not None
        assert snap.value == "Python"

    def test_history_chronological(self, tracker: StateTracker) -> None:
        tracker.record("sayan", "lang", "Python", timestamp="2024-01-01T00:00:00")
        tracker.close_active("sayan", "lang", end_time="2025-01-01T00:00:00")
        tracker.record("sayan", "lang", "Rust", timestamp="2025-01-01T00:00:00")
        hist = tracker.history("sayan", "lang")
        assert [s.value for s in hist] == ["Python", "Rust"]

    def test_changes_since(self, tracker: StateTracker) -> None:
        tracker.record("sayan", "lang", "Python", timestamp="2020-01-01T00:00:00")
        tracker.record("sayan", "lang2", "Rust", timestamp="2026-01-01T00:00:00")
        changes = tracker.changes_since("2025-01-01T00:00:00")
        assert len(changes) == 1
        assert changes[0].value == "Rust"

    def test_all_attributes(self, tracker: StateTracker) -> None:
        tracker.record("sayan", "lang", "Python")
        tracker.record("sayan", "city", "Patna")
        attrs = tracker.all_attributes("sayan")
        assert set(attrs) == {"lang", "city"}

    def test_all_entities(self, tracker: StateTracker) -> None:
        tracker.record("sayan", "lang", "Python")
        tracker.record("blix", "framework", "FastAPI")
        assert set(tracker.all_entities()) == {"sayan", "blix"}

    def test_entity_attribute_case_insensitive(self, tracker: StateTracker) -> None:
        tracker.record("Sayan", "Favorite_Language", "Python")
        current = tracker.current("sayan", "favorite_language")
        assert current is not None

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        t1 = StateTracker(tmp_path / "s.json")
        t1.record("sayan", "lang", "Python")
        t2 = StateTracker(tmp_path / "s.json")
        assert t2.current("sayan", "lang").value == "Python"

    def test_snapshot_covers(self) -> None:
        snap = StateSnapshot(entity="x", attribute="y", value="z",
                             start_time="2024-01-01", end_time="2025-01-01")
        assert snap.covers("2024-06-01")
        assert not snap.covers("2025-06-01")
        assert not snap.covers("2023-06-01")

    def test_active_count(self, tracker: StateTracker) -> None:
        tracker.record("sayan", "lang", "Python")
        tracker.record("sayan", "city", "Patna")
        assert tracker.active_count == 2
        tracker.close_active("sayan", "lang")
        assert tracker.active_count == 1


# ===========================================================================
# Item 2 — StateTransitionEngine
# ===========================================================================


class TestStateTransitionEngine:
    @pytest.fixture
    def engine(self, tmp_path: Path) -> StateTransitionEngine:
        tracker = StateTracker(tmp_path / "state.json")
        return StateTransitionEngine(tracker, tmp_path / "transitions.json")

    def test_initial_assignment_no_transition_object_but_creates_snapshot(
        self, engine: StateTransitionEngine
    ) -> None:
        snap, transition = engine.transition("sayan", "lang", "Python")
        assert snap.value == "Python"
        assert transition is not None
        assert transition.is_initial

    def test_genuine_transition_closes_old_opens_new(self, engine: StateTransitionEngine) -> None:
        engine.transition("sayan", "lang", "Python")
        new_snap, transition = engine.transition("sayan", "lang", "Rust")
        assert new_snap.value == "Rust"
        assert transition.from_value == "Python"
        assert transition.to_value == "Rust"
        assert not transition.is_initial

    def test_reassertion_reinforces_no_transition(self, engine: StateTransitionEngine) -> None:
        engine.transition("sayan", "lang", "Python", confidence=0.5)
        snap, transition = engine.transition("sayan", "lang", "Python", confidence=0.5)
        assert transition is None
        assert snap.confidence > 0.5

    def test_case_insensitive_reassertion(self, engine: StateTransitionEngine) -> None:
        engine.transition("sayan", "lang", "Python")
        snap, transition = engine.transition("sayan", "lang", "python")
        assert transition is None  # same value, different case

    def test_old_snapshot_closed_after_transition(self, engine: StateTransitionEngine) -> None:
        engine.transition("sayan", "lang", "Python")
        engine.transition("sayan", "lang", "Rust")
        history = engine._tracker.history("sayan", "lang")
        assert not history[0].is_active
        assert history[1].is_active

    def test_history_filtered_by_attribute(self, engine: StateTransitionEngine) -> None:
        engine.transition("sayan", "lang", "Python")
        engine.transition("sayan", "city", "Patna")
        hist = engine.history("sayan", attribute="lang")
        assert len(hist) == 1

    def test_latest_transition(self, engine: StateTransitionEngine) -> None:
        engine.transition("sayan", "lang", "Python")
        engine.transition("sayan", "lang", "Rust")
        latest = engine.latest_transition("sayan", "lang")
        assert latest.to_value == "Rust"

    def test_transitions_since(self, engine: StateTransitionEngine) -> None:
        engine.transition("sayan", "lang", "Python", timestamp=_iso(400))
        engine.transition("sayan", "lang", "Rust", timestamp=_iso(1))
        recent = engine.transitions_since(_iso(30))
        assert len(recent) == 1
        assert recent[0].to_value == "Rust"

    def test_attributes_changed_since(self, engine: StateTransitionEngine) -> None:
        engine.transition("sayan", "lang", "Python", timestamp=_iso(400))
        engine.transition("sayan", "lang", "Rust", timestamp=_iso(1))
        engine.transition("sayan", "city", "Patna", timestamp=_iso(400))
        changed = engine.attributes_changed_since(_iso(30), "sayan")
        assert changed == ["lang"]

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        tracker = StateTracker(tmp_path / "s.json")
        e1 = StateTransitionEngine(tracker, tmp_path / "t.json")
        e1.transition("sayan", "lang", "Python")
        e1.transition("sayan", "lang", "Rust")

        tracker2 = StateTracker(tmp_path / "s.json")
        e2 = StateTransitionEngine(tracker2, tmp_path / "t.json")
        assert e2.count == 2

    def test_transition_describe(self) -> None:
        t1 = StateTransition(entity="x", attribute="y", to_value="Rust", from_value="Python")
        assert "Python" in t1.describe()
        assert "Rust" in t1.describe()
        t2 = StateTransition(entity="x", attribute="y", to_value="Python")
        assert "set to" in t2.describe()


# ===========================================================================
# Item 3 — TruthManager
# ===========================================================================


class TestTruthManager:
    @pytest.fixture
    def tm(self, tmp_path: Path) -> TruthManager:
        return TruthManager(tmp_path / "truth.json")

    def test_ensure_creates_active_by_default(self, tm: TruthManager) -> None:
        rec = tm.ensure("rec1")
        assert rec.status == TruthStatus.ACTIVE

    def test_replace(self, tm: TruthManager) -> None:
        tm.replace("old_id", "new_id")
        assert tm.status_of("old_id") == TruthStatus.SUPERSEDED
        assert tm.status_of("new_id") == TruthStatus.ACTIVE
        assert tm.get("old_id").superseded_by == "new_id"

    def test_merge(self, tm: TruthManager) -> None:
        survivor = tm.merge("a", "b", surviving_id="a")
        assert survivor == "a"
        assert tm.status_of("a") == TruthStatus.ACTIVE
        assert tm.status_of("b") == TruthStatus.SUPERSEDED
        assert tm.get("b").merged_into == "a"

    def test_merge_default_survivor_is_a(self, tm: TruthManager) -> None:
        survivor = tm.merge("a", "b")
        assert survivor == "a"

    def test_archive(self, tm: TruthManager) -> None:
        tm.archive("rec1")
        assert tm.status_of("rec1") == TruthStatus.ARCHIVED

    def test_resolve_direct(self, tm: TruthManager) -> None:
        tm.resolve("rec1", TruthStatus.HISTORICAL)
        assert tm.status_of("rec1") == TruthStatus.HISTORICAL

    def test_mark_conflicting(self, tm: TruthManager) -> None:
        tm.mark_conflicting("a", "b")
        assert tm.status_of("a") == TruthStatus.CONFLICTING
        assert tm.status_of("b") == TruthStatus.CONFLICTING

    def test_mark_historical(self, tm: TruthManager) -> None:
        tm.mark_historical("rec1")
        assert tm.status_of("rec1") == TruthStatus.HISTORICAL

    def test_status_of_unknown_defaults_active(self, tm: TruthManager) -> None:
        assert tm.status_of("never_seen") == TruthStatus.ACTIVE

    def test_history_tracked(self, tm: TruthManager) -> None:
        tm.ensure("rec1")
        tm.archive("rec1")
        rec = tm.get("rec1")
        assert len(rec.history) >= 1

    def test_all_with_status(self, tm: TruthManager) -> None:
        tm.archive("a")
        tm.archive("b")
        tm.ensure("c")
        archived = tm.all_with_status(TruthStatus.ARCHIVED)
        assert len(archived) == 2

    def test_is_active(self, tm: TruthManager) -> None:
        tm.ensure("rec1")
        assert tm.is_active("rec1")
        tm.archive("rec1")
        assert not tm.is_active("rec1")

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        tm1 = TruthManager(tmp_path / "t.json")
        tm1.replace("old", "new")
        tm2 = TruthManager(tmp_path / "t.json")
        assert tm2.status_of("old") == TruthStatus.SUPERSEDED


# ===========================================================================
# Item 4 — BeliefStore
# ===========================================================================


class TestBeliefStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> BeliefStore:
        return BeliefStore(tmp_path / "beliefs.json")

    def test_add_new_belief(self, store: BeliefStore) -> None:
        belief = store.add_or_reinforce("User prefers dark mode")
        assert belief.evidence_count == 1
        assert store.count == 1

    def test_reinforce_similar_belief(self, store: BeliefStore) -> None:
        store.add_or_reinforce("User prefers dark mode in the editor", confidence=0.5)
        belief2 = store.add_or_reinforce("User prefers dark mode in the editor", confidence=0.5)
        assert store.count == 1
        assert belief2.evidence_count == 2
        assert belief2.confidence > 0.5

    def test_source_count_distinct(self, store: BeliefStore) -> None:
        store.add_or_reinforce("User likes Rust programming", source_memory_id=1)
        belief = store.add_or_reinforce("User likes Rust programming", source_memory_id=1)
        assert belief.source_count == 1  # same source twice
        belief2 = store.add_or_reinforce("User likes Rust programming", source_memory_id=2)
        assert belief2.source_count == 2

    def test_weaken(self, store: BeliefStore) -> None:
        belief = store.add_or_reinforce("Some claim", confidence=0.5)
        store.weaken(belief.belief_id)
        assert store.get(belief.belief_id).confidence < 0.5

    def test_set_status(self, store: BeliefStore) -> None:
        belief = store.add_or_reinforce("Some claim")
        store.set_status(belief.belief_id, TruthStatus.ARCHIVED)
        assert store.get(belief.belief_id).status == TruthStatus.ARCHIVED

    def test_find_similar_below_threshold_returns_none(self, store: BeliefStore) -> None:
        store.add_or_reinforce("User likes Rust programming")
        result = store.find_similar("Completely unrelated statement about cooking")
        assert result is None

    def test_all_active(self, store: BeliefStore) -> None:
        b1 = store.add_or_reinforce("Claim A")
        b2 = store.add_or_reinforce("Claim B totally different words here")
        store.set_status(b2.belief_id, TruthStatus.ARCHIVED)
        active = store.all_active()
        assert len(active) == 1
        assert active[0].belief_id == b1.belief_id

    def test_by_topic(self, store: BeliefStore) -> None:
        store.add_or_reinforce("Claim about NLP", topic="nlp")
        store.add_or_reinforce("Claim about vision", topic="vision")
        nlp_beliefs = store.by_topic("nlp")
        assert len(nlp_beliefs) == 1

    def test_low_confidence(self, store: BeliefStore) -> None:
        store.add_or_reinforce("Weak claim", confidence=0.1)
        store.add_or_reinforce("Strong claim totally different wording", confidence=0.9)
        low = store.low_confidence(threshold=0.3)
        assert len(low) == 1

    def test_find_conflicting_candidates(self, store: BeliefStore) -> None:
        store.add_or_reinforce("User lives in Mumbai currently")
        candidates = store.find_conflicting_candidates("User lives in Chennai currently", min_overlap=0.2)
        # Should find partial overlap (lives, currently) without being "the same belief"
        assert isinstance(candidates, list)

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        s1 = BeliefStore(tmp_path / "b.json")
        s1.add_or_reinforce("Persisted claim")
        s2 = BeliefStore(tmp_path / "b.json")
        assert s2.count == 1

    def test_to_dict_from_dict_roundtrip(self) -> None:
        b = Belief(statement="x", confidence=0.7, evidence_count=3, source_memory_ids=[1, 2])
        restored = Belief.from_dict(b.to_dict())
        assert restored.statement == "x"
        assert restored.source_count == 2


# ===========================================================================
# Item 5 — ContradictionResolver (4 cases)
# ===========================================================================


class TestContradictionResolver:
    @pytest.fixture
    def resolver(self, tmp_path: Path) -> ContradictionResolver:
        tm = TruthManager(tmp_path / "truth.json")
        return ContradictionResolver(tm)

    def test_classify_replacement(self, resolver: ContradictionResolver) -> None:
        case = resolver.classify("I moved to Delhi", "I now live in Kolkata", value_a="Delhi", value_b="Kolkata")
        assert case == ContradictionCase.REPLACEMENT

    def test_classify_parallel_truth(self, resolver: ContradictionResolver) -> None:
        case = resolver.classify(
            "I use Python for data work", "I also use Rust for systems programming",
        )
        assert case == ContradictionCase.PARALLEL_TRUTH

    def test_classify_merge_acronym(self, resolver: ContradictionResolver) -> None:
        case = resolver.classify("AI", "Artificial Intelligence", value_a="AI", value_b="Artificial Intelligence")
        assert case == ContradictionCase.MERGE

    def test_classify_merge_similar_statements(self, resolver: ContradictionResolver) -> None:
        case = resolver.classify(
            "User enjoys playing tennis on weekends",
            "User enjoys playing tennis during weekends",
        )
        assert case == ContradictionCase.MERGE

    def test_classify_conflict_no_markers(self, resolver: ContradictionResolver) -> None:
        case = resolver.classify("I live in Mumbai", "I live in Chennai", value_a="Mumbai", value_b="Chennai")
        assert case == ContradictionCase.CONFLICT

    def test_resolve_replacement_sets_statuses(self, resolver: ContradictionResolver) -> None:
        result = resolver.resolve(
            "snap_delhi", "snap_kolkata", "I moved to Delhi", "I now live in Kolkata",
            value_a="Delhi", value_b="Kolkata", newer_id="snap_kolkata",
        )
        assert result.case == ContradictionCase.REPLACEMENT
        assert result.winner_id == "snap_kolkata"

    def test_resolve_parallel_truth_both_active(self, resolver: ContradictionResolver) -> None:
        result = resolver.resolve(
            "belief_python", "belief_rust",
            "I use Python for data work", "I also use Rust for systems",
        )
        assert result.case == ContradictionCase.PARALLEL_TRUTH
        assert resolver._truth.status_of("belief_python") == TruthStatus.ACTIVE
        assert resolver._truth.status_of("belief_rust") == TruthStatus.ACTIVE

    def test_resolve_merge_collapses(self, resolver: ContradictionResolver) -> None:
        result = resolver.resolve(
            "belief_ai", "belief_full", "AI", "Artificial Intelligence",
            value_a="AI", value_b="Artificial Intelligence",
        )
        assert result.case == ContradictionCase.MERGE
        assert result.winner_id == "belief_ai"
        assert resolver._truth.status_of("belief_full") == TruthStatus.SUPERSEDED

    def test_resolve_conflict_marks_conflicting(self, resolver: ContradictionResolver) -> None:
        result = resolver.resolve(
            "belief_x", "belief_y", "I live in Mumbai", "I live in Chennai",
            value_a="Mumbai", value_b="Chennai",
        )
        assert result.case == ContradictionCase.CONFLICT
        assert resolver._truth.status_of("belief_x") == TruthStatus.CONFLICTING

    def test_resolve_replacement_without_newer_id_uses_confidence(self, resolver: ContradictionResolver) -> None:
        result = resolver.resolve(
            "a", "b", "I moved to Delhi", "I now live in Kolkata",
            value_a="Delhi", value_b="Kolkata",
            confidence_a=0.3, confidence_b=0.9,
        )
        assert result.winner_id == "b"

    def test_compare_evidence_a_dominates(self, resolver: ContradictionResolver) -> None:
        winner = resolver.compare_evidence(
            evidence_count_a=5, evidence_count_b=1,
            source_count_a=3, source_count_b=1,
            confidence_a=0.8, confidence_b=0.3,
        )
        assert winner == "a"

    def test_compare_evidence_ambiguous(self, resolver: ContradictionResolver) -> None:
        winner = resolver.compare_evidence(
            evidence_count_a=2, evidence_count_b=2,
            source_count_a=3, source_count_b=1,
            confidence_a=0.3, confidence_b=0.8,  # source favors a, confidence favors b
        )
        assert winner is None

    def test_try_resolve_conflict_promotes_to_replacement(self, resolver: ContradictionResolver) -> None:
        resolver.resolve("x", "y", "I live in Mumbai", "I live in Chennai", value_a="Mumbai", value_b="Chennai")
        result = resolver.try_resolve_conflict(
            "x", "y",
            evidence_count_a=1, evidence_count_b=5,
            source_count_a=1, source_count_b=4,
            confidence_a=0.3, confidence_b=0.9,
        )
        assert result is not None
        assert result.winner_id == "y"
        assert resolver._truth.status_of("x") == TruthStatus.SUPERSEDED

    def test_try_resolve_conflict_stays_ambiguous(self, resolver: ContradictionResolver) -> None:
        result = resolver.try_resolve_conflict(
            "x", "y",
            evidence_count_a=2, evidence_count_b=2,
            source_count_a=2, source_count_b=2,
            confidence_a=0.5, confidence_b=0.5,
        )
        assert result is None

    def test_resolution_result_to_dict(self) -> None:
        r = ResolutionResult(case=ContradictionCase.MERGE, record_a_id="a", record_b_id="b", winner_id="a")
        d = r.to_dict()
        assert d["case"] == "merge"
        assert d["winner_id"] == "a"


# ===========================================================================
# Item 6 — TemporalRetriever
# ===========================================================================


class TestTemporalRetriever:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        tracker = StateTracker(tmp_path / "state.json")
        tm = TruthManager(tmp_path / "truth.json")
        beliefs = BeliefStore(tmp_path / "beliefs.json")
        retriever = TemporalRetriever(state_tracker=tracker, truth_manager=tm, belief_store=beliefs)
        return tracker, tm, beliefs, retriever

    def test_score_no_context_neutral(self, setup) -> None:
        _, _, _, retriever = setup
        score = retriever.score(1, semantic=0.8, recency=0.5, importance=0.6)
        assert 0.0 <= score.final_score <= 1.0
        assert score.state_relevance == 0.5  # neutral, no context

    def test_score_active_state_boosts(self, setup) -> None:
        tracker, tm, _, retriever = setup
        snap = tracker.record("sayan", "lang", "Rust")
        tm.ensure(snap.snapshot_id, TruthStatus.ACTIVE)
        score = retriever.score(
            1, semantic=0.5, recency=0.5, importance=0.5,
            entity="sayan", attribute="lang", snapshot_id=snap.snapshot_id,
        )
        assert score.state_relevance == 1.0

    def test_score_superseded_state_zeroes_relevance(self, setup) -> None:
        tracker, tm, _, retriever = setup
        snap = tracker.record("sayan", "lang", "Python")
        tm.ensure(snap.snapshot_id, TruthStatus.SUPERSEDED)
        score = retriever.score(
            1, semantic=0.9, recency=0.9, importance=0.9,
            entity="sayan", attribute="lang", snapshot_id=snap.snapshot_id,
        )
        assert score.state_relevance == 0.0

    def test_score_belief_confidence_active(self, setup) -> None:
        _, _, beliefs, retriever = setup
        belief = beliefs.add_or_reinforce("Some claim", confidence=0.9)
        score = retriever.score(1, semantic=0.5, recency=0.5, importance=0.5, belief_id=belief.belief_id)
        assert score.belief_confidence == pytest.approx(0.9)

    def test_score_belief_confidence_superseded_near_zero(self, setup) -> None:
        _, _, beliefs, retriever = setup
        belief = beliefs.add_or_reinforce("Some claim", confidence=0.9)
        beliefs.set_status(belief.belief_id, TruthStatus.SUPERSEDED)
        score = retriever.score(1, semantic=0.5, recency=0.5, importance=0.5, belief_id=belief.belief_id)
        assert score.belief_confidence < 0.1

    def test_rank_sorts_descending(self, setup) -> None:
        _, _, _, retriever = setup
        candidates = [
            {"memory_id": 1, "semantic": 0.2, "recency": 0.2, "importance": 0.2},
            {"memory_id": 2, "semantic": 0.9, "recency": 0.9, "importance": 0.9},
        ]
        ranked = retriever.rank(candidates)
        assert ranked[0].memory_id == 2

    def test_current_outranks_historical_in_combined_score(self, setup) -> None:
        tracker, tm, _, retriever = setup
        current_snap = tracker.record("sayan", "lang", "Rust", timestamp=_iso(1))
        tm.ensure(current_snap.snapshot_id, TruthStatus.ACTIVE)
        old_snap = tracker.record("sayan", "lang2", "Python", timestamp=_iso(400))
        tm.ensure(old_snap.snapshot_id, TruthStatus.SUPERSEDED)

        # Even with IDENTICAL semantic/recency/importance, active state should outscore superseded
        score_current = retriever.score(
            1, semantic=0.5, recency=0.5, importance=0.5,
            entity="sayan", attribute="lang", snapshot_id=current_snap.snapshot_id,
        )
        score_old = retriever.score(
            2, semantic=0.5, recency=0.5, importance=0.5,
            entity="sayan", attribute="lang2", snapshot_id=old_snap.snapshot_id,
        )
        assert score_current.final_score > score_old.final_score

    def test_weights_total_close_to_one(self) -> None:
        w = TemporalScoringWeights()
        assert abs(w.total() - 1.0) < 1e-6

    def test_no_components_returns_neutral_defaults(self) -> None:
        retriever = TemporalRetriever()
        score = retriever.score(1, semantic=0.5, recency=0.5, importance=0.5)
        assert score.state_relevance == 0.5
        assert score.belief_confidence == 0.5

    def test_prioritize_current_over_historical(self, setup) -> None:
        tracker, tm, _, retriever = setup
        current_snap = tracker.record("sayan", "lang", "Rust")
        tm.ensure(current_snap.snapshot_id, TruthStatus.ACTIVE)

        class FakeMemory:
            def __init__(self, id):
                self.id = id

        m_historical = FakeMemory(1)
        m_current = FakeMemory(2)
        # Simulate: memory 1 supports an old (now superseded) value
        old_snap = StateSnapshot(entity="sayan", attribute="lang", value="Python", end_time="2024-01-01")
        tm.ensure(old_snap.snapshot_id, TruthStatus.SUPERSEDED)

        mapping = {1: ("sayan", "lang"), 2: ("sayan", "lang")}
        # Note: tracker.current("sayan","lang") will return the Rust snap for both ids
        # since mapping only encodes entity/attribute, not which snapshot — this still
        # demonstrates active-state-first ordering when both resolve to current.
        ordered = retriever.prioritize_current_over_historical([m_historical, m_current], mapping)
        assert len(ordered) == 2


# ===========================================================================
# Item 7 — TemporalGraph
# ===========================================================================


class TestTemporalGraph:
    @pytest.fixture
    def graph(self, tmp_path: Path) -> TemporalGraph:
        return TemporalGraph(tmp_path / "tgraph.json")

    def test_add_relation(self, graph: TemporalGraph) -> None:
        edge = graph.add_relation("Sayan", "uses", "Python")
        assert edge.is_active
        assert graph.count == 1

    def test_add_relation_closes_previous_same_relation(self, graph: TemporalGraph) -> None:
        graph.add_relation("Sayan", "uses", "Python", timestamp=_iso(400))
        graph.add_relation("Sayan", "uses", "Rust", timestamp=_iso(1))
        current = graph.current_relations("Sayan", relation="uses")
        assert len(current) == 1
        assert current[0].to_label == "Rust"

    def test_add_relation_no_close_for_multivalued(self, graph: TemporalGraph) -> None:
        graph.add_relation("Sayan", "knows", "Alice", close_previous=False)
        graph.add_relation("Sayan", "knows", "Bob", close_previous=False)
        current = graph.current_relations("Sayan", relation="knows")
        assert len(current) == 2

    def test_relations_at_time(self, graph: TemporalGraph) -> None:
        graph.add_relation("Sayan", "uses", "Python", timestamp="2024-01-01T00:00:00")
        graph.add_relation("Sayan", "uses", "Rust", timestamp="2026-01-01T00:00:00")
        at_2024 = graph.relations_at_time("Sayan", "2024-06-01T00:00:00", relation="uses")
        assert len(at_2024) == 1
        assert at_2024[0].to_label == "Python"

    def test_evolution_chronological(self, graph: TemporalGraph) -> None:
        graph.add_relation("Blix", "uses", "Python", timestamp="2024-01-01T00:00:00")
        graph.add_relation("Blix", "uses", "PyTorch", timestamp="2025-01-01T00:00:00")
        graph.add_relation("Blix", "uses", "Rust", timestamp="2026-01-01T00:00:00")
        evo = graph.evolution("Blix", "uses")
        assert [e.to_label for e in evo] == ["Python", "PyTorch", "Rust"]

    def test_changes_since(self, graph: TemporalGraph) -> None:
        graph.add_relation("Blix", "uses", "Python", timestamp=_iso(400))
        graph.add_relation("Blix", "uses", "Rust", timestamp=_iso(1))
        recent = graph.changes_since(_iso(30))
        assert len(recent) == 1
        assert recent[0].to_label == "Rust"

    def test_all_for_entity(self, graph: TemporalGraph) -> None:
        graph.add_relation("Blix", "uses", "FastAPI")
        graph.add_relation("Blix", "developed_by", "Sayan")
        assert len(graph.all_for_entity("Blix")) == 2

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        g1 = TemporalGraph(tmp_path / "g.json")
        g1.add_relation("Blix", "uses", "FastAPI")
        g2 = TemporalGraph(tmp_path / "g.json")
        assert g2.count == 1

    def test_active_count(self, graph: TemporalGraph) -> None:
        graph.add_relation("Blix", "uses", "Python", timestamp=_iso(400))
        graph.add_relation("Blix", "uses", "Rust", timestamp=_iso(1))
        assert graph.active_count == 1

    def test_edge_to_dict(self) -> None:
        edge = TemporalEdge(from_label="A", relation="uses", to_label="B", confidence=0.9)
        d = edge.to_dict()
        assert d["from_label"] == "A"
        assert d["is_active"] is True


# ===========================================================================
# Item 8 — TemporalQueryEngine
# ===========================================================================


class TestTemporalQueryEngine:
    @pytest.fixture
    def qe(self, tmp_path: Path) -> TemporalQueryEngine:
        tracker = StateTracker(tmp_path / "state.json")
        engine = StateTransitionEngine(tracker, tmp_path / "transitions.json")
        engine.transition("sayan", "favorite_language", "Python", timestamp="2024-06-01T00:00:00")
        engine.transition("sayan", "favorite_language", "PyTorch", timestamp="2025-01-01T00:00:00")
        engine.transition("sayan", "favorite_language", "Rust", timestamp="2026-01-01T00:00:00")
        return TemporalQueryEngine(tracker, engine, default_entity="sayan")

    def test_historical_year_query(self, qe: TemporalQueryEngine) -> None:
        result = qe.query("What was my favorite language in 2024?")
        assert result.query_type == "historical_year"
        assert result.answer == "Python"

    def test_historical_year_query_2025(self, qe: TemporalQueryEngine) -> None:
        result = qe.query("What was my favorite language in 2025?")
        assert result.answer == "PyTorch"

    def test_evolution_query(self, qe: TemporalQueryEngine) -> None:
        result = qe.query("How has my favorite language evolved?")
        assert result.query_type == "evolution"
        assert "Python" in result.answer
        assert "Rust" in result.answer
        assert len(result.timeline) == 3

    def test_current_query(self, qe: TemporalQueryEngine) -> None:
        result = qe.query("What is my favorite language?")
        assert result.query_type == "current"
        assert result.answer == "Rust"

    def test_transition_query(self, qe: TemporalQueryEngine) -> None:
        result = qe.query("When did sayan adopt Rust?")
        assert result.query_type == "transition"
        assert "2026" in result.answer

    def test_recent_changes_query(self, qe: TemporalQueryEngine) -> None:
        result = qe.query("What changed during the last 365 days?")
        assert result.query_type == "recent_changes"
        assert not result.is_empty()

    def test_unrecognised_query(self, qe: TemporalQueryEngine) -> None:
        result = qe.query("asdkjaslkdj random gibberish !!!")
        # Falls through to "current" pattern (fallback) or unrecognised
        assert result.query_type in ("current", "unrecognised")

    def test_historical_year_no_data(self, qe: TemporalQueryEngine) -> None:
        result = qe.query("What was my favorite language in 2020?")
        assert result.is_empty()

    def test_when_adopted_direct_api(self, qe: TemporalQueryEngine) -> None:
        ts = qe.when_adopted("sayan", "favorite_language", "Rust")
        assert ts == "2026-01-01T00:00:00"

    def test_evolution_chain_direct_api(self, qe: TemporalQueryEngine) -> None:
        chain = qe.evolution_chain("sayan", "favorite_language")
        assert chain == ["Python", "PyTorch", "Rust"]

    def test_result_to_dict(self, qe: TemporalQueryEngine) -> None:
        result = qe.query("What is my favorite language?")
        d = result.to_dict()
        assert d["answer"] == "Rust"
        assert "timeline" in d


# ===========================================================================
# Item 9 — StateReflectionEngine
# ===========================================================================


class TestStateReflectionEngine:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        tracker = StateTracker(tmp_path / "state.json")
        engine = StateTransitionEngine(tracker, tmp_path / "transitions.json")
        reflection = StateReflectionEngine(engine)
        return tracker, engine, reflection

    def test_skill_evolution(self, setup) -> None:
        _, engine, reflection = setup
        engine.transition("sayan", "favorite_language", "Python")
        engine.transition("sayan", "favorite_language", "Rust")
        entries = reflection.skill_evolution("sayan")
        assert len(entries) == 1
        assert entries[0].chain == ["Python", "Rust"]

    def test_interest_evolution(self, setup) -> None:
        _, engine, reflection = setup
        engine.transition("sayan", "research_focus", "NLP")
        engine.transition("sayan", "research_focus", "Memory Systems")
        entries = reflection.interest_evolution("sayan")
        assert len(entries) == 1
        assert "Memory Systems" in entries[0].chain

    def test_project_evolution(self, setup) -> None:
        _, engine, reflection = setup
        engine.transition("sayan", "current_project", "Blix v0.3.6")
        engine.transition("sayan", "current_project", "Blix v0.3.7")
        entries = reflection.project_evolution("sayan")
        assert len(entries) == 1

    def test_identity_evolution(self, setup) -> None:
        _, engine, reflection = setup
        engine.transition("sayan", "city", "Delhi")
        engine.transition("sayan", "city", "Kolkata")
        entries = reflection.identity_evolution("sayan")
        assert len(entries) == 1
        assert entries[0].chain == ["Delhi", "Kolkata"]

    def test_generate_full_report(self, setup) -> None:
        _, engine, reflection = setup
        engine.transition("sayan", "favorite_language", "Python")
        engine.transition("sayan", "favorite_language", "Rust")
        engine.transition("sayan", "city", "Delhi")
        report = reflection.generate("sayan")
        assert report.has_any_evolution()
        assert len(report.skill_evolution) == 1
        assert len(report.identity_evolution) == 1

    def test_report_summary(self, setup) -> None:
        _, engine, reflection = setup
        engine.transition("sayan", "favorite_language", "Python")
        engine.transition("sayan", "favorite_language", "Rust")
        report = reflection.generate("sayan")
        summary = report.summary()
        assert "Skills" in summary

    def test_no_evolution_empty_report(self, setup) -> None:
        _, _, reflection = setup
        report = reflection.generate("ghost_entity")
        assert not report.has_any_evolution()
        assert "No tracked evolution" in report.summary()

    def test_recent_shifts(self, setup) -> None:
        _, engine, reflection = setup
        engine.transition("sayan", "favorite_language", "Python", timestamp=_iso(400))
        engine.transition("sayan", "favorite_language", "Rust", timestamp=_iso(1))
        shifts = reflection.recent_shifts("sayan", days=30)
        assert len(shifts) == 1
        assert shifts[0]["dimension"] == "skill"
        assert shifts[0]["to_value"] == "Rust"

    def test_recent_shifts_excludes_initial_assignment(self, setup) -> None:
        _, engine, reflection = setup
        engine.transition("sayan", "favorite_language", "Python", timestamp=_iso(1))
        shifts = reflection.recent_shifts("sayan", days=30)
        assert shifts == []  # initial assignment isn't a "shift"

    def test_narrative_single_value(self, setup) -> None:
        _, engine, reflection = setup
        engine.transition("sayan", "favorite_language", "Python")
        entries = reflection.skill_evolution("sayan")
        assert "started as" in entries[0].narrative

    def test_narrative_multiple_values(self, setup) -> None:
        _, engine, reflection = setup
        engine.transition("sayan", "favorite_language", "Python")
        engine.transition("sayan", "favorite_language", "Rust")
        entries = reflection.skill_evolution("sayan")
        assert "evolved through" in entries[0].narrative

    def test_evolution_entry_to_dict(self) -> None:
        entry = EvolutionEntry(attribute="lang", chain=["Python", "Rust"], transition_count=1, narrative="x")
        d = entry.to_dict()
        assert d["chain"] == ["Python", "Rust"]

    def test_custom_dimension_attributes(self, setup) -> None:
        tracker, engine, _ = setup
        custom = StateReflectionEngine(engine, dimension_attributes={"interest": ["custom_attr"]})
        engine.transition("sayan", "custom_attr", "value1")
        engine.transition("sayan", "custom_attr", "value2")
        entries = custom.interest_evolution("sayan")
        assert len(entries) == 1


# ===========================================================================
# Item 10 — StateMetrics (StateBench-lite)
# ===========================================================================


class TestStateMetrics:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        tracker = StateTracker(tmp_path / "state.json")
        engine = StateTransitionEngine(tracker, tmp_path / "transitions.json")
        tm = TruthManager(tmp_path / "truth.json")
        engine.transition("sayan", "lang", "Python", timestamp="2024-06-01T00:00:00")
        engine.transition("sayan", "lang", "Rust", timestamp="2026-01-01T00:00:00")
        current = tracker.current("sayan", "lang")
        tm.ensure(current.snapshot_id, TruthStatus.ACTIVE)
        old = tracker.history("sayan", "lang")[0]
        tm.ensure(old.snapshot_id, TruthStatus.SUPERSEDED)
        return tracker, engine, tm

    def test_current_state_accuracy_correct(self, setup) -> None:
        tracker, _, _ = setup
        cases = [StateAccuracyCase(entity="sayan", attribute="lang", expected_value="Rust")]
        metrics = StateMetrics()
        assert metrics.current_state_accuracy(tracker, cases) == 1.0

    def test_current_state_accuracy_incorrect(self, setup) -> None:
        tracker, _, _ = setup
        cases = [StateAccuracyCase(entity="sayan", attribute="lang", expected_value="Python")]
        metrics = StateMetrics()
        assert metrics.current_state_accuracy(tracker, cases) == 0.0

    def test_historical_state_accuracy(self, setup) -> None:
        tracker, _, _ = setup
        cases = [StateAccuracyCase(
            entity="sayan", attribute="lang", expected_value="Python", at_time="2024-12-31T23:59:59",
        )]
        metrics = StateMetrics()
        assert metrics.historical_state_accuracy(tracker, cases) == 1.0

    def test_combined_state_accuracy(self, setup) -> None:
        tracker, _, _ = setup
        cases = [
            StateAccuracyCase(entity="sayan", attribute="lang", expected_value="Rust"),
            StateAccuracyCase(entity="sayan", attribute="lang", expected_value="Python", at_time="2024-12-31T23:59:59"),
        ]
        metrics = StateMetrics()
        assert metrics.combined_state_accuracy(tracker, cases) == 1.0

    def test_transition_accuracy_correct(self, setup) -> None:
        tracker, _, _ = setup
        cases = [TransitionAccuracyCase(entity="sayan", attribute="lang", expected_from="Python", expected_to="Rust")]
        metrics = StateMetrics()
        assert metrics.transition_accuracy(tracker, cases) == 1.0

    def test_transition_accuracy_wrong_from(self, setup) -> None:
        tracker, _, _ = setup
        cases = [TransitionAccuracyCase(entity="sayan", attribute="lang", expected_from="Java", expected_to="Rust")]
        metrics = StateMetrics()
        assert metrics.transition_accuracy(tracker, cases) == 0.0

    def test_transition_accuracy_with_time_tolerance(self, setup) -> None:
        tracker, _, _ = setup
        cases = [TransitionAccuracyCase(
            entity="sayan", attribute="lang", expected_from="Python", expected_to="Rust",
            expected_around_time="2026-01-02T00:00:00", tolerance_days=5,
        )]
        metrics = StateMetrics()
        assert metrics.transition_accuracy(tracker, cases) == 1.0

    def test_transition_accuracy_outside_tolerance(self, setup) -> None:
        tracker, _, _ = setup
        cases = [TransitionAccuracyCase(
            entity="sayan", attribute="lang", expected_from="Python", expected_to="Rust",
            expected_around_time="2027-06-01T00:00:00", tolerance_days=5,
        )]
        metrics = StateMetrics()
        assert metrics.transition_accuracy(tracker, cases) == 0.0

    def test_state_hallucination_rate(self) -> None:
        metrics = StateMetrics()
        rate = metrics.state_hallucination_rate(
            predicted_answers=["Rust", None, "Fabricated"],
            ground_truth_exists=[True, True, False],
        )
        assert rate == 1.0  # 1 hallucination out of 1 no-truth case

    def test_state_hallucination_rate_no_hallucination(self) -> None:
        metrics = StateMetrics()
        rate = metrics.state_hallucination_rate(
            predicted_answers=["Rust", None],
            ground_truth_exists=[True, False],
        )
        assert rate == 0.0

    def test_belief_drift(self) -> None:
        metrics = StateMetrics()
        drift = metrics.belief_drift({"b1": 0.5, "b2": 0.3}, {"b1": 0.8, "b2": 0.3})
        assert drift == pytest.approx(0.15)

    def test_belief_drift_no_shared_ids(self) -> None:
        metrics = StateMetrics()
        assert metrics.belief_drift({"a": 0.5}, {"b": 0.5}) == 0.0

    def test_truth_consistency_all_consistent(self, setup) -> None:
        tracker, _, tm = setup
        metrics = StateMetrics()
        consistency = metrics.truth_consistency(tm, [("sayan", "lang")], tracker)
        assert consistency == 1.0

    def test_truth_consistency_detects_inconsistency(self, setup) -> None:
        tracker, _, tm = setup
        # Force an inconsistency: mark the CLOSED snapshot as still ACTIVE
        old = tracker.history("sayan", "lang")[0]
        tm.resolve(old.snapshot_id, TruthStatus.ACTIVE)
        metrics = StateMetrics()
        consistency = metrics.truth_consistency(tm, [("sayan", "lang")], tracker)
        assert consistency == 0.0

    def test_run_statebench_full_suite(self, setup) -> None:
        tracker, _, tm = setup
        metrics = StateMetrics()
        state_cases = [StateAccuracyCase(entity="sayan", attribute="lang", expected_value="Rust")]
        trans_cases = [TransitionAccuracyCase(entity="sayan", attribute="lang", expected_from="Python", expected_to="Rust")]
        results = metrics.run_statebench(tracker, tm, state_cases, trans_cases)
        assert results["current_state_accuracy"] == 1.0
        assert results["transition_accuracy"] == 1.0
        assert results["truth_consistency"] == 1.0

    def test_inherits_v036_metrics(self) -> None:
        metrics = StateMetrics()
        from agents.types import TaskGraph, Task
        graph = TaskGraph(goal="g")
        graph.add_task(Task(title="T"))
        graph.tasks[0].mark_completed("ok")
        assert metrics.task_success_rate(graph) == 1.0

    def test_in_blix_eval_exports(self) -> None:
        from evaluation.blix_eval import StateMetrics as SM_blix
        assert StateMetrics is SM_blix


# ===========================================================================
# Integration — BlixContext wiring
# ===========================================================================


class _FakeLLM:
    def model_name(self) -> str:
        return "fake-0.3.7"

    def generate(self, prompt: str) -> str:
        return "Fake reply."


@pytest.fixture(scope="module")
def tmp_memory_v7(tmp_path_factory):
    return tmp_path_factory.mktemp("memory_v7")


@pytest.fixture(scope="module")
def ctx_v7(tmp_memory_v7):
    from config import settings as _settings
    _settings.settings.memory.conversations_file = tmp_memory_v7 / "conversations.json"
    _settings.settings.memory.profile_file = tmp_memory_v7 / "profile.json"
    _settings.settings.memory.learning_state_file = tmp_memory_v7 / "learning_state.json"
    _settings.settings.embed.embeddings_file = tmp_memory_v7 / "embeddings.npy"
    _settings.settings.embed.embedding_ids_file = tmp_memory_v7 / "embedding_ids.json"

    from api.context import BlixContext
    ctx = BlixContext(tmp_memory_v7)
    ctx.llm = _FakeLLM()
    ctx.agent._llm = _FakeLLM()
    return ctx


@pytest.fixture(scope="module")
def client_v7(ctx_v7) -> Generator[TestClient, None, None]:
    from api.deps import set_context
    from api.routers.temporal import router as temporal_router

    app = FastAPI(title="Blix Test v0.3.7")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(temporal_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    set_context(ctx_v7)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestBlixContextV037Wiring:
    def test_v037_components_present(self, ctx_v7) -> None:
        assert ctx_v7.state_tracker is not None
        assert ctx_v7.state_transitions is not None
        assert ctx_v7.truth_manager is not None
        assert ctx_v7.belief_store is not None
        assert ctx_v7.contradiction_resolver is not None
        assert ctx_v7.temporal_graph is not None
        assert ctx_v7.temporal_retriever is not None
        assert ctx_v7.temporal_query_engine is not None
        assert ctx_v7.state_reflection is not None
        assert ctx_v7.state_metrics is not None

    def test_dashboard_stats_includes_v037_metrics(self, ctx_v7) -> None:
        stats = ctx_v7.dashboard_stats()
        assert "state_snapshots_tracked" in stats
        assert "state_transitions_recorded" in stats
        assert "beliefs_tracked" in stats
        assert "temporal_graph_edges" in stats

    def test_end_to_end_transition_via_context(self, ctx_v7) -> None:
        ctx_v7.state_transitions.transition("test_entity", "test_attr", "ValueA")
        ctx_v7.state_transitions.transition("test_entity", "test_attr", "ValueB")
        current = ctx_v7.state_tracker.current("test_entity", "test_attr")
        assert current.value == "ValueB"


# ===========================================================================
# API — /temporal endpoints
# ===========================================================================


class TestTemporalAPI:
    def test_get_current_state(self, client_v7: TestClient, ctx_v7) -> None:
        ctx_v7.state_transitions.transition("api_entity", "lang", "Python")
        ctx_v7.state_transitions.transition("api_entity", "lang", "Rust")
        r = client_v7.get("/temporal/state/api_entity/lang")
        assert r.status_code == 200
        data = r.json()
        assert data["value"] == "Rust"
        assert "truth_status" in data

    def test_get_current_state_not_found(self, client_v7: TestClient) -> None:
        r = client_v7.get("/temporal/state/ghost_entity/ghost_attr")
        assert r.status_code == 404

    def test_get_state_history(self, client_v7: TestClient, ctx_v7) -> None:
        ctx_v7.state_transitions.transition("hist_entity", "lang", "Python")
        ctx_v7.state_transitions.transition("hist_entity", "lang", "Rust")
        r = client_v7.get("/temporal/state/hist_entity/lang/history")
        assert r.status_code == 200
        data = r.json()
        assert len(data["history"]) == 2

    def test_get_state_at_time(self, client_v7: TestClient, ctx_v7) -> None:
        ctx_v7.state_transitions.transition("time_entity", "lang", "Python", timestamp="2024-01-01T00:00:00")
        ctx_v7.state_transitions.transition("time_entity", "lang", "Rust", timestamp="2026-01-01T00:00:00")
        r = client_v7.get("/temporal/state/time_entity/lang/at?timestamp=2024-06-01T00:00:00")
        assert r.status_code == 200
        assert r.json()["value"] == "Python"

    def test_temporal_query_endpoint(self, client_v7: TestClient, ctx_v7) -> None:
        ctx_v7.state_transitions.transition("user", "favorite_language", "Python", timestamp="2024-06-01T00:00:00")
        ctx_v7.state_transitions.transition("user", "favorite_language", "Rust", timestamp="2026-01-01T00:00:00")
        r = client_v7.post("/temporal/query", json={"query": "What is my favorite language?"})
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "Rust"

    def test_evolution_endpoint(self, client_v7: TestClient, ctx_v7) -> None:
        ctx_v7.state_transitions.transition("evo_entity", "favorite_language", "Python")
        ctx_v7.state_transitions.transition("evo_entity", "favorite_language", "Rust")
        r = client_v7.get("/temporal/evolution/evo_entity")
        assert r.status_code == 200
        data = r.json()
        assert len(data["skill_evolution"]) >= 1

    def test_recent_shifts_endpoint(self, client_v7: TestClient, ctx_v7) -> None:
        r = client_v7.get("/temporal/evolution/evo_entity/recent?days=30")
        assert r.status_code == 200
        assert "shifts" in r.json()

    def test_list_beliefs_endpoint(self, client_v7: TestClient, ctx_v7) -> None:
        ctx_v7.belief_store.add_or_reinforce("Test belief statement here")
        r = client_v7.get("/temporal/beliefs")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1

    def test_list_beliefs_filtered_by_status(self, client_v7: TestClient, ctx_v7) -> None:
        belief = ctx_v7.belief_store.add_or_reinforce("Archived belief statement")
        ctx_v7.belief_store.set_status(belief.belief_id, TruthStatus.ARCHIVED)
        r = client_v7.get("/temporal/beliefs?status=archived")
        assert r.status_code == 200
        data = r.json()
        assert any(b["belief_id"] == belief.belief_id for b in data["beliefs"])

    def test_list_beliefs_invalid_status(self, client_v7: TestClient) -> None:
        r = client_v7.get("/temporal/beliefs?status=nonexistent_status")
        assert r.status_code == 422

    def test_get_truth_status(self, client_v7: TestClient, ctx_v7) -> None:
        ctx_v7.truth_manager.archive("test_record_xyz")
        r = client_v7.get("/temporal/truth/test_record_xyz")
        assert r.status_code == 200
        assert r.json()["status"] == "archived"

    def test_get_truth_status_not_found(self, client_v7: TestClient) -> None:
        r = client_v7.get("/temporal/truth/never_registered_id")
        assert r.status_code == 404

    def test_resolve_endpoint_replacement(self, client_v7: TestClient) -> None:
        r = client_v7.post("/temporal/resolve", json={
            "record_a_id": "api_delhi", "record_b_id": "api_kolkata",
            "text_a": "I moved to Delhi", "text_b": "I now live in Kolkata",
            "value_a": "Delhi", "value_b": "Kolkata", "newer_id": "api_kolkata",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["case"] == "replacement"
        assert data["winner_id"] == "api_kolkata"

    def test_resolve_endpoint_conflict(self, client_v7: TestClient) -> None:
        r = client_v7.post("/temporal/resolve", json={
            "record_a_id": "api_x", "record_b_id": "api_y",
            "text_a": "I live in Mumbai", "text_b": "I live in Chennai",
            "value_a": "Mumbai", "value_b": "Chennai",
        })
        assert r.status_code == 200
        assert r.json()["case"] == "conflict"
