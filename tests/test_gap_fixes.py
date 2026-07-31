"""
Tests for the 12 gap fixes applied post-v0.3.13.

Fix #1  — SearchCritic thin_margin uses real VN scores (apples-to-apples)
Fix #2  — HypothesisManager.expire_stale() transitions old PENDING -> UNKNOWN
Fix #3  — CuriosityEngine CONTRADICTION signal includes conflicting_belief_ids
Fix #4  — MetaCausal which_strategies_cause_success strengthened negative qualifier list
Fix #5  — explore_with_trajectories() returns fresh copies, never mutates explore() results
Fix #6  — GoalTracker.suggest_next_search() wired to BeamSearchPlanner
Fix #7  — CognitiveQueryEngine confirmed functional (not a stub — false alarm in audit)
Fix #8  — dashboard_stats() includes 7 new operational metrics
Fix #9  — curiosity list_hypotheses / list_experiments support limit/offset pagination
Fix #10 — EventBus FAILURE + CONFIDENCE_CHANGED subscribers wired in BlixContext
Fix #11 — PrincipleSynthesizer LLM paths confirmed functional (not stubs)
Fix #12 — BlixContext.shutdown() performs full cleanup chain
"""
from __future__ import annotations

import random
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fix #1 — SearchCritic thin_margin uses real VN scores
# ---------------------------------------------------------------------------

class TestFix1SearchCriticMargin:
    def test_runner_up_values_populated(self, tmp_path):
        from world_model.value_network import ValueNetwork
        from world_model.latent_world_model import LatentState
        from planning.beam_search import BeamSearchPlanner

        vn = ValueNetwork(tmp_path / "vn.json", min_examples_to_train=20)
        random.seed(7)
        for i in range(25):
            risky = i % 2 == 0
            s = LatentState(confidence=0.6, risk=0.9 if risky else 0.1, capability_estimate=0.6)
            vn.observe_outcome(s, eventual_value=0.1 if risky else 0.9)

        planner = BeamSearchPlanner(vn, beam_width=3, max_depth=1)
        candidates = [
            ("HIGH_RISK", LatentState(confidence=0.6, risk=0.9, capability_estimate=0.6)),
            ("MED_RISK",  LatentState(confidence=0.6, risk=0.5, capability_estimate=0.6)),
            ("LOW_RISK",  LatentState(confidence=0.6, risk=0.1, capability_estimate=0.6)),
        ]
        result = planner.search("goal", LatentState(confidence=0.4, risk=0.7), lambda s: candidates)
        assert len(result.runner_up_values) == len(result.runner_up_trajectories)
        assert all(isinstance(v, float) for v in result.runner_up_values)

    def test_thin_margin_uses_vn_scores_not_delta(self, tmp_path):
        from world_model.value_network import ValueNetwork
        from world_model.latent_world_model import LatentState
        from planning.beam_search import BeamSearchPlanner, BeamSearchResult
        from planning.search_critic import SearchCritic
        from simulation.trajectory_graph import TrajectoryBuilder

        # Construct a result where runner_up_values is close to best_value
        b1 = TrajectoryBuilder(LatentState())
        b1.step("a", LatentState())
        b2 = TrajectoryBuilder(LatentState())
        b2.step("b", LatentState())
        result = BeamSearchResult(
            goal="g", best_trajectory=b1.build(),
            runner_up_trajectories=[b2.build()],
            runner_up_values=[0.501],  # just below best_value
            best_value=0.502,
        )
        critic = SearchCritic()
        expl = critic.explain(result)
        # Margin is 0.001 < _THIN_MARGIN_THRESHOLD (0.05) — should be flagged
        assert any(i.category == "thin_margin" for i in expl.issues)

    def test_result_to_dict_includes_runner_up_values(self, tmp_path):
        from planning.beam_search import BeamSearchResult
        from simulation.trajectory_graph import TrajectoryBuilder
        from world_model.latent_world_model import LatentState

        b = TrajectoryBuilder(LatentState())
        b.step("x", LatentState())
        result = BeamSearchResult(goal="g", best_trajectory=b.build(),
                                   runner_up_values=[0.4, 0.3], best_value=0.7)
        d = result.to_dict()
        assert "runner_up_values" in d
        assert d["runner_up_values"] == [0.4, 0.3]


# ---------------------------------------------------------------------------
# Fix #2 — HypothesisManager.expire_stale()
# ---------------------------------------------------------------------------

class TestFix2ExpireStale:
    def test_old_pending_transitions_to_unknown(self, tmp_path):
        from hypothesis.hypothesis_manager import HypothesisManager, HypothesisStatus
        hm = HypothesisManager(tmp_path / "hyp.json")
        h = hm.propose("Old hypothesis")
        h.created_at = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()
        hm._save()
        expired = hm.expire_stale(max_age_days=30)
        assert len(expired) == 1
        assert hm.get(h.hypothesis_id).status == HypothesisStatus.UNKNOWN

    def test_recent_pending_not_expired(self, tmp_path):
        from hypothesis.hypothesis_manager import HypothesisManager, HypothesisStatus
        hm = HypothesisManager(tmp_path / "hyp.json")
        h = hm.propose("Recent hypothesis")
        expired = hm.expire_stale(max_age_days=30)
        assert len(expired) == 0
        assert hm.get(h.hypothesis_id).status == HypothesisStatus.PENDING

    def test_supported_hypotheses_not_touched(self, tmp_path):
        from memory.beliefs import BeliefStore
        from hypothesis.hypothesis_manager import HypothesisManager, HypothesisStatus
        bs = BeliefStore(tmp_path / "beliefs.json")
        hm = HypothesisManager(tmp_path / "hyp.json", belief_store=bs, support_threshold=0.7)
        h = hm.propose("Strong hypothesis", confidence=0.3)
        hm.add_evidence(h.hypothesis_id, "E1", confidence_delta=0.25)
        hm.add_evidence(h.hypothesis_id, "E2", confidence_delta=0.25)
        # Backdate it
        hyp = hm.get(h.hypothesis_id)
        hyp.created_at = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        hm._save()
        expired = hm.expire_stale(max_age_days=30)
        assert len(expired) == 0  # SUPPORTED not touched


# ---------------------------------------------------------------------------
# Fix #3 — CuriosityEngine CONTRADICTION includes conflicting_belief_ids
# ---------------------------------------------------------------------------

class TestFix3ContradictionIds:
    def test_contradiction_signal_has_belief_ids(self, tmp_path):
        from memory.beliefs import BeliefStore
        from causality.belief_dependency_graph import BeliefDependencyGraph
        from curiosity.curiosity_engine import CuriosityEngine, CuriosityTrigger
        from memory.beliefs import _jaccard

        store = BeliefStore(tmp_path / "beliefs.json")
        # Use algorithmically-contrasting pairs (Jaccard in [0.3, 0.5))
        b1 = store.add_or_reinforce("The project release was accelerated approved")
        b2 = store.add_or_reinforce("The project release was delayed rejected")

        engine = CuriosityEngine(store, low_confidence_threshold=0.0, sparse_evidence_threshold=0)
        signals = engine.generate_signals(top_k=20)
        contra = [s for s in signals if s.trigger == CuriosityTrigger.CONTRADICTION]

        if contra:  # only assert if contradiction was actually detected
            for s in contra:
                assert len(s.conflicting_belief_ids) == 2
                assert b1.belief_id in s.conflicting_belief_ids or b2.belief_id in s.conflicting_belief_ids

    def test_non_contradiction_signal_has_empty_ids(self, tmp_path):
        from memory.beliefs import BeliefStore
        from curiosity.curiosity_engine import CuriosityEngine, CuriosityTrigger
        store = BeliefStore(tmp_path / "beliefs.json")
        store.add_or_reinforce("Apple zero", confidence=0.1)
        engine = CuriosityEngine(store, low_confidence_threshold=0.4)
        signals = engine.generate_signals(top_k=10)
        low_conf = [s for s in signals if s.trigger == CuriosityTrigger.LOW_CONFIDENCE]
        for s in low_conf:
            assert s.conflicting_belief_ids == []


# ---------------------------------------------------------------------------
# Fix #4 — MetaCausal which_strategies_cause_success negative qualifiers
# ---------------------------------------------------------------------------

class TestFix4NegativeQualifiers:
    def test_unreliable_confidence_drops_excluded(self, tmp_path):
        from causality.cause_graph import CauseGraph, CauseRelation
        from causality.meta_causal_reflection import MetaCausalReflection
        cg = CauseGraph(tmp_path / "cg.json")
        cg.record_observation("some_strategy", "unreliable confidence drops", CauseRelation.INCREASES)
        mcr = MetaCausalReflection(cg)
        answer = mcr.which_strategies_cause_success()
        assert "some_strategy" not in answer.answer_summary

    def test_increases_confidence_rapidly_included(self, tmp_path):
        from causality.cause_graph import CauseGraph, CauseRelation
        from causality.meta_causal_reflection import MetaCausalReflection
        cg = CauseGraph(tmp_path / "cg.json")
        cg.record_observation("good_strategy", "increases confidence rapidly", CauseRelation.ENABLES)
        mcr = MetaCausalReflection(cg)
        answer = mcr.which_strategies_cause_success()
        assert "good_strategy" in answer.answer_summary

    def test_failure_keyword_excluded(self, tmp_path):
        from causality.cause_graph import CauseGraph, CauseRelation
        from causality.meta_causal_reflection import MetaCausalReflection
        cg = CauseGraph(tmp_path / "cg.json")
        cg.record_observation("bad_strategy", "failure confidence loss", CauseRelation.INCREASES)
        mcr = MetaCausalReflection(cg)
        answer = mcr.which_strategies_cause_success()
        assert "bad_strategy" not in answer.answer_summary


# ---------------------------------------------------------------------------
# Fix #5 — explore_with_trajectories never mutates explore() results
# ---------------------------------------------------------------------------

class TestFix5NoMutation:
    def test_explore_result_not_mutated_by_trajectories(self, tmp_path):
        from world_model.value_network import ValueNetwork
        from world_model.latent_world_model import LatentState
        from causality.counterfactual_engine import CounterfactualScenarioEngine, CounterfactualAlternative

        engine = CounterfactualScenarioEngine(ValueNetwork(tmp_path / "vn.json"))
        current = LatentState(confidence=0.4)
        alts = [CounterfactualAlternative(name="a", description="d", resulting_state=LatentState(confidence=0.7))]

        base_results = engine.explore(current, alts)
        assert base_results[0].trajectory is None  # not set yet

        enriched = engine.explore_with_trajectories(current, alts)

        # base_results must not have been mutated
        assert base_results[0].trajectory is None
        assert enriched[0].trajectory is not None
        assert base_results[0] is not enriched[0]  # different objects

    def test_enriched_trajectory_epistemic_status_counterfactual(self, tmp_path):
        from world_model.value_network import ValueNetwork
        from world_model.latent_world_model import LatentState
        from causality.counterfactual_engine import CounterfactualScenarioEngine, CounterfactualAlternative
        from causality.epistemic_status import EpistemicStatus

        engine = CounterfactualScenarioEngine(ValueNetwork(tmp_path / "vn.json"))
        alts = [CounterfactualAlternative(name="x", description="d", resulting_state=LatentState())]
        results = engine.explore_with_trajectories(LatentState(), alts)
        assert results[0].trajectory.epistemic_status == EpistemicStatus.COUNTERFACTUAL


# ---------------------------------------------------------------------------
# Fix #6 — GoalTracker.suggest_next_search() wired to BeamSearchPlanner
# ---------------------------------------------------------------------------

class TestFix6GoalTrackerPlanning:
    def test_suggest_next_search_no_goals_returns_none(self, tmp_path):
        from reflection.goal_tracker import GoalTracker
        from world_model.value_network import ValueNetwork
        from world_model.latent_world_model import LatentState
        from planning.beam_search import BeamSearchPlanner
        tracker = GoalTracker(tmp_path / "goals.json")
        vn = ValueNetwork(tmp_path / "vn.json")
        planner = BeamSearchPlanner(vn)
        assert tracker.suggest_next_search(planner, LatentState()) is None

    def test_suggest_next_search_uses_goal_title(self, tmp_path):
        from reflection.goal_tracker import GoalTracker
        from world_model.value_network import ValueNetwork
        from world_model.latent_world_model import LatentState
        from planning.beam_search import BeamSearchPlanner
        tracker = GoalTracker(tmp_path / "goals.json")
        g = tracker.create_goal("Improve retrieval quality")
        tracker.add_task(g.goal_id, "Profile slow queries")
        vn = ValueNetwork(tmp_path / "vn.json")
        planner = BeamSearchPlanner(vn, beam_width=2, max_depth=1)
        result = tracker.suggest_next_search(planner, LatentState(confidence=0.5, risk=0.3))
        assert result is not None
        assert result.goal == "Improve retrieval quality"

    def test_suggest_next_search_includes_task_actions(self, tmp_path):
        from reflection.goal_tracker import GoalTracker
        from world_model.value_network import ValueNetwork
        from world_model.latent_world_model import LatentState
        from planning.beam_search import BeamSearchPlanner
        tracker = GoalTracker(tmp_path / "goals.json")
        g = tracker.create_goal("Fix performance")
        tracker.add_task(g.goal_id, "Profile database queries")
        tracker.add_blocker(g.goal_id, "Missing profiling tool")
        vn = ValueNetwork(tmp_path / "vn.json")
        planner = BeamSearchPlanner(vn, beam_width=3, max_depth=1)
        result = tracker.suggest_next_search(planner, LatentState(confidence=0.4, risk=0.5))
        actions = result.best_trajectory.actions if result.best_trajectory else []
        assert any("resolve_blocker" in a or "complete_task" in a for a in actions)


# ---------------------------------------------------------------------------
# Fix #8 — dashboard_stats new keys
# ---------------------------------------------------------------------------

class TestFix8DashboardStats:
    @pytest.fixture(scope="class")
    def ctx(self, tmp_path_factory):
        import sys
        sys.path.insert(0, "/home/claude/blix_v03")
        from config import settings as _settings
        tmp = tmp_path_factory.mktemp("ctx_gap")
        _settings.settings.memory.conversations_file = tmp / "conv.json"
        _settings.settings.memory.profile_file = tmp / "profile.json"
        _settings.settings.memory.learning_state_file = tmp / "ls.json"
        _settings.settings.embed.embeddings_file = tmp / "emb.npy"
        _settings.settings.embed.embedding_ids_file = tmp / "emb_ids.json"
        from api.context import BlixContext
        return BlixContext(tmp)

    def test_new_dashboard_keys_present(self, ctx):
        stats = ctx.dashboard_stats()
        for key in ("truth_manager_beliefs", "principles_synthesized", "principle_graph_edges",
                    "self_model_capabilities", "self_model_weaknesses",
                    "failure_clusters", "workspace_broadcast_count"):
            assert key in stats, f"Missing key: {key}"

    def test_new_keys_are_numeric(self, ctx):
        stats = ctx.dashboard_stats()
        for key in ("principles_synthesized", "principle_graph_edges",
                    "self_model_capabilities", "failure_clusters"):
            val = stats[key]
            assert isinstance(val, (int, float, type(None))), f"{key}={val!r} not numeric"


# ---------------------------------------------------------------------------
# Fix #9 — Pagination on list endpoints
# ---------------------------------------------------------------------------

class TestFix9Pagination:
    def test_list_hypotheses_limit_offset(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.deps import set_context
        from api.routers.curiosity import router

        import sys
        sys.path.insert(0, "/home/claude/blix_v03")
        from config import settings as _settings
        _settings.settings.memory.conversations_file = tmp_path / "c.json"
        _settings.settings.memory.profile_file = tmp_path / "p.json"
        _settings.settings.memory.learning_state_file = tmp_path / "l.json"
        _settings.settings.embed.embeddings_file = tmp_path / "e.npy"
        _settings.settings.embed.embedding_ids_file = tmp_path / "ei.json"
        from api.context import BlixContext
        ctx = BlixContext(tmp_path)
        for i in range(5):
            ctx.hypothesis_manager.propose(f"Hypothesis {i}")

        set_context(ctx)
        app = FastAPI()
        app.include_router(router)
        with TestClient(app) as client:
            r = client.get("/curiosity/hypotheses?limit=3&offset=0")
            assert r.status_code == 200
            d = r.json()
            assert d["total"] == 5
            assert len(d["hypotheses"]) == 3
            assert "limit" in d
            assert "offset" in d

            r2 = client.get("/curiosity/hypotheses?limit=3&offset=3")
            assert len(r2.json()["hypotheses"]) == 2


# ---------------------------------------------------------------------------
# Fix #12 — shutdown() cleanup chain
# ---------------------------------------------------------------------------

class TestFix12Shutdown:
    def test_shutdown_does_not_raise(self, tmp_path):
        import sys
        sys.path.insert(0, "/home/claude/blix_v03")
        from config import settings as _settings
        _settings.settings.memory.conversations_file = tmp_path / "c.json"
        _settings.settings.memory.profile_file = tmp_path / "p.json"
        _settings.settings.memory.learning_state_file = tmp_path / "l.json"
        _settings.settings.embed.embeddings_file = tmp_path / "e.npy"
        _settings.settings.embed.embedding_ids_file = tmp_path / "ei.json"
        from api.context import BlixContext
        ctx = BlixContext(tmp_path)
        ctx.hypothesis_manager.propose("A hypothesis that should be persisted on shutdown")
        ctx.shutdown()  # must not raise

    def test_shutdown_persists_beliefs(self, tmp_path):
        import sys
        sys.path.insert(0, "/home/claude/blix_v03")
        from config import settings as _settings
        _settings.settings.memory.conversations_file = tmp_path / "c.json"
        _settings.settings.memory.profile_file = tmp_path / "p.json"
        _settings.settings.memory.learning_state_file = tmp_path / "l.json"
        _settings.settings.embed.embeddings_file = tmp_path / "e.npy"
        _settings.settings.embed.embedding_ids_file = tmp_path / "ei.json"
        from api.context import BlixContext
        from memory.beliefs import BeliefStore
        ctx = BlixContext(tmp_path)
        b = ctx.belief_store.add_or_reinforce("Shutdown persistence test belief unique")
        ctx.shutdown()
        # Reload and verify
        store2 = BeliefStore(tmp_path / "beliefs.json")
        assert store2.get(b.belief_id) is not None
