"""
Tests for Blix v0.3.12 — "Imagination + Search".

Covers:
  simulation.trajectory_graph
  planning.beam_search
  planning.search_critic
  causality.counterfactual_engine (explore_with_trajectories extension)
  evaluation.prediction_evaluator
  Safeguard: counterfactual_engine zero memory.beliefs imports after extension
  Integration  — BlixContext wiring
  API          — /search endpoints

Python 3.10 compatible — fully offline.
"""

from __future__ import annotations

import inspect
import random
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from causality.counterfactual_engine import CounterfactualAlternative, CounterfactualResult, CounterfactualScenarioEngine
from causality.epistemic_status import EpistemicStatus
from evaluation.prediction_evaluator import PredictionDrift, PredictionEvaluator
from memory.future_memory import FutureMemoryStore
from planning.beam_search import BeamSearchPlanner, BeamSearchResult, _clone_and_step
from planning.search_critic import DecisionExplanation, IssueSeverity, SearchCritic, SearchIssue
from simulation.trajectory_graph import (
    ActionEdge, StateNode, Trajectory, TrajectoryBuilder, TrajectoryGraph,
)
from world_model.latent_world_model import LatentState
from world_model.value_network import ValueNetwork


# helpers

def _trained_vn(tmp_path: Path, n: int = 25) -> ValueNetwork:
    vn = ValueNetwork(tmp_path / "vn.json", min_examples_to_train=n)
    random.seed(42)
    for i in range(n + 5):
        risky = i % 2 == 0
        s = LatentState(confidence=0.6, risk=0.9 if risky else 0.1, capability_estimate=0.6)
        vn.observe_outcome(s, eventual_value=0.1 if risky else 0.9)
    return vn


def _actions(state: LatentState):
    return [
        ("DIRECT", LatentState(confidence=state.confidence, risk=0.9)),
        ("TOT", LatentState(confidence=state.confidence + 0.1, risk=0.5)),
        ("DECOMPOSE", LatentState(confidence=state.confidence + 0.1, risk=0.1)),
    ]


# TrajectoryGraph

class TestTrajectoryGraph:
    def test_builder_creates_start_node(self):
        traj = TrajectoryBuilder(LatentState()).build()
        assert len(traj.nodes) == 1
        assert traj.depth == 0

    def test_step_increments_depth(self):
        b = TrajectoryBuilder(LatentState())
        b.step("a", LatentState())
        b.step("b", LatentState())
        traj = b.build()
        assert traj.depth == 2
        assert len(traj.nodes) == 3

    def test_actions(self):
        b = TrajectoryBuilder(LatentState())
        b.step("DIRECT", LatentState(risk=0.9))
        b.step("TOT", LatentState(risk=0.5))
        assert b.build().actions == ["DIRECT", "TOT"]

    def test_start_and_end_nodes(self):
        b = TrajectoryBuilder(LatentState(confidence=0.2))
        b.step("improve", LatentState(confidence=0.8))
        traj = b.build()
        assert traj.start_node.state.confidence == pytest.approx(0.2)
        assert traj.end_node.state.confidence == pytest.approx(0.8)

    def test_value_delta_sum(self):
        b = TrajectoryBuilder(LatentState())
        b.step("a", LatentState(), predicted_value_delta=0.2)
        b.step("b", LatentState(), predicted_value_delta=0.1)
        assert b.build().total_predicted_value_delta == pytest.approx(0.3)

    def test_default_epistemic_status_predicted(self):
        assert TrajectoryBuilder(LatentState()).build().epistemic_status == EpistemicStatus.PREDICTED

    def test_counterfactual_status(self):
        traj = TrajectoryBuilder(LatentState(), epistemic_status=EpistemicStatus.COUNTERFACTUAL).build()
        assert traj.epistemic_status == EpistemicStatus.COUNTERFACTUAL

    def test_to_dict(self):
        b = TrajectoryBuilder(LatentState(confidence=0.5))
        b.step("act", LatentState(confidence=0.7))
        d = b.build().to_dict()
        assert len(d["nodes"]) == 2
        assert d["depth"] == 1
        assert d["epistemic_status"] == "predicted"

    def test_node_depth_set_correctly(self):
        b = TrajectoryBuilder(LatentState())
        b.step("a", LatentState())
        b.step("b", LatentState())
        traj = b.build()
        assert traj.nodes[0].depth == 0
        assert traj.nodes[1].depth == 1
        assert traj.nodes[2].depth == 2

    def test_graph_add_and_get(self):
        tg = TrajectoryGraph()
        traj = TrajectoryBuilder(LatentState()).build()
        tg.add(traj)
        assert tg.get(traj.trajectory_id) is traj

    def test_graph_count(self):
        tg = TrajectoryGraph()
        assert tg.count == 0
        tg.add(TrajectoryBuilder(LatentState()).build())
        assert tg.count == 1

    def test_graph_deepest(self):
        tg = TrajectoryGraph()
        tg.add(TrajectoryBuilder(LatentState()).build())
        deep = TrajectoryBuilder(LatentState())
        deep.step("a", LatentState())
        deep.step("b", LatentState())
        tg.add(deep.build())
        assert tg.deepest().depth == 2

    def test_graph_clear(self):
        tg = TrajectoryGraph()
        tg.add(TrajectoryBuilder(LatentState()).build())
        tg.clear()
        assert tg.count == 0

    def test_not_persisted_to_disk(self):
        assert not hasattr(TrajectoryGraph(), "_file")

    def test_no_belief_imports(self):
        import simulation.trajectory_graph as mod
        source = inspect.getsource(mod)
        assert "from memory.beliefs" not in source
        assert "import memory.beliefs" not in source


# BeamSearchPlanner

class TestBeamSearchPlanner:
    def test_returns_best_trajectory(self, tmp_path):
        vn = _trained_vn(tmp_path)
        result = BeamSearchPlanner(vn, beam_width=3, max_depth=2).search("goal", LatentState(confidence=0.4, risk=0.6), _actions)
        assert result.best_trajectory is not None

    def test_chooses_low_risk_action(self, tmp_path):
        vn = _trained_vn(tmp_path)
        result = BeamSearchPlanner(vn, beam_width=3, max_depth=2).search("goal", LatentState(confidence=0.4, risk=0.6), _actions)
        assert "DECOMPOSE" in result.best_trajectory.actions

    def test_depth_matches_max_depth(self, tmp_path):
        vn = ValueNetwork(tmp_path / "vn.json")
        result = BeamSearchPlanner(vn, beam_width=2, max_depth=3).search("goal", LatentState(), _actions)
        assert result.best_trajectory.depth == 3

    def test_runner_ups_count(self, tmp_path):
        vn = ValueNetwork(tmp_path / "vn.json")
        result = BeamSearchPlanner(vn, beam_width=3, max_depth=1).search("goal", LatentState(), _actions)
        assert len(result.runner_up_trajectories) == 2

    def test_result_epistemic_status_predicted(self, tmp_path):
        vn = ValueNetwork(tmp_path / "vn.json")
        result = BeamSearchPlanner(vn).search("goal", LatentState(), _actions)
        assert result.epistemic_status == EpistemicStatus.PREDICTED

    def test_no_actions_returns_none_trajectory(self, tmp_path):
        vn = ValueNetwork(tmp_path / "vn.json")
        result = BeamSearchPlanner(vn).search("goal", LatentState(), lambda s: [])
        # when no actions are generated, search returns the start beam (depth 0) not None
        assert result.best_trajectory is not None
        assert result.best_trajectory.depth == 0

    def test_beam_width_limits_runner_ups(self, tmp_path):
        vn = ValueNetwork(tmp_path / "vn.json")
        result = BeamSearchPlanner(vn, beam_width=2, max_depth=1).search("goal", LatentState(), _actions)
        assert len(result.runner_up_trajectories) <= 1

    def test_result_to_dict(self, tmp_path):
        vn = ValueNetwork(tmp_path / "vn.json")
        d = BeamSearchPlanner(vn).search("goal", LatentState(), _actions).to_dict()
        assert "best_trajectory" in d
        assert d["epistemic_status"] == "predicted"

    def test_clone_and_step_independence(self):
        b1 = TrajectoryBuilder(LatentState())
        b1.step("first", LatentState())
        b2 = _clone_and_step(b1, "branch_a", LatentState(), value_delta=0.1)
        b3 = _clone_and_step(b1, "branch_b", LatentState(), value_delta=0.2)
        assert b2.build().actions[-1] == "branch_a"
        assert b3.build().actions[-1] == "branch_b"


# SearchCritic

class TestSearchCritic:
    def _result(self, tmp_path, trained=True):
        vn = _trained_vn(tmp_path) if trained else ValueNetwork(tmp_path / "vn.json")
        return BeamSearchPlanner(vn, beam_width=3, max_depth=2).search("test goal", LatentState(confidence=0.4, risk=0.6), _actions), vn

    def test_returns_decision_explanation(self, tmp_path):
        result, vn = self._result(tmp_path)
        expl = SearchCritic(value_network=vn).explain(result)
        assert isinstance(expl, DecisionExplanation)
        assert expl.goal == "test goal"

    def test_chosen_actions_match_trajectory(self, tmp_path):
        result, vn = self._result(tmp_path)
        expl = SearchCritic(value_network=vn).explain(result)
        assert expl.chosen_actions == result.best_trajectory.actions

    def test_assumptions_always_present(self, tmp_path):
        result, vn = self._result(tmp_path)
        assert len(SearchCritic().explain(result).assumptions) >= 2

    def test_no_trajectory_produces_critical(self):
        empty = BeamSearchResult(goal="impossible", best_trajectory=None)
        expl = SearchCritic().explain(empty)
        assert expl.has_critical
        assert any(i.category == "no_trajectory_found" for i in expl.issues)

    def test_untrained_vn_flagged(self, tmp_path):
        vn = ValueNetwork(tmp_path / "vn.json")
        result = BeamSearchPlanner(vn, beam_width=2, max_depth=1).search("goal", LatentState(), _actions)
        expl = SearchCritic(value_network=vn).explain(result)
        assert any(i.category == "untrained_value_network" for i in expl.issues)

    def test_high_risk_swing_flagged(self):
        b = TrajectoryBuilder(LatentState(risk=0.1))
        b.step("risky", LatentState(risk=0.8))
        result = BeamSearchResult(goal="g", best_trajectory=b.build(), best_value=0.5)
        expl = SearchCritic().explain(result)
        assert any(i.category == "high_risk_swing" for i in expl.issues)

    def test_shallow_search_flagged(self):
        b = TrajectoryBuilder(LatentState())
        b.step("s", LatentState())
        result = BeamSearchResult(goal="g", best_trajectory=b.build(), best_value=0.5)
        expl = SearchCritic().explain(result)
        assert any(i.category == "shallow_search" for i in expl.issues)

    def test_to_dict_fields(self, tmp_path):
        result, vn = self._result(tmp_path)
        d = SearchCritic(value_network=vn).explain(result).to_dict()
        for key in ("chosen_actions", "why_chosen", "assumptions", "issues"):
            assert key in d


# CounterfactualScenarioEngine extension

class TestCounterfactualEngineExtension:
    @pytest.fixture
    def engine(self, tmp_path):
        return CounterfactualScenarioEngine(ValueNetwork(tmp_path / "vn.json"))

    def test_returns_trajectories(self, engine):
        alts = [CounterfactualAlternative(name="a", description="d", resulting_state=LatentState())]
        results = engine.explore_with_trajectories(LatentState(), alts)
        assert results[0].trajectory is not None

    def test_trajectory_depth_one(self, engine):
        alts = [CounterfactualAlternative(name="a", description="d", resulting_state=LatentState())]
        results = engine.explore_with_trajectories(LatentState(), alts)
        assert results[0].trajectory.depth == 1

    def test_trajectory_action_matches_name(self, engine):
        alts = [CounterfactualAlternative(name="switch_to_tot", description="d", resulting_state=LatentState())]
        results = engine.explore_with_trajectories(LatentState(), alts)
        assert results[0].trajectory.actions == ["switch_to_tot"]

    def test_trajectory_counterfactual_status(self, engine):
        alts = [CounterfactualAlternative(name="a", description="d", resulting_state=LatentState())]
        results = engine.explore_with_trajectories(LatentState(), alts)
        assert results[0].trajectory.epistemic_status == EpistemicStatus.COUNTERFACTUAL

    def test_old_explore_no_trajectory(self, engine):
        alts = [CounterfactualAlternative(name="a", description="d", resulting_state=LatentState())]
        assert engine.explore(LatentState(), alts)[0].trajectory is None

    def test_to_dict_includes_trajectory(self, engine):
        alts = [CounterfactualAlternative(name="a", description="d", resulting_state=LatentState())]
        d = engine.explore_with_trajectories(LatentState(), alts)[0].to_dict()
        assert d["trajectory"] is not None
        assert "nodes" in d["trajectory"]

    def test_safeguard_intact_after_extension(self):
        import causality.counterfactual_engine as mod
        source = inspect.getsource(mod)
        assert "import memory.beliefs" not in source
        assert "from memory.beliefs" not in source
        assert "from memory import beliefs" not in source


# PredictionEvaluator

class TestPredictionEvaluator:
    @pytest.fixture
    def store(self, tmp_path):
        return FutureMemoryStore(tmp_path / "future.json")

    def test_empty_returns_null_report(self, store):
        report = PredictionEvaluator(store).calibration_report()
        assert report["sample_count"] == 0
        assert report["brier_score"] is None

    def test_spec_example_predicted_08_actual_failure(self, store):
        p = store.predict("task_success", confidence=0.8)
        store.resolve(p.expected_state_id, actual_outcome=False)
        report = PredictionEvaluator(store).calibration_report()
        assert report["brier_score"] == pytest.approx(0.64)
        assert report["overconfidence_rate"] == 1.0

    def test_calibration_report_fields(self, store):
        p = store.predict("x", confidence=0.5)
        store.resolve(p.expected_state_id, actual_outcome=True)
        report = PredictionEvaluator(store).calibration_report()
        for key in ("sample_count", "brier_score", "expected_calibration_error", "overconfidence_rate", "underconfidence_rate", "buckets"):
            assert key in report

    def test_drift_none_when_too_few(self, store):
        for i in range(3):
            p = store.predict(f"x{i}", confidence=0.7)
            store.resolve(p.expected_state_id, actual_outcome=True)
        assert PredictionEvaluator(store).prediction_drift(min_samples_per_half=3) is None

    def test_drift_computed_with_enough_samples(self, store):
        for i in range(10):
            p = store.predict(f"x{i}", confidence=0.7)
            store.resolve(p.expected_state_id, actual_outcome=(i % 2 == 0))
        drift = PredictionEvaluator(store).prediction_drift(min_samples_per_half=3)
        assert drift is not None
        assert isinstance(drift.drift, float)

    def test_drift_to_dict(self, store):
        for i in range(8):
            p = store.predict(f"x{i}", confidence=0.6)
            store.resolve(p.expected_state_id, actual_outcome=(i % 2 == 0))
        drift = PredictionEvaluator(store).prediction_drift(min_samples_per_half=3)
        assert drift is not None
        d = drift.to_dict()
        assert "earlier_brier" in d
        assert "improving" in d

    def test_calibration_for_subject_empty(self, store):
        report = PredictionEvaluator(store).calibration_for_subject("unknown")
        assert report["sample_count"] == 0

    def test_calibration_for_subject_scoped(self, store):
        p1 = store.predict("subject_a", confidence=0.8)
        store.resolve(p1.expected_state_id, actual_outcome=True)
        store.predict("subject_b", confidence=0.3)
        assert PredictionEvaluator(store).calibration_for_subject("subject_a")["sample_count"] == 1

    def test_perfect_calibration_zero_brier(self, store):
        p = store.predict("x", confidence=1.0)
        store.resolve(p.expected_state_id, actual_outcome=True)
        report = PredictionEvaluator(store).calibration_report()
        assert report["brier_score"] == pytest.approx(0.0)


# Integration

class _FakeLLM:
    def model_name(self): return "fake-0.3.12"
    def generate(self, prompt): return "Fake reply."


@pytest.fixture(scope="module")
def tmp_memory_v12(tmp_path_factory):
    return tmp_path_factory.mktemp("memory_v12")


@pytest.fixture(scope="module")
def ctx_v12(tmp_memory_v12):
    from config import settings as _settings
    _settings.settings.memory.conversations_file = tmp_memory_v12 / "conversations.json"
    _settings.settings.memory.profile_file = tmp_memory_v12 / "profile.json"
    _settings.settings.memory.learning_state_file = tmp_memory_v12 / "learning_state.json"
    _settings.settings.embed.embeddings_file = tmp_memory_v12 / "embeddings.npy"
    _settings.settings.embed.embedding_ids_file = tmp_memory_v12 / "embedding_ids.json"
    from api.context import BlixContext
    ctx = BlixContext(tmp_memory_v12)
    ctx.llm = _FakeLLM()
    ctx.agent._llm = _FakeLLM()
    return ctx


@pytest.fixture(scope="module")
def client_v12(ctx_v12) -> Generator[TestClient, None, None]:
    from api.deps import set_context
    from api.routers.search import router as search_router
    app = FastAPI(title="Blix Test v0.3.12")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(search_router)

    @app.get("/health")
    def health(): return {"status": "ok"}

    set_context(ctx_v12)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


_BEAM_REQ = {
    "goal": "fix low confidence",
    "start_state": {"confidence": 0.4, "risk": 0.6},
    "candidate_actions": [
        {"action": "DIRECT", "resulting_state": {"confidence": 0.4, "risk": 0.9}},
        {"action": "TOT", "resulting_state": {"confidence": 0.5, "risk": 0.5}},
        {"action": "DECOMPOSE", "resulting_state": {"confidence": 0.5, "risk": 0.1}},
    ],
    "beam_width": 3, "max_depth": 2,
}


class TestBlixContextV0312Wiring:
    def test_components_present(self, ctx_v12):
        assert ctx_v12.trajectory_graph is not None
        assert ctx_v12.beam_search_planner is not None
        assert ctx_v12.search_critic is not None
        assert ctx_v12.prediction_evaluator is not None

    def test_dashboard_stats(self, ctx_v12):
        stats = ctx_v12.dashboard_stats()
        assert "active_trajectories" in stats
        assert "resolved_predictions" in stats

    def test_trajectory_graph_in_memory(self, ctx_v12):
        ctx_v12.trajectory_graph.add(TrajectoryBuilder(LatentState()).build())
        assert ctx_v12.trajectory_graph.count >= 1

    def test_beam_search_via_context(self, ctx_v12):
        result = ctx_v12.beam_search_planner.search("integration goal", LatentState(confidence=0.4, risk=0.6), _actions)
        assert result.best_trajectory is not None

    def test_search_critic_via_context(self, ctx_v12):
        result = ctx_v12.beam_search_planner.search("explain goal", LatentState(), _actions)
        expl = ctx_v12.search_critic.explain(result)
        assert expl.goal == "explain goal"

    def test_prediction_evaluator_wraps_future_memory(self, ctx_v12):
        p = ctx_v12.future_memory.predict("wiring_test", confidence=0.7)
        ctx_v12.future_memory.resolve(p.expected_state_id, actual_outcome=True)
        report = ctx_v12.prediction_evaluator.calibration_report()
        assert report["sample_count"] >= 1


class TestSearchAPI:
    def test_beam_search(self, client_v12):
        r = client_v12.post("/search/beam", json=_BEAM_REQ)
        assert r.status_code == 200
        d = r.json()
        assert "best_trajectory" in d
        assert d["epistemic_status"] == "predicted"

    def test_beam_search_requires_actions(self, client_v12):
        bad = {**_BEAM_REQ, "candidate_actions": []}
        assert client_v12.post("/search/beam", json=bad).status_code == 422

    def test_explain(self, client_v12):
        r = client_v12.post("/search/explain", json=_BEAM_REQ)
        assert r.status_code == 200
        d = r.json()
        for key in ("chosen_actions", "why_chosen", "assumptions", "issues"):
            assert key in d

    def test_counterfactual_trajectories(self, client_v12):
        r = client_v12.post("/search/counterfactual/trajectories", json={
            "current_state": {"confidence": 0.4, "risk": 0.6},
            "alternatives": [
                {"name": "tot", "description": "What if ToT?", "resulting_state": {"confidence": 0.6, "risk": 0.3}},
                {"name": "direct", "description": "Stayed", "resulting_state": {"confidence": 0.4, "risk": 0.6}},
            ],
        })
        assert r.status_code == 200
        d = r.json()
        assert d["total"] == 2
        assert all(s["trajectory"] is not None for s in d["scenarios"])
        assert all(s["epistemic_status"] == "counterfactual" for s in d["scenarios"])
        assert all(s["validated_causally"] is False for s in d["scenarios"])

    def test_counterfactual_trajectories_requires_alternatives(self, client_v12):
        r = client_v12.post("/search/counterfactual/trajectories", json={"alternatives": []})
        assert r.status_code == 422

    def test_prediction_calibration(self, client_v12):
        r = client_v12.get("/search/predictions/calibration")
        assert r.status_code == 200
        assert "sample_count" in r.json()

    def test_prediction_drift(self, client_v12):
        r = client_v12.get("/search/predictions/drift")
        assert r.status_code == 200
        assert "available" in r.json()

    def test_prediction_calibration_subject(self, client_v12, ctx_v12):
        p = ctx_v12.future_memory.predict("api_subj", confidence=0.6)
        ctx_v12.future_memory.resolve(p.expected_state_id, actual_outcome=True)
        r = client_v12.get("/search/predictions/calibration/api_subj")
        assert r.status_code == 200
        assert r.json()["subject"] == "api_subj"

    def test_health(self, client_v12):
        assert client_v12.get("/health").status_code == 200
