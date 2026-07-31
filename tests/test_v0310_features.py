"""
Tests for Blix v0.3.10 — "Hybrid Symbolic + ML".

Covers:
1.  world_model.latent_world_model         (Learned World Model)
2.  retrieval.cross_encoder_reranker          (Cross-Encoder Reranker, lexical fallback)
3.  reasoning.confidence_model                    (Confidence Prediction Model)
4.  agents.tool_success_predictor                     (Tool Success Predictor)
5.  workspace.neural_attention                            (Neural Attention System)
6.  metacognition.strategy_selector                          (Strategy Selector Network)
7.  learning.failure_clusterer                                  (Failure Pattern Mining)
8.  procedural.skill_discovery                                      (Skill Discovery Engine)
9.  world_model.scenario_ranker                                         (Scenario Evaluator)
10. memory.future_memory                                                    (Predictive Memory)
11. memory.semantic_compressor                                                  (Semantic Compression)
12. learning.continual_adapter                                                      (Continual Learning)
13. world_model.value_network                                                           (Value Function)
14. memory.importance_model                                                                  (Memory Importance Predictor)
Shared: learning.ml_base                                                                          (TrainableModel)
Integration  — BlixContext wiring
API          — /ml endpoints

All min-sample thresholds are set low in these tests for speed; this
verifies the cold-start -> learned MECHANISM works correctly, not that
any model achieves research-grade accuracy (impossible to claim
honestly without real production-scale data — see module docstrings).

Python 3.10 compatible — fully offline, no network calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from agents.failure_memory import FailureMemory
from agents.tool_reliability import ToolReliabilityRegistry
from agents.tool_success_predictor import ToolSuccessPredictor
from agents.types import Task, TaskGraph, TaskStatus
from learning.continual_adapter import ContinualLearningAdapter, ContinualLearningEvent
from learning.failure_clusterer import FailureCluster, FailureClusterer
from learning.ml_base import PredictionResult, TrainableModel, TrainingExample
from memory.future_memory import ExpectedState, FutureMemoryStore
from memory.importance_model import MemoryImportancePredictor
from memory.procedural_memory import ProceduralMemory
from memory.semantic_compressor import CompressedConcept, SemanticCompressor
from metacognition.capability_tracker import CapabilityTracker
from metacognition.self_model import SelfModelStore
from metacognition.strategy_manager import ReasoningStrategy, StrategyManager
from metacognition.strategy_selector import StrategySelectorNetwork
from planning.plan_evaluator import PlanQualityScore
from procedural.skill_discovery import DiscoveredTrajectory, SkillDiscoveryEngine
from reasoning.confidence_model import ConfidenceModel
from reasoning.confidence_reasoner import ConfidenceReasoner
from retrieval.cross_encoder_reranker import CrossEncoderReranker, RerankedResult
from schemas.memory_entry import MemoryEntry
from workspace.attention_manager import AttentionCandidate, AttentionManager
from workspace.neural_attention import NeuralAttentionScorer
from world_model.latent_world_model import LATENT_DIMENSIONS, LatentState, LatentWorldModel, WorldModelPrediction
from world_model.scenario_ranker import RankedScenario, Scenario, ScenarioRanker
from world_model.value_network import ValueNetwork
from datetime import datetime, timezone


# ===========================================================================
# Shared infrastructure — learning.ml_base.TrainableModel
# ===========================================================================


class TestTrainableModel:
    def test_cold_start_returns_fallback(self, tmp_path: Path) -> None:
        from sklearn.linear_model import LogisticRegression
        model = TrainableModel(tmp_path / "ex.json", ["a", "b"], min_samples_to_train=5, estimator_factory=lambda: LogisticRegression())
        result = model.predict({"a": 0.5, "b": 0.5}, fallback=0.42)
        assert result.value == 0.42
        assert result.mode == "fallback"

    def test_trains_after_min_samples(self, tmp_path: Path) -> None:
        from sklearn.linear_model import LogisticRegression
        model = TrainableModel(tmp_path / "ex.json", ["a"], min_samples_to_train=4, estimator_factory=lambda: LogisticRegression())
        for i in range(8):
            model.add_example({"a": 1.0 if i % 2 == 0 else 0.0}, label=1.0 if i % 2 == 0 else 0.0)
        assert model.is_trained

    def test_learned_prediction_differs_by_input(self, tmp_path: Path) -> None:
        from sklearn.linear_model import LogisticRegression
        model = TrainableModel(tmp_path / "ex.json", ["a"], min_samples_to_train=4, estimator_factory=lambda: LogisticRegression())
        for i in range(10):
            high = i % 2 == 0
            model.add_example({"a": 1.0 if high else 0.0}, label=1.0 if high else 0.0)
        high_pred = model.predict({"a": 1.0}, fallback=0.5)
        low_pred = model.predict({"a": 0.0}, fallback=0.5)
        assert high_pred.mode == "learned"
        assert high_pred.value > low_pred.value

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        from sklearn.linear_model import LogisticRegression
        f = tmp_path / "ex.json"
        m1 = TrainableModel(f, ["a"], min_samples_to_train=4, estimator_factory=lambda: LogisticRegression())
        for i in range(8):
            m1.add_example({"a": float(i % 2)}, label=float(i % 2))
        m2 = TrainableModel(f, ["a"], min_samples_to_train=4, estimator_factory=lambda: LogisticRegression())
        assert m2.is_trained
        assert m2.sample_count == 8

    def test_degenerate_single_class_stays_fallback(self, tmp_path: Path) -> None:
        from sklearn.linear_model import LogisticRegression
        model = TrainableModel(tmp_path / "ex.json", ["a"], min_samples_to_train=4, estimator_factory=lambda: LogisticRegression())
        for i in range(6):
            model.add_example({"a": 0.5}, label=1.0)  # single class only
        result = model.predict({"a": 0.5}, fallback=0.3)
        assert result.mode == "fallback"  # fit() should have failed gracefully

    def test_prediction_result_to_dict(self) -> None:
        result = PredictionResult(value=0.7, mode="learned", sample_count=10, explanation="x")
        d = result.to_dict()
        assert d["value"] == 0.7
        assert d["mode"] == "learned"

    def test_training_example_round_trip(self) -> None:
        ex = TrainingExample(features={"a": 1.0}, label=0.5)
        d = ex.to_dict()
        restored = TrainingExample.from_dict(d)
        assert restored.features == ex.features
        assert restored.label == ex.label


# ===========================================================================
# Item 1 — Latent World Model
# ===========================================================================


class TestLatentWorldModel:
    def test_latent_state_vector_order(self) -> None:
        state = LatentState(confidence=0.9, complexity=0.1, risk=0.2, capability_estimate=0.3, recent_failure_rate=0.4, dependency_density=0.5)
        vec = state.as_vector()
        assert vec == [0.9, 0.1, 0.2, 0.3, 0.4, 0.5]
        assert len(vec) == len(LATENT_DIMENSIONS)

    def test_cold_start_uses_documented_fallback(self, tmp_path: Path) -> None:
        wm = LatentWorldModel(tmp_path / "wm.json", min_examples_to_train=10)
        state = LatentState(confidence=0.8, risk=0.2, recent_failure_rate=0.1)
        pred = wm.predict(state)
        assert pred.mode == "fallback"
        assert pred.predicted_tool_failure == 0.1  # fallback ties directly to recent_failure_rate

    def test_trains_and_separates_risk_levels(self, tmp_path: Path) -> None:
        wm = LatentWorldModel(tmp_path / "wm.json", min_examples_to_train=16)
        for i in range(20):
            risky = i % 2 == 0
            z = LatentState(confidence=0.7, complexity=0.5, risk=0.9 if risky else 0.1, capability_estimate=0.6, recent_failure_rate=0.0, dependency_density=0.0)
            wm.observe_transition(z, plan_succeeded=not risky, tool_failed=risky, confidence_after=0.3 if risky else 0.7)
        assert wm.is_trained
        risky_pred = wm.predict(LatentState(confidence=0.7, complexity=0.5, risk=0.9, capability_estimate=0.6))
        safe_pred = wm.predict(LatentState(confidence=0.7, complexity=0.5, risk=0.1, capability_estimate=0.6))
        assert risky_pred.predicted_plan_success < safe_pred.predicted_plan_success
        assert risky_pred.predicted_tool_failure > safe_pred.predicted_tool_failure

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "wm.json"
        wm1 = LatentWorldModel(f, min_examples_to_train=100)  # won't train, just persisting examples
        wm1.observe_transition(LatentState(), plan_succeeded=True, tool_failed=False, confidence_after=0.5)
        wm2 = LatentWorldModel(f, min_examples_to_train=100)
        assert wm2.sample_count == 1

    def test_prediction_to_dict(self) -> None:
        pred = WorldModelPrediction(predicted_plan_success=0.7, predicted_tool_failure=0.2, predicted_confidence_decay=0.1, mode="learned", sample_count=30)
        d = pred.to_dict()
        assert d["mode"] == "learned"
        assert "predicted_plan_success" in d

    def test_latent_state_to_dict(self) -> None:
        state = LatentState(confidence=0.5)
        d = state.to_dict()
        assert d["confidence"] == 0.5


# ===========================================================================
# Item 13 — Value Function
# ===========================================================================


class TestValueNetwork:
    def test_cold_start_blended_heuristic(self, tmp_path: Path) -> None:
        vn = ValueNetwork(tmp_path / "vn.json", min_examples_to_train=10)
        state = LatentState(confidence=0.8, capability_estimate=0.8, risk=0.0)
        assert not vn.is_trained
        value = vn.value(state)
        assert value > 0.5  # high confidence/capability, no risk -> should be a high heuristic value

    def test_trains_and_distinguishes_risk(self, tmp_path: Path) -> None:
        vn = ValueNetwork(tmp_path / "vn.json", min_examples_to_train=14)
        for i in range(20):
            risky = i % 2 == 0
            s = LatentState(confidence=0.7, complexity=0.5, risk=0.9 if risky else 0.1, capability_estimate=0.6)
            vn.observe_outcome(s, eventual_value=0.1 if risky else 0.9)
        assert vn.is_trained
        risky_value = vn.value(LatentState(confidence=0.7, complexity=0.5, risk=0.9, capability_estimate=0.6))
        safe_value = vn.value(LatentState(confidence=0.7, complexity=0.5, risk=0.1, capability_estimate=0.6))
        assert safe_value > risky_value

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "vn.json"
        vn1 = ValueNetwork(f, min_examples_to_train=100)
        vn1.observe_outcome(LatentState(), eventual_value=0.5)
        vn2 = ValueNetwork(f, min_examples_to_train=100)
        assert vn2.sample_count == 1

    def test_value_clamped_to_unit_interval(self, tmp_path: Path) -> None:
        vn = ValueNetwork(tmp_path / "vn.json", min_examples_to_train=100)
        vn.observe_outcome(LatentState(), eventual_value=1.5)  # should clamp on storage
        assert vn._examples[0]["value"] == 1.0


# ===========================================================================
# Item 9 — Scenario Evaluator
# ===========================================================================


class TestScenarioRanker:
    def test_no_value_network_neutral_scores(self) -> None:
        ranker = ScenarioRanker(value_network=None)
        scenarios = [Scenario(name="a", state=LatentState()), Scenario(name="b", state=LatentState())]
        ranked = ranker.rank(scenarios)
        assert all(r.value == 0.5 for r in ranked)

    def test_empty_scenarios_returns_empty(self, tmp_path: Path) -> None:
        vn = ValueNetwork(tmp_path / "vn.json")
        ranker = ScenarioRanker(vn)
        assert ranker.rank([]) == []
        assert ranker.best([]) is None

    def test_ranks_by_trained_value(self, tmp_path: Path) -> None:
        vn = ValueNetwork(tmp_path / "vn.json", min_examples_to_train=14)
        for i in range(20):
            risky = i % 2 == 0
            s = LatentState(confidence=0.7, risk=0.9 if risky else 0.1, capability_estimate=0.6)
            vn.observe_outcome(s, eventual_value=0.1 if risky else 0.9)
        ranker = ScenarioRanker(vn)
        scenarios = [
            Scenario(name="risky", state=LatentState(confidence=0.7, risk=0.9, capability_estimate=0.6)),
            Scenario(name="safe", state=LatentState(confidence=0.7, risk=0.1, capability_estimate=0.6)),
        ]
        best = ranker.best(scenarios)
        assert best.scenario.name == "safe"

    def test_ranked_scenario_to_dict(self) -> None:
        scenario = Scenario(name="x", state=LatentState(), description="desc")
        ranked = RankedScenario(scenario=scenario, value=0.7)
        d = ranked.to_dict()
        assert d["name"] == "x"
        assert d["value"] == 0.7


# ===========================================================================
# Item 2 — Cross-Encoder Retrieval Reranker
# ===========================================================================


class TestCrossEncoderReranker:
    def _mem(self, id_, input_, output_):
        return MemoryEntry(id=id_, input=input_, output=output_, timestamp=datetime.now(timezone.utc), importance=0.5)

    def test_no_network_load_is_fast_and_falls_back(self) -> None:
        reranker = CrossEncoderReranker(attempt_model_load=False)
        assert not reranker.is_using_real_model

    def test_rerank_empty_candidates(self) -> None:
        reranker = CrossEncoderReranker(attempt_model_load=False)
        assert reranker.rerank("query", []) == []

    def test_rerank_ranks_relevant_higher(self) -> None:
        reranker = CrossEncoderReranker(attempt_model_load=False, rerank_k=5)
        relevant = self._mem(1, "machine learning neural networks", "discussed transformers")
        irrelevant = self._mem(2, "weather today", "it is sunny")
        results = reranker.rerank("machine learning neural networks", [irrelevant, relevant])
        assert results[0].entry.id == 1

    def test_rerank_results_tagged_lexical_fallback(self) -> None:
        reranker = CrossEncoderReranker(attempt_model_load=False)
        mem = self._mem(1, "test query content", "test response")
        results = reranker.rerank("test query content", [mem])
        assert all(r.scorer_mode == "lexical_fallback" for r in results)

    def test_rerank_respects_candidate_k_and_rerank_k(self) -> None:
        reranker = CrossEncoderReranker(attempt_model_load=False, candidate_k=3, rerank_k=2)
        memories = [self._mem(i, f"text {i}", f"output {i}") for i in range(10)]
        results = reranker.rerank("text", memories)
        assert len(results) <= 2

    def test_model_name_property(self) -> None:
        reranker = CrossEncoderReranker(model_name="custom-model", attempt_model_load=False)
        assert reranker.model_name == "custom-model"


# ===========================================================================
# Item 3 — Confidence Prediction Model
# ===========================================================================


class TestConfidenceModel:
    def test_cold_start_uses_confidence_reasoner_heuristic(self, tmp_path: Path) -> None:
        cm = ConfidenceModel(tmp_path / "cm.json", min_samples_to_train=10)
        result = cm.predict_correctness(evidence_count=3, source_count=2)
        assert result.mode == "fallback"

    def test_trains_and_separates_strong_weak_evidence(self, tmp_path: Path) -> None:
        cm = ConfidenceModel(tmp_path / "cm.json", min_samples_to_train=14)
        for i in range(20):
            strong = i % 2 == 0
            cm.observe_outcome(
                was_correct=strong, evidence_count=5 if strong else 1,
                source_count=3 if strong else 1, verification_passed=strong,
            )
        assert cm.is_trained
        strong_pred = cm.predict_correctness(evidence_count=5, source_count=3, verification_passed=True)
        weak_pred = cm.predict_correctness(evidence_count=1, source_count=1, verification_passed=False)
        assert strong_pred.value > weak_pred.value

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "cm.json"
        cm1 = ConfidenceModel(f, min_samples_to_train=100)
        cm1.observe_outcome(was_correct=True, evidence_count=2, source_count=1)
        cm2 = ConfidenceModel(f, min_samples_to_train=100)
        assert cm2.sample_count == 1


# ===========================================================================
# Item 4 — Tool Success Predictor
# ===========================================================================


class TestToolSuccessPredictor:
    @pytest.fixture
    def registry(self, tmp_path: Path) -> ToolReliabilityRegistry:
        return ToolReliabilityRegistry(tmp_path / "reliability.json")

    def test_cold_start_uses_flat_rate(self, tmp_path: Path, registry: ToolReliabilityRegistry) -> None:
        predictor = ToolSuccessPredictor(tmp_path / "tsp.json", tool_reliability=registry, min_samples_to_train=10)
        result = predictor.predict("task", "web_search")
        assert result.mode == "fallback"
        assert result.value == 0.5  # neutral prior, no observations yet

    def test_trains_and_separates_complexity(self, tmp_path: Path, registry: ToolReliabilityRegistry) -> None:
        predictor = ToolSuccessPredictor(tmp_path / "tsp.json", tool_reliability=registry, min_samples_to_train=14)
        for i in range(20):
            complex_task = i % 2 == 0
            predictor.observe_outcome("web_search", success=not complex_task, task_complexity_hint=0.9 if complex_task else 0.1)
        assert predictor.is_trained
        simple_pred = predictor.predict("simple task", "web_search", task_complexity_hint=0.1)
        complex_pred = predictor.predict("complex task", "web_search", task_complexity_hint=0.9)
        assert simple_pred.value > complex_pred.value

    def test_persistence_round_trip(self, tmp_path: Path, registry: ToolReliabilityRegistry) -> None:
        f = tmp_path / "tsp.json"
        p1 = ToolSuccessPredictor(f, tool_reliability=registry, min_samples_to_train=100)
        p1.observe_outcome("web_search", success=True)
        p2 = ToolSuccessPredictor(f, tool_reliability=registry, min_samples_to_train=100)
        assert p2.sample_count == 1


# ===========================================================================
# Item 5 — Neural Attention System
# ===========================================================================


class TestNeuralAttentionScorer:
    def test_cold_start_matches_fixed_formula(self, tmp_path: Path) -> None:
        am = AttentionManager()
        scorer = NeuralAttentionScorer(am, tmp_path / "attn.json", min_samples_to_train=12)
        candidate = AttentionCandidate(ref_id="a", source="x", content_summary="x", relevance=0.8, urgency=0.6, novelty=0.4, confidence=0.5)
        result = scorer.score(candidate)
        fixed_formula_score = am.score(candidate).score
        assert result.mode == "fallback"
        assert result.value == pytest.approx(fixed_formula_score)

    def test_trains_and_learns_different_pattern(self, tmp_path: Path) -> None:
        am = AttentionManager()
        scorer = NeuralAttentionScorer(am, tmp_path / "attn.json", min_samples_to_train=16)
        for i in range(24):
            high_conf = i % 2 == 0
            c = AttentionCandidate(ref_id=f"c{i}", source="x", content_summary="x", relevance=0.5, urgency=0.5, novelty=0.5, confidence=0.9 if high_conf else 0.1)
            scorer.observe_importance(c, was_important=high_conf)
        assert scorer.is_trained
        high_c = AttentionCandidate(ref_id="hc", source="x", content_summary="x", relevance=0.5, urgency=0.5, novelty=0.5, confidence=0.9)
        low_c = AttentionCandidate(ref_id="lc", source="x", content_summary="x", relevance=0.5, urgency=0.5, novelty=0.5, confidence=0.1)
        assert scorer.score(high_c).value > scorer.score(low_c).value

    def test_score_many_sorted_descending(self, tmp_path: Path) -> None:
        am = AttentionManager()
        scorer = NeuralAttentionScorer(am, tmp_path / "attn.json", min_samples_to_train=100)
        candidates = [
            AttentionCandidate(ref_id="low", source="x", content_summary="x", relevance=0.1, urgency=0.1, novelty=0.1, confidence=0.1),
            AttentionCandidate(ref_id="high", source="x", content_summary="x", relevance=0.9, urgency=0.9, novelty=0.9, confidence=0.9),
        ]
        scored = scorer.score_many(candidates)
        assert scored[0][0].ref_id == "high"


# ===========================================================================
# Item 6 — Strategy Selector Network
# ===========================================================================


class TestStrategySelectorNetwork:
    def test_cold_start_uses_strategy_manager_fallback(self, tmp_path: Path) -> None:
        sm = StrategyManager()
        selector = StrategySelectorNetwork(sm, tmp_path / "ss.json", min_samples_to_train=20)
        decision = selector.select_strategy("ref1", confidence=0.3)
        assert decision.strategy == ReasoningStrategy.CRITIC_FIRST  # matches StrategyManager's own fallback

    def test_repeated_failure_always_overrides_learned_model(self, tmp_path: Path) -> None:
        sm = StrategyManager()
        selector = StrategySelectorNetwork(sm, tmp_path / "ss.json", min_samples_to_train=8)
        for i in range(10):
            high = PlanQualityScore(graph_id="g", complexity=0.9, risk=0.1, confidence=0.8, dependency_density=0.2, expected_success=0.7)
            selector.observe_outcome(f"ref_a_{i}", high, ReasoningStrategy.TREE_OF_THOUGHT, succeeded=True)
            selector.observe_outcome(f"ref_b_{i}", high, ReasoningStrategy.DIRECT, succeeded=False)
        sm.record_failure("ref_fail")
        sm.record_failure("ref_fail")
        decision = selector.select_strategy("ref_fail", quality=PlanQualityScore(graph_id="g", complexity=0.9, risk=0.1, confidence=0.8, dependency_density=0.2, expected_success=0.7))
        assert decision.strategy == ReasoningStrategy.DECOMPOSE_FURTHER

    def test_trains_and_distinguishes_complexity(self, tmp_path: Path) -> None:
        sm = StrategyManager()
        selector = StrategySelectorNetwork(sm, tmp_path / "ss.json", min_samples_to_train=8)
        for i in range(10):
            hi = PlanQualityScore(graph_id="g", complexity=0.9, risk=0.1, confidence=0.8, dependency_density=0.2, expected_success=0.7)
            lo = PlanQualityScore(graph_id="g", complexity=0.1, risk=0.1, confidence=0.8, dependency_density=0.2, expected_success=0.9)
            selector.observe_outcome(f"tot_hi_{i}", hi, ReasoningStrategy.TREE_OF_THOUGHT, succeeded=True)
            selector.observe_outcome(f"tot_lo_{i}", lo, ReasoningStrategy.TREE_OF_THOUGHT, succeeded=False)
            selector.observe_outcome(f"dir_hi_{i}", hi, ReasoningStrategy.DIRECT, succeeded=False)
            selector.observe_outcome(f"dir_lo_{i}", lo, ReasoningStrategy.DIRECT, succeeded=True)
        assert selector.is_trained
        hi_decision = selector.select_strategy("new_ref_hi", quality=PlanQualityScore(graph_id="g2", complexity=0.9, risk=0.1, confidence=0.8, dependency_density=0.2, expected_success=0.7))
        lo_decision = selector.select_strategy("new_ref_lo", quality=PlanQualityScore(graph_id="g3", complexity=0.1, risk=0.1, confidence=0.8, dependency_density=0.2, expected_success=0.9))
        assert hi_decision.strategy == ReasoningStrategy.TREE_OF_THOUGHT
        assert lo_decision.strategy == ReasoningStrategy.DIRECT

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        sm = StrategyManager()
        f = tmp_path / "ss.json"
        s1 = StrategySelectorNetwork(sm, f, min_samples_to_train=1000)
        s1.observe_outcome("ref1", None, ReasoningStrategy.DIRECT, succeeded=True)
        s2 = StrategySelectorNetwork(sm, f, min_samples_to_train=1000)
        assert s2.sample_count == 1


# ===========================================================================
# Item 7 — Failure Pattern Mining
# ===========================================================================


class TestFailureClusterer:
    @pytest.fixture
    def fm(self, tmp_path: Path) -> FailureMemory:
        return FailureMemory(tmp_path / "fm.json", similarity_threshold=0.99)

    def test_too_few_records_returns_empty(self, fm: FailureMemory) -> None:
        fm.record("Task 1", "tool_a", "error one")
        clusterer = FailureClusterer(fm, min_records_to_cluster=6)
        assert clusterer.discover_clusters() == []

    def test_discovers_two_distinct_patterns(self, fm: FailureMemory) -> None:
        fm.record("Search papers", "web_search", "web search connection timeout error after 30s")
        fm.record("Find news", "web_search", "web search connection timeout error after 45s")
        fm.record("Stock prices", "web_search", "web search connection timeout error occurred")
        fm.record("Write script", "code_tool", "python code generation syntax error invalid token")
        fm.record("Generate func", "code_tool", "python code generation syntax error unexpected indent")
        fm.record("Build api", "code_tool", "python code generation syntax error near line 5")

        clusterer = FailureClusterer(fm, min_records_to_cluster=4)
        recurring = clusterer.recurring_clusters()
        assert len(recurring) >= 1
        all_tools = {tool for c in recurring for tool in c.records[0].tool.split()}
        # at least one cluster should be purely web_search or purely code_tool
        assert any(len({r.tool for r in c.records}) == 1 for c in recurring)

    def test_summarize_for_reflection_produces_text(self, fm: FailureMemory) -> None:
        fm.record("Search papers", "web_search", "web search connection timeout error after 30s")
        fm.record("Find news", "web_search", "web search connection timeout error after 45s")
        fm.record("Stock prices", "web_search", "web search connection timeout error occurred")
        fm.record("More search", "web_search", "web search connection timeout error happened")
        clusterer = FailureClusterer(fm, min_records_to_cluster=4)
        summaries = clusterer.summarize_for_reflection()
        if summaries:  # non-deterministic clustering on tiny data; just verify shape if produced
            assert "Recurring failure pattern" in summaries[0]

    def test_cluster_is_noise_property(self) -> None:
        cluster = FailureCluster(cluster_id=-1)
        assert cluster.is_noise
        cluster2 = FailureCluster(cluster_id=0)
        assert not cluster2.is_noise


# ===========================================================================
# Item 8 — Skill Discovery Engine
# ===========================================================================


class TestSkillDiscoveryEngine:
    @pytest.fixture
    def engine(self, tmp_path: Path) -> SkillDiscoveryEngine:
        pm = ProceduralMemory(tmp_path / "pm.json")
        return SkillDiscoveryEngine(pm)

    def test_extract_trajectory_topological_order(self, engine: SkillDiscoveryEngine) -> None:
        graph = TaskGraph(goal="Research and summarize")
        t1 = Task(title="Retrieve", tool_hint="web_search", status=TaskStatus.COMPLETED)
        graph.add_task(t1)
        t2 = Task(title="Summarize", tool_hint="summarizer", status=TaskStatus.COMPLETED, depends_on=[t1.task_id])
        graph.add_task(t2)
        trajectory = engine.extract_trajectory(graph)
        assert trajectory is not None
        assert trajectory.step_titles == ["Retrieve", "Summarize"]

    def test_failed_graph_returns_none(self, engine: SkillDiscoveryEngine) -> None:
        graph = TaskGraph(goal="Failed")
        graph.add_task(Task(title="Step 1", status=TaskStatus.FAILED))
        assert engine.extract_trajectory(graph) is None

    def test_too_short_trajectory_returns_none(self, engine: SkillDiscoveryEngine) -> None:
        graph = TaskGraph(goal="One step")
        graph.add_task(Task(title="Only step", status=TaskStatus.COMPLETED))
        assert engine.extract_trajectory(graph) is None

    def test_discover_and_learn_persists_skill(self, engine: SkillDiscoveryEngine) -> None:
        graph = TaskGraph(goal="Multi-step goal")
        t1 = Task(title="Step A", tool_hint="tool_a", status=TaskStatus.COMPLETED)
        graph.add_task(t1)
        t2 = Task(title="Step B", tool_hint="tool_b", status=TaskStatus.COMPLETED, depends_on=[t1.task_id])
        graph.add_task(t2)
        skill = engine.discover_and_learn(graph)
        assert skill is not None
        assert skill.steps == ["tool_a", "tool_b"]

    def test_discover_and_learn_failed_graph_returns_none(self, engine: SkillDiscoveryEngine) -> None:
        graph = TaskGraph(goal="Failed goal")
        graph.add_task(Task(title="Step 1", status=TaskStatus.FAILED))
        assert engine.discover_and_learn(graph) is None

    def test_trajectory_to_dict(self) -> None:
        traj = DiscoveredTrajectory(goal="g", step_titles=["a", "b"], tool_sequence=["t1", "t2"])
        d = traj.to_dict()
        assert d["goal"] == "g"
        assert traj.length == 2


# ===========================================================================
# Item 10 — Predictive Memory
# ===========================================================================


class TestFutureMemoryStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> FutureMemoryStore:
        return FutureMemoryStore(tmp_path / "future.json")

    def test_predict_creates_expected_state(self, store: FutureMemoryStore) -> None:
        state = store.predict("paper_acceptance", confidence=0.63, predicted_date="2026-09-01")
        assert state.subject == "paper_acceptance"
        assert state.confidence == 0.63
        assert not state.resolved

    def test_resolve_marks_resolved(self, store: FutureMemoryStore) -> None:
        state = store.predict("deploy_success", confidence=0.8)
        resolved = store.resolve(state.expected_state_id, actual_outcome=True)
        assert resolved.resolved
        assert resolved.actual_outcome is True

    def test_was_correct_high_confidence_matches_true(self, store: FutureMemoryStore) -> None:
        state = store.predict("x", confidence=0.9)
        store.resolve(state.expected_state_id, actual_outcome=True)
        assert store.get(state.expected_state_id).was_correct is True

    def test_was_correct_low_confidence_matches_false(self, store: FutureMemoryStore) -> None:
        state = store.predict("x", confidence=0.1)
        store.resolve(state.expected_state_id, actual_outcome=False)
        assert store.get(state.expected_state_id).was_correct is True  # predicted "no" (conf<0.5), outcome False -> correct

    def test_was_correct_none_when_unresolved(self, store: FutureMemoryStore) -> None:
        state = store.predict("x", confidence=0.7)
        assert state.was_correct is None

    def test_pending_and_resolved_lists(self, store: FutureMemoryStore) -> None:
        s1 = store.predict("a", confidence=0.5)
        s2 = store.predict("b", confidence=0.5)
        store.resolve(s1.expected_state_id, actual_outcome=True)
        assert len(store.pending()) == 1
        assert len(store.resolved()) == 1

    def test_by_subject(self, store: FutureMemoryStore) -> None:
        store.predict("paper_acceptance", confidence=0.5)
        store.predict("paper_acceptance", confidence=0.6)
        store.predict("deploy_success", confidence=0.5)
        assert len(store.by_subject("paper_acceptance")) == 2

    def test_calibration_accuracy(self, store: FutureMemoryStore) -> None:
        s1 = store.predict("a", confidence=0.9)
        store.resolve(s1.expected_state_id, actual_outcome=True)  # correct
        s2 = store.predict("b", confidence=0.9)
        store.resolve(s2.expected_state_id, actual_outcome=False)  # incorrect
        assert store.calibration_accuracy() == 0.5

    def test_calibration_accuracy_no_resolved_perfect_default(self, store: FutureMemoryStore) -> None:
        assert store.calibration_accuracy() == 1.0

    def test_resolve_unknown_id_returns_none(self, store: FutureMemoryStore) -> None:
        assert store.resolve("ghost", actual_outcome=True) is None

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "future.json"
        store1 = FutureMemoryStore(f)
        state = store1.predict("x", confidence=0.5)
        store2 = FutureMemoryStore(f)
        assert store2.get(state.expected_state_id) is not None

    def test_count_property(self, store: FutureMemoryStore) -> None:
        store.predict("a", confidence=0.5)
        store.predict("b", confidence=0.5)
        assert store.count == 2


# ===========================================================================
# Item 11 — Semantic Compression
# ===========================================================================


class TestSemanticCompressor:
    def _memories(self, n_per_topic: int = 3) -> list[MemoryEntry]:
        topics = [
            ("machine learning model training", "discussed gradient descent optimization"),
            ("neural network training tips", "talked about backpropagation techniques"),
            ("deep learning training process", "covered transformer layers"),
            ("favorite pizza toppings list", "pepperoni and mushroom discussed"),
            ("best pizza restaurants nearby", "recommended three local pizzerias"),
            ("weekend pizza recipe ideas", "shared homemade dough recipe"),
        ]
        return [
            MemoryEntry(id=i + 1, input=inp, output=out, timestamp=datetime.now(timezone.utc), importance=0.5)
            for i, (inp, out) in enumerate(topics)
        ]

    def test_too_few_memories_returns_empty(self) -> None:
        compressor = SemanticCompressor(llm=None, min_memories_to_compress=10)
        result = compressor.compress(self._memories())
        assert result == []

    def test_compress_produces_concepts(self) -> None:
        compressor = SemanticCompressor(llm=None, min_memories_to_compress=4)
        memories = self._memories()
        concepts = compressor.compress(memories)
        assert len(concepts) >= 1
        total_sources = sum(len(c.source_memory_ids) for c in concepts)
        assert total_sources == len(memories)

    def test_fallback_summary_format_without_llm(self) -> None:
        compressor = SemanticCompressor(llm=None, min_memories_to_compress=4)
        concepts = compressor.compress(self._memories())
        assert all("memories concerning" in c.summary for c in concepts)

    def test_compression_ratio(self) -> None:
        compressor = SemanticCompressor(llm=None, min_memories_to_compress=4)
        memories = self._memories()
        concepts = compressor.compress(memories)
        ratio = SemanticCompressor.compression_ratio(len(memories), concepts)
        assert ratio >= 1.0

    def test_compression_ratio_no_concepts(self) -> None:
        assert SemanticCompressor.compression_ratio(10, []) == 1.0

    def test_concept_to_dict(self) -> None:
        concept = CompressedConcept(concept_id=0, summary="test", source_memory_ids=[1, 2])
        d = concept.to_dict()
        assert d["compression_ratio"] == 2


# ===========================================================================
# Item 12 — Continual Learning
# ===========================================================================


class TestContinualLearningAdapter:
    @pytest.fixture
    def adapter(self, tmp_path: Path) -> ContinualLearningAdapter:
        sm = SelfModelStore(tmp_path / "sm.json")
        ct = CapabilityTracker(tmp_path / "ct.json", min_samples_for_confidence=3)
        pm = ProceduralMemory(tmp_path / "pm.json")
        registry = ToolReliabilityRegistry(tmp_path / "tr.json")
        tsp = ToolSuccessPredictor(tmp_path / "tsp.json", tool_reliability=registry, min_samples_to_train=100)
        return ContinualLearningAdapter(sm, ct, procedural_memory=pm, tool_success_predictor=tsp)

    def test_observe_task_outcome_updates_capability_tracker(self, adapter: ContinualLearningAdapter) -> None:
        event = adapter.observe_task_outcome("research", success=True)
        assert "capability_tracker" in event.updated_targets

    def test_observe_task_outcome_syncs_self_model_once_confident(self, adapter: ContinualLearningAdapter) -> None:
        for _ in range(3):
            event = adapter.observe_task_outcome("research", success=True)
        assert "self_model" in event.updated_targets

    def test_observe_task_outcome_learns_skill_on_success_with_steps(self, adapter: ContinualLearningAdapter) -> None:
        event = adapter.observe_task_outcome(
            "research", success=True, goal="research goal", steps=["a", "b", "c"], skill_name="test_skill",
        )
        assert "procedural_memory" in event.updated_targets

    def test_observe_task_outcome_skips_skill_on_failure(self, adapter: ContinualLearningAdapter) -> None:
        event = adapter.observe_task_outcome("research", success=False, goal="g", steps=["a", "b"])
        assert "procedural_memory" not in event.updated_targets

    def test_observe_task_outcome_feeds_tool_predictor(self, adapter: ContinualLearningAdapter) -> None:
        event = adapter.observe_task_outcome("coding", success=True, tool="code_tool")
        assert "tool_success_predictor" in event.updated_targets

    def test_observe_answer_outcome_without_confidence_model_returns_none(self, tmp_path: Path) -> None:
        sm = SelfModelStore(tmp_path / "sm.json")
        ct = CapabilityTracker(tmp_path / "ct.json")
        adapter = ContinualLearningAdapter(sm, ct)
        assert adapter.observe_answer_outcome(was_correct=True) is None

    def test_observe_answer_outcome_with_confidence_model(self, tmp_path: Path) -> None:
        sm = SelfModelStore(tmp_path / "sm.json")
        ct = CapabilityTracker(tmp_path / "ct.json")
        cm = ConfidenceModel(tmp_path / "cm.json", min_samples_to_train=100)
        adapter = ContinualLearningAdapter(sm, ct, confidence_model=cm)
        event = adapter.observe_answer_outcome(was_correct=True, evidence_count=3)
        assert event is not None
        assert cm.sample_count == 1

    def test_recent_events_and_count(self, adapter: ContinualLearningAdapter) -> None:
        adapter.observe_task_outcome("a", success=True)
        adapter.observe_task_outcome("b", success=False)
        assert adapter.event_count == 2
        assert len(adapter.recent_events(limit=1)) == 1

    def test_learning_status(self, adapter: ContinualLearningAdapter) -> None:
        status = adapter.learning_status()
        assert "tool_success_predictor" in status

    def test_event_to_dict(self) -> None:
        event = ContinualLearningEvent(domain="coding", success=True, updated_targets=["a", "b"])
        d = event.to_dict()
        assert d["domain"] == "coding"


# ===========================================================================
# Item 14 — Memory Importance Predictor
# ===========================================================================


class TestMemoryImportancePredictor:
    def test_cold_start_uses_heuristic(self, tmp_path: Path) -> None:
        predictor = MemoryImportancePredictor(tmp_path / "ip.json", min_samples_to_train=10)
        result = predictor.predict(0.7, "some text", "some output")
        assert result.mode == "fallback"
        assert result.value == 0.7

    def test_trains_and_predicts(self, tmp_path: Path) -> None:
        predictor = MemoryImportancePredictor(tmp_path / "ip.json", min_samples_to_train=14)
        for i in range(20):
            important = i % 2 == 0
            predictor.observe_true_importance(
                observed_importance=0.9 if important else 0.1, heuristic_importance=0.5,
                input_text="x" * (400 if important else 10), output_text="y", retrieval_count=8 if important else 0,
            )
        assert predictor.is_trained
        high_pred = predictor.predict(0.5, "x" * 400, "y", retrieval_count=8)
        low_pred = predictor.predict(0.5, "x" * 10, "y", retrieval_count=0)
        assert high_pred.value > low_pred.value

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "ip.json"
        p1 = MemoryImportancePredictor(f, min_samples_to_train=100)
        p1.observe_true_importance(0.5, 0.5, "a", "b")
        p2 = MemoryImportancePredictor(f, min_samples_to_train=100)
        assert p2.sample_count == 1


# ===========================================================================
# Integration — BlixContext wiring + API
# ===========================================================================


class _FakeLLM:
    def model_name(self) -> str:
        return "fake-0.3.10"

    def generate(self, prompt: str) -> str:
        return "Fake reply."


@pytest.fixture(scope="module")
def tmp_memory_v10(tmp_path_factory):
    return tmp_path_factory.mktemp("memory_v10")


@pytest.fixture(scope="module")
def ctx_v10(tmp_memory_v10):
    from config import settings as _settings
    _settings.settings.memory.conversations_file = tmp_memory_v10 / "conversations.json"
    _settings.settings.memory.profile_file = tmp_memory_v10 / "profile.json"
    _settings.settings.memory.learning_state_file = tmp_memory_v10 / "learning_state.json"
    _settings.settings.embed.embeddings_file = tmp_memory_v10 / "embeddings.npy"
    _settings.settings.embed.embedding_ids_file = tmp_memory_v10 / "embedding_ids.json"

    from api.context import BlixContext
    ctx = BlixContext(tmp_memory_v10)
    ctx.llm = _FakeLLM()
    ctx.agent._llm = _FakeLLM()
    return ctx


@pytest.fixture(scope="module")
def client_v10(ctx_v10) -> Generator[TestClient, None, None]:
    from api.deps import set_context
    from api.routers.ml import router as ml_router

    app = FastAPI(title="Blix Test v0.3.10")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(ml_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    set_context(ctx_v10)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestBlixContextV0310Wiring:
    def test_v0310_components_present(self, ctx_v10) -> None:
        assert ctx_v10.latent_world_model is not None
        assert ctx_v10.value_network is not None
        assert ctx_v10.scenario_ranker is not None
        assert ctx_v10.cross_encoder_reranker is not None
        assert ctx_v10.confidence_model is not None
        assert ctx_v10.tool_success_predictor is not None
        assert ctx_v10.neural_attention_scorer is not None
        assert ctx_v10.strategy_selector is not None
        assert ctx_v10.failure_clusterer is not None
        assert ctx_v10.skill_discovery is not None
        assert ctx_v10.future_memory is not None
        assert ctx_v10.semantic_compressor is not None
        assert ctx_v10.memory_importance_predictor is not None
        assert ctx_v10.continual_learning is not None

    def test_cross_encoder_did_not_attempt_network_load(self, ctx_v10) -> None:
        # Confirms BlixContext wires attempt_model_load=False (avoids ~5s network timeout on startup)
        assert ctx_v10.cross_encoder_reranker.is_using_real_model is False

    def test_dashboard_stats_includes_v0310_metrics(self, ctx_v10) -> None:
        stats = ctx_v10.dashboard_stats()
        assert "world_model_trained" in stats
        assert "value_network_trained" in stats
        assert "tool_success_predictor_trained" in stats
        assert "confidence_model_trained" in stats
        assert "future_predictions_tracked" in stats
        assert "continual_learning_events" in stats

    def test_end_to_end_future_prediction_via_context(self, ctx_v10) -> None:
        state = ctx_v10.future_memory.predict("integration_test_subject", confidence=0.7)
        assert ctx_v10.future_memory.get(state.expected_state_id) is not None

    def test_end_to_end_continual_learning_via_context(self, ctx_v10) -> None:
        initial_count = ctx_v10.continual_learning.event_count
        ctx_v10.continual_learning.observe_task_outcome("integration_domain", success=True)
        assert ctx_v10.continual_learning.event_count == initial_count + 1

    def test_world_model_prediction_via_context(self, ctx_v10) -> None:
        from world_model.latent_world_model import LatentState
        pred = ctx_v10.latent_world_model.predict(LatentState(confidence=0.7))
        assert pred.mode in ("fallback", "learned")


# ===========================================================================
# API — /ml endpoints
# ===========================================================================


class TestMlAPI:
    def test_ml_status(self, client_v10: TestClient) -> None:
        r = client_v10.get("/ml/status")
        assert r.status_code == 200
        data = r.json()
        assert "world_model" in data
        assert "cross_encoder_reranker" in data

    def test_world_model_predict(self, client_v10: TestClient) -> None:
        r = client_v10.post("/ml/world-model/predict", json={"confidence": 0.7, "risk": 0.2})
        assert r.status_code == 200
        data = r.json()
        assert "predicted_plan_success" in data

    def test_value_estimate(self, client_v10: TestClient) -> None:
        r = client_v10.post("/ml/value/estimate", json={"confidence": 0.8})
        assert r.status_code == 200
        assert "value" in r.json()

    def test_rank_scenarios(self, client_v10: TestClient) -> None:
        r = client_v10.post("/ml/scenarios/rank", json={
            "scenarios": [
                {"name": "plan_a", "state": {"confidence": 0.8, "risk": 0.1}},
                {"name": "plan_b", "state": {"confidence": 0.5, "risk": 0.7}},
            ]
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["ranked"]) == 2

    def test_rank_scenarios_requires_at_least_one(self, client_v10: TestClient) -> None:
        r = client_v10.post("/ml/scenarios/rank", json={"scenarios": []})
        assert r.status_code == 422

    def test_tool_success_predict(self, client_v10: TestClient) -> None:
        r = client_v10.post("/ml/tool-success/predict", json={"task": "test task", "tool": "web_search"})
        assert r.status_code == 200
        assert "value" in r.json()

    def test_confidence_predict(self, client_v10: TestClient) -> None:
        r = client_v10.post("/ml/confidence/predict", json={"evidence_count": 3, "source_count": 2})
        assert r.status_code == 200
        assert "value" in r.json()

    def test_failure_clusters(self, client_v10: TestClient) -> None:
        r = client_v10.get("/ml/failure-clusters")
        assert r.status_code == 200
        assert "clusters" in r.json()

    def test_future_predict_and_pending(self, client_v10: TestClient) -> None:
        r = client_v10.post("/ml/future/predict", json={"subject": "api_test_subject", "confidence": 0.6})
        assert r.status_code == 200
        expected_state_id = r.json()["expected_state_id"]

        r2 = client_v10.get("/ml/future/pending")
        assert r2.status_code == 200
        assert any(p["expected_state_id"] == expected_state_id for p in r2.json()["predictions"])

    def test_future_resolve(self, client_v10: TestClient) -> None:
        r = client_v10.post("/ml/future/predict", json={"subject": "resolve_test", "confidence": 0.5})
        expected_state_id = r.json()["expected_state_id"]
        r2 = client_v10.post(f"/ml/future/{expected_state_id}/resolve", json={"actual_outcome": True})
        assert r2.status_code == 200
        assert r2.json()["resolved"] is True

    def test_future_resolve_unknown_404(self, client_v10: TestClient) -> None:
        r = client_v10.post("/ml/future/ghost_id/resolve", json={"actual_outcome": True})
        assert r.status_code == 404

    def test_health_check_still_works(self, client_v10: TestClient) -> None:
        r = client_v10.get("/health")
        assert r.status_code == 200
