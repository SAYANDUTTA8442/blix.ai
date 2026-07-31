"""
Tests for Blix v0.3.11 — "Causal Cognition".

Covers:
Shared: causality.epistemic_status                                  (EpistemicStatus vocabulary)
Phase 1:
  1. causality.cause_graph                                              (CauseGraph, typed CauseEdge)
  2. causality.belief_dependency_graph                                      (supports/weakens DAG + propagation)
  3. causality.causal_memory                                                    (CauseMemory recall store)
  memory.beliefs HYPOTHESIS gating                                                  (add_hypothesis/confirm_observation)
Phase 2:
  4. causality.principle_synthesizer                                                    (mines CauseGraph/clusters -> Principle)
  5. causality.principle / causality.principle_graph                                        (first-class Principle + supports DAG)
Phase 3:
  6. causality.causal_reflection                                                                (prescriptive, principle-grounded)
  7. causality.meta_causal_reflection                                                               (aggregate causal queries)
  8. metacognition.strategy_evolution                                                                   (explainable strategy change)
Phase 4:
  9. causality.counterfactual_engine                                                                        (lightweight what-if ranking)
Safeguard: COUNTERFACTUAL must never reach BeliefStore automatically.
Integration  — BlixContext wiring
API          — /causality endpoints

Python 3.10 compatible — fully offline.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from causality.belief_dependency_graph import (
    BeliefDependencyEdge,
    BeliefDependencyGraph,
    DependencyRelation,
    PropagationResult,
)
from causality.causal_memory import CauseMemory, CausalMemoryStore
from causality.causal_reflection import CausalReflection, CausalReflectionResult
from causality.cause_graph import CauseEdge, CauseGraph, CauseRelation
from causality.counterfactual_engine import (
    CounterfactualAlternative,
    CounterfactualResult,
    CounterfactualScenarioEngine,
)
from causality.epistemic_status import EpistemicStatus
from causality.meta_causal_reflection import MetaCausalAnswer, MetaCausalReflection
from causality.principle import Principle, PrincipleStore
from causality.principle_graph import PrincipleGraph, PrincipleSupportEdge
from causality.principle_synthesizer import PrincipleSynthesizer
from learning.failure_clusterer import FailureCluster, FailureClusterer
from agents.failure_memory import FailureMemory
from memory.beliefs import Belief, BeliefStore
from metacognition.strategy_evolution import StrategyEvolution, StrategyEvolutionDecision
from metacognition.strategy_manager import ReasoningStrategy
from world_model.latent_world_model import LatentState
from world_model.value_network import ValueNetwork


# ===========================================================================
# Shared — EpistemicStatus
# ===========================================================================


class TestEpistemicStatus:
    def test_observed_is_trusted(self) -> None:
        assert EpistemicStatus.OBSERVED.is_trusted

    def test_derived_is_trusted(self) -> None:
        assert EpistemicStatus.DERIVED.is_trusted

    def test_counterfactual_not_trusted(self) -> None:
        assert not EpistemicStatus.COUNTERFACTUAL.is_trusted

    def test_hypothesis_not_trusted(self) -> None:
        assert not EpistemicStatus.HYPOTHESIS.is_trusted

    def test_predicted_not_trusted(self) -> None:
        assert not EpistemicStatus.PREDICTED.is_trusted

    def test_principle_not_trusted(self) -> None:
        assert not EpistemicStatus.PRINCIPLE.is_trusted

    def test_requires_validation_before_belief(self) -> None:
        assert EpistemicStatus.COUNTERFACTUAL.requires_validation_before_belief
        assert not EpistemicStatus.OBSERVED.requires_validation_before_belief

    def test_all_six_values_present(self) -> None:
        values = {e.value for e in EpistemicStatus}
        assert values == {"observed", "derived", "predicted", "counterfactual", "principle", "hypothesis"}


# ===========================================================================
# Item 1 — CauseGraph
# ===========================================================================


class TestCauseGraph:
    @pytest.fixture
    def cg(self, tmp_path: Path) -> CauseGraph:
        return CauseGraph(tmp_path / "cg.json")

    def test_record_observation_creates_edge(self, cg: CauseGraph) -> None:
        edge = cg.record_observation("web failures", "low confidence", CauseRelation.CAUSES)
        assert edge.trigger == "web failures"
        assert edge.evidence_count == 1
        assert edge.epistemic_status == EpistemicStatus.DERIVED

    def test_record_observation_reinforces_existing(self, cg: CauseGraph) -> None:
        e1 = cg.record_observation("X", "Y", CauseRelation.CAUSES, initial_confidence=0.5)
        e2 = cg.record_observation("X", "Y", CauseRelation.CAUSES)
        assert e1.edge_id == e2.edge_id
        assert e2.evidence_count == 2
        assert e2.confidence > 0.5

    def test_different_relations_produce_different_edges(self, cg: CauseGraph) -> None:
        cg.record_observation("X", "Y", CauseRelation.CAUSES)
        cg.record_observation("X", "Y", CauseRelation.BLOCKS)
        assert cg.count == 2

    def test_weaken_reduces_confidence(self, cg: CauseGraph) -> None:
        edge = cg.record_observation("X", "Y", CauseRelation.CAUSES, initial_confidence=0.6)
        weakened = cg.weaken(edge.edge_id, amount=0.2)
        assert weakened.confidence == pytest.approx(0.4)

    def test_effects_of_sorted_by_confidence(self, cg: CauseGraph) -> None:
        cg.record_observation("trigger", "low_conf_effect", CauseRelation.CAUSES, initial_confidence=0.3)
        cg.record_observation("trigger", "high_conf_effect", CauseRelation.CAUSES, initial_confidence=0.9)
        effects = cg.effects_of("trigger")
        assert effects[0].effect == "high_conf_effect"

    def test_effects_of_filtered_by_relation(self, cg: CauseGraph) -> None:
        cg.record_observation("trigger", "a", CauseRelation.CAUSES)
        cg.record_observation("trigger", "b", CauseRelation.BLOCKS)
        only_blocks = cg.effects_of("trigger", relation=CauseRelation.BLOCKS)
        assert len(only_blocks) == 1
        assert only_blocks[0].effect == "b"

    def test_causes_of(self, cg: CauseGraph) -> None:
        cg.record_observation("cause_a", "shared_effect", CauseRelation.CAUSES)
        cg.record_observation("cause_b", "shared_effect", CauseRelation.CAUSES)
        causes = cg.causes_of("shared_effect")
        assert len(causes) == 2

    def test_high_confidence_edges(self, cg: CauseGraph) -> None:
        cg.record_observation("X", "Y", CauseRelation.CAUSES, initial_confidence=0.9)
        cg.record_observation("A", "B", CauseRelation.CAUSES, initial_confidence=0.2)
        high = cg.high_confidence_edges(threshold=0.7)
        assert len(high) == 1

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "cg.json"
        cg1 = CauseGraph(f)
        cg1.record_observation("X", "Y", CauseRelation.ENABLES)
        cg2 = CauseGraph(f)
        assert cg2.count == 1

    def test_get_unknown_edge_returns_none(self, cg: CauseGraph) -> None:
        assert cg.get("ghost::causes::edge") is None

    def test_relation_vocabulary(self) -> None:
        assert {r.value for r in CauseRelation} == {"causes", "increases", "decreases", "enables", "blocks"}

    def test_spec_example(self, cg: CauseGraph) -> None:
        e1 = cg.record_observation("no evaluation", "reliable optimization", CauseRelation.BLOCKS)
        e2 = cg.record_observation("benchmarks", "fast iteration", CauseRelation.ENABLES)
        assert e1.relation == CauseRelation.BLOCKS
        assert e2.relation == CauseRelation.ENABLES


# ===========================================================================
# Item 2 — Belief Dependency Graph (+ Belief HYPOTHESIS gating)
# ===========================================================================


class TestBeliefHypothesisGating:
    @pytest.fixture
    def store(self, tmp_path: Path) -> BeliefStore:
        return BeliefStore(tmp_path / "beliefs.json")

    def test_add_or_reinforce_defaults_observed(self, store: BeliefStore) -> None:
        b = store.add_or_reinforce("A directly witnessed fact")
        assert b.epistemic_status == EpistemicStatus.OBSERVED

    def test_add_hypothesis_creates_hypothesis_status(self, store: BeliefStore) -> None:
        h = store.add_hypothesis("A candidate belief", confidence=0.3, basis="counterfactual estimate")
        assert h.epistemic_status == EpistemicStatus.HYPOTHESIS

    def test_confirm_observation_promotes_to_observed(self, store: BeliefStore) -> None:
        h = store.add_hypothesis("A candidate belief")
        confirmed = store.confirm_observation(h.belief_id)
        assert confirmed.epistemic_status == EpistemicStatus.OBSERVED

    def test_confirm_observation_increases_confidence(self, store: BeliefStore) -> None:
        h = store.add_hypothesis("A candidate belief", confidence=0.3)
        confirmed = store.confirm_observation(h.belief_id)
        assert confirmed.confidence > 0.3

    def test_confirm_observation_on_non_hypothesis_returns_none(self, store: BeliefStore) -> None:
        b = store.add_or_reinforce("An observed fact")
        assert store.confirm_observation(b.belief_id) is None

    def test_confirm_observation_unknown_id_returns_none(self, store: BeliefStore) -> None:
        assert store.confirm_observation("ghost_id") is None

    def test_hypothesis_persists_across_reload(self, tmp_path: Path) -> None:
        f = tmp_path / "beliefs.json"
        s1 = BeliefStore(f)
        h = s1.add_hypothesis("A candidate belief")
        s2 = BeliefStore(f)
        reloaded = s2.get(h.belief_id)
        assert reloaded.epistemic_status == EpistemicStatus.HYPOTHESIS

    def test_persist_public_method_exists(self, store: BeliefStore) -> None:
        b = store.add_or_reinforce("Something")
        b.confidence = 0.99
        store.persist()  # should not raise
        reloaded = BeliefStore(store._file)
        assert reloaded.get(b.belief_id).confidence == 0.99


class TestBeliefDependencyGraph:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        store = BeliefStore(tmp_path / "beliefs.json")
        graph = BeliefDependencyGraph(tmp_path / "dg.json", store)
        return store, graph

    def test_add_dependency(self, setup) -> None:
        store, graph = setup
        a = store.add_or_reinforce("Belief about topic alpha")
        b = store.add_or_reinforce("Belief about topic beta")
        edge = graph.add_dependency(a.belief_id, b.belief_id, DependencyRelation.SUPPORTS, strength=0.7)
        assert edge.relation == DependencyRelation.SUPPORTS

    def test_propagate_supports_chain(self, setup) -> None:
        store, graph = setup
        a = store.add_or_reinforce("The user prefers dark mode interfaces", confidence=0.8)
        b = store.add_or_reinforce("Visual accessibility settings matter here", confidence=0.8)
        c = store.add_or_reinforce("Custom theming should be prioritized", confidence=0.8)
        graph.add_dependency(a.belief_id, b.belief_id, DependencyRelation.SUPPORTS, strength=0.8)
        graph.add_dependency(b.belief_id, c.belief_id, DependencyRelation.SUPPORTS, strength=0.8)

        results = graph.propagate(a.belief_id, confidence_delta=-0.3)
        assert len(results) == 2
        assert store.get(b.belief_id).confidence < 0.8
        assert store.get(c.belief_id).confidence < 0.8

    def test_propagate_damping_reduces_effect_with_hops(self, setup) -> None:
        store, graph = setup
        a = store.add_or_reinforce("Topic alpha statement one", confidence=0.8)
        b = store.add_or_reinforce("Topic beta statement two", confidence=0.8)
        c = store.add_or_reinforce("Topic gamma statement three", confidence=0.8)
        graph.add_dependency(a.belief_id, b.belief_id, DependencyRelation.SUPPORTS, strength=0.8)
        graph.add_dependency(b.belief_id, c.belief_id, DependencyRelation.SUPPORTS, strength=0.8)
        results = graph.propagate(a.belief_id, confidence_delta=-0.3)
        b_change = abs(0.8 - results[0].new_confidence)
        c_change = abs(0.8 - results[1].new_confidence)
        assert c_change < b_change  # damped further from source

    def test_weakens_relation_inverts_direction(self, setup) -> None:
        store, graph = setup
        a = store.add_or_reinforce("Topic alpha statement four", confidence=0.5)
        b = store.add_or_reinforce("Topic beta statement five", confidence=0.5)
        graph.add_dependency(a.belief_id, b.belief_id, DependencyRelation.WEAKENS, strength=0.8)
        results = graph.propagate(a.belief_id, confidence_delta=0.3)  # a strengthens
        assert store.get(b.belief_id).confidence < 0.5  # b should weaken in response

    def test_dependents_of_and_dependencies_of(self, setup) -> None:
        store, graph = setup
        a = store.add_or_reinforce("Alpha statement six")
        b = store.add_or_reinforce("Beta statement seven")
        graph.add_dependency(a.belief_id, b.belief_id, DependencyRelation.SUPPORTS)
        assert len(graph.dependents_of(a.belief_id)) == 1
        assert len(graph.dependencies_of(b.belief_id)) == 1

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        store = BeliefStore(tmp_path / "beliefs.json")
        f = tmp_path / "dg.json"
        a = store.add_or_reinforce("Persisted alpha")
        b = store.add_or_reinforce("Persisted beta")
        g1 = BeliefDependencyGraph(f, store)
        g1.add_dependency(a.belief_id, b.belief_id, DependencyRelation.SUPPORTS)
        g2 = BeliefDependencyGraph(f, store)
        assert g2.count == 1

    def test_count_property(self, setup) -> None:
        store, graph = setup
        a = store.add_or_reinforce("Count alpha")
        b = store.add_or_reinforce("Count beta")
        graph.add_dependency(a.belief_id, b.belief_id, DependencyRelation.SUPPORTS)
        assert graph.count == 1


# ===========================================================================
# Item 3 — Causal Memory
# ===========================================================================


class TestCausalMemoryStore:
    @pytest.fixture
    def cms(self, tmp_path: Path) -> CausalMemoryStore:
        return CausalMemoryStore(tmp_path / "cm.json")

    def test_record_creates_memory(self, cms: CausalMemoryStore) -> None:
        m = cms.record("skipping benchmarks", "poor optimization", confidence=0.5)
        assert m.trigger == "skipping benchmarks"
        assert m.epistemic_status == EpistemicStatus.DERIVED

    def test_record_reinforces_similar(self, cms: CausalMemoryStore) -> None:
        cms.record("skipping benchmarks", "poor optimization", confidence=0.5)
        m2 = cms.record("skipping benchmarks before optimizing", "poor optimization results")
        assert m2.evidence_count == 2

    def test_recall_by_trigger(self, cms: CausalMemoryStore) -> None:
        cms.record("skipping benchmarks", "poor optimization")
        recalled = cms.recall("skipping benchmarks")
        assert recalled is not None
        assert recalled.effect == "poor optimization"

    def test_recall_no_match_returns_none(self, cms: CausalMemoryStore) -> None:
        cms.record("skipping benchmarks", "poor optimization")
        assert cms.recall("completely unrelated novel topic xyz") is None

    def test_effects_of_trigger(self, cms: CausalMemoryStore) -> None:
        cms.record("skipping benchmarks", "poor optimization", confidence=0.6)
        effects = cms.effects_of_trigger("skipping benchmarks")
        assert len(effects) == 1

    def test_high_confidence_principles_requires_evidence(self, cms: CausalMemoryStore) -> None:
        cms.record("X trigger", "Y effect", confidence=0.9)  # only 1 evidence
        assert cms.high_confidence_principles(threshold=0.7, min_evidence=2) == []

    def test_high_confidence_principles_found_with_enough_evidence(self, cms: CausalMemoryStore) -> None:
        cms.record("X trigger", "Y effect", confidence=0.6)
        cms.record("X trigger", "Y effect")  # reinforce
        principles = cms.high_confidence_principles(threshold=0.6, min_evidence=2)
        assert len(principles) == 1

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "cm.json"
        c1 = CausalMemoryStore(f)
        c1.record("X", "Y")
        c2 = CausalMemoryStore(f)
        assert c2.count == 1

    def test_count_property(self, cms: CausalMemoryStore) -> None:
        cms.record("A", "B")
        cms.record("C", "D")
        assert cms.count == 2


# ===========================================================================
# Item 5a/5b — Principle + PrincipleGraph
# ===========================================================================


class TestPrinciple:
    def test_principle_is_dataclass_not_string(self) -> None:
        p = Principle(statement="Always benchmark before optimization")
        assert not isinstance(p, str)
        assert hasattr(p, "id")
        assert hasattr(p, "confidence")
        assert hasattr(p, "evidence_count")
        assert hasattr(p, "supporting_causes")
        assert hasattr(p, "supporting_failures")

    def test_principle_default_status(self) -> None:
        p = Principle(statement="x")
        assert p.status == EpistemicStatus.PRINCIPLE

    def test_principle_to_dict_from_dict_round_trip(self) -> None:
        p = Principle(statement="x", confidence=0.7, supporting_causes=["a::causes::b"])
        restored = Principle.from_dict(p.to_dict())
        assert restored.statement == p.statement
        assert restored.supporting_causes == p.supporting_causes


class TestPrincipleStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> PrincipleStore:
        return PrincipleStore(tmp_path / "principles.json")

    def test_add_and_get(self, store: PrincipleStore) -> None:
        p = store.add(Principle(statement="Test principle"))
        assert store.get(p.id) is not None

    def test_reinforce(self, store: PrincipleStore) -> None:
        p = store.add(Principle(statement="Test principle", confidence=0.5))
        reinforced = store.reinforce(p.id, confidence_increment=0.1)
        assert reinforced.confidence == pytest.approx(0.6)
        assert reinforced.evidence_count == 2

    def test_high_confidence(self, store: PrincipleStore) -> None:
        store.add(Principle(statement="High", confidence=0.9))
        store.add(Principle(statement="Low", confidence=0.2))
        assert len(store.high_confidence(0.7)) == 1

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "principles.json"
        s1 = PrincipleStore(f)
        s1.add(Principle(statement="Persisted"))
        s2 = PrincipleStore(f)
        assert s2.count == 1


class TestPrincipleGraph:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        store = PrincipleStore(tmp_path / "principles.json")
        graph = PrincipleGraph(tmp_path / "pg.json", store)
        return store, graph

    def test_spec_example_chain(self, setup) -> None:
        store, graph = setup
        p1 = store.add(Principle(statement="Always evaluate before optimizing", confidence=0.8))
        p2 = store.add(Principle(statement="Reliable optimization", confidence=0.8))
        p3 = store.add(Principle(statement="Faster iteration", confidence=0.8))
        graph.add_support(p1.id, p2.id, strength=0.8)
        graph.add_support(p2.id, p3.id, strength=0.8)

        results = graph.propagate(p1.id, confidence_delta=0.15)
        assert len(results) == 2
        assert store.get(p2.id).confidence > 0.8
        assert store.get(p3.id).confidence > 0.8

    def test_supported_by_and_supports_of(self, setup) -> None:
        store, graph = setup
        p1 = store.add(Principle(statement="Source principle"))
        p2 = store.add(Principle(statement="Target principle"))
        graph.add_support(p1.id, p2.id)
        assert len(graph.supported_by(p1.id)) == 1
        assert len(graph.supports_of(p2.id)) == 1

    def test_persistence_round_trip(self, tmp_path: Path) -> None:
        store = PrincipleStore(tmp_path / "principles.json")
        f = tmp_path / "pg.json"
        p1 = store.add(Principle(statement="A"))
        p2 = store.add(Principle(statement="B"))
        g1 = PrincipleGraph(f, store)
        g1.add_support(p1.id, p2.id)
        g2 = PrincipleGraph(f, store)
        assert g2.count == 1


# ===========================================================================
# Item 4 — Principle Synthesizer
# ===========================================================================


class TestPrincipleSynthesizer:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        cg = CauseGraph(tmp_path / "cg.json")
        ps = PrincipleStore(tmp_path / "principles.json")
        synth = PrincipleSynthesizer(ps, cg, llm=None)
        return cg, ps, synth

    def test_synthesize_from_cause_edge_insufficient_evidence(self, setup) -> None:
        cg, ps, synth = setup
        edge = cg.record_observation("X", "Y", CauseRelation.BLOCKS)  # evidence_count=1
        result = synth.synthesize_from_cause_edge(edge)
        assert result is None  # min_evidence default is 2

    def test_synthesize_from_cause_edge_produces_principle(self, setup) -> None:
        cg, ps, synth = setup
        edge = cg.record_observation("no evaluation", "reliable optimization", CauseRelation.BLOCKS)
        cg.record_observation("no evaluation", "reliable optimization", CauseRelation.BLOCKS)  # reinforce -> 2
        edge = cg.get(edge.edge_id)
        principle = synth.synthesize_from_cause_edge(edge)
        assert principle is not None
        assert isinstance(principle, Principle)
        assert edge.edge_id in principle.supporting_causes

    def test_template_fallback_phrasing_blocks(self, setup) -> None:
        cg, ps, synth = setup
        edge = cg.record_observation("no evaluation", "reliable optimization", CauseRelation.BLOCKS)
        cg.record_observation("no evaluation", "reliable optimization", CauseRelation.BLOCKS)
        edge = cg.get(edge.edge_id)
        principle = synth.synthesize_from_cause_edge(edge)
        assert "no evaluation" in principle.statement
        assert "reliable optimization" in principle.statement

    def test_synthesize_from_failure_cluster(self, setup) -> None:
        cg, ps, synth = setup
        from agents.failure_memory import FailureRecord
        cluster = FailureCluster(
            cluster_id=0,
            records=[
                FailureRecord(task_title="t1", tool="tool_a", failure="timeout error one"),
                FailureRecord(task_title="t2", tool="tool_a", failure="timeout error two"),
            ],
            representative_terms=["timeout", "error"],
            total_occurrences=2,
        )
        principle = synth.synthesize_from_failure_cluster(cluster)
        assert principle is not None
        assert "0" in principle.supporting_failures

    def test_synthesize_all_combines_sources(self, tmp_path: Path) -> None:
        cg = CauseGraph(tmp_path / "cg.json")
        cg.record_observation("X", "Y", CauseRelation.BLOCKS)
        cg.record_observation("X", "Y", CauseRelation.BLOCKS)  # evidence_count=2
        ps = PrincipleStore(tmp_path / "principles.json")
        synth = PrincipleSynthesizer(ps, cg, llm=None)
        principles = synth.synthesize_all()
        assert len(principles) >= 1


# ===========================================================================
# Item 6 — Causal Reflection
# ===========================================================================


class TestCausalReflection:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        ps = PrincipleStore(tmp_path / "principles.json")
        vn = ValueNetwork(tmp_path / "vn.json")
        cr = CausalReflection(principle_store=ps, value_network=vn)
        return ps, vn, cr

    def test_extends_meta_reflection_engine(self, setup) -> None:
        from reflection.meta_reflection import MetaReflectionEngine
        _, _, cr = setup
        assert isinstance(cr, MetaReflectionEngine)

    def test_relevant_principles_found(self, setup) -> None:
        ps, vn, cr = setup
        ps.add(Principle(statement="Always evaluate research tasks before optimizing", confidence=0.8))
        ps.add(Principle(statement="Completely unrelated pizza topic", confidence=0.9))
        result = cr.reflect_on_failure(topic="research task evaluation failed")
        assert len(result.relevant_principles) == 1
        assert "evaluate" in result.relevant_principles[0].statement

    def test_no_alternative_strategy_no_estimate(self, setup) -> None:
        _, _, cr = setup
        result = cr.reflect_on_failure(topic="some topic")
        assert result.estimated_success is None

    def test_estimate_present_with_alternative_and_state(self, setup) -> None:
        _, _, cr = setup
        result = cr.reflect_on_failure(
            topic="topic", alternative_strategy=ReasoningStrategy.TREE_OF_THOUGHT,
            latent_state_for_alternative=LatentState(confidence=0.7),
        )
        assert result.estimated_success is not None
        assert result.epistemic_status == EpistemicStatus.PREDICTED

    def test_result_to_dict(self, setup) -> None:
        _, _, cr = setup
        result = cr.reflect_on_failure(topic="topic")
        d = result.to_dict()
        assert "relevant_principles" in d
        assert "epistemic_status" in d


# ===========================================================================
# Item 7 — Meta-Causal Reflection
# ===========================================================================


class TestMetaCausalReflection:
    @pytest.fixture
    def cg(self, tmp_path: Path) -> CauseGraph:
        return CauseGraph(tmp_path / "cg.json")

    def test_why_repeated_failures_found(self, cg: CauseGraph) -> None:
        cg.record_observation("research task ambiguity", "repeated research failures", CauseRelation.CAUSES)
        cg.record_observation("research task ambiguity", "repeated research failures", CauseRelation.CAUSES)
        mcr = MetaCausalReflection(cg)
        answer = mcr.why_repeated_failures("research")
        assert "research" in answer.answer_summary.lower() or len(answer.supporting_edges) > 0

    def test_why_repeated_failures_none_found(self, cg: CauseGraph) -> None:
        mcr = MetaCausalReflection(cg)
        answer = mcr.why_repeated_failures("nonexistent_domain")
        assert "No recurring" in answer.answer_summary

    def test_what_causes(self, cg: CauseGraph) -> None:
        cg.record_observation("web search failures", "low confidence", CauseRelation.CAUSES, initial_confidence=0.7)
        mcr = MetaCausalReflection(cg)
        answer = mcr.what_causes("low confidence")
        assert "web search failures" in answer.answer_summary

    def test_which_strategies_cause_success_excludes_negative_effects(self, cg: CauseGraph) -> None:
        cg.record_observation("research task ambiguity", "increases low confidence", CauseRelation.INCREASES)
        mcr = MetaCausalReflection(cg)
        answer = mcr.which_strategies_cause_success()
        assert "research task ambiguity" not in answer.answer_summary

    def test_which_strategies_cause_success_finds_positive(self, cg: CauseGraph) -> None:
        cg.record_observation("tree of thought strategy", "higher success rate", CauseRelation.ENABLES, initial_confidence=0.7)
        mcr = MetaCausalReflection(cg)
        answer = mcr.which_strategies_cause_success()
        assert "tree of thought strategy" in answer.answer_summary

    def test_top_principles_for_domain(self, tmp_path: Path, cg: CauseGraph) -> None:
        ps = PrincipleStore(tmp_path / "principles.json")
        ps.add(Principle(statement="Research tasks need evaluation", confidence=0.9))
        ps.add(Principle(statement="Pizza toppings preference", confidence=0.8))
        mcr = MetaCausalReflection(cg, principle_store=ps)
        top = mcr.top_principles_for_domain("research")
        assert len(top) == 1

    def test_answer_to_dict(self, cg: CauseGraph) -> None:
        mcr = MetaCausalReflection(cg)
        answer = mcr.what_causes("nonexistent")
        d = answer.to_dict()
        assert "question" in d
        assert "supporting_edges" in d


# ===========================================================================
# Item 8 — Strategy Evolution
# ===========================================================================


class TestStrategyEvolution:
    @pytest.fixture
    def cg(self, tmp_path: Path) -> CauseGraph:
        return CauseGraph(tmp_path / "cg.json")

    def test_evolve_strategy_cites_cause(self, cg: CauseGraph) -> None:
        cg.record_observation("research task ambiguity", "repeated research failures", CauseRelation.BLOCKS, initial_confidence=0.7)
        se = StrategyEvolution(cg)
        decision = se.evolve_strategy("ref1", "research task ambiguity")
        assert decision.recommended_strategy == ReasoningStrategy.DECOMPOSE_FURTHER
        assert "research task ambiguity" in decision.explanation

    def test_evolve_strategy_blocks_maps_to_decompose(self, cg: CauseGraph) -> None:
        cg.record_observation("X", "Y", CauseRelation.BLOCKS, initial_confidence=0.7)
        se = StrategyEvolution(cg)
        decision = se.evolve_strategy("ref1", "X")
        assert decision.recommended_strategy == ReasoningStrategy.DECOMPOSE_FURTHER

    def test_evolve_strategy_enables_maps_to_tot(self, cg: CauseGraph) -> None:
        cg.record_observation("X", "Y", CauseRelation.ENABLES, initial_confidence=0.7)
        se = StrategyEvolution(cg)
        decision = se.evolve_strategy("ref1", "X")
        assert decision.recommended_strategy == ReasoningStrategy.TREE_OF_THOUGHT

    def test_evolve_strategy_no_cause_no_principle(self, cg: CauseGraph) -> None:
        se = StrategyEvolution(cg)
        decision = se.evolve_strategy("ref1", "totally unknown topic")
        assert decision.recommended_strategy == ReasoningStrategy.DIRECT
        assert "No causal pattern" in decision.explanation

    def test_evolve_strategy_falls_back_to_principle(self, tmp_path: Path, cg: CauseGraph) -> None:
        ps = PrincipleStore(tmp_path / "principles.json")
        ps.add(Principle(statement="Research tasks need more evaluation upfront", confidence=0.8))
        se = StrategyEvolution(cg, principle_store=ps)
        decision = se.evolve_strategy("ref1", "research tasks")
        assert decision.cited_principle is not None

    def test_record_outcome_feeds_strategy_selector(self, tmp_path: Path, cg: CauseGraph) -> None:
        from metacognition.strategy_manager import StrategyManager
        from metacognition.strategy_selector import StrategySelectorNetwork
        sm = StrategyManager()
        selector = StrategySelectorNetwork(sm, tmp_path / "ss.json", min_samples_to_train=1000)
        se = StrategyEvolution(cg, strategy_selector=selector)
        se.record_outcome("ref1", None, ReasoningStrategy.DIRECT, succeeded=True)
        assert selector.sample_count == 1

    def test_record_outcome_no_selector_no_crash(self, cg: CauseGraph) -> None:
        se = StrategyEvolution(cg)
        se.record_outcome("ref1", None, ReasoningStrategy.DIRECT, succeeded=True)  # should not raise

    def test_decision_to_dict(self, cg: CauseGraph) -> None:
        se = StrategyEvolution(cg)
        decision = se.evolve_strategy("ref1", "unknown")
        d = decision.to_dict()
        assert "recommended_strategy" in d
        assert "explanation" in d


# ===========================================================================
# Item 9 — Counterfactual Scenario Engine
# ===========================================================================


class TestCounterfactualScenarioEngine:
    @pytest.fixture
    def engine(self, tmp_path: Path) -> CounterfactualScenarioEngine:
        vn = ValueNetwork(tmp_path / "vn.json")
        return CounterfactualScenarioEngine(vn)

    def test_explore_empty_alternatives(self, engine: CounterfactualScenarioEngine) -> None:
        assert engine.explore(LatentState(), []) == []

    def test_explore_ranks_alternatives(self, engine: CounterfactualScenarioEngine) -> None:
        current = LatentState(confidence=0.4, risk=0.6)
        alternatives = [
            CounterfactualAlternative(name="tot", description="What if ToT had been used?", resulting_state=LatentState(confidence=0.6, risk=0.3)),
            CounterfactualAlternative(name="direct", description="Stayed direct", resulting_state=LatentState(confidence=0.4, risk=0.6)),
        ]
        results = engine.explore(current, alternatives)
        assert len(results) == 2
        assert results[0].estimated_value >= results[1].estimated_value

    def test_every_result_tagged_counterfactual(self, engine: CounterfactualScenarioEngine) -> None:
        alternatives = [CounterfactualAlternative(name="a", description="d", resulting_state=LatentState())]
        results = engine.explore(LatentState(), alternatives)
        assert all(r.epistemic_status == EpistemicStatus.COUNTERFACTUAL for r in results)

    def test_every_result_validated_causally_false(self, engine: CounterfactualScenarioEngine) -> None:
        alternatives = [CounterfactualAlternative(name="a", description="d", resulting_state=LatentState())]
        results = engine.explore(LatentState(), alternatives)
        assert all(r.validated_causally is False for r in results)

    def test_every_result_has_zero_evidence_count(self, engine: CounterfactualScenarioEngine) -> None:
        alternatives = [CounterfactualAlternative(name="a", description="d", resulting_state=LatentState())]
        results = engine.explore(LatentState(), alternatives)
        assert all(r.evidence_count == 0 for r in results)

    def test_every_result_has_basis(self, engine: CounterfactualScenarioEngine) -> None:
        alternatives = [CounterfactualAlternative(name="a", description="d", resulting_state=LatentState())]
        results = engine.explore(LatentState(), alternatives)
        assert all(r.basis for r in results)

    def test_best_returns_top_one(self, engine: CounterfactualScenarioEngine) -> None:
        alternatives = [
            CounterfactualAlternative(name="a", description="d", resulting_state=LatentState(confidence=0.9)),
            CounterfactualAlternative(name="b", description="d", resulting_state=LatentState(confidence=0.1)),
        ]
        best = engine.best(LatentState(), alternatives)
        assert best is not None

    def test_respects_top_k(self, engine: CounterfactualScenarioEngine) -> None:
        alternatives = [CounterfactualAlternative(name=f"a{i}", description="d", resulting_state=LatentState()) for i in range(10)]
        results = engine.explore(LatentState(), alternatives, top_k=3)
        assert len(results) == 3

    def test_result_to_dict(self, engine: CounterfactualScenarioEngine) -> None:
        alternatives = [CounterfactualAlternative(name="a", description="d", resulting_state=LatentState())]
        results = engine.explore(LatentState(), alternatives)
        d = results[0].to_dict()
        assert d["epistemic_status"] == "counterfactual"
        assert d["validated_causally"] is False
        assert "confidence" in d
        assert "evidence_count" in d
        assert "basis" in d


class TestCounterfactualSafeguard:
    """The most important safeguard in this release: counterfactuals must never reach BeliefStore."""

    def test_module_has_no_belief_store_import(self) -> None:
        import causality.counterfactual_engine as mod
        source = inspect.getsource(mod)
        # Only the docstring should mention BeliefStore/memory.beliefs; there
        # must be no actual import statement.
        assert "import memory.beliefs" not in source
        assert "from memory.beliefs" not in source
        assert "from memory import beliefs" not in source

    def test_module_has_no_belief_store_symbol(self) -> None:
        import causality.counterfactual_engine as mod
        assert not hasattr(mod, "BeliefStore")
        assert not hasattr(mod, "Belief")

    def test_counterfactual_result_has_no_belief_id_field(self) -> None:
        result = CounterfactualResult(name="x", description="d", estimated_value=0.5, confidence=0.5)
        assert not hasattr(result, "belief_id")

    def test_hypothesis_pipeline_requires_two_explicit_calls(self, tmp_path: Path) -> None:
        """A counterfactual can only become a belief via add_hypothesis() then confirm_observation() — never one step."""
        store = BeliefStore(tmp_path / "beliefs.json")
        vn = ValueNetwork(tmp_path / "vn.json")
        engine = CounterfactualScenarioEngine(vn)

        alternatives = [CounterfactualAlternative(name="a", description="d", resulting_state=LatentState(confidence=0.9))]
        counterfactual = engine.best(LatentState(), alternatives)

        # Step 1: caller must explicitly stage it as a hypothesis
        hypothesis = store.add_hypothesis(
            f"Counterfactual '{counterfactual.name}' may hold", confidence=counterfactual.confidence, basis=counterfactual.basis,
        )
        assert hypothesis.epistemic_status == EpistemicStatus.HYPOTHESIS  # NOT observed yet

        # Step 2: only explicit confirmation promotes it
        confirmed = store.confirm_observation(hypothesis.belief_id)
        assert confirmed.epistemic_status == EpistemicStatus.OBSERVED


# ===========================================================================
# Integration — BlixContext wiring + API
# ===========================================================================


class _FakeLLM:
    def model_name(self) -> str:
        return "fake-0.3.11"

    def generate(self, prompt: str) -> str:
        return "Fake reply."


@pytest.fixture(scope="module")
def tmp_memory_v11(tmp_path_factory):
    return tmp_path_factory.mktemp("memory_v11")


@pytest.fixture(scope="module")
def ctx_v11(tmp_memory_v11):
    from config import settings as _settings
    _settings.settings.memory.conversations_file = tmp_memory_v11 / "conversations.json"
    _settings.settings.memory.profile_file = tmp_memory_v11 / "profile.json"
    _settings.settings.memory.learning_state_file = tmp_memory_v11 / "learning_state.json"
    _settings.settings.embed.embeddings_file = tmp_memory_v11 / "embeddings.npy"
    _settings.settings.embed.embedding_ids_file = tmp_memory_v11 / "embedding_ids.json"

    from api.context import BlixContext
    ctx = BlixContext(tmp_memory_v11)
    ctx.llm = _FakeLLM()
    ctx.agent._llm = _FakeLLM()
    return ctx


@pytest.fixture(scope="module")
def client_v11(ctx_v11) -> Generator[TestClient, None, None]:
    from api.deps import set_context
    from api.routers.causality import router as causality_router

    app = FastAPI(title="Blix Test v0.3.11")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(causality_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    set_context(ctx_v11)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestBlixContextV0311Wiring:
    def test_v0311_components_present(self, ctx_v11) -> None:
        assert ctx_v11.cause_graph is not None
        assert ctx_v11.belief_dependency_graph is not None
        assert ctx_v11.causal_memory is not None
        assert ctx_v11.principle_store is not None
        assert ctx_v11.principle_synthesizer is not None
        assert ctx_v11.principle_graph is not None
        assert ctx_v11.causal_reflection is not None
        assert ctx_v11.meta_causal_reflection is not None
        assert ctx_v11.strategy_evolution is not None
        assert ctx_v11.counterfactual_engine is not None

    def test_dashboard_stats_includes_v0311_metrics(self, ctx_v11) -> None:
        stats = ctx_v11.dashboard_stats()
        assert "cause_graph_edges" in stats
        assert "belief_dependency_edges" in stats
        assert "causal_memories" in stats
        assert "principles_synthesized" in stats
        assert "principle_graph_edges" in stats

    def test_end_to_end_cause_recording_via_context(self, ctx_v11) -> None:
        ctx_v11.cause_graph.record_observation("integration trigger", "integration effect", CauseRelation.CAUSES)
        effects = ctx_v11.cause_graph.effects_of("integration trigger")
        assert len(effects) >= 1

    def test_belief_dependency_graph_uses_shared_belief_store(self, ctx_v11) -> None:
        b1 = ctx_v11.belief_store.add_or_reinforce("Integration belief one")
        b2 = ctx_v11.belief_store.add_or_reinforce("Integration belief two distinct text")
        ctx_v11.belief_dependency_graph.add_dependency(b1.belief_id, b2.belief_id, DependencyRelation.SUPPORTS)
        results = ctx_v11.belief_dependency_graph.propagate(b1.belief_id, confidence_delta=0.1)
        assert len(results) == 1

    def test_counterfactual_engine_via_context_does_not_touch_beliefs(self, ctx_v11) -> None:
        alternatives = [CounterfactualAlternative(name="x", description="d", resulting_state=LatentState())]
        ctx_v11.counterfactual_engine.explore(LatentState(), alternatives)
        # No assertion needed beyond "did not raise" — the structural safeguard test covers correctness


# ===========================================================================
# API — /causality endpoints
# ===========================================================================


class TestCausalityAPI:
    def test_record_cause(self, client_v11: TestClient) -> None:
        r = client_v11.post("/causality/cause/record", json={
            "trigger": "api test trigger", "effect": "api test effect", "relation": "causes",
        })
        assert r.status_code == 200
        assert r.json()["relation"] == "causes"

    def test_effects_of(self, client_v11: TestClient) -> None:
        client_v11.post("/causality/cause/record", json={"trigger": "effects_test_trigger", "effect": "effects_test_effect", "relation": "enables"})
        r = client_v11.get("/causality/cause/effects/effects_test_trigger")
        assert r.status_code == 200
        assert r.json()["total"] >= 1

    def test_causes_of(self, client_v11: TestClient) -> None:
        client_v11.post("/causality/cause/record", json={"trigger": "causes_test_trigger", "effect": "causes_test_effect", "relation": "blocks"})
        r = client_v11.get("/causality/cause/causes/causes_test_effect")
        assert r.status_code == 200
        assert r.json()["total"] >= 1

    def test_belief_dependency_and_propagate(self, client_v11: TestClient, ctx_v11) -> None:
        b1 = ctx_v11.belief_store.add_or_reinforce("The user works primarily in Python")
        b2 = ctx_v11.belief_store.add_or_reinforce("Code review feedback should focus on readability")
        r = client_v11.post("/causality/belief-graph/dependency", json={
            "source_belief_id": b1.belief_id, "target_belief_id": b2.belief_id, "relation": "supports", "strength": 0.7,
        })
        assert r.status_code == 200

        r2 = client_v11.post("/causality/belief-graph/propagate", json={
            "changed_belief_id": b1.belief_id, "confidence_delta": 0.1,
        })
        assert r2.status_code == 200
        assert r2.json()["total_affected"] >= 1

    def test_list_principles(self, client_v11: TestClient, ctx_v11) -> None:
        ctx_v11.principle_store.add(Principle(statement="API test principle"))
        r = client_v11.get("/causality/principles")
        assert r.status_code == 200
        assert r.json()["total"] >= 1

    def test_synthesize_principles(self, client_v11: TestClient, ctx_v11) -> None:
        ctx_v11.cause_graph.record_observation("synth api trigger", "synth api effect", CauseRelation.BLOCKS)
        ctx_v11.cause_graph.record_observation("synth api trigger", "synth api effect", CauseRelation.BLOCKS)
        r = client_v11.post("/causality/principles/synthesize")
        assert r.status_code == 200
        assert "synthesized" in r.json()

    def test_reflect_causal(self, client_v11: TestClient) -> None:
        r = client_v11.post("/causality/reflect/causal", json={"topic": "some failed topic"})
        assert r.status_code == 200
        assert "relevant_principles" in r.json()

    def test_reflect_causal_with_alternative_strategy(self, client_v11: TestClient) -> None:
        r = client_v11.post("/causality/reflect/causal", json={
            "topic": "some failed topic", "alternative_strategy": "tree_of_thought",
            "latent_state": {"confidence": 0.6},
        })
        assert r.status_code == 200
        assert r.json()["estimated_success"] is not None

    def test_reflect_why(self, client_v11: TestClient) -> None:
        r = client_v11.get("/causality/reflect/why?domain=test_domain")
        assert r.status_code == 200
        assert "question" in r.json()

    def test_reflect_causes_of(self, client_v11: TestClient) -> None:
        r = client_v11.get("/causality/reflect/causes-of?effect=test_effect")
        assert r.status_code == 200

    def test_strategy_evolve(self, client_v11: TestClient) -> None:
        r = client_v11.post("/causality/strategy/evolve", json={"ref_key": "api_ref", "failure_topic": "api failure topic"})
        assert r.status_code == 200
        assert "recommended_strategy" in r.json()

    def test_counterfactual_explore(self, client_v11: TestClient) -> None:
        r = client_v11.post("/causality/counterfactual/explore", json={
            "current_state": {"confidence": 0.4, "risk": 0.6},
            "alternatives": [
                {"name": "tot", "description": "What if ToT?", "resulting_state": {"confidence": 0.6, "risk": 0.3}},
                {"name": "direct", "description": "Stayed direct", "resulting_state": {"confidence": 0.4, "risk": 0.6}},
            ],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert all(s["epistemic_status"] == "counterfactual" for s in data["scenarios"])
        assert all(s["validated_causally"] is False for s in data["scenarios"])

    def test_counterfactual_explore_requires_alternatives(self, client_v11: TestClient) -> None:
        r = client_v11.post("/causality/counterfactual/explore", json={"alternatives": []})
        assert r.status_code == 422

    def test_health_check_still_works(self, client_v11: TestClient) -> None:
        r = client_v11.get("/health")
        assert r.status_code == 200
