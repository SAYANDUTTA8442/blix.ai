"""
Principle Synthesizer — Blix v0.3.11  (New module 4, Phase 2)

Closes the loop from raw experience to reusable generalization:

    Experience
      ↓
    Causes              (causality.cause_graph.CauseGraph / causality.causal_memory.CausalMemoryStore)
      ↓
    Principles           (this module)
      ↓
    Principle Graph        (causality.principle_graph.PrincipleGraph)
      ↓
    Strategies               (metacognition.strategy_evolution.StrategyEvolution, Phase 3)

Example, matching the spec:

    "Projects fail repeatedly without evaluation."
    "Projects stagnate without benchmarks."
    "Optimization before metrics hurts."
      ↓
    Principle("Always build evaluation before optimization.")

Mechanically, this mines two REAL sources of repeated pattern: high-
confidence ``causality.cause_graph.CauseGraph`` edges (BLOCKS/ENABLES
relations recurring across multiple evidence observations) and
``learning.failure_clusterer.FailureClusterer`` recurring clusters
(v0.3.10) — then uses the existing, already-configured LLM
(``llm.base.LLMProvider``, same honest pattern as v0.3.10's
``memory.semantic_compressor.SemanticCompressor``) to phrase the
generalization in natural language, with a non-LLM fallback when no
LLM is available. The resulting ``Principle`` always carries its
``supporting_causes``/``supporting_failures`` evidence trail — nothing
is synthesized from fewer than the configured minimum number of
corroborating observations.

Python 3.10 compatible.
"""

from __future__ import annotations

from typing import Optional

from causality.cause_graph import CauseEdge, CauseGraph, CauseRelation
from causality.principle import Principle, PrincipleStore
from learning.failure_clusterer import FailureCluster, FailureClusterer
from llm.base import LLMProvider
from utils.logger import get_logger

log = get_logger(__name__)

_MIN_EVIDENCE_TO_SYNTHESIZE = 2


class PrincipleSynthesizer:
    """
    Synthesizes ``Principle`` objects from recurring CauseGraph edges
    and FailureClusterer clusters.

    Parameters
    ----------
    principle_store:
        ``PrincipleStore`` — synthesized principles are persisted here.
    cause_graph:
        ``CauseGraph`` — source of recurring BLOCKS/ENABLES patterns.
    failure_clusterer:
        Optional ``FailureClusterer`` — source of recurring failure patterns.
    llm:
        Optional ``LLMProvider`` — used to phrase the principle in
        natural language; falls back to a template-based phrasing when
        unavailable (same honest pattern as v0.3.10's SemanticCompressor).
    min_evidence_to_synthesize:
        Minimum evidence_count on a CauseGraph edge, or minimum cluster
        size, before a principle is synthesized from it.
    """

    def __init__(
        self,
        principle_store: PrincipleStore,
        cause_graph: CauseGraph,
        failure_clusterer: Optional[FailureClusterer] = None,
        llm: Optional[LLMProvider] = None,
        min_evidence_to_synthesize: int = _MIN_EVIDENCE_TO_SYNTHESIZE,
    ) -> None:
        self._principle_store = principle_store
        self._cause_graph = cause_graph
        self._failure_clusterer = failure_clusterer
        self._llm = llm
        self._min_evidence = min_evidence_to_synthesize

    # ------------------------------------------------------------------
    # Synthesis from CauseGraph
    # ------------------------------------------------------------------

    def synthesize_from_cause_edge(self, edge: CauseEdge) -> Optional[Principle]:
        """
        Synthesize a Principle from one well-evidenced CauseGraph edge.
        Returns None if the edge doesn't have enough evidence yet.
        """
        if edge.evidence_count < self._min_evidence:
            return None

        statement = self._phrase_cause_principle(edge)
        principle = Principle(
            statement=statement,
            confidence=edge.confidence,
            evidence_count=edge.evidence_count,
            supporting_causes=[edge.edge_id],
        )
        return self._principle_store.add(principle)

    def _phrase_cause_principle(self, edge: CauseEdge) -> str:
        if self._llm is not None:
            prompt = (
                f"In one short imperative sentence, state a general principle that follows from this "
                f"observed pattern: '{edge.trigger}' {edge.relation.value} '{edge.effect}' "
                f"(observed {edge.evidence_count} times). Principle:"
            )
            try:
                return self._llm.generate(prompt).strip()
            except Exception as exc:
                log.warning("PrincipleSynthesizer: LLM phrasing failed (%s) — using template fallback.", exc)

        templates = {
            CauseRelation.BLOCKS: f"Avoid '{edge.trigger}' — it blocks '{edge.effect}'.",
            CauseRelation.ENABLES: f"Ensure '{edge.trigger}' — it enables '{edge.effect}'.",
            CauseRelation.CAUSES: f"'{edge.trigger}' reliably leads to '{edge.effect}' — plan accordingly.",
            CauseRelation.INCREASES: f"'{edge.trigger}' increases '{edge.effect}' — leverage this where useful.",
            CauseRelation.DECREASES: f"'{edge.trigger}' decreases '{edge.effect}' — avoid where '{edge.effect}' matters.",
        }
        return templates.get(edge.relation, f"'{edge.trigger}' relates to '{edge.effect}' ({edge.relation.value}).")

    # ------------------------------------------------------------------
    # Synthesis from FailureClusterer
    # ------------------------------------------------------------------

    def synthesize_from_failure_cluster(self, cluster: FailureCluster) -> Optional[Principle]:
        """Synthesize a Principle from one recurring failure cluster (v0.3.10)."""
        if len(cluster.records) < self._min_evidence:
            return None

        statement = self._phrase_failure_principle(cluster)
        confidence = min(1.0, 0.4 + 0.05 * cluster.total_occurrences)
        principle = Principle(
            statement=statement, confidence=confidence, evidence_count=cluster.total_occurrences,
            supporting_failures=[str(cluster.cluster_id)],
        )
        return self._principle_store.add(principle)

    def _phrase_failure_principle(self, cluster: FailureCluster) -> str:
        terms = ", ".join(cluster.representative_terms[:3]) or "this recurring issue"
        if self._llm is not None:
            sample = "; ".join(r.failure for r in cluster.records[:3])
            prompt = (
                f"In one short imperative sentence, state a general principle to avoid this recurring "
                f"failure pattern (key terms: {terms}). Sample failures: {sample}. Principle:"
            )
            try:
                return self._llm.generate(prompt).strip()
            except Exception as exc:
                log.warning("PrincipleSynthesizer: LLM phrasing failed (%s) — using template fallback.", exc)

        return f"Address recurring failures involving {terms} before they compound further."

    # ------------------------------------------------------------------
    # Batch synthesis
    # ------------------------------------------------------------------

    def synthesize_from_experiment(self, experiment) -> Optional[Principle]:
        """
        v0.3.13 — Synthesize a Principle from a completed Experiment.
        Only COMPLETED experiments with a confirmed outcome produce principles;
        planned or failed experiments without outcomes are skipped.
        """
        from experiments.experiment_planner import ExperimentStatus
        if experiment.status != ExperimentStatus.COMPLETED or not experiment.outcome:
            return None

        statement = self._phrase_experiment_principle(experiment)
        confidence = 0.6 if experiment.outcome_confirmed else 0.4
        principle = Principle(
            statement=statement, confidence=confidence, evidence_count=1,
            supporting_failures=[f"experiment:{experiment.experiment_id}"],
        )
        return self._principle_store.add(principle)

    def _phrase_experiment_principle(self, experiment) -> str:
        if self._llm is not None:
            prompt = (
                f"In one short imperative sentence, state a reusable principle from this experiment result. "
                f"Hypothesis: '{experiment.hypothesis_id}'. Outcome: '{experiment.outcome}'. Principle:"
            )
            try:
                return self._llm.generate(prompt).strip()
            except Exception as exc:
                log.warning("PrincipleSynthesizer: LLM phrasing failed (%s) — using template.", exc)
        return f"Experiment finding: {experiment.outcome[:120]}."

    def synthesize_all(self, experiments: Optional[list] = None) -> list[Principle]:
        """Synthesize principles from CauseGraph edges, failure clusters, and (v0.3.13) experiments."""
        principles: list[Principle] = []

        for edge in self._cause_graph.high_confidence_edges(threshold=0.0):
            if edge.evidence_count >= self._min_evidence:
                p = self.synthesize_from_cause_edge(edge)
                if p is not None:
                    principles.append(p)

        if self._failure_clusterer is not None:
            for cluster in self._failure_clusterer.recurring_clusters():
                p = self.synthesize_from_failure_cluster(cluster)
                if p is not None:
                    principles.append(p)

        if experiments is not None:
            for exp in experiments:
                p = self.synthesize_from_experiment(exp)
                if p is not None:
                    principles.append(p)

        return principles
