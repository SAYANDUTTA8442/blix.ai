"""
Evaluation Framework v2 — Blix v0.3.4  (Feature 7)

Extends ``CognitiveEvaluator`` (v0.3.3) with reasoning-specific metrics:

Metric                Purpose
──────────────────────────────────────────────────
ReasoningAccuracy     Did multi-hop / cognitive queries return the right answer?
GraphCoverage         What fraction of expected entities/relations are in the graph?
PathAccuracy          Did path queries find the correct shortest path?
InferenceRecall       Did transitive inference recover all expected nodes?
ExplainabilityScore   How complete is the evidence chain (memory + fact + graph)?

These sit on top of the v0.3 → v0.3.3 metric stack, giving a full
7-level evaluation tower.

Delivery: metrics exposed via ``ReasoningEvaluator`` (extends
``CognitiveEvaluator``) and re-exported through ``blix_eval``.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from evaluation.cognitive import CognitiveEvaluator


# ---------------------------------------------------------------------------
# Reasoning evaluation case
# ---------------------------------------------------------------------------


import dataclasses as _dc
from typing import Optional


@_dc.dataclass
class ReasoningCase:
    """
    One evaluation instance for the reasoning engine.

    Fields
    ------
    case_id:
        Unique identifier.
    query:
        Natural-language query (for ``CognitiveQueryEngine.query()``).
    expected_answers:
        Ground-truth answer labels (e.g. ["ChromaDB", "FastAPI"]).
    start_entity:
        Source entity for multi-hop / transitive queries.
    end_entity:
        Target entity for multi-hop / path queries.
    expected_path_hops:
        Expected number of hops in the shortest path (None = any).
    expected_intermediates:
        Expected intermediate nodes in a multi-hop query.
    transitive_relation:
        Relation for transitive inference queries.
    transitive_depth:
        Depth for transitive inference.
    expected_transitive_nodes:
        All nodes expected in transitive closure.
    """

    case_id: str = ""
    query: str = ""
    expected_answers: list = _dc.field(default_factory=list)
    start_entity: str = ""
    end_entity: str = ""
    expected_path_hops: Optional[int] = None
    expected_intermediates: list = _dc.field(default_factory=list)
    transitive_relation: str = ""
    transitive_depth: int = 2
    expected_transitive_nodes: list = _dc.field(default_factory=list)


# ---------------------------------------------------------------------------
# Reasoning evaluator
# ---------------------------------------------------------------------------


class ReasoningEvaluator(CognitiveEvaluator):
    """
    Extends ``CognitiveEvaluator`` with v0.3.4 reasoning-specific metrics.

    All new metrics follow the same pattern: static methods that accept
    inputs and return floats, composable in ``evaluate_reasoning()``.
    """

    # ------------------------------------------------------------------
    # Reasoning accuracy (Feature 1/2 evaluation)
    # ------------------------------------------------------------------

    @staticmethod
    def reasoning_accuracy(
        predicted_answers: list[str],
        expected_answers: list[str],
        case_insensitive: bool = True,
    ) -> float:
        """
        Fraction of expected answers found in predicted answers.

        Uses substring/case-insensitive matching so "chromadb" matches
        "ChromaDB" and "ChromaDB v1.0".
        """
        if not expected_answers:
            return 1.0
        if not predicted_answers:
            return 0.0

        def _normalise(s: str) -> str:
            return s.lower().strip() if case_insensitive else s.strip()

        pred_norm = [_normalise(p) for p in predicted_answers]

        hits = 0
        for exp in expected_answers:
            en = _normalise(exp)
            if any(en in pn or pn in en for pn in pred_norm):
                hits += 1
        return hits / len(expected_answers)

    @staticmethod
    def reasoning_precision(
        predicted_answers: list[str],
        expected_answers: list[str],
        case_insensitive: bool = True,
    ) -> float:
        """Fraction of predicted answers that are correct (anti-hallucination)."""
        if not predicted_answers:
            return 1.0

        def _normalise(s: str) -> str:
            return s.lower().strip() if case_insensitive else s.strip()

        exp_norm = [_normalise(e) for e in expected_answers]

        hits = 0
        for pred in predicted_answers:
            pn = _normalise(pred)
            if any(pn in en or en in pn for en in exp_norm):
                hits += 1
        return hits / len(predicted_answers)

    # ------------------------------------------------------------------
    # Graph coverage (Feature 2 / Knowledge Graph Foundation evaluation)
    # ------------------------------------------------------------------

    @staticmethod
    def graph_coverage(
        graph: object,          # MemoryGraph
        expected_entities: list[str],
        expected_edges: Optional[list[tuple[str, str, str]]] = None,
    ) -> float:
        """
        Fraction of expected entities AND edges present in the graph.

        Parameters
        ----------
        graph:
            ``MemoryGraph`` instance.
        expected_entities:
            Labels of entities expected to be in the graph.
        expected_edges:
            Optional list of (from_label, relation, to_label) triples.

        Returns
        -------
        float
            Combined coverage score (0–1). Entity coverage weighted 0.6,
            edge coverage weighted 0.4.
        """
        total = 0
        correct = 0

        for label in expected_entities:
            total += 1
            if graph.find_node_by_label(label) is not None:  # type: ignore[union-attr]
                correct += 1

        entity_coverage = correct / total if total else 1.0

        if not expected_edges:
            return entity_coverage

        edge_total = 0
        edge_correct = 0
        for from_label, relation, to_label in expected_edges:
            edge_total += 1
            from_node = graph.find_node_by_label(from_label)  # type: ignore[union-attr]
            to_node = graph.find_node_by_label(to_label)  # type: ignore[union-attr]
            if from_node is None or to_node is None:
                continue
            edges = graph.get_edges(from_id=from_node.id, to_id=to_node.id)  # type: ignore[union-attr]
            if any(e.relation == relation or relation in e.relation for e in edges):
                edge_correct += 1

        edge_coverage = edge_correct / edge_total if edge_total else 1.0
        return round(0.6 * entity_coverage + 0.4 * edge_coverage, 4)

    # ------------------------------------------------------------------
    # Path accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def path_accuracy(
        predicted_hops: Optional[int],
        expected_hops: Optional[int],
        tolerance: int = 1,
    ) -> float:
        """
        Whether the predicted path hop count matches the expected count
        within ``tolerance``.

        Returns 1.0 if expected_hops is None (any path accepted).
        """
        if expected_hops is None:
            return 1.0 if predicted_hops is not None else 0.0
        if predicted_hops is None:
            return 0.0
        return 1.0 if abs(predicted_hops - expected_hops) <= tolerance else 0.0

    # ------------------------------------------------------------------
    # Transitive inference recall
    # ------------------------------------------------------------------

    @staticmethod
    def inference_recall(
        predicted_nodes: list[str],
        expected_nodes: list[str],
        case_insensitive: bool = True,
    ) -> float:
        """
        Fraction of expected transitive-closure nodes found in predicted output.
        """
        return ReasoningEvaluator.reasoning_accuracy(
            predicted_nodes, expected_nodes, case_insensitive
        )

    # ------------------------------------------------------------------
    # Explainability score
    # ------------------------------------------------------------------

    @staticmethod
    def explainability_score(explained_response: object) -> float:
        """
        How complete is the evidence chain in an ``ExplainedResponse``?

        Score = weighted fraction of non-empty evidence categories:
            - memory_evidence    weight 1.0
            - fact_evidence      weight 1.5
            - graph_evidence     weight 1.2
            - reasoning_trace    weight 1.0

        Max score = 1.0 (all four populated).
        """
        weights = {
            "memory_evidence": 1.0,
            "fact_evidence": 1.5,
            "graph_evidence": 1.2,
            "reasoning_trace": 1.0,
        }
        total = sum(weights.values())
        score = 0.0
        for field, w in weights.items():
            val = getattr(explained_response, field, None)
            if val is not None and (isinstance(val, list) and len(val) > 0
                                    or not isinstance(val, list)):
                score += w
        return round(score / total, 4)

    # ------------------------------------------------------------------
    # Combined reasoning evaluation pass
    # ------------------------------------------------------------------

    def evaluate_reasoning(
        self,
        cases: list[ReasoningCase],
        *,
        query_fn: Optional[object] = None,      # callable(query) → QueryResult
        multihop_fn: Optional[object] = None,   # callable(start, end) → QueryResult
        infer_fn: Optional[object] = None,      # callable(entity, relation, depth) → QueryResult
        graph: Optional[object] = None,
        expected_graph_entities: Optional[list[str]] = None,
        expected_graph_edges: Optional[list[tuple[str, str, str]]] = None,
    ) -> dict[str, float]:
        """
        Run reasoning-specific evaluation over a list of ``ReasoningCase``s.

        Returns a summary dict of all computed metrics.
        """
        results: dict[str, float] = {}

        # 1. Reasoning accuracy / precision from cognitive queries
        if query_fn is not None:
            acc_vals: list[float] = []
            prec_vals: list[float] = []
            for case in cases:
                if not case.query or not case.expected_answers:
                    continue
                qr = query_fn(case.query)  # type: ignore[operator]
                predicted = getattr(qr, "answer", [])
                acc_vals.append(self.reasoning_accuracy(predicted, case.expected_answers))
                prec_vals.append(self.reasoning_precision(predicted, case.expected_answers))
            if acc_vals:
                results["reasoning_accuracy"] = sum(acc_vals) / len(acc_vals)
                results["reasoning_precision"] = sum(prec_vals) / len(prec_vals)

        # 2. Path accuracy from multi-hop queries
        if multihop_fn is not None:
            path_vals: list[float] = []
            inter_recalls: list[float] = []
            for case in cases:
                if not case.start_entity or not case.end_entity:
                    continue
                qr = multihop_fn(case.start_entity, case.end_entity)  # type: ignore[operator]
                predicted_intermediates = getattr(qr, "answer", [])
                # We don't have hop count easily here; use intermediates as proxy
                if case.expected_intermediates:
                    ir = self.inference_recall(predicted_intermediates, case.expected_intermediates)
                    inter_recalls.append(ir)
            if inter_recalls:
                results["multihop_recall"] = sum(inter_recalls) / len(inter_recalls)

        # 3. Transitive inference recall
        if infer_fn is not None:
            infer_vals: list[float] = []
            for case in cases:
                if not case.start_entity or not case.transitive_relation:
                    continue
                if not case.expected_transitive_nodes:
                    continue
                qr = infer_fn(case.start_entity, case.transitive_relation, case.transitive_depth)  # type: ignore[operator]
                predicted = getattr(qr, "answer", [])
                infer_vals.append(self.inference_recall(predicted, case.expected_transitive_nodes))
            if infer_vals:
                results["inference_recall"] = sum(infer_vals) / len(infer_vals)

        # 4. Graph coverage
        if graph is not None and expected_graph_entities:
            results["graph_coverage"] = self.graph_coverage(
                graph, expected_graph_entities, expected_graph_edges
            )

        return results


# ---------------------------------------------------------------------------
# Re-export ReasoningEvaluator from blix_eval
# ---------------------------------------------------------------------------
# Import in evaluation/blix_eval/__init__.py for the deliverable namespace.

