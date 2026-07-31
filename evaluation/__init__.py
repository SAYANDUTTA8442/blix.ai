"""
Memory Evaluation Framework — Blix v0.3  (Feature 7)

Provides a research-grade evaluation module for the memory system.

Metrics
-------
* Retrieval Precision        — fraction of retrieved memories that are relevant
* Retrieval Recall           — fraction of relevant memories that are retrieved
* Fact Accuracy              — fraction of extracted facts confirmed in ground truth
* Hallucinated Memory Rate   — fraction of extracted facts NOT in ground truth
* Profile Accuracy           — fraction of profile fields matching ground truth
* Graph Consistency          — fraction of asserted edges that are correct
* Summary Quality            — BLEU-inspired n-gram overlap for summaries

Usage
-----
    from evaluation.evaluator import MemoryEvaluator, EvalDataset, EvalCase
    ev = MemoryEvaluator()
    result = ev.evaluate(dataset, retriever, memory_manager, scorer)
    ev.print_report(result)
    ev.save_report(result, Path("eval_results.json"))

CLI
---
    python -m blix.evaluation.cli --dataset data.json --output report.json

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset types
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    """One evaluation instance."""

    case_id: str
    query: str
    relevant_memory_ids: list[int]                        # ground-truth relevant ids
    ground_truth_facts: list[str] = field(default_factory=list)
    ground_truth_profile: dict = field(default_factory=dict)
    ground_truth_edges: list[tuple[str, str, str]] = field(default_factory=list)  # (from, rel, to)
    reference_summary: Optional[str] = None


@dataclass
class EvalDataset:
    """A collection of eval cases with a name for tracking."""

    name: str
    cases: list[EvalCase]
    version: str = "1.0"


# ---------------------------------------------------------------------------
# Metric results
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    name: str
    value: float
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "value": round(self.value, 4), "details": self.details}


@dataclass
class EvalReport:
    """Full evaluation run report."""

    dataset_name: str
    dataset_version: str
    timestamp: str
    metrics: list[MetricResult]
    per_case: list[dict]

    def summary_dict(self) -> dict:
        return {m.name: m.value for m in self.metrics}

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "version": self.dataset_version,
            "timestamp": self.timestamp,
            "metrics": [m.to_dict() for m in self.metrics],
            "per_case": self.per_case,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class MemoryEvaluator:
    """
    Runs reproducible evaluation passes and produces ``EvalReport`` objects.

    Designed to be used standalone (import) or via the CLI.
    """

    # ------------------------------------------------------------------
    # Retrieval metrics
    # ------------------------------------------------------------------

    @staticmethod
    def precision_at_k(retrieved: list[int], relevant: list[int]) -> float:
        """P@k — fraction of retrieved that are relevant."""
        if not retrieved:
            return 0.0
        hits = sum(1 for r in retrieved if r in relevant)
        return hits / len(retrieved)

    @staticmethod
    def recall_at_k(retrieved: list[int], relevant: list[int]) -> float:
        """R@k — fraction of relevant that are retrieved."""
        if not relevant:
            return 1.0
        hits = sum(1 for r in relevant if r in retrieved)
        return hits / len(relevant)

    @staticmethod
    def f1(precision: float, recall: float) -> float:
        if precision + recall == 0.0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # ------------------------------------------------------------------
    # Fact metrics
    # ------------------------------------------------------------------

    @staticmethod
    def fact_accuracy(extracted: list[str], ground_truth: list[str]) -> float:
        """
        Fraction of extracted facts that are confirmed by ground truth.

        Uses simple substring containment (case-insensitive) as a proxy
        for semantic equivalence — suitable for research-paper baselines.
        """
        if not extracted:
            return 1.0
        confirmed = sum(
            1 for f in extracted
            if any(f.lower() in gt.lower() or gt.lower() in f.lower() for gt in ground_truth)
        )
        return confirmed / len(extracted)

    @staticmethod
    def hallucination_rate(extracted: list[str], ground_truth: list[str]) -> float:
        """Fraction of extracted facts NOT supported by ground truth."""
        if not extracted:
            return 0.0
        return 1.0 - MemoryEvaluator.fact_accuracy(extracted, ground_truth)

    # ------------------------------------------------------------------
    # Profile accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def profile_accuracy(actual_profile: dict, ground_truth_profile: dict) -> float:
        """
        Fraction of ground-truth profile fields that are correctly set.

        For list fields, checks subset containment.
        """
        if not ground_truth_profile:
            return 1.0
        correct = 0
        total = len(ground_truth_profile)
        for key, expected in ground_truth_profile.items():
            actual = actual_profile.get(key)
            if isinstance(expected, list):
                if isinstance(actual, list):
                    correct += int(all(e in actual for e in expected))
                # else: miss
            else:
                correct += int(str(actual).lower() == str(expected).lower())
        return correct / total

    # ------------------------------------------------------------------
    # Graph consistency
    # ------------------------------------------------------------------

    @staticmethod
    def graph_consistency(
        actual_edges: list[tuple[str, str, str]],
        ground_truth_edges: list[tuple[str, str, str]],
    ) -> float:
        """Fraction of asserted edges that appear in ground truth."""
        if not actual_edges:
            return 1.0
        gt_set = {(f.lower(), r.lower(), t.lower()) for f, r, t in ground_truth_edges}
        hits = sum(
            1 for f, r, t in actual_edges
            if (f.lower(), r.lower(), t.lower()) in gt_set
        )
        return hits / len(actual_edges)

    # ------------------------------------------------------------------
    # Summary quality (lightweight BLEU-1 proxy)
    # ------------------------------------------------------------------

    @staticmethod
    def summary_quality(generated: str, reference: str) -> float:
        """
        Unigram precision (BLEU-1 without brevity penalty) as a summary
        quality proxy.  Returns 0.0–1.0.
        """
        if not reference:
            return 0.0
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()
        if not gen_tokens:
            return 0.0
        ref_counter = Counter(ref_tokens)
        hits = sum(min(count, ref_counter.get(tok, 0)) for tok, count in Counter(gen_tokens).items())
        return hits / len(gen_tokens)

    # ------------------------------------------------------------------
    # Full evaluation pass
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset: EvalDataset,
        *,
        retrieval_fn: Optional[object] = None,   # callable(query) → list[int]
        extracted_facts_fn: Optional[object] = None,  # callable(query) → list[str]
        profile_fn: Optional[object] = None,      # callable() → dict
        graph_fn: Optional[object] = None,        # callable() → list[(f,r,t)]
        summary_fn: Optional[object] = None,      # callable(query) → str
    ) -> EvalReport:
        """
        Run a full evaluation pass over *dataset*.

        All callable parameters are optional — pass only what you want
        to measure.  Missing callables produce NaN for that metric.

        Parameters are typed as ``object`` so the caller isn't forced to
        import the full Blix stack.
        """
        precision_vals: list[float] = []
        recall_vals: list[float] = []
        fact_acc_vals: list[float] = []
        hall_vals: list[float] = []
        profile_acc_vals: list[float] = []
        graph_cons_vals: list[float] = []
        summary_q_vals: list[float] = []
        per_case: list[dict] = []

        for case in dataset.cases:
            case_result: dict = {"case_id": case.case_id, "query": case.query}

            # Retrieval
            if retrieval_fn is not None:
                retrieved: list[int] = retrieval_fn(case.query)  # type: ignore[operator]
                p = self.precision_at_k(retrieved, case.relevant_memory_ids)
                r = self.recall_at_k(retrieved, case.relevant_memory_ids)
                precision_vals.append(p)
                recall_vals.append(r)
                case_result["retrieval_precision"] = p
                case_result["retrieval_recall"] = r

            # Fact accuracy
            if extracted_facts_fn is not None:
                facts: list[str] = extracted_facts_fn(case.query)  # type: ignore[operator]
                fa = self.fact_accuracy(facts, case.ground_truth_facts)
                hr = self.hallucination_rate(facts, case.ground_truth_facts)
                fact_acc_vals.append(fa)
                hall_vals.append(hr)
                case_result["fact_accuracy"] = fa
                case_result["hallucination_rate"] = hr

            # Profile accuracy
            if profile_fn is not None and case.ground_truth_profile:
                actual_profile: dict = profile_fn()  # type: ignore[operator]
                pa = self.profile_accuracy(actual_profile, case.ground_truth_profile)
                profile_acc_vals.append(pa)
                case_result["profile_accuracy"] = pa

            # Graph consistency
            if graph_fn is not None and case.ground_truth_edges:
                actual_edges: list = graph_fn()  # type: ignore[operator]
                gc = self.graph_consistency(actual_edges, case.ground_truth_edges)
                graph_cons_vals.append(gc)
                case_result["graph_consistency"] = gc

            # Summary quality
            if summary_fn is not None and case.reference_summary:
                generated: str = summary_fn(case.query)  # type: ignore[operator]
                sq = self.summary_quality(generated, case.reference_summary)
                summary_q_vals.append(sq)
                case_result["summary_quality"] = sq

            per_case.append(case_result)

        def _avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else float("nan")

        metrics = [
            MetricResult("retrieval_precision", _avg(precision_vals)),
            MetricResult("retrieval_recall",    _avg(recall_vals)),
            MetricResult("retrieval_f1", self.f1(_avg(precision_vals), _avg(recall_vals))),
            MetricResult("fact_accuracy",        _avg(fact_acc_vals)),
            MetricResult("hallucination_rate",   _avg(hall_vals)),
            MetricResult("profile_accuracy",     _avg(profile_acc_vals)),
            MetricResult("graph_consistency",    _avg(graph_cons_vals)),
            MetricResult("summary_quality",      _avg(summary_q_vals)),
        ]

        return EvalReport(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            per_case=per_case,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, report: EvalReport) -> None:
        print(f"\n=== Eval Report: {report.dataset_name} v{report.dataset_version} ===")
        print(f"  Timestamp: {report.timestamp}")
        print(f"  Cases: {len(report.per_case)}\n")
        for m in report.metrics:
            if m.value != m.value:  # NaN
                continue
            print(f"  {m.name:28s}  {m.value:.4f}")
        print()

    def save_report(self, report: EvalReport, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
        log.info("Eval report saved to %s", output)
