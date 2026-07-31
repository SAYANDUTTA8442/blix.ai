"""
Research-Grade Evaluation Extensions — Blix v0.3.1  (Issues 10 & 14)

Addresses:
  Issue 10: "Evaluation framework is weak for research publication —
             missing memory-specific metrics."
  Issue 14: "No central research hypothesis tying the system together."

Part 1: Extended metrics
------------------------
    retention_over_time     — fraction of memories correctly retrievable N days later
    forgetting_curve        — retrieval accuracy vs age curve (Ebbinghaus-style)
    contradiction_rate      — fraction of retrieved pairs that contradict each other
    memory_drift            — semantic shift in similar memories over time
    profile_drift           — magnitude of profile change per unit time
    temporal_consistency    — does retrieval rank newer more-accurate facts higher?

Part 2: ResearchHypothesis framework
-------------------------------------
    A structured way to define, track, and evaluate scientific hypotheses
    about Blix's memory system.

    Example hypotheses:
    H1: "Hierarchical memory + profile evolution improves long-horizon
         personalization by X% vs flat retrieval baseline."
    H2: "Graph-augmented memory retrieval reduces memory drift by X%."
    H3: "MMR diversification improves fact coverage without hurting precision."

    Each hypothesis specifies:
    - independent variable (what we change)
    - dependent variable (what we measure)
    - baseline condition
    - treatment condition
    - metric (from MemoryEvaluator)

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from evaluation import EvalReport, MemoryEvaluator
from utils.logger import get_logger

log = get_logger(__name__)


# ===========================================================================
# Part 1 — Extended research metrics
# ===========================================================================


class ExtendedMemoryEvaluator(MemoryEvaluator):
    """
    Extends the base ``MemoryEvaluator`` with memory-system-specific metrics
    required for research publication.
    """

    # ------------------------------------------------------------------
    # Retention over time
    # ------------------------------------------------------------------

    @staticmethod
    def retention_over_time(
        retrieval_fn: object,            # callable(query, age_cutoff_days) → list[int]
        cases: list,                     # list[EvalCase]
        age_buckets: list[float],        # e.g. [7, 30, 90, 365] days
    ) -> dict[float, float]:
        """
        Measure retrieval precision at different memory ages.

        Returns dict: age_days → mean precision.
        """
        results: dict[float, float] = {}
        for age in age_buckets:
            precisions: list[float] = []
            for case in cases:
                retrieved: list[int] = retrieval_fn(case.query, age)  # type: ignore[operator]
                p = ExtendedMemoryEvaluator.precision_at_k(
                    retrieved, case.relevant_memory_ids
                )
                precisions.append(p)
            results[age] = sum(precisions) / len(precisions) if precisions else 0.0
        return results

    # ------------------------------------------------------------------
    # Forgetting curve
    # ------------------------------------------------------------------

    @staticmethod
    def forgetting_curve(
        retrieval_fn: object,
        cases: list,
        age_days_sequence: list[float],
    ) -> list[tuple[float, float]]:
        """
        Generate an Ebbinghaus-style forgetting curve.

        Returns list of (age_days, retention_rate) pairs.
        """
        retention = ExtendedMemoryEvaluator.retention_over_time(
            retrieval_fn, cases, age_days_sequence
        )
        return [(age, rate) for age, rate in sorted(retention.items())]

    # ------------------------------------------------------------------
    # Contradiction rate
    # ------------------------------------------------------------------

    @staticmethod
    def contradiction_rate(
        contradiction_detector: object,  # ContradictionDetector
        memories: list,
    ) -> float:
        """
        Fraction of memory pairs that contain contradictions.

        Useful as a data-quality metric for the memory store.
        """
        n = len(memories)
        if n < 2:
            return 0.0
        total_pairs = n * (n - 1) / 2
        contradictions = contradiction_detector.detect(memories)  # type: ignore[union-attr]
        return len(contradictions) / total_pairs

    # ------------------------------------------------------------------
    # Memory drift
    # ------------------------------------------------------------------

    @staticmethod
    def memory_drift(
        embeddings_by_time: list[tuple[float, object]],  # [(timestamp_days, np.ndarray)]
        topic: str = "all",
    ) -> float:
        """
        Measure how much the embedding centroid of topic-related memories
        shifts over time (semantic drift).

        Parameters
        ----------
        embeddings_by_time:
            List of (age_days, embedding_vector) for memories on the same topic.
        topic:
            Label for logging.

        Returns
        -------
        float
            Mean cosine distance between consecutive time-sorted centroids.
            0 = no drift, 1 = complete rotation.
        """
        import numpy as np

        if len(embeddings_by_time) < 2:
            return 0.0

        sorted_embs = sorted(embeddings_by_time, key=lambda x: x[0])
        vecs = [np.array(e, dtype=np.float32) for _, e in sorted_embs]

        drifts: list[float] = []
        for i in range(1, len(vecs)):
            a = vecs[i - 1]
            b = vecs[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-8 or nb < 1e-8:
                continue
            cos_sim = float(np.dot(a / na, b / nb))
            cos_dist = 1.0 - cos_sim
            drifts.append(cos_dist)

        result = sum(drifts) / len(drifts) if drifts else 0.0
        log.debug("memory_drift(topic=%r): %.4f over %d steps", topic, result, len(drifts))
        return result

    # ------------------------------------------------------------------
    # Profile drift
    # ------------------------------------------------------------------

    @staticmethod
    def profile_drift(
        audit_entries: list,    # list[ProfileAuditEntry]
        time_window_days: float = 30.0,
    ) -> float:
        """
        Rate of profile change per day over ``time_window_days``.

        Computed as: number_of_audit_changes / time_window_days.
        Higher = more volatile profile.
        """
        if not audit_entries or time_window_days <= 0:
            return 0.0
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff_days = time_window_days
        recent = [
            e for e in audit_entries
            if (now - e.timestamp).total_seconds() / 86400.0 <= cutoff_days
        ]
        return len(recent) / time_window_days

    # ------------------------------------------------------------------
    # Temporal consistency
    # ------------------------------------------------------------------

    @staticmethod
    def temporal_consistency(
        retrieved: list,    # list[MemoryEntry] in retrieval order
        ground_truth_newer_ids: set[int],
    ) -> float:
        """
        Measure whether the retrieval pipeline correctly ranks newer
        (more accurate) facts above older (possibly stale) ones.

        Returns fraction of retrieved memories that are from ``ground_truth_newer_ids``
        in the top-half of the retrieved list.
        """
        if not retrieved:
            return 0.0
        top_half = retrieved[: max(1, len(retrieved) // 2)]
        hits = sum(1 for m in top_half if getattr(m, "id") in ground_truth_newer_ids)
        return hits / len(top_half)

    # ------------------------------------------------------------------
    # Full extended evaluation pass
    # ------------------------------------------------------------------

    def evaluate_extended(
        self,
        memories: list,
        audit_entries: Optional[list] = None,
        contradiction_detector: Optional[object] = None,
        embeddings_by_time: Optional[list] = None,
        time_window_days: float = 30.0,
    ) -> dict[str, float]:
        """
        Run all extended metrics and return a summary dict.
        """
        results: dict[str, float] = {}

        if contradiction_detector is not None:
            results["contradiction_rate"] = self.contradiction_rate(
                contradiction_detector, memories
            )

        if audit_entries is not None:
            results["profile_drift_per_day"] = self.profile_drift(
                audit_entries, time_window_days
            )

        if embeddings_by_time is not None:
            results["memory_drift"] = self.memory_drift(embeddings_by_time)

        return results


# ===========================================================================
# Part 2 — Research Hypothesis Framework
# ===========================================================================


class HypothesisStatus(str, Enum):
    UNTESTED = "untested"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


@dataclass
class ResearchHypothesis:
    """
    A structured scientific hypothesis about Blix's memory system.

    Fields
    ------
    id:
        Short identifier, e.g. "H1".
    statement:
        Full natural-language hypothesis statement.
    independent_variable:
        What is being changed (e.g. "hierarchical compression enabled").
    dependent_variable:
        What is being measured (e.g. "retrieval_precision at 30 days").
    metric_name:
        Key in the EvalReport.summary_dict().
    baseline_condition:
        Description of the control condition.
    treatment_condition:
        Description of the experimental condition.
    expected_direction:
        "higher" or "lower" — expected direction of metric change.
    results:
        List of (condition, metric_value) tuples from experiment runs.
    status:
        Current support status.
    notes:
        Free-text notes.
    """

    id: str
    statement: str
    independent_variable: str
    dependent_variable: str
    metric_name: str
    baseline_condition: str
    treatment_condition: str
    expected_direction: str = "higher"
    results: list[dict] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.UNTESTED
    notes: str = ""

    def record_result(
        self,
        condition: str,
        metric_value: float,
        report: Optional[EvalReport] = None,
    ) -> None:
        """Record one experimental result."""
        self.results.append({
            "condition": condition,
            "metric_value": metric_value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "report_dataset": report.dataset_name if report else None,
        })

    def evaluate_support(self) -> HypothesisStatus:
        """
        Auto-evaluate hypothesis support from recorded results.

        Requires at least one baseline and one treatment result.
        """
        baseline_vals = [r["metric_value"] for r in self.results if r["condition"] == "baseline"]
        treatment_vals = [r["metric_value"] for r in self.results if r["condition"] == "treatment"]

        if not baseline_vals or not treatment_vals:
            self.status = HypothesisStatus.UNTESTED
            return self.status

        mean_base = sum(baseline_vals) / len(baseline_vals)
        mean_treat = sum(treatment_vals) / len(treatment_vals)
        delta = mean_treat - mean_base

        # Effect size threshold: 5% relative improvement required
        threshold = abs(mean_base) * 0.05

        if self.expected_direction == "higher":
            if delta > threshold:
                self.status = HypothesisStatus.SUPPORTED
            elif delta < -threshold:
                self.status = HypothesisStatus.REFUTED
            else:
                self.status = HypothesisStatus.INCONCLUSIVE
        else:  # "lower"
            if delta < -threshold:
                self.status = HypothesisStatus.SUPPORTED
            elif delta > threshold:
                self.status = HypothesisStatus.REFUTED
            else:
                self.status = HypothesisStatus.INCONCLUSIVE

        log.info(
            "Hypothesis %s: baseline=%.4f treatment=%.4f → %s",
            self.id, mean_base, mean_treat, self.status.value,
        )
        return self.status

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "independent_variable": self.independent_variable,
            "dependent_variable": self.dependent_variable,
            "metric_name": self.metric_name,
            "baseline_condition": self.baseline_condition,
            "treatment_condition": self.treatment_condition,
            "expected_direction": self.expected_direction,
            "results": self.results,
            "status": self.status.value,
            "notes": self.notes,
        }


class HypothesisRegistry:
    """
    Stores and manages all research hypotheses for Blix.

    Built-in hypotheses correspond to the paper's core claims.
    """

    # Pre-defined hypotheses from the reviewer's feedback
    BUILT_IN: list[dict] = [
        {
            "id": "H1",
            "statement": (
                "Hierarchical memory + profile evolution improves long-horizon "
                "personalization recall by ≥10% vs flat retrieval baseline."
            ),
            "independent_variable": "hierarchical compression + profile evolver enabled",
            "dependent_variable": "retrieval_recall at 30+ day memory age",
            "metric_name": "retrieval_recall",
            "baseline_condition": "flat retrieval (v0.2)",
            "treatment_condition": "hierarchical + profile (v0.3)",
            "expected_direction": "higher",
        },
        {
            "id": "H2",
            "statement": (
                "Graph-augmented memory retrieval reduces memory drift "
                "by ≥15% compared to non-graph retrieval."
            ),
            "independent_variable": "memory graph + graph-biased retrieval enabled",
            "dependent_variable": "memory_drift (cosine drift over 30 days)",
            "metric_name": "memory_drift",
            "baseline_condition": "semantic retrieval only",
            "treatment_condition": "graph-augmented retrieval",
            "expected_direction": "lower",
        },
        {
            "id": "H3",
            "statement": (
                "MMR diversification improves fact coverage without reducing "
                "retrieval precision by more than 5%."
            ),
            "independent_variable": "MMR reranker enabled (λ=0.5)",
            "dependent_variable": "fact_accuracy and retrieval_precision",
            "metric_name": "fact_accuracy",
            "baseline_condition": "greedy top-k retrieval",
            "treatment_condition": "MMR-diversified retrieval",
            "expected_direction": "higher",
        },
        {
            "id": "H4",
            "statement": (
                "Confidence-propagated extractions produce fewer hallucinated facts "
                "than unfiltered extractions."
            ),
            "independent_variable": "FactVerifier + ConfidencePropagator enabled",
            "dependent_variable": "hallucination_rate",
            "metric_name": "hallucination_rate",
            "baseline_condition": "raw CoT extraction without verification",
            "treatment_condition": "verified extraction with confidence propagation",
            "expected_direction": "lower",
        },
    ]

    def __init__(self, registry_file: Optional[Path] = None) -> None:
        self._file = registry_file
        self._hypotheses: dict[str, ResearchHypothesis] = {}
        self._load_built_in()
        if registry_file:
            self._load_file()

    def _load_built_in(self) -> None:
        for h in self.BUILT_IN:
            self._hypotheses[h["id"]] = ResearchHypothesis(**h)  # type: ignore[arg-type]

    def _load_file(self) -> None:
        if not self._file or not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            for raw in data:
                h = ResearchHypothesis(**raw)
                self._hypotheses[h.id] = h
        except Exception as exc:
            log.warning("HypothesisRegistry load failed: %s", exc)

    def save(self) -> None:
        if self._file is None:
            return
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([h.to_dict() for h in self._hypotheses.values()], fh, indent=2)

    def get(self, hypothesis_id: str) -> Optional[ResearchHypothesis]:
        return self._hypotheses.get(hypothesis_id)

    def list_all(self) -> list[ResearchHypothesis]:
        return list(self._hypotheses.values())

    def add(self, hypothesis: ResearchHypothesis) -> None:
        self._hypotheses[hypothesis.id] = hypothesis
        if self._file:
            self.save()

    def evaluate_all(self) -> dict[str, HypothesisStatus]:
        return {hid: h.evaluate_support() for hid, h in self._hypotheses.items()}

    def print_summary(self) -> None:
        print("\n=== Research Hypothesis Status ===")
        for h in self._hypotheses.values():
            print(f"  [{h.status.value.upper():12s}] {h.id}: {h.statement[:70]}…")
        print()
