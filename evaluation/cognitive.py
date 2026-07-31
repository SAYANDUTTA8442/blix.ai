"""
Cognitive Evaluation Extensions — Blix v0.3.2  (Feature 5)

Extends ``ExtendedMemoryEvaluator`` (v0.3.1) with the metric families
required to evaluate the v0.3.2 "cognitive knowledge system":

Retrieval
    Recall@K, MRR (Mean Reciprocal Rank), Precision@K (already present
    as ``precision_at_k`` in the base evaluator)

Memory
    Retention Rate, Forgetting Rate, Memory Drift (Retention/Drift
    already present in v0.3.1; Forgetting Rate added here)

Profile
    Profile Accuracy (v0.3), Profile Stability (new)

Projects
    Project Accuracy, Milestone Accuracy

Reflection
    Insight Accuracy, Reflection Consistency

Deliverable namespace: ``blix_eval`` — this module is the implementation;
``evaluation/blix_eval/__init__.py`` re-exports it under that name for the
spec-mandated import path ``from blix_eval import CognitiveEvaluator``.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from evaluation.research import ExtendedMemoryEvaluator
from utils.logger import get_logger

log = get_logger(__name__)


class CognitiveEvaluator(ExtendedMemoryEvaluator):
    """
    Full v0.3.2 evaluation suite. Extends ``ExtendedMemoryEvaluator``
    (which itself extends the v0.3 ``MemoryEvaluator``) with retrieval
    ranking metrics and cognitive-layer (project/reflection) metrics.
    """

    # ------------------------------------------------------------------
    # Retrieval — Recall@K, MRR
    # ------------------------------------------------------------------

    @staticmethod
    def recall_at_k(retrieved: list[int], relevant: list[int], k: Optional[int] = None) -> float:
        """
        Recall@K: fraction of relevant items found in the top-K retrieved.

        If ``k`` is None, uses the full ``retrieved`` list (equivalent to
        the base ``recall_at_k``).
        """
        if not relevant:
            return 1.0
        top = retrieved[:k] if k is not None else retrieved
        hits = sum(1 for r in relevant if r in top)
        return hits / len(relevant)

    @staticmethod
    def mean_reciprocal_rank(retrieved: list[int], relevant: list[int]) -> float:
        """
        MRR: reciprocal of the rank of the first relevant item, 0 if none found.

        Rank is 1-indexed (first position → reciprocal rank 1.0).
        """
        relevant_set = set(relevant)
        for i, item in enumerate(retrieved, start=1):
            if item in relevant_set:
                return 1.0 / i
        return 0.0

    @staticmethod
    def mrr_batch(results: list[tuple[list[int], list[int]]]) -> float:
        """Mean MRR across multiple (retrieved, relevant) pairs."""
        if not results:
            return 0.0
        scores = [
            CognitiveEvaluator.mean_reciprocal_rank(r, rel)
            for r, rel in results
        ]
        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # Memory — Forgetting Rate
    # ------------------------------------------------------------------

    @staticmethod
    def forgetting_rate(
        lifecycle_manager: object,  # MemoryLifecycleManager
    ) -> float:
        """
        Fraction of all tracked memories that have transitioned out of
        ACTIVE (i.e. compressed, archived, or deleted).

        Complements ``retention_over_time`` (retrieval-based) with a
        storage-based forgetting measure.
        """
        counts = lifecycle_manager.state_counts()  # type: ignore[union-attr]
        total = sum(counts.values())
        if total == 0:
            return 0.0
        forgotten = counts.get("compressed", 0) + counts.get("archived", 0) + counts.get("deleted", 0)
        return forgotten / total

    # ------------------------------------------------------------------
    # Profile — Stability
    # ------------------------------------------------------------------

    @staticmethod
    def profile_stability(
        audit_entries: list,    # list[ProfileAuditEntry]
        total_turns: int,
    ) -> float:
        """
        Profile Stability: 1 - (profile_changes / total_conversation_turns).

        Higher = more stable profile (fewer changes per turn).
        Returns 1.0 if there have been no turns yet.
        """
        if total_turns <= 0:
            return 1.0
        change_rate = len(audit_entries) / total_turns
        return max(0.0, 1.0 - change_rate)

    # ------------------------------------------------------------------
    # Projects — Accuracy, Milestone Accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def project_accuracy(
        actual_states: list,        # list[ProjectState]
        ground_truth: dict[str, dict],  # project_name -> {focus, risk_level, progress}
    ) -> float:
        """
        Fraction of ground-truth project fields (focus, risk_level,
        progress within tolerance) that match the actual ``ProjectState``.

        Progress is considered correct if within ±10 of ground truth.
        """
        if not ground_truth:
            return 1.0
        actual_by_name = {p.project_name.lower(): p for p in actual_states}
        total = 0
        correct = 0
        for name, gt in ground_truth.items():
            actual = actual_by_name.get(name.lower())
            for field, expected in gt.items():
                total += 1
                if actual is None:
                    continue
                actual_val = getattr(actual, field, None)
                if field == "progress":
                    if actual_val is not None and abs(actual_val - expected) <= 10:
                        correct += 1
                elif field == "risk_level":
                    actual_str = getattr(actual_val, "value", actual_val)
                    if str(actual_str).lower() == str(expected).lower():
                        correct += 1
                else:
                    if str(actual_val).lower() == str(expected).lower():
                        correct += 1
        return correct / total if total else 1.0

    @staticmethod
    def milestone_accuracy(
        actual_goals: list,         # list[Goal]
        ground_truth_milestones: dict[str, list[str]],  # goal_title -> [completed milestone titles]
    ) -> float:
        """
        Fraction of ground-truth-completed milestones that are marked
        DONE in the actual ``Goal`` objects.
        """
        if not ground_truth_milestones:
            return 1.0
        from reflection.goal_tracker import ItemStatus

        actual_by_title = {g.title.lower(): g for g in actual_goals}
        total = 0
        correct = 0
        for goal_title, completed_titles in ground_truth_milestones.items():
            goal = actual_by_title.get(goal_title.lower())
            for m_title in completed_titles:
                total += 1
                if goal is None:
                    continue
                for m in goal.milestones:
                    if m.title.lower() == m_title.lower() and m.status == ItemStatus.DONE:
                        correct += 1
                        break
        return correct / total if total else 1.0

    # ------------------------------------------------------------------
    # Reflection — Insight Accuracy, Consistency
    # ------------------------------------------------------------------

    @staticmethod
    def insight_accuracy(
        insights: list,             # list[Insight]
        ground_truth_insights: list[str],
        min_overlap_words: int = 3,
    ) -> float:
        """
        Fraction of generated insights that overlap meaningfully
        (≥ min_overlap_words shared content words) with at least one
        ground-truth insight.
        """
        if not insights:
            return 1.0
        if not ground_truth_insights:
            return 1.0

        gt_token_sets = [_content_words(g) for g in ground_truth_insights]
        hits = 0
        for ins in insights:
            ins_tokens = _content_words(getattr(ins, "insight", str(ins)))
            if any(len(ins_tokens & gt) >= min_overlap_words for gt in gt_token_sets):
                hits += 1
        return hits / len(insights)

    @staticmethod
    def reflection_consistency(
        insights_run_a: list,   # list[Insight] from one reflection run
        insights_run_b: list,   # list[Insight] from a re-run on the same material
        min_overlap_words: int = 3,
    ) -> float:
        """
        Reflection Consistency: stability of reflection output across
        repeated runs on the same material.

        Computed as the fraction of insights in run A that have a
        semantically-overlapping counterpart in run B (symmetric mean
        of both directions).
        """
        if not insights_run_a and not insights_run_b:
            return 1.0
        if not insights_run_a or not insights_run_b:
            return 0.0

        def _coverage(src: list, tgt: list) -> float:
            tgt_sets = [_content_words(getattr(i, "insight", str(i))) for i in tgt]
            hits = 0
            for ins in src:
                ins_tokens = _content_words(getattr(ins, "insight", str(ins)))
                if any(len(ins_tokens & t) >= min_overlap_words for t in tgt_sets):
                    hits += 1
            return hits / len(src)

        a_to_b = _coverage(insights_run_a, insights_run_b)
        b_to_a = _coverage(insights_run_b, insights_run_a)
        return (a_to_b + b_to_a) / 2

    # ------------------------------------------------------------------
    # Combined cognitive evaluation pass
    # ------------------------------------------------------------------

    def evaluate_cognitive(
        self,
        *,
        retrieval_results: Optional[list[tuple[list[int], list[int]]]] = None,
        lifecycle_manager: Optional[object] = None,
        audit_entries: Optional[list] = None,
        total_turns: int = 0,
        project_states: Optional[list] = None,
        project_ground_truth: Optional[dict] = None,
        goals: Optional[list] = None,
        milestone_ground_truth: Optional[dict] = None,
        insights: Optional[list] = None,
        insight_ground_truth: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """
        Run all applicable v0.3.2 cognitive metrics and return a summary dict.

        Every parameter is optional; only metrics whose required inputs
        are provided will appear in the result.
        """
        results: dict[str, float] = {}

        if retrieval_results is not None:
            results["mrr"] = self.mrr_batch(retrieval_results)
            recalls = [self.recall_at_k(r, rel) for r, rel in retrieval_results]
            results["recall_at_k"] = sum(recalls) / len(recalls) if recalls else 0.0

        if lifecycle_manager is not None:
            results["forgetting_rate"] = self.forgetting_rate(lifecycle_manager)

        if audit_entries is not None and total_turns > 0:
            results["profile_stability"] = self.profile_stability(audit_entries, total_turns)

        if project_states is not None and project_ground_truth is not None:
            results["project_accuracy"] = self.project_accuracy(project_states, project_ground_truth)

        if goals is not None and milestone_ground_truth is not None:
            results["milestone_accuracy"] = self.milestone_accuracy(goals, milestone_ground_truth)

        if insights is not None and insight_ground_truth is not None:
            results["insight_accuracy"] = self.insight_accuracy(insights, insight_ground_truth)

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_words(text: str) -> set[str]:
    import re
    stop = {"the", "a", "an", "is", "are", "was", "were", "user", "users",
            "to", "of", "in", "on", "and", "or", "this", "that", "has", "have"}
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in stop and len(w) > 2}
