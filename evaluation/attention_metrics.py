"""
Attention Metrics — Blix v0.3.9  (New module 9b)

Measures whether ``workspace.attention_manager.AttentionManager`` is
actually directing attention well: does it correctly let high-value
candidates in and correctly keep low-value candidates out, relative to
some ground truth of what SHOULD have mattered.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass

from workspace.attention_manager import AttentionScore


@dataclass
class AttentionGroundTruthCase:
    """One (candidate ref_id, should it have entered the workspace?) ground-truth pair."""

    ref_id: str
    should_have_entered: bool


class AttentionMetrics:
    """Attention-scoring and selection-quality metrics."""

    @staticmethod
    def selection_accuracy(
        scored: list[AttentionScore], ground_truth: list[AttentionGroundTruthCase], threshold: float,
    ) -> float:
        """
        Fraction of candidates where (score >= threshold) matches
        ``should_have_entered``.
        """
        if not ground_truth:
            return 1.0
        score_by_id = {s.candidate.ref_id: s.score for s in scored}
        correct = 0
        for case in ground_truth:
            actual_score = score_by_id.get(case.ref_id, 0.0)
            predicted_entry = actual_score >= threshold
            if predicted_entry == case.should_have_entered:
                correct += 1
        return round(correct / len(ground_truth), 4)

    @staticmethod
    def mean_score(scored: list[AttentionScore]) -> float:
        if not scored:
            return 0.0
        return round(sum(s.score for s in scored) / len(scored), 4)

    @staticmethod
    def score_variance(scored: list[AttentionScore]) -> float:
        """Variance of attention scores — low variance suggests the manager isn't discriminating well."""
        if not scored:
            return 0.0
        mean = sum(s.score for s in scored) / len(scored)
        return round(sum((s.score - mean) ** 2 for s in scored) / len(scored), 4)

    @staticmethod
    def false_admission_rate(scored: list[AttentionScore], ground_truth: list[AttentionGroundTruthCase], threshold: float) -> float:
        """Fraction of items that entered (score>=threshold) but shouldn't have, among all that shouldn't have."""
        score_by_id = {s.candidate.ref_id: s.score for s in scored}
        should_not_enter = [c for c in ground_truth if not c.should_have_entered]
        if not should_not_enter:
            return 0.0
        false_admissions = sum(1 for c in should_not_enter if score_by_id.get(c.ref_id, 0.0) >= threshold)
        return round(false_admissions / len(should_not_enter), 4)

    @staticmethod
    def false_rejection_rate(scored: list[AttentionScore], ground_truth: list[AttentionGroundTruthCase], threshold: float) -> float:
        """Fraction of items that were rejected (score<threshold) but shouldn't have been, among all that should have entered."""
        score_by_id = {s.candidate.ref_id: s.score for s in scored}
        should_enter = [c for c in ground_truth if c.should_have_entered]
        if not should_enter:
            return 0.0
        false_rejections = sum(1 for c in should_enter if score_by_id.get(c.ref_id, 0.0) < threshold)
        return round(false_rejections / len(should_enter), 4)
