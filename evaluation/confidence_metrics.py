"""
Confidence Metrics — Blix v0.3.8  (New module 10a)

Measures whether Blix's stated confidence actually tracks reality —
"confidence calibration" in the forecasting-evaluation sense. A
perfectly calibrated system that says "80% confident" should be right
about 80% of the time across many such predictions; this module
quantifies how far Blix is from that ideal.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CalibrationCase:
    """One (stated confidence, actual outcome) pair."""

    confidence: float       # 0-1, what Blix said
    was_correct: bool        # whether the prediction/answer was actually right


@dataclass
class CalibrationBucketResult:
    """Calibration result for one confidence bucket."""

    bucket_low: float
    bucket_high: float
    mean_confidence: float
    actual_accuracy: float
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "bucket": f"{self.bucket_low:.1f}-{self.bucket_high:.1f}",
            "mean_confidence": round(self.mean_confidence, 4),
            "actual_accuracy": round(self.actual_accuracy, 4),
            "sample_count": self.sample_count,
            "gap": round(abs(self.mean_confidence - self.actual_accuracy), 4),
        }


class ConfidenceMetrics:
    """Confidence calibration metrics."""

    @staticmethod
    def brier_score(cases: list[CalibrationCase]) -> float:
        """
        Mean squared error between stated confidence and outcome (0/1).
        Lower is better; 0 is perfect calibration, 0.25 is the score of
        always guessing 0.5, 1.0 is maximally wrong.
        """
        if not cases:
            return 0.0
        total = sum((c.confidence - (1.0 if c.was_correct else 0.0)) ** 2 for c in cases)
        return round(total / len(cases), 4)

    @staticmethod
    def calibration_buckets(cases: list[CalibrationCase], bucket_size: float = 0.2) -> list[CalibrationBucketResult]:
        """
        Group cases into confidence buckets (e.g. 0.0-0.2, 0.2-0.4, ...)
        and compare mean stated confidence vs. actual accuracy within
        each bucket — the classic calibration-curve view.
        """
        if not cases:
            return []
        n_buckets = max(1, round(1.0 / bucket_size))
        buckets: list[list[CalibrationCase]] = [[] for _ in range(n_buckets)]
        for c in cases:
            idx = min(n_buckets - 1, int(c.confidence / bucket_size))
            buckets[idx].append(c)

        results = []
        for i, bucket_cases in enumerate(buckets):
            if not bucket_cases:
                continue
            low, high = i * bucket_size, (i + 1) * bucket_size
            mean_conf = sum(c.confidence for c in bucket_cases) / len(bucket_cases)
            accuracy = sum(1 for c in bucket_cases if c.was_correct) / len(bucket_cases)
            results.append(CalibrationBucketResult(
                bucket_low=low, bucket_high=high, mean_confidence=mean_conf,
                actual_accuracy=accuracy, sample_count=len(bucket_cases),
            ))
        return results

    @staticmethod
    def expected_calibration_error(cases: list[CalibrationCase], bucket_size: float = 0.2) -> float:
        """
        Weighted mean absolute gap between confidence and accuracy across
        buckets (ECE) — a single summary number for overall calibration
        quality. 0 = perfectly calibrated.
        """
        buckets = ConfidenceMetrics.calibration_buckets(cases, bucket_size)
        if not buckets:
            return 0.0
        total_samples = sum(b.sample_count for b in buckets)
        weighted_gap = sum(abs(b.mean_confidence - b.actual_accuracy) * b.sample_count for b in buckets)
        return round(weighted_gap / total_samples, 4)

    @staticmethod
    def overconfidence_rate(cases: list[CalibrationCase], margin: float = 0.2) -> float:
        """
        Fraction of cases where stated confidence exceeded actual
        correctness by more than ``margin`` (high confidence, wrong answer).
        """
        if not cases:
            return 0.0
        overconfident = sum(
            1 for c in cases
            if not c.was_correct and c.confidence > margin
        )
        return round(overconfident / len(cases), 4)

    @staticmethod
    def underconfidence_rate(cases: list[CalibrationCase], margin: float = 0.2) -> float:
        """
        Fraction of cases where Blix was right but stated confidence was
        surprisingly low (under-claiming correct answers).
        """
        if not cases:
            return 0.0
        underconfident = sum(
            1 for c in cases
            if c.was_correct and c.confidence < (1.0 - margin)
        )
        return round(underconfident / len(cases), 4)
