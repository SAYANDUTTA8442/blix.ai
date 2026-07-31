"""
Metacognition Metrics — Blix v0.3.8  (New module 10c)

Completes the evaluation tower:

    MemoryEvaluator → ... → AdaptiveAgentEvaluator → StateMetrics → MetacognitionMetrics

Measures adaptation ability and strategy-switching behavior — does the
``metacognition.controller.MetaCognitiveController`` actually catch
problems and respond appropriately, and does that response correlate
with better outcomes?

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from evaluation.state_metrics import StateMetrics
from metacognition.controller import AdaptationAction, AdaptationDecision, CognitiveIssue, CognitiveMonitorReport


@dataclass
class AdaptationCase:
    """One ground-truth case: was an adaptation warranted, and did it happen?"""

    issue_present: bool          # was there actually a problem (ground truth)?
    issue_detected: bool          # did the controller's monitor report flag an issue?
    adaptation_taken: bool          # did the controller decide to act (action != NONE)?
    outcome_improved: Optional[bool] = None   # did the situation improve after adapting? (None = unknown/not applicable)


class MetacognitionMetrics(StateMetrics):
    """
    Extends ``StateMetrics`` (v0.3.7) with v0.3.8 meta-cognitive
    evaluation: self-awareness, adaptation ability, and strategy
    switching effectiveness.
    """

    # ------------------------------------------------------------------
    # Detection quality
    # ------------------------------------------------------------------

    @staticmethod
    def issue_detection_accuracy(cases: list[AdaptationCase]) -> float:
        """Fraction of cases where issue_detected matches issue_present (true positives + true negatives)."""
        if not cases:
            return 1.0
        correct = sum(1 for c in cases if c.issue_detected == c.issue_present)
        return round(correct / len(cases), 4)

    @staticmethod
    def false_alarm_rate(cases: list[AdaptationCase]) -> float:
        """Fraction of no-issue cases that were incorrectly flagged (false positives)."""
        no_issue_cases = [c for c in cases if not c.issue_present]
        if not no_issue_cases:
            return 0.0
        false_alarms = sum(1 for c in no_issue_cases if c.issue_detected)
        return round(false_alarms / len(no_issue_cases), 4)

    @staticmethod
    def missed_detection_rate(cases: list[AdaptationCase]) -> float:
        """Fraction of actual-issue cases that went undetected (false negatives)."""
        issue_cases = [c for c in cases if c.issue_present]
        if not issue_cases:
            return 0.0
        missed = sum(1 for c in issue_cases if not c.issue_detected)
        return round(missed / len(issue_cases), 4)

    # ------------------------------------------------------------------
    # Adaptation ability
    # ------------------------------------------------------------------

    @staticmethod
    def adaptation_responsiveness(cases: list[AdaptationCase]) -> float:
        """Of cases where an issue was detected, fraction where an adaptation was actually taken."""
        detected_cases = [c for c in cases if c.issue_detected]
        if not detected_cases:
            return 1.0
        responded = sum(1 for c in detected_cases if c.adaptation_taken)
        return round(responded / len(detected_cases), 4)

    @staticmethod
    def adaptation_effectiveness(cases: list[AdaptationCase]) -> float:
        """
        Of cases where an adaptation was taken AND the outcome is known,
        fraction where the outcome actually improved. Cases with
        ``outcome_improved=None`` are excluded (not just counted as failures).
        """
        adapted_with_known_outcome = [
            c for c in cases if c.adaptation_taken and c.outcome_improved is not None
        ]
        if not adapted_with_known_outcome:
            return 1.0
        improved = sum(1 for c in adapted_with_known_outcome if c.outcome_improved)
        return round(improved / len(adapted_with_known_outcome), 4)

    # ------------------------------------------------------------------
    # Strategy switching
    # ------------------------------------------------------------------

    @staticmethod
    def strategy_switch_rate(decisions: list[AdaptationDecision]) -> float:
        """Fraction of adaptation decisions that resulted in an actual action (not NONE)."""
        if not decisions:
            return 0.0
        switched = sum(1 for d in decisions if d.action != AdaptationAction.NONE)
        return round(switched / len(decisions), 4)

    @staticmethod
    def action_distribution(decisions: list[AdaptationDecision]) -> dict[str, float]:
        """Fraction of decisions falling into each ``AdaptationAction`` category."""
        if not decisions:
            return {}
        counts: dict[str, int] = {}
        for d in decisions:
            counts[d.action.value] = counts.get(d.action.value, 0) + 1
        return {k: round(v / len(decisions), 4) for k, v in counts.items()}

    # ------------------------------------------------------------------
    # Combined pass
    # ------------------------------------------------------------------

    def run_metacognition_bench(self, cases: list[AdaptationCase], decisions: Optional[list[AdaptationDecision]] = None) -> dict[str, float]:
        """Run the full v0.3.8 meta-cognition metric suite."""
        results = {
            "issue_detection_accuracy": self.issue_detection_accuracy(cases),
            "false_alarm_rate": self.false_alarm_rate(cases),
            "missed_detection_rate": self.missed_detection_rate(cases),
            "adaptation_responsiveness": self.adaptation_responsiveness(cases),
            "adaptation_effectiveness": self.adaptation_effectiveness(cases),
        }
        if decisions:
            results["strategy_switch_rate"] = self.strategy_switch_rate(decisions)
        return results
