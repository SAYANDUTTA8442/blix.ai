"""
StateBench-lite Evaluation — Blix v0.3.7  (New module 10)

Without benchmarks, temporal correctness is unverifiable. ``StateMetrics``
provides the six metrics the spec calls for, aligned with the user's
StateBench research roadmap:

    Current State Accuracy     — does StateTracker.current() match ground truth?
    Historical State Accuracy   — does StateTracker.at_time() match ground truth?
    Transition Accuracy          — were transitions detected at the right times?
    State Hallucination Rate      — fraction of queries answered with an
                                     unsupported/fabricated value
    Belief Drift                   — how much has belief confidence shifted
                                     over a window (volatility, not just decay)
    Truth Consistency                — fraction of TruthStatus assignments
                                        that are internally consistent
                                        (no ACTIVE+SUPERSEDED both claiming
                                        to be the live value, etc.)

``StateMetrics`` extends ``AdaptiveAgentEvaluator`` (v0.3.6) so it slots
into the same evaluation tower used throughout the project, without
requiring any agent-specific machinery to compute purely temporal metrics.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.state_tracker import StateTracker
from core.truth_manager import TruthManager, TruthStatus
from evaluation.agent_benchmark import AdaptiveAgentEvaluator
from memory.beliefs import BeliefStore


# ---------------------------------------------------------------------------
# Ground-truth case models
# ---------------------------------------------------------------------------


@dataclass
class StateAccuracyCase:
    """One ground-truth check for current or historical state."""

    entity: str
    attribute: str
    expected_value: str
    at_time: Optional[str] = None    # None = check current state


@dataclass
class TransitionAccuracyCase:
    """One ground-truth check for whether/when a transition occurred."""

    entity: str
    attribute: str
    expected_from: Optional[str]
    expected_to: str
    expected_around_time: Optional[str] = None   # ISO timestamp, for tolerance check
    tolerance_days: float = 7.0


# ---------------------------------------------------------------------------
# StateMetrics
# ---------------------------------------------------------------------------


class StateMetrics(AdaptiveAgentEvaluator):
    """
    Extends ``AdaptiveAgentEvaluator`` (v0.3.6) with v0.3.7 temporal/truth
    correctness metrics — completing the evaluation tower:

        MemoryEvaluator → ... → AgentEvaluator → AdaptiveAgentEvaluator → StateMetrics
    """

    # ------------------------------------------------------------------
    # Current / Historical State Accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def current_state_accuracy(
        tracker: StateTracker, cases: list[StateAccuracyCase],
    ) -> float:
        """Fraction of current-state cases where StateTracker matches ground truth."""
        current_cases = [c for c in cases if c.at_time is None]
        if not current_cases:
            return 1.0
        correct = 0
        for case in current_cases:
            snap = tracker.current(case.entity, case.attribute)
            if snap is not None and _values_match(snap.value, case.expected_value):
                correct += 1
        return correct / len(current_cases)

    @staticmethod
    def historical_state_accuracy(
        tracker: StateTracker, cases: list[StateAccuracyCase],
    ) -> float:
        """Fraction of historical-state cases where StateTracker.at_time() matches ground truth."""
        historical_cases = [c for c in cases if c.at_time is not None]
        if not historical_cases:
            return 1.0
        correct = 0
        for case in historical_cases:
            snap = tracker.at_time(case.entity, case.attribute, case.at_time)
            if snap is not None and _values_match(snap.value, case.expected_value):
                correct += 1
        return correct / len(historical_cases)

    @staticmethod
    def combined_state_accuracy(
        tracker: StateTracker, cases: list[StateAccuracyCase],
    ) -> float:
        """All cases (current + historical) combined into one accuracy figure."""
        if not cases:
            return 1.0
        correct = 0
        for case in cases:
            if case.at_time is None:
                snap = tracker.current(case.entity, case.attribute)
            else:
                snap = tracker.at_time(case.entity, case.attribute, case.at_time)
            if snap is not None and _values_match(snap.value, case.expected_value):
                correct += 1
        return correct / len(cases)

    # ------------------------------------------------------------------
    # Transition Accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def transition_accuracy(
        tracker: StateTracker, cases: list[TransitionAccuracyCase],
    ) -> float:
        """
        Fraction of expected transitions that are actually present in
        ``StateTracker`` history with the right from/to values (and,
        if ``expected_around_time`` is given, within tolerance).
        """
        if not cases:
            return 1.0
        correct = 0
        for case in cases:
            history = tracker.history(case.entity, case.attribute)
            match = None
            for i, snap in enumerate(history):
                if not _values_match(snap.value, case.expected_to):
                    continue
                prev_value = history[i - 1].value if i > 0 else None
                if case.expected_from is not None and not _values_match(prev_value or "", case.expected_from):
                    continue
                if case.expected_from is None and prev_value is not None:
                    continue
                match = snap
                break
            if match is None:
                continue
            if case.expected_around_time is not None:
                delta_days = abs(_days_between(match.start_time, case.expected_around_time))
                if delta_days > case.tolerance_days:
                    continue
            correct += 1
        return correct / len(cases)

    # ------------------------------------------------------------------
    # State Hallucination Rate
    # ------------------------------------------------------------------

    @staticmethod
    def state_hallucination_rate(
        predicted_answers: list[Optional[str]],
        ground_truth_exists: list[bool],
    ) -> float:
        """
        Fraction of queries where Blix CONFIDENTLY answered (returned a
        non-empty value) despite there being no supporting ground truth
        — i.e. fabricated a value rather than saying "unknown".

        ``predicted_answers[i]``: what Blix answered (None/"" = declined to answer).
        ``ground_truth_exists[i]``: whether a ground-truth value actually exists.
        """
        if not predicted_answers or len(predicted_answers) != len(ground_truth_exists):
            return 0.0
        hallucinations = sum(
            1 for ans, exists in zip(predicted_answers, ground_truth_exists)
            if ans and not exists
        )
        # Hallucination rate is only meaningful relative to cases where
        # no ground truth exists (the only cases where confidently
        # answering would BE a hallucination).
        no_truth_count = sum(1 for exists in ground_truth_exists if not exists)
        if no_truth_count == 0:
            return 0.0
        return hallucinations / no_truth_count

    # ------------------------------------------------------------------
    # Belief Drift
    # ------------------------------------------------------------------

    @staticmethod
    def belief_drift(confidence_before: dict[str, float], confidence_after: dict[str, float]) -> float:
        """
        Mean absolute change in confidence across beliefs present in both
        snapshots — measures volatility, not direction. High drift means
        beliefs are swinging around a lot (possibly a sign of noisy
        evidence or an unstable ContradictionResolver threshold);
        near-zero drift means beliefs are settling.
        """
        shared_ids = set(confidence_before) & set(confidence_after)
        if not shared_ids:
            return 0.0
        deltas = [abs(confidence_after[bid] - confidence_before[bid]) for bid in shared_ids]
        return round(sum(deltas) / len(deltas), 4)

    @staticmethod
    def belief_drift_from_store(
        belief_store: BeliefStore, previous_snapshot: dict[str, float],
    ) -> float:
        """Convenience wrapper: compute drift between a saved confidence snapshot and the live BeliefStore."""
        current = {b.belief_id: b.confidence for b in belief_store.all_active()}
        return StateMetrics.belief_drift(previous_snapshot, current)

    # ------------------------------------------------------------------
    # Truth Consistency
    # ------------------------------------------------------------------

    @staticmethod
    def truth_consistency(
        truth_manager: TruthManager, entity_attribute_pairs: list[tuple], tracker: StateTracker,
    ) -> float:
        """
        Fraction of (entity, attribute) pairs where TruthStatus assignments
        are internally consistent: exactly the currently-active
        StateSnapshot is ACTIVE, and all closed (non-active) snapshots for
        that pair are NOT ACTIVE (i.e. SUPERSEDED/HISTORICAL/ARCHIVED, not
        left dangling as ACTIVE which would mean two "current" values
        compete).
        """
        if not entity_attribute_pairs:
            return 1.0
        consistent = 0
        for entity, attribute in entity_attribute_pairs:
            history = tracker.history(entity, attribute)
            if not history:
                consistent += 1  # vacuously consistent — nothing to check
                continue

            active_snaps = [s for s in history if s.is_active]
            closed_snaps = [s for s in history if not s.is_active]

            # At most one snapshot should be the live (graph-active) one,
            # and its TruthStatus should be ACTIVE (not SUPERSEDED/ARCHIVED).
            ok = True
            if len(active_snaps) > 1:
                ok = False  # multiple "currently active" snapshots — inconsistent
            for s in active_snaps:
                if truth_manager.status_of(s.snapshot_id) not in (TruthStatus.ACTIVE, TruthStatus.CONFLICTING):
                    ok = False
            for s in closed_snaps:
                if truth_manager.status_of(s.snapshot_id) == TruthStatus.ACTIVE:
                    ok = False  # closed snapshot still claims ACTIVE — inconsistent

            if ok:
                consistent += 1
        return consistent / len(entity_attribute_pairs)

    # ------------------------------------------------------------------
    # Combined StateBench-lite pass
    # ------------------------------------------------------------------

    def run_statebench(
        self,
        tracker: StateTracker,
        truth_manager: TruthManager,
        state_cases: list[StateAccuracyCase],
        transition_cases: list[TransitionAccuracyCase],
        entity_attribute_pairs: Optional[list[tuple]] = None,
    ) -> dict[str, float]:
        """Run the full StateBench-lite metric suite and return a summary dict."""
        pairs = entity_attribute_pairs or [(c.entity, c.attribute) for c in state_cases]
        return {
            "current_state_accuracy": self.current_state_accuracy(tracker, state_cases),
            "historical_state_accuracy": self.historical_state_accuracy(tracker, state_cases),
            "combined_state_accuracy": self.combined_state_accuracy(tracker, state_cases),
            "transition_accuracy": self.transition_accuracy(tracker, transition_cases),
            "truth_consistency": self.truth_consistency(truth_manager, pairs, tracker),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _values_match(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def _days_between(iso_a: str, iso_b: str) -> float:
    from datetime import datetime
    try:
        da = datetime.fromisoformat(iso_a.replace("Z", "+00:00"))
        db = datetime.fromisoformat(iso_b.replace("Z", "+00:00"))
        return (da - db).total_seconds() / 86400.0
    except Exception:
        return 9999.0
