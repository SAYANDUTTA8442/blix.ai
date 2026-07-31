"""
Coordination Metrics — Blix v0.3.9  (New module 9c)

Completes the evaluation tower:

    ... → AdaptiveAgentEvaluator → StateMetrics → MetacognitionMetrics → CoordinationMetrics

Measures whether previously-isolated subsystems are actually acting as
a coordinated cognitive system: does specialist consensus converge or
stay contested, do registered subsystems actually participate in
broadcasts, and how much of the system's activity flows through the
shared workspace vs. happening in isolation.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.metacognition_metrics import MetacognitionMetrics
from specialists.consensus import ConsensusResult
from workspace.broadcast_bus import BroadcastBus


@dataclass
class SubsystemParticipation:
    """How much one named subsystem participated in broadcasts."""

    subsystem: str
    events_received: int


class CoordinationMetrics(MetacognitionMetrics):
    """
    Extends ``MetacognitionMetrics`` (v0.3.8) with v0.3.9 coordination
    evaluation: consensus convergence, subsystem participation, and
    workspace-mediated vs. isolated activity.
    """

    # ------------------------------------------------------------------
    # Consensus convergence
    # ------------------------------------------------------------------

    @staticmethod
    def consensus_convergence_rate(results: list[ConsensusResult]) -> float:
        """Fraction of consensus decisions that were NOT contested (agreement_ratio >= 2/3)."""
        if not results:
            return 1.0
        converged = sum(1 for r in results if not r.is_contested)
        return round(converged / len(results), 4)

    @staticmethod
    def mean_agreement_ratio(results: list[ConsensusResult]) -> float:
        if not results:
            return 0.0
        return round(sum(r.agreement_ratio for r in results) / len(results), 4)

    @staticmethod
    def no_opinion_rate(results: list[ConsensusResult]) -> float:
        """Fraction of all specialist opinions across all decisions that were 'no_opinion'."""
        all_opinions = [o for r in results for o in r.opinions]
        if not all_opinions:
            return 0.0
        no_opinion = sum(1 for o in all_opinions if not o.has_opinion)
        return round(no_opinion / len(all_opinions), 4)

    # ------------------------------------------------------------------
    # Broadcast / subsystem participation
    # ------------------------------------------------------------------

    @staticmethod
    def subsystem_participation_rate(broadcast_bus: BroadcastBus) -> float:
        """
        Fraction of registered subsystems that have received at least
        one broadcast (vs. registered but never actually notified —
        a coordination gap, e.g. a mismatched event-type subscription).

        Returns 0.0 if no subsystems are registered (nothing CAN
        participate) and 1.0 only if subsystems are registered AND no
        broadcast went unheard.
        """
        registered = broadcast_bus.registered_subsystems()
        if not registered:
            return 0.0
        if broadcast_bus.broadcast_count == 0:
            return 0.0
        zero_listener_broadcasts = broadcast_bus.broadcasts_with_zero_listeners()
        participating_fraction = 1.0 - (len(zero_listener_broadcasts) / broadcast_bus.broadcast_count)
        return round(participating_fraction, 4)

    @staticmethod
    def isolation_rate(broadcast_bus: BroadcastBus) -> float:
        """Fraction of broadcasts that reached zero listeners — the inverse coordination-gap signal."""
        total = broadcast_bus.broadcast_count
        if total == 0:
            return 0.0
        return round(len(broadcast_bus.broadcasts_with_zero_listeners()) / total, 4)

    # ------------------------------------------------------------------
    # Combined pass
    # ------------------------------------------------------------------

    def run_coordination_bench(
        self, consensus_results: list[ConsensusResult], broadcast_bus: BroadcastBus | None = None,
    ) -> dict[str, float]:
        """Run the full v0.3.9 coordination metric suite."""
        results = {
            "consensus_convergence_rate": self.consensus_convergence_rate(consensus_results),
            "mean_agreement_ratio": self.mean_agreement_ratio(consensus_results),
            "no_opinion_rate": self.no_opinion_rate(consensus_results),
        }
        if broadcast_bus is not None:
            results["subsystem_participation_rate"] = self.subsystem_participation_rate(broadcast_bus)
            results["isolation_rate"] = self.isolation_rate(broadcast_bus)
        return results
