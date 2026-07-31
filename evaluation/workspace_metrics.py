"""
Workspace Metrics — Blix v0.3.9  (New module 9a)

Measures the health of the ``workspace.global_workspace.GlobalWorkspace``
cycle itself: how often does important information actually make it
into the workspace, how often is it wasted (broadcast to nobody), and
how stable is the workspace's contents over time.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass

from workspace.broadcast_bus import BroadcastBus
from workspace.global_workspace import WorkspaceCycleResult


@dataclass
class WorkspaceCycleStats:
    """Summary stats across a batch of workspace cycles."""

    total_cycles: int
    total_entered: int
    total_rejected: int
    mean_entries_per_cycle: float
    entry_rate: float        # fraction of all submitted candidates that entered the workspace

    def to_dict(self) -> dict:
        return {
            "total_cycles": self.total_cycles,
            "total_entered": self.total_entered,
            "total_rejected": self.total_rejected,
            "mean_entries_per_cycle": round(self.mean_entries_per_cycle, 4),
            "entry_rate": round(self.entry_rate, 4),
        }


class WorkspaceMetrics:
    """Broadcast-quality and workspace-cycle-health metrics."""

    @staticmethod
    def cycle_stats(cycle_results: list[WorkspaceCycleResult]) -> WorkspaceCycleStats:
        if not cycle_results:
            return WorkspaceCycleStats(total_cycles=0, total_entered=0, total_rejected=0, mean_entries_per_cycle=0.0, entry_rate=0.0)

        total_entered = sum(len(c.entered) for c in cycle_results)
        total_rejected = sum(c.rejected_count for c in cycle_results)
        total_submitted = total_entered + total_rejected
        return WorkspaceCycleStats(
            total_cycles=len(cycle_results),
            total_entered=total_entered,
            total_rejected=total_rejected,
            mean_entries_per_cycle=total_entered / len(cycle_results),
            entry_rate=(total_entered / total_submitted) if total_submitted else 0.0,
        )

    @staticmethod
    def broadcast_quality(broadcast_bus: BroadcastBus) -> dict:
        """
        Broadcast quality from a ``BroadcastBus``'s log: mean listener
        count and fraction of broadcasts that reached nobody (wasted).
        """
        total = broadcast_bus.broadcast_count
        if total == 0:
            return {"total_broadcasts": 0, "mean_listener_count": 0.0, "zero_listener_rate": 0.0}
        zero_listener = len(broadcast_bus.broadcasts_with_zero_listeners())
        return {
            "total_broadcasts": total,
            "mean_listener_count": round(broadcast_bus.mean_listener_count(), 4),
            "zero_listener_rate": round(zero_listener / total, 4),
        }

    @staticmethod
    def utilization(entered_count: int, capacity: int) -> float:
        """How full the workspace is relative to its capacity (0-1)."""
        if capacity <= 0:
            return 0.0
        return round(min(1.0, entered_count / capacity), 4)
