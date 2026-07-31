"""
Global Workspace — Blix v0.3.9  (New module 1)

The brain's "working stage." Through v0.3.8, Memory, Planner,
Reflection, and SelfModel operated as independent subsystems — each
intelligent on its own, but with no shared stage where information
from one could become visible to all the others. ``GlobalWorkspace``
is that shared stage:

    Global Workspace
                          ↓
         --------------------------------
         ↓       ↓        ↓       ↓
     Memory Planner Reflection SelfModel
         ↓       ↓        ↓       ↓
                 Broadcast

Concretely, ``GlobalWorkspace`` composes three things that already
exist as standalone modules in this release:

    workspace.attention_manager.AttentionManager   — decides what's important enough to enter
    workspace.workspace_memory.WorkspaceMemory       — holds what's currently in the workspace
    workspace.broadcast_bus.BroadcastBus               — notifies registered subsystems of entries

One cognitive cycle through the workspace is:

    1. Candidates are submitted (from any subsystem) via ``submit_candidate()``.
    2. ``run_cycle()`` scores all pending candidates with AttentionManager,
       selects the winners (above-threshold, within capacity),
       installs them into WorkspaceMemory, and broadcasts each one.

This module does not replace any subsystem's own internal logic — it
is the coordination layer that lets previously-isolated subsystems
become aware of each other's important discoveries.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from events.event_types import CognitiveEvent, EventType
from workspace.attention_manager import AttentionCandidate, AttentionManager, AttentionScore
from workspace.broadcast_bus import BroadcastBus
from workspace.workspace_memory import WorkspaceItem, WorkspaceMemory
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class WorkspaceCycleResult:
    """Result of one ``run_cycle()`` pass."""

    entered: list[WorkspaceItem] = field(default_factory=list)
    rejected_count: int = 0
    broadcasts_sent: int = 0
    cycle_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "entered": [i.to_dict() for i in self.entered],
            "rejected_count": self.rejected_count,
            "broadcasts_sent": self.broadcasts_sent,
            "cycle_at": self.cycle_at,
        }


class GlobalWorkspace:
    """
    Central coordination point: attention-gated entry into a shared
    workspace, with broadcast notification to registered subsystems.

    Parameters
    ----------
    attention_manager:
        ``AttentionManager`` — decides which candidates enter.
    workspace_memory:
        ``WorkspaceMemory`` — holds current workspace contents.
    broadcast_bus:
        ``BroadcastBus`` — notifies subsystems when something enters.
    """

    def __init__(
        self,
        attention_manager: Optional[AttentionManager] = None,
        workspace_memory: Optional[WorkspaceMemory] = None,
        broadcast_bus: Optional[BroadcastBus] = None,
    ) -> None:
        self._attention = attention_manager or AttentionManager()
        self._memory = workspace_memory or WorkspaceMemory()
        self._broadcast = broadcast_bus or BroadcastBus()
        self._pending: list[AttentionCandidate] = []
        self._cycle_count = 0

    # ------------------------------------------------------------------
    # Candidate submission
    # ------------------------------------------------------------------

    def submit_candidate(self, candidate: AttentionCandidate) -> None:
        """Submit a candidate item for consideration in the next cycle."""
        self._pending.append(candidate)

    def submit_many(self, candidates: list[AttentionCandidate]) -> None:
        self._pending.extend(candidates)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    # ------------------------------------------------------------------
    # Cognitive cycle
    # ------------------------------------------------------------------

    def run_cycle(self, active_goal: Optional[str] = None) -> WorkspaceCycleResult:
        """
        Run one attention → entry → broadcast cycle over all pending
        candidates, then clear the pending queue.
        """
        if active_goal is not None:
            self._memory.set_active_goal(active_goal)

        scored = self._attention.select_for_workspace(self._pending)
        rejected_count = len(self._pending) - len(scored)

        items = [WorkspaceItem.from_attention_score(s) for s in scored]
        self._memory.set_items(items)

        broadcasts_sent = 0
        for item in items:
            self._attention.mark_seen(item.ref_id)
            event = CognitiveEvent(
                event_type=EventType.WORKSPACE_BROADCAST, source=item.source,
                payload={"ref_id": item.ref_id, "content_summary": item.content_summary, "attention_score": item.attention_score},
            )
            self._broadcast.broadcast(event)
            broadcasts_sent += 1

        self._pending = []
        self._cycle_count += 1

        return WorkspaceCycleResult(entered=items, rejected_count=rejected_count, broadcasts_sent=broadcasts_sent)

    # ------------------------------------------------------------------
    # Subsystem registration (delegates to BroadcastBus)
    # ------------------------------------------------------------------

    def register_subsystem(self, name: str, event_type: EventType, handler) -> None:
        self._broadcast.register_subsystem(name, event_type, handler)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def memory(self) -> WorkspaceMemory:
        return self._memory

    @property
    def attention(self) -> AttentionManager:
        return self._attention

    @property
    def broadcast_bus(self) -> BroadcastBus:
        return self._broadcast

    @property
    def cycle_count(self) -> int:
        return self._cycle_count
