"""
Broadcast Bus — Blix v0.3.9  (New module 3)

Where ``events.event_bus.EventBus`` is a generic pub/sub mechanism,
``BroadcastBus`` is the specific Global-Workspace-Theory pattern built
on top of it: when something important happens anywhere in the system,
it gets BROADCAST to every registered subsystem at once, rather than
requiring the originating module to know who might care.

    Planner discovers failure
      ↓
    Broadcast
      ↓
    Reflection, Self Model, Belief Layer, Failure Memory  (all notified)

This is "cooperative cognition" — subsystems sharing discoveries —
rather than each module operating in isolation and only ever being
explicitly called by some orchestrator. ``BroadcastBus`` wraps an
``EventBus`` and adds: (1) a simple "register a subsystem listener"
API that doesn't require callers to know ``EventType`` values, (2) a
log of what was broadcast and to how many listeners, used by
``evaluation.coordination_metrics`` to measure broadcast quality.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from events.event_bus import EventBus
from events.event_types import CognitiveEvent, EventType
from utils.logger import get_logger

log = get_logger(__name__)

BroadcastHandler = Callable[[CognitiveEvent], None]


@dataclass
class BroadcastRecord:
    """One logged broadcast — what was sent and how many listeners received it."""

    event_type: str
    source: str
    listener_count: int
    broadcast_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type, "source": self.source,
            "listener_count": self.listener_count, "broadcast_at": self.broadcast_at,
        }


class BroadcastBus:
    """
    GWT-style broadcast layer on top of an ``EventBus``.

    Parameters
    ----------
    event_bus:
        The underlying ``EventBus`` used for actual dispatch. If not
        provided, a private one is created.
    """

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self._bus = event_bus or EventBus()
        self._listeners: dict[str, list[tuple[EventType, BroadcastHandler]]] = {}
        self._log: list[BroadcastRecord] = []

    # ------------------------------------------------------------------
    # Subsystem registration
    # ------------------------------------------------------------------

    def register_subsystem(self, name: str, event_type: EventType, handler: BroadcastHandler) -> None:
        """
        Register a named subsystem to listen for a specific event type.
        Multiple registrations for the same subsystem name (different
        event types) are allowed.
        """
        self._bus.subscribe(event_type, handler)
        self._listeners.setdefault(name, []).append((event_type, handler))

    def registered_subsystems(self) -> list[str]:
        return list(self._listeners.keys())

    def unregister_subsystem(self, name: str) -> int:
        """Remove all of a subsystem's registrations. Returns count removed."""
        entries = self._listeners.pop(name, [])
        for event_type, handler in entries:
            self._bus.unsubscribe(event_type, handler)
        return len(entries)

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    def broadcast(self, event: CognitiveEvent) -> BroadcastRecord:
        """
        Broadcast an event to every registered subsystem listening for
        its type, logging the broadcast for coordination-quality metrics.
        """
        listener_count = self._bus.subscriber_count(event.event_type)
        self._bus.publish(event)
        record = BroadcastRecord(
            event_type=event.event_type.value, source=event.source, listener_count=listener_count,
        )
        self._log.append(record)
        return record

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def recent_broadcasts(self, limit: int = 20) -> list[BroadcastRecord]:
        return self._log[-limit:]

    def mean_listener_count(self) -> float:
        """Average number of listeners per broadcast — a coarse signal of how 'connected' the system is."""
        if not self._log:
            return 0.0
        return sum(r.listener_count for r in self._log) / len(self._log)

    def broadcasts_with_zero_listeners(self) -> list[BroadcastRecord]:
        """Broadcasts that nobody was listening for — a coordination gap."""
        return [r for r in self._log if r.listener_count == 0]

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    @property
    def broadcast_count(self) -> int:
        return len(self._log)
