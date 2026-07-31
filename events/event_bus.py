"""
Event Bus — Blix v0.3.9  (New module 4b)

Pure publish/subscribe dispatch. ``EventBus`` knows nothing about
memory, planning, or reflection specifically — it just routes
``CognitiveEvent`` instances from publishers to subscribers by
``EventType``, synchronously, in registration order.

This is intentionally minimal (no async, no external message broker) —
Blix is a single-process system and a synchronous in-memory bus is the
right level of complexity for "modules can react to each other's
events" without introducing concurrency hazards.

Python 3.10 compatible.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional

from events.event_types import CognitiveEvent, EventType
from utils.logger import get_logger

log = get_logger(__name__)

EventHandler = Callable[[CognitiveEvent], None]


class EventBus:
    """
    Synchronous, in-memory publish/subscribe bus for ``CognitiveEvent``.

    Parameters
    ----------
    event_store:
        Optional ``events.event_store.EventStore`` — if provided, every
        published event is also persisted there automatically.
    """

    def __init__(self, event_store=None) -> None:
        self._subscribers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._wildcard_subscribers: list[EventHandler] = []
        self._event_store = event_store
        self._publish_count = 0

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register ``handler`` to be called for every event of ``event_type``."""
        self._subscribers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Register ``handler`` to be called for every event, regardless of type."""
        self._wildcard_subscribers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> bool:
        """Remove a specific handler from a specific event type. Returns True if removed."""
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)
            return True
        return False

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, event: CognitiveEvent) -> int:
        """
        Publish an event to all matching subscribers (type-specific +
        wildcard), in registration order. Handler exceptions are caught
        and logged so one broken subscriber can't crash a publish.

        Returns the number of handlers invoked.
        """
        self._publish_count += 1
        if self._event_store is not None:
            self._event_store.append(event)

        invoked = 0
        for handler in list(self._subscribers.get(event.event_type, [])):
            try:
                handler(event)
                invoked += 1
            except Exception as exc:
                log.warning("EventBus: subscriber raised for %s: %s", event.event_type.value, exc)

        for handler in list(self._wildcard_subscribers):
            try:
                handler(event)
                invoked += 1
            except Exception as exc:
                log.warning("EventBus: wildcard subscriber raised for %s: %s", event.event_type.value, exc)

        return invoked

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def subscriber_count(self, event_type: Optional[EventType] = None) -> int:
        if event_type is None:
            return sum(len(h) for h in self._subscribers.values()) + len(self._wildcard_subscribers)
        return len(self._subscribers.get(event_type, [])) + len(self._wildcard_subscribers)

    @property
    def publish_count(self) -> int:
        return self._publish_count
