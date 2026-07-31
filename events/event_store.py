"""
Event Store — Blix v0.3.9  (New module 4c)

Persists ``CognitiveEvent`` history to disk. Separate from
``EventBus`` (dispatch) by design — the bus doesn't need to know how
or whether events are stored, and the store doesn't need to know how
events are routed. Used for audit, debugging "why did Blix do that",
and as the data source for ``evaluation.coordination_metrics``.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from events.event_types import CognitiveEvent, EventType
from utils.logger import get_logger

log = get_logger(__name__)

# Persisted log is capped to keep the file bounded — this is an audit
# trail, not a long-term memory store (that's what hierarchical memory
# / belief / state-tracker layers are for).
_MAX_PERSISTED_EVENTS = 5000


class EventStore:
    """
    Append-only persisted log of ``CognitiveEvent`` instances.

    Parameters
    ----------
    event_log_file:
        Path to ``event_log.json``.
    """

    def __init__(self, event_log_file: Path) -> None:
        self._file = event_log_file
        self._events: list[CognitiveEvent] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._events = [CognitiveEvent.from_dict(e) for e in raw]
            log.info("EventStore: loaded %d event(s).", len(self._events))
        except Exception as exc:
            log.warning("EventStore: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([e.to_dict() for e in self._events[-_MAX_PERSISTED_EVENTS:]], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Append / query
    # ------------------------------------------------------------------

    def append(self, event: CognitiveEvent) -> None:
        self._events.append(event)
        self._save()

    def recent(self, limit: int = 50, event_type: Optional[EventType] = None) -> list[CognitiveEvent]:
        events = self._events
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def by_source(self, source: str, limit: int = 50) -> list[CognitiveEvent]:
        matching = [e for e in self._events if e.source == source]
        return matching[-limit:]

    def count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._events:
            counts[e.event_type.value] = counts.get(e.event_type.value, 0) + 1
        return counts

    @property
    def count(self) -> int:
        return len(self._events)
