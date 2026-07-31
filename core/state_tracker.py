"""
State Tracker — Blix v0.3.7  (New module 1)

Tracks how an entity's attributes change over time, instead of treating
every new fact as an isolated, competing memory.

    Python (2024)     ← StateSnapshot(end_time=2025-01-01)
    Rust   (2025–now)  ← StateSnapshot(end_time=None)

Each ``StateSnapshot`` records one (entity, attribute) → value binding
valid over a known time window. The ``StateTracker`` is the single
source of truth for "what does Blix currently believe about X" and
"what did Blix believe about X at time T" — both v0.3.7's flagship
queries.

This module does NOT decide how snapshots transition (that's
``core.state_transition.StateTransitionEngine``) or how conflicting
snapshots get resolved (that's ``core.truth_manager.TruthManager`` and
``core.contradiction_resolver.ContradictionResolver``). It is the
storage and query layer underneath all three.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# StateSnapshot
# ---------------------------------------------------------------------------


@dataclass
class StateSnapshot:
    """
    One (entity, attribute) → value binding, valid over a time window.

    Fields
    ------
    entity:
        The subject, e.g. "sayan", "blix" (lowercase slug, matches
        ``core.memory_graph.GraphNode.id`` conventions where applicable).
    attribute:
        What's being tracked, e.g. "favorite_language", "tech_stack",
        "city", "research_focus".
    value:
        The current value, e.g. "Rust", "FastAPI".
    start_time:
        When this value became true (ISO 8601).
    end_time:
        When this value stopped being true, or ``None`` if still active.
    confidence:
        0–1 confidence in this snapshot, independent of TruthStatus.
    snapshot_id:
        Unique id.
    source_memory_ids:
        Memory ids that support this snapshot, for explainability.
    """

    entity: str
    attribute: str
    value: str
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: Optional[str] = None
    confidence: float = 0.7
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    source_memory_ids: list[int] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.end_time is None

    def covers(self, timestamp: str) -> bool:
        """True if ``timestamp`` falls within [start_time, end_time)."""
        if timestamp < self.start_time:
            return False
        if self.end_time is not None and timestamp >= self.end_time:
            return False
        return True

    def close(self, end_time: Optional[str] = None) -> None:
        """Mark this snapshot as no longer active, as of ``end_time`` (default: now)."""
        self.end_time = end_time or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "entity": self.entity,
            "attribute": self.attribute,
            "value": self.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_active": self.is_active,
            "confidence": round(self.confidence, 3),
            "source_memory_ids": self.source_memory_ids,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StateSnapshot":
        return cls(
            entity=d["entity"],
            attribute=d["attribute"],
            value=d["value"],
            start_time=d.get("start_time", ""),
            end_time=d.get("end_time"),
            confidence=d.get("confidence", 0.7),
            snapshot_id=d.get("snapshot_id", uuid.uuid4().hex[:8]),
            source_memory_ids=d.get("source_memory_ids", []),
        )


# ---------------------------------------------------------------------------
# State Tracker
# ---------------------------------------------------------------------------


class StateTracker:
    """
    Persists and queries ``StateSnapshot`` history for every
    (entity, attribute) pair Blix tracks.

    Parameters
    ----------
    state_file:
        Path to ``state_snapshots.json``.
    """

    def __init__(self, state_file: Path) -> None:
        self._file = state_file
        self._snapshots: list[StateSnapshot] = []
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
            self._snapshots = [StateSnapshot.from_dict(s) for s in raw]
            log.info("StateTracker: loaded %d snapshot(s).", len(self._snapshots))
        except Exception as exc:
            log.warning("StateTracker: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([s.to_dict() for s in self._snapshots], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def record(
        self,
        entity: str,
        attribute: str,
        value: str,
        confidence: float = 0.7,
        source_memory_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> StateSnapshot:
        """
        Record a new value for (entity, attribute), WITHOUT automatically
        closing any prior active snapshot.

        Use ``StateTransitionEngine.transition()`` (the normal entry
        point from the rest of the system) if you want the prior active
        snapshot for this (entity, attribute) to be closed automatically.
        This low-level ``record()`` is intentionally dumb — it only
        appends — so higher-level modules retain full control over
        transition semantics.
        """
        entity = entity.lower().strip()
        attribute = attribute.lower().strip()
        snap = StateSnapshot(
            entity=entity, attribute=attribute, value=value,
            start_time=timestamp or datetime.now(timezone.utc).isoformat(),
            confidence=confidence,
            source_memory_ids=[source_memory_id] if source_memory_id is not None else [],
        )
        self._snapshots.append(snap)
        self._save()
        log.info("StateTracker: recorded %s.%s = %r", entity, attribute, value)
        return snap

    def close_active(
        self, entity: str, attribute: str, end_time: Optional[str] = None
    ) -> list[StateSnapshot]:
        """Close all currently-active snapshots for (entity, attribute). Returns those closed."""
        entity, attribute = entity.lower().strip(), attribute.lower().strip()
        closed = []
        for snap in self._snapshots:
            if snap.entity == entity and snap.attribute == attribute and snap.is_active:
                snap.close(end_time)
                closed.append(snap)
        if closed:
            self._save()
        return closed

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def current(self, entity: str, attribute: str) -> Optional[StateSnapshot]:
        """The currently-active snapshot for (entity, attribute), if any."""
        entity, attribute = entity.lower().strip(), attribute.lower().strip()
        active = [
            s for s in self._snapshots
            if s.entity == entity and s.attribute == attribute and s.is_active
        ]
        if not active:
            return None
        # If multiple are somehow active (shouldn't normally happen),
        # prefer the most recently started.
        return max(active, key=lambda s: s.start_time)

    def all_current(self, entity: str) -> list[StateSnapshot]:
        """All currently-active snapshots for an entity, across attributes."""
        entity = entity.lower().strip()
        return [s for s in self._snapshots if s.entity == entity and s.is_active]

    def at_time(self, entity: str, attribute: str, timestamp: str) -> Optional[StateSnapshot]:
        """The snapshot that was active for (entity, attribute) at a given ISO timestamp."""
        entity, attribute = entity.lower().strip(), attribute.lower().strip()
        candidates = [
            s for s in self._snapshots
            if s.entity == entity and s.attribute == attribute and s.covers(timestamp)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.start_time)

    def history(self, entity: str, attribute: str) -> list[StateSnapshot]:
        """Full chronological history of values for (entity, attribute), oldest first."""
        entity, attribute = entity.lower().strip(), attribute.lower().strip()
        matches = [
            s for s in self._snapshots
            if s.entity == entity and s.attribute == attribute
        ]
        return sorted(matches, key=lambda s: s.start_time)

    def changes_since(self, since_timestamp: str, entity: Optional[str] = None) -> list[StateSnapshot]:
        """All snapshots that started on or after ``since_timestamp``, optionally filtered by entity."""
        matches = [s for s in self._snapshots if s.start_time >= since_timestamp]
        if entity:
            entity = entity.lower().strip()
            matches = [s for s in matches if s.entity == entity]
        return sorted(matches, key=lambda s: s.start_time)

    def all_attributes(self, entity: str) -> list[str]:
        """All distinct attributes ever tracked for an entity."""
        entity = entity.lower().strip()
        return sorted({s.attribute for s in self._snapshots if s.entity == entity})

    def all_entities(self) -> list[str]:
        return sorted({s.entity for s in self._snapshots})

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._snapshots)

    @property
    def active_count(self) -> int:
        return sum(1 for s in self._snapshots if s.is_active)
