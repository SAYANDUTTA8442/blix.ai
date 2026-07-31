"""
State Transition Engine — Blix v0.3.7  (New module 2)

The normal entry point for updating tracked state. Where
``StateTracker.record()`` is a dumb append, ``StateTransitionEngine``
encodes the actual semantics of "this attribute changed":

    Python (2024) → Rust (2025)

is a TRANSITION, not two competing facts. The engine:

1. Looks up the current active snapshot for (entity, attribute).
2. If the new value differs, closes the old snapshot (end_time = now)
   and opens a new one — recording a ``StateTransition`` describing the
   change for reflection/evolution queries (Items 4 and 8 in the spec).
3. If the new value is the SAME as the current one, just reinforces
   confidence (no spurious transition).
4. If there's no current value yet, this is an initial assignment, not
   a transition.

This module decides WHETHER something is a transition vs. a brand-new
fact. It deliberately does NOT decide what to do about genuinely
conflicting parallel truths (e.g. "Python AND Rust, both still true") —
that nuance belongs to ``core.contradiction_resolver.ContradictionResolver``,
which this engine can optionally delegate to when the situation looks
like parallel truth rather than a clean transition.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.state_tracker import StateSnapshot, StateTracker
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Transition record
# ---------------------------------------------------------------------------


@dataclass
class StateTransition:
    """
    A recorded change from one value to another for (entity, attribute).

    Fields
    ------
    entity / attribute:
        What changed.
    from_value:
        Previous value, or ``None`` if this was an initial assignment.
    to_value:
        New value.
    transitioned_at:
        ISO timestamp of the transition.
    reason:
        Optional human-readable note (e.g. "explicit update", "inferred from memory #42").
    """

    entity: str
    attribute: str
    to_value: str
    from_value: Optional[str] = None
    transitioned_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reason: str = ""
    transition_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def is_initial(self) -> bool:
        return self.from_value is None

    def to_dict(self) -> dict:
        return {
            "transition_id": self.transition_id,
            "entity": self.entity,
            "attribute": self.attribute,
            "from_value": self.from_value,
            "to_value": self.to_value,
            "transitioned_at": self.transitioned_at,
            "reason": self.reason,
            "is_initial": self.is_initial,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StateTransition":
        return cls(
            entity=d["entity"], attribute=d["attribute"],
            from_value=d.get("from_value"), to_value=d["to_value"],
            transitioned_at=d.get("transitioned_at", ""),
            reason=d.get("reason", ""),
            transition_id=d.get("transition_id", uuid.uuid4().hex[:8]),
        )

    def describe(self) -> str:
        if self.is_initial:
            return f"{self.entity}.{self.attribute} set to '{self.to_value}'"
        return f"{self.entity}.{self.attribute}: '{self.from_value}' → '{self.to_value}'"


# ---------------------------------------------------------------------------
# Transition Engine
# ---------------------------------------------------------------------------


class StateTransitionEngine:
    """
    Applies state changes through proper transition semantics rather
    than letting new facts silently compete with old ones.

    Parameters
    ----------
    tracker:
        ``StateTracker`` to read/write snapshots through.
    transitions_file:
        Path to ``state_transitions.json`` — persisted transition log.
    reinforcement_increment:
        How much to bump confidence when the same value is reasserted.
    """

    def __init__(
        self,
        tracker: StateTracker,
        transitions_file: Path,
        reinforcement_increment: float = 0.05,
    ) -> None:
        self._tracker = tracker
        self._file = transitions_file
        self._reinforcement = reinforcement_increment
        self._transitions: list[StateTransition] = []
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
            self._transitions = [StateTransition.from_dict(t) for t in raw]
            log.info("StateTransitionEngine: loaded %d transition(s).", len(self._transitions))
        except Exception as exc:
            log.warning("StateTransitionEngine: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([t.to_dict() for t in self._transitions], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    def transition(
        self,
        entity: str,
        attribute: str,
        new_value: str,
        confidence: float = 0.7,
        source_memory_id: Optional[int] = None,
        reason: str = "",
        timestamp: Optional[str] = None,
    ) -> tuple[StateSnapshot, Optional[StateTransition]]:
        """
        Update (entity, attribute) to ``new_value`` using proper
        transition semantics.

        Returns
        -------
        (new_snapshot, transition_or_none)
            ``transition_or_none`` is ``None`` only when the value is
            unchanged from the current active snapshot (pure
            reinforcement — no transition recorded).
        """
        current = self._tracker.current(entity, attribute)

        if current is not None and _normalise(current.value) == _normalise(new_value):
            # Same value reasserted — reinforce confidence, no transition.
            current.confidence = min(1.0, current.confidence + self._reinforcement)
            if source_memory_id is not None and source_memory_id not in current.source_memory_ids:
                current.source_memory_ids.append(source_memory_id)
            self._tracker._save()  # tracker owns persistence of snapshot mutations
            log.debug(
                "StateTransitionEngine: reinforced %s.%s = %r (confidence=%.2f)",
                entity, attribute, new_value, current.confidence,
            )
            return current, None

        # Genuine change (or initial assignment) — close old, open new.
        if current is not None:
            self._tracker.close_active(entity, attribute, end_time=timestamp)

        new_snapshot = self._tracker.record(
            entity, attribute, new_value,
            confidence=confidence, source_memory_id=source_memory_id, timestamp=timestamp,
        )

        transition = StateTransition(
            entity=entity.lower().strip(),
            attribute=attribute.lower().strip(),
            from_value=current.value if current is not None else None,
            to_value=new_value,
            transitioned_at=timestamp or datetime.now(timezone.utc).isoformat(),
            reason=reason,
        )
        self._transitions.append(transition)
        self._save()
        log.info("StateTransitionEngine: %s", transition.describe())
        return new_snapshot, transition

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def history(self, entity: str, attribute: Optional[str] = None) -> list[StateTransition]:
        """Chronological transition history for an entity (optionally filtered by attribute)."""
        entity = entity.lower().strip()
        matches = [t for t in self._transitions if t.entity == entity]
        if attribute:
            attribute = attribute.lower().strip()
            matches = [t for t in matches if t.attribute == attribute]
        return sorted(matches, key=lambda t: t.transitioned_at)

    def latest_transition(self, entity: str, attribute: str) -> Optional[StateTransition]:
        hist = self.history(entity, attribute)
        return hist[-1] if hist else None

    def transitions_since(self, since_timestamp: str, entity: Optional[str] = None) -> list[StateTransition]:
        matches = [t for t in self._transitions if t.transitioned_at >= since_timestamp]
        if entity:
            entity = entity.lower().strip()
            matches = [t for t in matches if t.entity == entity]
        return sorted(matches, key=lambda t: t.transitioned_at)

    def attributes_changed_since(self, since_timestamp: str, entity: str) -> list[str]:
        """Distinct attributes of ``entity`` that changed since a given time."""
        changes = self.transitions_since(since_timestamp, entity=entity)
        return sorted({t.attribute for t in changes if not t.is_initial})

    @property
    def count(self) -> int:
        return len(self._transitions)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(value: str) -> str:
    return value.strip().lower()
