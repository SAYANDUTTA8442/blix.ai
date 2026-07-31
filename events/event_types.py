"""
Event Types — Blix v0.3.9  (New module 4a)

The typed vocabulary for "everything becomes events". Rather than every
subsystem calling every other subsystem's methods directly (the
implicit "isolated cognition" pattern through v0.3.8), modules emit
typed events onto the ``events.event_bus.EventBus`` and other modules
subscribe to the event types they care about.

This module defines the event taxonomy only — no dispatch logic lives
here (that's ``event_bus.py``) and no persistence lives here (that's
``event_store.py``).

Python 3.10 compatible.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
    TASK_COMPLETED = "task_completed"
    FAILURE = "failure"
    BELIEF_UPDATED = "belief_updated"
    STATE_CHANGED = "state_changed"
    REFLECTION_GENERATED = "reflection_generated"
    PLAN_CREATED = "plan_created"
    CONFIDENCE_CHANGED = "confidence_changed"
    STRATEGY_SWITCHED = "strategy_switched"
    WORKSPACE_BROADCAST = "workspace_broadcast"


@dataclass
class CognitiveEvent:
    """
    A single typed cognitive event.

    Fields
    ------
    event_type:
        What kind of event this is (see ``EventType``).
    source:
        Name of the module/component that emitted it (e.g. "planner",
        "executor", "belief_store") — used for provenance and for
        ``BroadcastBus`` filtering ("don't echo an event back to its
        own source").
    payload:
        Free-form dict of event-specific data. Each ``EventType`` has
        a conventional (but not strictly enforced) payload shape —
        see the factory functions below for the canonical shapes.
    event_id:
        Unique id.
    created_at:
        ISO timestamp.
    """

    event_type: EventType
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "payload": self.payload,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CognitiveEvent":
        return cls(
            event_type=EventType(d["event_type"]), source=d["source"],
            payload=d.get("payload", {}), event_id=d.get("event_id", uuid.uuid4().hex[:12]),
            created_at=d.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Convenience factory functions — canonical payload shapes per event type.
# These are optional sugar; callers may also construct CognitiveEvent
# directly with an arbitrary payload.
# ---------------------------------------------------------------------------


def task_completed_event(source: str, task_title: str, success: bool, domain: str = "") -> CognitiveEvent:
    return CognitiveEvent(
        event_type=EventType.TASK_COMPLETED, source=source,
        payload={"task_title": task_title, "success": success, "domain": domain},
    )


def failure_event(source: str, task_title: str, reason: str, tool: str = "") -> CognitiveEvent:
    return CognitiveEvent(
        event_type=EventType.FAILURE, source=source,
        payload={"task_title": task_title, "reason": reason, "tool": tool},
    )


def belief_updated_event(source: str, belief_id: str, statement: str, confidence: float) -> CognitiveEvent:
    return CognitiveEvent(
        event_type=EventType.BELIEF_UPDATED, source=source,
        payload={"belief_id": belief_id, "statement": statement, "confidence": confidence},
    )


def state_changed_event(source: str, entity: str, attribute: str, old_value: Optional[str], new_value: str) -> CognitiveEvent:
    return CognitiveEvent(
        event_type=EventType.STATE_CHANGED, source=source,
        payload={"entity": entity, "attribute": attribute, "old_value": old_value, "new_value": new_value},
    )


def reflection_generated_event(source: str, scope: str, insight_text: str) -> CognitiveEvent:
    return CognitiveEvent(
        event_type=EventType.REFLECTION_GENERATED, source=source,
        payload={"scope": scope, "insight_text": insight_text},
    )


def plan_created_event(source: str, graph_id: str, goal: str, step_count: int) -> CognitiveEvent:
    return CognitiveEvent(
        event_type=EventType.PLAN_CREATED, source=source,
        payload={"graph_id": graph_id, "goal": goal, "step_count": step_count},
    )


def confidence_changed_event(source: str, namespace: str, ref_id: str, score: float) -> CognitiveEvent:
    return CognitiveEvent(
        event_type=EventType.CONFIDENCE_CHANGED, source=source,
        payload={"namespace": namespace, "ref_id": ref_id, "score": score},
    )


def strategy_switched_event(source: str, ref_key: str, strategy: str, reason: str) -> CognitiveEvent:
    return CognitiveEvent(
        event_type=EventType.STRATEGY_SWITCHED, source=source,
        payload={"ref_key": ref_key, "strategy": strategy, "reason": reason},
    )
