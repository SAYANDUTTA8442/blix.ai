"""
Hierarchical memory layer schemas — Blix v0.3

Memory hierarchy (coarsest to finest):
    Raw Memory → Session Summary → Daily Summary → Weekly Summary → Project Summary

Each layer is a Pydantic model.  The base class ``BaseMemorySummary`` carries
the fields common to all summarised layers so callers can treat them uniformly.

Python 3.10 compatible.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MemoryLayerKind(str, Enum):
    RAW = "raw"
    SESSION = "session"
    DAILY = "daily"
    WEEKLY = "weekly"
    PROJECT = "project"


class BaseMemorySummary(BaseModel):
    """
    Fields common to every summarised memory layer.

    Subclasses add layer-specific span information (session_id, date, …).
    """

    id: str = Field(..., description="Unique string id for this summary (e.g. 'session-42').")
    kind: MemoryLayerKind
    summary: str = Field(..., min_length=1, description="Human-readable compressed summary.")
    source_ids: list[str] = Field(
        default_factory=list,
        description="Ids of the lower-level objects that were compressed to produce this summary.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    topics: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------


class SessionSummary(BaseMemorySummary):
    """
    Aggregates all raw MemoryEntry objects from a single chat session.

    A session is delimited by a configurable idle-gap (default 30 min) or
    an explicit ``/new-session`` command.
    """

    kind: MemoryLayerKind = MemoryLayerKind.SESSION
    session_index: int = Field(..., description="Monotonic session counter starting at 1.")
    raw_memory_ids: list[int] = Field(
        default_factory=list,
        description="MemoryEntry.id values compressed into this session.",
    )
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    turn_count: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Daily summary
# ---------------------------------------------------------------------------


class DailySummary(BaseMemorySummary):
    """Aggregates SessionSummary objects for a single calendar day (UTC)."""

    kind: MemoryLayerKind = MemoryLayerKind.DAILY
    date: str = Field(..., description="ISO date string, e.g. '2025-07-15'.")
    session_ids: list[str] = Field(default_factory=list)
    session_count: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Weekly summary
# ---------------------------------------------------------------------------


class WeeklySummary(BaseMemorySummary):
    """Aggregates DailySummary objects for a calendar week (Mon–Sun, UTC)."""

    kind: MemoryLayerKind = MemoryLayerKind.WEEKLY
    week_label: str = Field(..., description="ISO week label, e.g. '2025-W29'.")
    daily_ids: list[str] = Field(default_factory=list)
    daily_count: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Project summary
# ---------------------------------------------------------------------------


class ProjectSummary(BaseMemorySummary):
    """
    Long-running project memory object.

    Projects are first-class entities: they aggregate sessions, track
    milestones, and persist across weeks.  Any session referencing a project
    name triggers an update.
    """

    kind: MemoryLayerKind = MemoryLayerKind.PROJECT
    project_name: str = Field(..., description="Canonical project name.")
    description: str = Field(default="")
    goals: list[str] = Field(default_factory=list)
    milestones: list[str] = Field(default_factory=list)
    completed_work: list[str] = Field(default_factory=list)
    current_status: str = Field(default="active")
    next_actions: list[str] = Field(default_factory=list)
    related_session_ids: list[str] = Field(default_factory=list)
    last_active: Optional[datetime] = None
