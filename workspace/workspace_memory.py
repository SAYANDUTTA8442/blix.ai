"""
Workspace Memory — Blix v0.3.9  (Part of New module 1, Global Workspace)

The "working stage" itself — what's currently held in conscious focus,
as opposed to ``agents.working_memory.WorkingMemory`` (v0.3.5, scoped
to a single agent execution's scratch state) or any of the long-term
memory layers. ``WorkspaceMemory`` holds the small set of items that
``workspace.attention_manager.AttentionManager`` has selected as
currently important, plus the active goal and attention focus —
exactly the slate that ``workspace.snapshot`` later needs to capture
for suspend/resume.

This module is deliberately NOT a new long-term memory layer (out of
scope per spec) — it's transient, capacity-limited, and gets replaced
as attention shifts, which is precisely what distinguishes a
"workspace" from a memory store.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from workspace.attention_manager import AttentionScore


@dataclass
class WorkspaceItem:
    """One item currently held in the workspace."""

    ref_id: str
    source: str
    content_summary: str
    attention_score: float
    entered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "ref_id": self.ref_id, "source": self.source, "content_summary": self.content_summary,
            "attention_score": round(self.attention_score, 4), "entered_at": self.entered_at,
        }

    @classmethod
    def from_attention_score(cls, scored: AttentionScore) -> "WorkspaceItem":
        return cls(
            ref_id=scored.candidate.ref_id, source=scored.candidate.source,
            content_summary=scored.candidate.content_summary, attention_score=scored.score,
        )


class WorkspaceMemory:
    """
    Holds the current contents of the global workspace: active items,
    active goal, and attention focus.

    This is in-memory only (no persistence) by design — workspace
    contents are transient and reconstructed each cognitive cycle.
    Use ``workspace.snapshot.WorkspaceSnapshot`` for durable capture.
    """

    def __init__(self) -> None:
        self._items: dict[str, WorkspaceItem] = {}
        self._active_goal: Optional[str] = None
        self._attention_focus: Optional[str] = None   # ref_id of the single highest-priority item

    # ------------------------------------------------------------------
    # Item management
    # ------------------------------------------------------------------

    def set_items(self, items: list[WorkspaceItem]) -> None:
        """Replace the full set of workspace items (typical: after an attention cycle)."""
        self._items = {item.ref_id: item for item in items}
        self._attention_focus = items[0].ref_id if items else None

    def add_item(self, item: WorkspaceItem) -> None:
        self._items[item.ref_id] = item
        if self._attention_focus is None or item.attention_score > self._items[self._attention_focus].attention_score:
            self._attention_focus = item.ref_id

    def remove_item(self, ref_id: str) -> bool:
        existed = ref_id in self._items
        self._items.pop(ref_id, None)
        if self._attention_focus == ref_id:
            self._attention_focus = max(self._items, key=lambda k: self._items[k].attention_score, default=None)
        return existed

    def clear(self) -> None:
        self._items = {}
        self._attention_focus = None

    # ------------------------------------------------------------------
    # Goal management
    # ------------------------------------------------------------------

    def set_active_goal(self, goal: Optional[str]) -> None:
        self._active_goal = goal

    @property
    def active_goal(self) -> Optional[str]:
        return self._active_goal

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    @property
    def items(self) -> list[WorkspaceItem]:
        return list(self._items.values())

    @property
    def attention_focus(self) -> Optional[str]:
        return self._attention_focus

    def get(self, ref_id: str) -> Optional[WorkspaceItem]:
        return self._items.get(ref_id)

    def items_from_source(self, source: str) -> list[WorkspaceItem]:
        return [i for i in self._items.values() if i.source == source]

    @property
    def count(self) -> int:
        return len(self._items)

    def to_dict(self) -> dict:
        return {
            "active_goal": self._active_goal,
            "attention_focus": self._attention_focus,
            "items": [i.to_dict() for i in self._items.values()],
        }
