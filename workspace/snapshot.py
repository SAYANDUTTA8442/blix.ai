"""
Workspace Snapshotting — Blix v0.3.9  (New module 7)

Captures the current state of the ``workspace.global_workspace.GlobalWorkspace``
— active goal, important beliefs, current plan, current failures, and
attention focus — into a single durable, persistable record. This is
what makes task suspension and resumption possible: a long-running
goal can be paused mid-thought and later restored to roughly the same
cognitive state, rather than starting over.

Deliberately a thin capture/restore layer — it does not duplicate
storage that already exists elsewhere (beliefs still live in
``memory.beliefs.BeliefStore``, plans in ``agents.types.TaskGraph``,
etc.). A snapshot stores REFERENCES and SUMMARIES of that state at a
point in time, not full copies of every subsystem's data.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from workspace.global_workspace import GlobalWorkspace
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class WorkspaceSnapshot:
    """One captured workspace state, suitable for later resumption."""

    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    active_goal: Optional[str] = None
    important_beliefs: list[str] = field(default_factory=list)       # belief_id or statement summaries
    current_plan_graph_id: Optional[str] = None
    current_plan_summary: str = ""
    current_failures: list[str] = field(default_factory=list)        # short failure descriptions
    attention_focus: Optional[str] = None
    workspace_items: list[dict] = field(default_factory=list)        # WorkspaceItem.to_dict() snapshots
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "active_goal": self.active_goal,
            "important_beliefs": self.important_beliefs,
            "current_plan_graph_id": self.current_plan_graph_id,
            "current_plan_summary": self.current_plan_summary,
            "current_failures": self.current_failures,
            "attention_focus": self.attention_focus,
            "workspace_items": self.workspace_items,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkspaceSnapshot":
        return cls(
            snapshot_id=d.get("snapshot_id", uuid.uuid4().hex[:10]),
            active_goal=d.get("active_goal"),
            important_beliefs=d.get("important_beliefs", []),
            current_plan_graph_id=d.get("current_plan_graph_id"),
            current_plan_summary=d.get("current_plan_summary", ""),
            current_failures=d.get("current_failures", []),
            attention_focus=d.get("attention_focus"),
            workspace_items=d.get("workspace_items", []),
            created_at=d.get("created_at", ""),
        )


class WorkspaceSnapshotStore:
    """
    Captures and persists ``WorkspaceSnapshot`` instances, keyed by id,
    to support task suspension and later resumption.

    Parameters
    ----------
    snapshot_file:
        Path to ``workspace_snapshots.json``.
    """

    def __init__(self, snapshot_file: Path) -> None:
        self._file = snapshot_file
        self._snapshots: dict[str, WorkspaceSnapshot] = {}
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
            for item in raw:
                snap = WorkspaceSnapshot.from_dict(item)
                self._snapshots[snap.snapshot_id] = snap
            log.info("WorkspaceSnapshotStore: loaded %d snapshot(s).", len(self._snapshots))
        except Exception as exc:
            log.warning("WorkspaceSnapshotStore: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([s.to_dict() for s in self._snapshots.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(
        self,
        workspace: GlobalWorkspace,
        important_beliefs: Optional[list[str]] = None,
        current_plan_graph_id: Optional[str] = None,
        current_plan_summary: str = "",
        current_failures: Optional[list[str]] = None,
    ) -> WorkspaceSnapshot:
        """
        Capture the current state of ``workspace`` plus any
        externally-supplied context (beliefs/plan/failures aren't owned
        by GlobalWorkspace itself, so callers supply them).
        """
        snapshot = WorkspaceSnapshot(
            active_goal=workspace.memory.active_goal,
            important_beliefs=important_beliefs or [],
            current_plan_graph_id=current_plan_graph_id,
            current_plan_summary=current_plan_summary,
            current_failures=current_failures or [],
            attention_focus=workspace.memory.attention_focus,
            workspace_items=[i.to_dict() for i in workspace.memory.items],
        )
        self._snapshots[snapshot.snapshot_id] = snapshot
        self._save()
        return snapshot

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(self, snapshot_id: str, workspace: GlobalWorkspace) -> Optional[WorkspaceSnapshot]:
        """
        Restore a previously-captured snapshot's active_goal back into
        ``workspace``. Workspace items themselves are NOT re-entered
        automatically (they'd need to be re-scored by AttentionManager
        against current conditions) — restore() re-establishes the
        goal/focus context; callers can re-submit candidates as needed.
        """
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None:
            return None
        workspace.memory.set_active_goal(snapshot.active_goal)
        return snapshot

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, snapshot_id: str) -> Optional[WorkspaceSnapshot]:
        return self._snapshots.get(snapshot_id)

    def all_snapshots(self) -> list[WorkspaceSnapshot]:
        return list(self._snapshots.values())

    def most_recent(self) -> Optional[WorkspaceSnapshot]:
        if not self._snapshots:
            return None
        return max(self._snapshots.values(), key=lambda s: s.created_at)

    def delete(self, snapshot_id: str) -> bool:
        existed = snapshot_id in self._snapshots
        self._snapshots.pop(snapshot_id, None)
        if existed:
            self._save()
        return existed

    @property
    def count(self) -> int:
        return len(self._snapshots)
