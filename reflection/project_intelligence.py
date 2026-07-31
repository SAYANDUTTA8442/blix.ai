"""
Project Intelligence Engine — Blix v0.3.2  (Feature 4)

Makes projects first-class COGNITIVE objects by extending v0.3's
``ProjectSummary`` (name, goals, milestones, completed_work, status,
next_actions) with a richer ``ProjectState``:

    name, description, status, priority, progress, milestones,
    risks, next_steps, related_memories, focus

Example
-------
    {
      "project": "Blix",
      "focus": "Reflection Engine",
      "progress": 68,
      "risk_level": "medium"
    }

Design
------
``ProjectIntelligenceEngine`` wraps a v0.3 ``ProjectManager`` (storage
unchanged) and layers ``ProjectState`` records in a separate
``project_intelligence.json`` file — fully additive, no breaking changes
to ``ProjectSummary``.

Risk assessment is heuristic v0.3.2 (blocker count + stagnation), and is
designed to be upgraded to LLM-based reasoning in v0.4.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# ProjectState
# ---------------------------------------------------------------------------


@dataclass
class ProjectState:
    """
    Cognitive state layer for a project, complementing ``ProjectSummary``.

    Fields
    ------
    project_name:
        Matches ``ProjectSummary.project_name`` (join key).
    focus:
        What the project is currently centred on (e.g. "Reflection Engine").
    priority:
        1 (highest) – 5 (lowest).
    progress:
        0-100, mirrors/derives from associated ``Goal.progress`` if linked.
    risks:
        List of risk descriptions.
    risk_level:
        Auto-computed or manually set overall risk.
    next_steps:
        Forward-looking action items (distinct from completed_work).
    related_memory_ids:
        MemoryEntry ids most relevant to this project (for retrieval bias).
    related_goal_id:
        Optional link to a ``Goal`` in GoalTracker.
    """

    project_name: str
    focus: str = ""
    priority: int = 3
    progress: int = 0
    risks: list[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    next_steps: list[str] = field(default_factory=list)
    related_memory_ids: list[int] = field(default_factory=list)
    related_goal_id: Optional[str] = None
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "focus": self.focus,
            "priority": self.priority,
            "progress": self.progress,
            "risks": self.risks,
            "risk_level": self.risk_level.value,
            "next_steps": self.next_steps,
            "related_memory_ids": self.related_memory_ids,
            "related_goal_id": self.related_goal_id,
            "updated_at": self.updated_at,
        }

    def to_summary_dict(self) -> dict:
        """Compact summary matching the spec example format."""
        return {
            "project": self.project_name,
            "focus": self.focus,
            "progress": self.progress,
            "risk_level": self.risk_level.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectState":
        return cls(
            project_name=d["project_name"],
            focus=d.get("focus", ""),
            priority=d.get("priority", 3),
            progress=d.get("progress", 0),
            risks=d.get("risks", []),
            risk_level=RiskLevel(d.get("risk_level", "low")),
            next_steps=d.get("next_steps", []),
            related_memory_ids=d.get("related_memory_ids", []),
            related_goal_id=d.get("related_goal_id"),
            updated_at=d.get("updated_at", ""),
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ProjectIntelligenceEngine:
    """
    Manages ``ProjectState`` records layered on top of ``ProjectManager``.

    Parameters
    ----------
    states_file:
        Path to ``project_intelligence.json``.
    project_manager:
        Optional v0.3 ``ProjectManager`` — used to sync progress/next_steps
        from ``ProjectSummary`` when available.
    stagnation_days:
        If a project hasn't been updated within this many days and has
        unresolved risks, risk_level escalates.
    """

    def __init__(
        self,
        states_file: Path,
        project_manager: Optional[object] = None,
        stagnation_days: float = 14.0,
    ) -> None:
        self._file = states_file
        self._pm = project_manager
        self._stagnation_days = stagnation_days
        self._states: dict[str, ProjectState] = {}
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
                ps = ProjectState.from_dict(item)
                self._states[ps.project_name.lower()] = ps
            log.info("ProjectIntelligenceEngine: loaded %d state(s).", len(self._states))
        except Exception as exc:
            log.warning("ProjectIntelligenceEngine: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([s.to_dict() for s in self._states.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get_or_create(self, project_name: str) -> ProjectState:
        key = project_name.lower()
        if key not in self._states:
            self._states[key] = ProjectState(project_name=project_name)
            self._save()
        return self._states[key]

    def get(self, project_name: str) -> Optional[ProjectState]:
        return self._states.get(project_name.lower())

    def list_all(self) -> list[ProjectState]:
        return list(self._states.values())

    def update(self, project_name: str, **fields: object) -> ProjectState:
        """Update fields on a ProjectState, creating it if necessary."""
        ps = self.get_or_create(project_name)
        for k, v in fields.items():
            if hasattr(ps, k):
                setattr(ps, k, v)
        ps.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return ps

    def set_focus(self, project_name: str, focus: str) -> ProjectState:
        return self.update(project_name, focus=focus)

    def add_risk(self, project_name: str, risk: str) -> ProjectState:
        ps = self.get_or_create(project_name)
        if risk not in ps.risks:
            ps.risks.append(risk)
        ps.updated_at = datetime.now(timezone.utc).isoformat()
        self._recompute_risk_level(ps)
        self._save()
        return ps

    def resolve_risk(self, project_name: str, risk: str) -> ProjectState:
        ps = self.get_or_create(project_name)
        if risk in ps.risks:
            ps.risks.remove(risk)
        ps.updated_at = datetime.now(timezone.utc).isoformat()
        self._recompute_risk_level(ps)
        self._save()
        return ps

    def link_memories(self, project_name: str, memory_ids: list[int]) -> ProjectState:
        ps = self.get_or_create(project_name)
        for mid in memory_ids:
            if mid not in ps.related_memory_ids:
                ps.related_memory_ids.append(mid)
        ps.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return ps

    def link_goal(self, project_name: str, goal_id: str) -> ProjectState:
        return self.update(project_name, related_goal_id=goal_id)

    def sync_progress_from_goal(self, project_name: str, goal: object) -> ProjectState:
        """Sync ``progress`` and ``next_steps`` from a linked ``Goal``."""
        ps = self.get_or_create(project_name)
        ps.progress = getattr(goal, "progress", ps.progress)
        active_blockers = getattr(goal, "active_blockers", [])
        ps.risks = [getattr(b, "description", str(b)) for b in active_blockers]
        pending_tasks = [
            t.title for t in getattr(goal, "tasks", [])
            if getattr(t, "status", None) and t.status.value != "done"
        ]
        ps.next_steps = pending_tasks
        ps.updated_at = datetime.now(timezone.utc).isoformat()
        self._recompute_risk_level(ps)
        self._save()
        return ps

    def sync_from_project_summary(self, summary: object) -> ProjectState:
        """Sync ``next_steps`` from a v0.3 ``ProjectSummary.next_actions``."""
        ps = self.get_or_create(getattr(summary, "project_name"))
        ps.next_steps = list(getattr(summary, "next_actions", []))
        ps.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return ps

    # ------------------------------------------------------------------
    # Risk assessment (heuristic)
    # ------------------------------------------------------------------

    def _recompute_risk_level(self, ps: ProjectState) -> None:
        """
        Heuristic risk scoring:
            0 risks            → LOW
            1-2 risks          → MEDIUM
            3+ risks           → HIGH
        Stagnation (no update for stagnation_days with any open risk)
        escalates by one level.
        """
        n = len(ps.risks)
        if n == 0:
            level = RiskLevel.LOW
        elif n <= 2:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.HIGH

        if n > 0:
            try:
                updated = datetime.fromisoformat(ps.updated_at).replace(tzinfo=None)
                age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - updated).total_seconds() / 86400.0
                if age_days > self._stagnation_days:
                    level = _escalate(level)
            except ValueError:
                pass

        ps.risk_level = level

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def project_report(self, project_name: str) -> dict:
        """Combined report merging ``ProjectState`` with ``ProjectSummary`` if available."""
        ps = self.get_or_create(project_name)
        report = ps.to_summary_dict()
        if self._pm is not None:
            summary = self._pm.get(project_name)  # type: ignore[union-attr]
            if summary is not None:
                report["status"] = getattr(summary, "current_status", "active")
                report["goals"] = list(getattr(summary, "goals", []))
                report["completed_work"] = list(getattr(summary, "completed_work", []))
        report["risks"] = list(ps.risks)
        report["next_steps"] = list(ps.next_steps)
        return report

    def at_risk_projects(self) -> list[ProjectState]:
        return [p for p in self._states.values() if p.risk_level != RiskLevel.LOW]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._states)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _escalate(level: RiskLevel) -> RiskLevel:
    order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
    idx = order.index(level)
    return order[min(idx + 1, len(order) - 1)]
