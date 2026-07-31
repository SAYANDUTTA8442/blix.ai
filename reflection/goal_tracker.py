"""
Goal Tracking System — Blix v0.3.2  (Feature 3)

Enables long-term objective management with first-class entities:

    Goal
     ├── Milestone (checkpoints toward the goal)
     ├── Task      (concrete actionable items)
     ├── Blocker   (obstacles preventing progress)
     └── Progress  (0-100 percent, derived or manually set)

Example
-------
    {
      "goal": "Build Blix v0.4",
      "progress": 72,
      "status": "active",
      "blockers": ["evaluation framework"]
    }

Progress is computed from milestone/task completion ratios unless
explicitly overridden, so it stays consistent with underlying work items.

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


class GoalStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ItemStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"


# ---------------------------------------------------------------------------
# Sub-entities
# ---------------------------------------------------------------------------


@dataclass
class Milestone:
    title: str
    status: ItemStatus = ItemStatus.PENDING

    def to_dict(self) -> dict:
        return {"title": self.title, "status": self.status.value}

    @classmethod
    def from_dict(cls, d: dict) -> "Milestone":
        return cls(title=d["title"], status=ItemStatus(d.get("status", "pending")))


@dataclass
class Task:
    title: str
    status: ItemStatus = ItemStatus.PENDING

    def to_dict(self) -> dict:
        return {"title": self.title, "status": self.status.value}

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(title=d["title"], status=ItemStatus(d.get("status", "pending")))


@dataclass
class Blocker:
    description: str
    resolved: bool = False
    raised_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "resolved": self.resolved,
            "raised_at": self.raised_at,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Blocker":
        return cls(
            description=d["description"],
            resolved=d.get("resolved", False),
            raised_at=d.get("raised_at", ""),
            resolved_at=d.get("resolved_at"),
        )


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------


@dataclass
class Goal:
    """
    A long-term objective tracked by Blix.

    ``progress`` is auto-computed from milestone + task completion ratios
    unless ``progress_override`` is set (e.g. by direct user statement).
    """

    goal_id: str
    title: str
    description: str = ""
    status: GoalStatus = GoalStatus.ACTIVE
    priority: int = 3                      # 1 (highest) – 5 (lowest)
    milestones: list[Milestone] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    blockers: list[Blocker] = field(default_factory=list)
    progress_override: Optional[int] = None
    related_project: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ------------------------------------------------------------------
    # Progress computation
    # ------------------------------------------------------------------

    @property
    def progress(self) -> int:
        """0-100 integer. Uses override if set, else computed from items."""
        if self.progress_override is not None:
            return max(0, min(100, self.progress_override))
        items = self.milestones + self.tasks
        if not items:
            return 0
        done = sum(1 for it in items if it.status == ItemStatus.DONE)
        return round(100 * done / len(items))

    @property
    def active_blockers(self) -> list[Blocker]:
        return [b for b in self.blockers if not b.resolved]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
            "milestones": [m.to_dict() for m in self.milestones],
            "tasks": [t.to_dict() for t in self.tasks],
            "blockers": [b.to_dict() for b in self.blockers],
            "progress_override": self.progress_override,
            "related_project": self.related_project,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_summary_dict(self) -> dict:
        """Compact summary matching the spec example format."""
        return {
            "goal": self.title,
            "progress": self.progress,
            "status": self.status.value,
            "blockers": [b.description for b in self.active_blockers],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Goal":
        return cls(
            goal_id=d["goal_id"],
            title=d["title"],
            description=d.get("description", ""),
            status=GoalStatus(d.get("status", "active")),
            priority=d.get("priority", 3),
            milestones=[Milestone.from_dict(m) for m in d.get("milestones", [])],
            tasks=[Task.from_dict(t) for t in d.get("tasks", [])],
            blockers=[Blocker.from_dict(b) for b in d.get("blockers", [])],
            progress_override=d.get("progress_override"),
            related_project=d.get("related_project", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


# ---------------------------------------------------------------------------
# Goal Tracker
# ---------------------------------------------------------------------------


class GoalTracker:
    """
    Manages the collection of ``Goal`` objects.

    Parameters
    ----------
    goals_file:
        Path to ``goals.json``.
    """

    def __init__(self, goals_file: Path) -> None:
        self._file = goals_file
        self._goals: dict[str, Goal] = {}
        self._next_id = 0
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
                g = Goal.from_dict(item)
                self._goals[g.goal_id] = g
            if self._goals:
                self._next_id = max(
                    int(gid.replace("goal_", "")) for gid in self._goals
                ) + 1
            log.info("GoalTracker: loaded %d goal(s).", len(self._goals))
        except Exception as exc:
            log.warning("GoalTracker: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([g.to_dict() for g in self._goals.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_goal(
        self,
        title: str,
        description: str = "",
        priority: int = 3,
        related_project: str = "",
    ) -> Goal:
        """Create a new goal. Returns the created ``Goal``."""
        gid = f"goal_{self._next_id}"
        self._next_id += 1
        goal = Goal(
            goal_id=gid, title=title, description=description,
            priority=max(1, min(5, priority)), related_project=related_project,
        )
        self._goals[gid] = goal
        self._save()
        log.info("GoalTracker: created %s %r", gid, title)
        return goal

    def get(self, goal_id: str) -> Optional[Goal]:
        return self._goals.get(goal_id)

    def find_by_title(self, title: str) -> Optional[Goal]:
        lower = title.lower()
        for g in self._goals.values():
            if g.title.lower() == lower:
                return g
        return None

    def list_goals(
        self,
        status: Optional[GoalStatus] = None,
        related_project: Optional[str] = None,
    ) -> list[Goal]:
        goals = list(self._goals.values())
        if status is not None:
            goals = [g for g in goals if g.status == status]
        if related_project is not None:
            goals = [g for g in goals if g.related_project == related_project]
        return goals

    def prioritized_goals(self) -> list[Goal]:
        """Active goals sorted by priority (1=highest) then progress (lowest first)."""
        active = self.list_goals(status=GoalStatus.ACTIVE)
        return sorted(active, key=lambda g: (g.priority, g.progress))

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update_status(self, goal_id: str, status: GoalStatus) -> Optional[Goal]:
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        goal.status = status
        goal.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return goal

    def set_progress_override(self, goal_id: str, progress: int) -> Optional[Goal]:
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        goal.progress_override = max(0, min(100, progress))
        goal.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return goal

    def add_milestone(self, goal_id: str, title: str) -> Optional[Goal]:
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        goal.milestones.append(Milestone(title=title))
        goal.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return goal

    def add_task(self, goal_id: str, title: str) -> Optional[Goal]:
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        goal.tasks.append(Task(title=title))
        goal.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return goal

    def complete_item(self, goal_id: str, title: str) -> Optional[Goal]:
        """Mark a milestone or task with matching title as DONE."""
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        found = False
        for item in goal.milestones + goal.tasks:
            if item.title == title:
                item.status = ItemStatus.DONE
                found = True
        if found:
            goal.updated_at = datetime.now(timezone.utc).isoformat()
            if goal.progress == 100 and goal.status == GoalStatus.ACTIVE:
                goal.status = GoalStatus.COMPLETED
            self._save()
        return goal

    def add_blocker(self, goal_id: str, description: str) -> Optional[Goal]:
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        goal.blockers.append(Blocker(description=description))
        goal.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return goal

    def resolve_blocker(self, goal_id: str, description: str) -> Optional[Goal]:
        goal = self._goals.get(goal_id)
        if goal is None:
            return None
        for b in goal.blockers:
            if b.description == description and not b.resolved:
                b.resolved = True
                b.resolved_at = datetime.now(timezone.utc).isoformat()
        goal.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return goal

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._goals)

    def summary(self) -> str:
        active = self.list_goals(status=GoalStatus.ACTIVE)
        if not active:
            return "No active goals."
        parts = [f'"{g.title}" ({g.progress}%)' for g in active[:3]]
        return f"{len(active)} active goal(s): " + ", ".join(parts)

    # ------------------------------------------------------------------
    # Planner integration (v0.3.13 gap fix)
    # ------------------------------------------------------------------

    def suggest_next_search(self, beam_search_planner, start_state) -> Optional["BeamSearchResult"]:  # type: ignore[name-defined]
        """
        Take the highest-priority active goal and run a
        ``BeamSearchPlanner`` pass to suggest the best next action.

        Action candidates are derived from the goal's pending tasks and
        active blockers — each becomes a named action whose resulting
        LatentState reflects reduced risk (for blocker-resolution actions)
        or increased confidence (for task-completion actions).

        Returns ``None`` when no active goal exists.
        """
        from world_model.latent_world_model import LatentState

        priority_goals = self.prioritized_goals()
        if not priority_goals:
            return None
        top_goal = priority_goals[0]

        pending_tasks = [t for t in top_goal.tasks if t.status.value == "pending"]
        active_blockers = top_goal.active_blockers

        def _action_generator(state: LatentState):
            candidates = []
            for blocker in active_blockers[:3]:
                action_name = f"resolve_blocker:{blocker.description[:40]}"
                resulting = LatentState(
                    confidence=state.confidence + 0.05,
                    complexity=state.complexity,
                    risk=max(0.0, state.risk - 0.2),
                    capability_estimate=state.capability_estimate,
                    recent_failure_rate=state.recent_failure_rate,
                    dependency_density=state.dependency_density,
                )
                candidates.append((action_name, resulting))
            for task in pending_tasks[:3]:
                action_name = f"complete_task:{task.title[:40]}"
                resulting = LatentState(
                    confidence=min(1.0, state.confidence + 0.1),
                    complexity=max(0.0, state.complexity - 0.05),
                    risk=state.risk,
                    capability_estimate=min(1.0, state.capability_estimate + 0.05),
                    recent_failure_rate=state.recent_failure_rate,
                    dependency_density=state.dependency_density,
                )
                candidates.append((action_name, resulting))
            if not candidates:
                # No specific actions — offer a generic exploration step
                candidates.append(("explore_goal_domain",
                    LatentState(confidence=min(1.0, state.confidence + 0.05),
                                complexity=state.complexity, risk=state.risk,
                                capability_estimate=state.capability_estimate,
                                recent_failure_rate=state.recent_failure_rate,
                                dependency_density=state.dependency_density)))
            return candidates

        return beam_search_planner.search(top_goal.title, start_state, _action_generator)
