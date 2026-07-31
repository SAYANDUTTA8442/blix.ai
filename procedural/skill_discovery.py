"""
Skill Discovery Engine — Blix v0.3.10  (New module 8)

Inspired by Voyager: automatically discovers reusable skills from
SUCCESSFUL EXECUTION TRAJECTORIES (completed ``TaskGraph`` runs),
rather than requiring an explicit ``learn_from_success(goal, steps)``
call as ``memory.procedural_memory.ProceduralMemory`` (v0.3.8) does.

Where v0.3.8's ``ProceduralMemory`` is the STORE (it holds ``Skill``
objects and matches goals against them), ``SkillDiscoveryEngine`` is a
DISCOVERY pass: given a completed, successful ``TaskGraph``, extract
the actual sequence of completed steps (in dependency order) and feed
it to ``ProceduralMemory.learn_from_success()`` automatically — so
skill learning happens passively from normal agent execution instead
of requiring every caller to remember to report it.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from agents.types import Task, TaskGraph, TaskStatus
from memory.procedural_memory import ProceduralMemory, Skill
from utils.logger import get_logger

log = get_logger(__name__)

_MIN_TRAJECTORY_LENGTH = 2   # trajectories shorter than this aren't worth distilling into a skill


@dataclass
class DiscoveredTrajectory:
    """One extracted (ordered) sequence of completed steps from a successful run."""

    goal: str
    step_titles: list[str] = field(default_factory=list)
    tool_sequence: list[str] = field(default_factory=list)
    graph_id: str = ""
    discovered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "goal": self.goal, "step_titles": self.step_titles, "tool_sequence": self.tool_sequence,
            "graph_id": self.graph_id, "discovered_at": self.discovered_at,
        }

    @property
    def length(self) -> int:
        return len(self.step_titles)


class SkillDiscoveryEngine:
    """
    Extracts reusable skills from successful ``TaskGraph`` trajectories.

    Parameters
    ----------
    procedural_memory:
        ``ProceduralMemory`` — discovered skills are persisted here,
        reusing its existing similarity-matching and reinforcement logic.
    min_trajectory_length:
        Minimum number of completed steps before a trajectory is
        considered worth distilling into a skill.
    """

    def __init__(self, procedural_memory: ProceduralMemory, min_trajectory_length: int = _MIN_TRAJECTORY_LENGTH) -> None:
        self._procedural_memory = procedural_memory
        self._min_length = min_trajectory_length

    # ------------------------------------------------------------------
    # Trajectory extraction
    # ------------------------------------------------------------------

    def extract_trajectory(self, graph: TaskGraph) -> Optional[DiscoveredTrajectory]:
        """
        Extract the ordered sequence of COMPLETED steps from a
        ``TaskGraph``, in dependency order (topological — each step
        appears after everything it depends on).

        Returns ``None`` if the graph wasn't fully successful (any task
        FAILED) or is too short to be worth distilling.
        """
        if any(t.status == TaskStatus.FAILED for t in graph.tasks):
            return None

        completed = [t for t in graph.tasks if t.status == TaskStatus.COMPLETED]
        if len(completed) < self._min_length:
            return None

        ordered = self._topological_order(completed)
        return DiscoveredTrajectory(
            goal=graph.goal, step_titles=[t.title for t in ordered],
            tool_sequence=[t.tool_hint or t.title for t in ordered], graph_id=graph.graph_id,
        )

    @staticmethod
    def _topological_order(tasks: list[Task]) -> list[Task]:
        """Simple Kahn's-algorithm-style topological sort over a task subset by depends_on."""
        by_id = {t.task_id: t for t in tasks}
        in_degree = {t.task_id: 0 for t in tasks}
        for t in tasks:
            for dep in t.depends_on:
                if dep in by_id:
                    in_degree[t.task_id] += 1

        ready = [tid for tid, deg in in_degree.items() if deg == 0]
        ordered: list[Task] = []
        visited: set[str] = set()

        while ready:
            tid = ready.pop(0)
            if tid in visited:
                continue
            visited.add(tid)
            ordered.append(by_id[tid])
            for t in tasks:
                if tid in t.depends_on and t.task_id not in visited:
                    in_degree[t.task_id] -= 1
                    if in_degree[t.task_id] <= 0 and t.task_id not in ready:
                        ready.append(t.task_id)

        # Any tasks not reached (e.g. cyclic edges referencing tasks
        # outside this subset) are appended in original order as a
        # safe fallback, rather than silently dropping steps.
        remaining = [t for t in tasks if t.task_id not in visited]
        return ordered + remaining

    # ------------------------------------------------------------------
    # Discovery + learning
    # ------------------------------------------------------------------

    def discover_and_learn(self, graph: TaskGraph) -> Optional[Skill]:
        """
        Extract a trajectory from ``graph`` (if successful and long
        enough) and feed it into ``ProceduralMemory`` as a learned or
        reinforced skill. Returns the resulting ``Skill``, or ``None``
        if the graph wasn't eligible for skill discovery.
        """
        trajectory = self.extract_trajectory(graph)
        if trajectory is None:
            return None
        return self._procedural_memory.learn_from_success(trajectory.goal, trajectory.tool_sequence)
