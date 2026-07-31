"""
Agent State — Blix v0.3.6  (Upgrade 10)

A single unified state object threaded through every cognitive module in
the v0.3.6 loop:

    Goal → Planner → Critic → DAG Runtime → Verifier → Observation
         → Reflection → Replanner → Memory

Before v0.3.6, state was scattered: ``WorkingMemory`` held tool outputs,
``TaskGraph`` held task status, ``ReflectionLoop`` held history — each
module only saw its own slice. ``AgentState`` gives every module the
full picture: the active plan, what succeeded/failed, what was observed,
and how confident the agent currently is in reaching the goal.

This does NOT replace ``WorkingMemory`` (still used for TTL-scoped
ephemeral key-value data) — ``AgentState`` is the structural/cognitive
record, ``WorkingMemory`` is the scratch pad.

Python 3.10 compatible.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from agents.types import ExecutionHistoryEntry, Observation, Task, TaskGraph


@dataclass
class AgentState:
    """
    Unified cognitive state for one agent run.

    Fields
    ------
    state_id:
        Unique id for this run.
    goal:
        The original natural-language goal.
    active_plan:
        Current ``TaskGraph`` (replaced wholesale on replan, see Upgrade 1).
    plan_version:
        Incremented every time the plan is replaced by the Replanner.
    completed_tasks:
        Task ids that finished successfully.
    failed_tasks:
        Task ids that failed (after exhausting retries).
    observations:
        All ``Observation`` objects produced so far, in order.
    failure_records:
        Structured failure records for Failure Memory (Upgrade 4).
    tool_reliability:
        Running success-rate per tool name (Upgrade 5), updated live.
    confidence:
        0–1 running estimate of "will this run reach the goal successfully".
    replan_count:
        How many times the Replanner has intervened.
    cost:
        Running ``ExecutionCostModel`` totals (Upgrade 7).
    """

    state_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    goal: str = ""
    active_plan: Optional[TaskGraph] = None
    plan_version: int = 1
    completed_tasks: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    failure_records: list[dict] = field(default_factory=list)
    tool_reliability: dict[str, "ToolReliabilityStats"] = field(default_factory=dict)
    confidence: float = 0.5
    replan_count: int = 0
    cost: "ExecutionCostModel" = field(default_factory=lambda: ExecutionCostModel())
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def set_plan(self, graph: TaskGraph, is_replan: bool = False) -> None:
        """Install a new plan. If ``is_replan``, bumps plan_version and replan_count."""
        self.active_plan = graph
        if is_replan:
            self.plan_version += 1
            self.replan_count += 1
        self._touch()

    def record_observation(self, observation: Observation) -> None:
        self.observations.append(observation)
        self._touch()

    def record_completion(self, task_id: str) -> None:
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)
        self._touch()

    def record_failure(self, task_id: str, failure_record: Optional[dict] = None) -> None:
        if task_id not in self.failed_tasks:
            self.failed_tasks.append(task_id)
        if failure_record:
            self.failure_records.append(failure_record)
        self._touch()

    def update_confidence(self, new_value: float) -> None:
        self.confidence = max(0.0, min(1.0, new_value))
        self._touch()

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def progress(self) -> int:
        return self.active_plan.progress if self.active_plan else 0

    @property
    def is_stalled(self) -> bool:
        """True if the plan has failures but no path to completion remains."""
        if self.active_plan is None:
            return False
        return self.active_plan.has_failures and not self.active_plan.is_complete

    def recent_observations(self, n: int = 5) -> list[Observation]:
        return self.observations[-n:]

    def to_dict(self) -> dict:
        return {
            "state_id": self.state_id,
            "goal": self.goal,
            "plan_version": self.plan_version,
            "progress": self.progress,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "observation_count": len(self.observations),
            "failure_record_count": len(self.failure_records),
            "tool_reliability": {k: v.to_dict() for k, v in self.tool_reliability.items()},
            "confidence": round(self.confidence, 3),
            "replan_count": self.replan_count,
            "cost": self.cost.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------------------
# Tool reliability stats (Upgrade 5) — lives alongside AgentState
# ---------------------------------------------------------------------------


@dataclass
class ToolReliabilityStats:
    """Running success-rate tracker for one tool."""

    tool_name: str
    successes: int = 0
    failures: int = 0

    def record(self, success: bool) -> None:
        if success:
            self.successes += 1
        else:
            self.failures += 1

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.5  # neutral prior — unknown reliability
        return self.successes / self.total

    def to_dict(self) -> dict:
        return {
            "tool": self.tool_name,
            "success_rate": round(self.success_rate, 3),
            "successes": self.successes,
            "failures": self.failures,
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# Execution cost model (Upgrade 7) — lives alongside AgentState
# ---------------------------------------------------------------------------


@dataclass
class ExecutionCostModel:
    """Running cost totals for one agent run."""

    token_cost: int = 0
    execution_time_secs: float = 0.0
    tool_calls: int = 0
    retry_count: int = 0

    def record_call(
        self,
        tokens: int = 0,
        duration_secs: float = 0.0,
        is_retry: bool = False,
    ) -> None:
        self.tool_calls += 1
        self.token_cost += tokens
        self.execution_time_secs += duration_secs
        if is_retry:
            self.retry_count += 1

    def to_dict(self) -> dict:
        return {
            "token_cost": self.token_cost,
            "execution_time_secs": round(self.execution_time_secs, 2),
            "tool_calls": self.tool_calls,
            "retry_count": self.retry_count,
        }

    def efficiency_score(self) -> float:
        """
        Rough efficiency heuristic: successful work per retry.
        Higher is better; 1.0 = no retries needed at all.
        """
        if self.tool_calls == 0:
            return 1.0
        return round(1.0 - (self.retry_count / max(1, self.tool_calls)), 3)
