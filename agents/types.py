"""
Agent shared types — Blix v0.3.5

Defines the core data structures used across planning, execution,
tools, observation, and reflection:

    Task / TaskGraph / TaskStatus
    ExecutionResult / Observation
    WorkingMemoryEntry

Python 3.10 compatible.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """
    A single atomic unit of agent work.

    Fields
    ------
    task_id:
        Unique id (uuid4 hex prefix by default).
    title:
        Short human-readable label.
    description:
        Full natural-language description of what needs to be done.
    depends_on:
        List of task_ids that must complete before this task can start.
    status:
        Current lifecycle status.
    tool_hint:
        Optional name of the tool most likely needed (e.g. "web_search").
    result:
        Result text once the task is completed.
    error:
        Error message if the task failed.
    attempts:
        How many times execution was attempted.
    created_at / completed_at:
        ISO timestamps.
    metadata:
        Arbitrary key-value store for tool-specific context.
    """

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str = ""
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    tool_hint: Optional[str] = None
    result: str = ""
    error: str = ""
    attempts: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def is_ready(self, completed_ids: set[str]) -> bool:
        """Return True if all dependencies are satisfied."""
        return all(dep in completed_ids for dep in self.depends_on)

    def mark_completed(self, result: str) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def mark_failed(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "tool_hint": self.tool_hint,
            "result": self.result[:500] if self.result else "",
            "error": self.error,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Task Graph
# ---------------------------------------------------------------------------


@dataclass
class TaskGraph:
    """
    A directed acyclic graph (DAG) of Tasks for one agent goal.

    Fields
    ------
    graph_id:
        Unique id for this execution plan.
    goal:
        The high-level goal this graph is solving.
    tasks:
        Ordered list of Task objects.
    """

    graph_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    goal: str = ""
    tasks: list[Task] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def get_task(self, task_id: str) -> Optional[Task]:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def ready_tasks(self) -> list[Task]:
        """Return tasks whose dependencies are all completed."""
        completed = {t.task_id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        return [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING and t.is_ready(completed)
        ]

    def next_task(self) -> Optional[Task]:
        """Return the first ready task (topological order)."""
        ready = self.ready_tasks()
        return ready[0] if ready else None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
                   for t in self.tasks)

    @property
    def has_failures(self) -> bool:
        return any(t.status == TaskStatus.FAILED for t in self.tasks)

    @property
    def progress(self) -> int:
        if not self.tasks:
            return 0
        done = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return round(100 * done / len(self.tasks))

    def status_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {s.value: 0 for s in TaskStatus}
        for t in self.tasks:
            counts[t.status.value] += 1
        return counts

    def to_dict(self) -> dict:
        return {
            "graph_id": self.graph_id,
            "goal": self.goal,
            "progress": self.progress,
            "is_complete": self.is_complete,
            "has_failures": self.has_failures,
            "status_summary": self.status_summary(),
            "tasks": [t.to_dict() for t in self.tasks],
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Execution result & Observation
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """
    Raw output of a single tool call or task execution step.

    Produced by tools and consumed by the Observation layer.
    """

    task_id: str
    tool_name: str
    status: ExecutionStatus
    output: str = ""
    raw: Any = None               # tool-specific raw output (JSON, bytes, etc.)
    error: str = ""
    duration_ms: float = 0.0
    tokens_used: int = 0
    executed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def is_success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "output": self.output[:1000],
            "error": self.error,
            "duration_ms": round(self.duration_ms, 1),
            "tokens_used": self.tokens_used,
            "executed_at": self.executed_at,
        }


@dataclass
class Observation:
    """
    Structured interpretation of an ExecutionResult.

    The Observation layer transforms raw tool outputs into structured
    facts that the Reflection loop can reason over.

    Fields
    ------
    success:
        Whether the task produced useful output.
    summary:
        Natural-language summary of what was observed.
    extracted_facts:
        Key factual items extracted from the output.
    quality_score:
        0–1 estimate of output quality (for reflection).
    retry_suggested:
        Whether the reflection engine suggests retrying.
    retry_hint:
        Suggested modification if retry is recommended.
    """

    task_id: str
    tool_name: str
    success: bool
    summary: str = ""
    extracted_facts: list[str] = field(default_factory=list)
    quality_score: float = 0.5
    retry_suggested: bool = False
    retry_hint: str = ""
    raw_result: Optional[ExecutionResult] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "summary": self.summary,
            "extracted_facts": self.extracted_facts,
            "quality_score": round(self.quality_score, 3),
            "retry_suggested": self.retry_suggested,
            "retry_hint": self.retry_hint,
        }


# ---------------------------------------------------------------------------
# Execution History Entry
# ---------------------------------------------------------------------------


@dataclass
class ExecutionHistoryEntry:
    """
    Persisted record of one completed task execution.

    Every action becomes a searchable entry in long-term memory.
    """

    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    goal: str = ""
    task_id: str = ""
    task_title: str = ""
    tool: str = ""
    result_summary: str = ""
    success: bool = False
    quality_score: float = 0.0
    reflection_note: str = ""
    executed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "goal": self.goal,
            "task_id": self.task_id,
            "task_title": self.task_title,
            "tool": self.tool,
            "result_summary": self.result_summary[:300],
            "success": self.success,
            "quality_score": round(self.quality_score, 3),
            "reflection_note": self.reflection_note,
            "executed_at": self.executed_at,
        }


# ---------------------------------------------------------------------------
# Working Memory Entry
# ---------------------------------------------------------------------------


@dataclass
class WorkingMemoryEntry:
    """
    One slot in the agent's working (short-term) memory.

    Stores intermediate reasoning, tool outputs, and state during
    an active task graph execution.
    """

    key: str                        # e.g. "task_3_output", "search_results"
    value: Any
    task_id: Optional[str] = None
    ttl_steps: int = 10            # evict after this many agent steps
    age_steps: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def is_expired(self) -> bool:
        return self.age_steps >= self.ttl_steps

    def to_dict(self) -> dict:
        val = self.value
        if not isinstance(val, (str, int, float, bool, list, dict, type(None))):
            val = str(val)
        return {
            "key": self.key,
            "value": val,
            "task_id": self.task_id,
            "ttl_steps": self.ttl_steps,
            "age_steps": self.age_steps,
        }
