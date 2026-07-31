"""
Execution DAG Runtime — Blix v0.3.6  (Upgrade 3)

Upgrades task scheduling from sequential ready-task picking (v0.3.5
``TaskGraph.next_task()`` returns one task at a time) to proper DAG
semantics:

    Task A
     ├── Task B
     ├── Task C
     └── Task D

with:
* dependency tracking      (already in v0.3.5 Task.depends_on)
* parallel execution batches (NEW — ``next_batch()`` returns ALL
  currently-ready tasks, not just one, so independent branches can run
  concurrently)
* failure propagation       (NEW — if A fails, B/C/D that depend on A
  are automatically marked BLOCKED rather than waiting forever)

This module wraps a ``TaskGraph`` rather than replacing it — fully
backwards compatible with v0.3.5's sequential ``AgentExecutor`` loop,
which can continue to call ``graph.next_task()`` and ignore this module
entirely if parallelism isn't needed.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agents.types import Task, TaskGraph, TaskStatus
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class DAGRuntimeStats:
    """Summary of one DAG runtime pass."""

    batches_executed: int = 0
    max_batch_size: int = 0
    blocked_tasks: int = 0

    def to_dict(self) -> dict:
        return {
            "batches_executed": self.batches_executed,
            "max_batch_size": self.max_batch_size,
            "blocked_tasks": self.blocked_tasks,
        }


class TaskRuntime:
    """
    Dependency-aware scheduler wrapping a ``TaskGraph``.

    Parameters
    ----------
    graph:
        The ``TaskGraph`` to schedule.
    max_parallel:
        Maximum tasks returned per batch (caps fan-out concurrency).
    """

    def __init__(self, graph: TaskGraph, max_parallel: int = 4) -> None:
        self._graph = graph
        self._max_parallel = max_parallel
        self._stats = DAGRuntimeStats()

    # ------------------------------------------------------------------
    # Batch scheduling
    # ------------------------------------------------------------------

    def next_batch(self) -> list[Task]:
        """
        Return up to ``max_parallel`` ready tasks (PENDING with all
        dependencies COMPLETED) that can run concurrently right now.
        """
        self.propagate_failures()
        ready = self._graph.ready_tasks()
        batch = ready[: self._max_parallel]
        if batch:
            self._stats.batches_executed += 1
            self._stats.max_batch_size = max(self._stats.max_batch_size, len(batch))
        return batch

    def has_runnable_work(self) -> bool:
        """True if there is at least one PENDING task that could still run."""
        self.propagate_failures()
        return len(self._graph.ready_tasks()) > 0

    # ------------------------------------------------------------------
    # Failure propagation
    # ------------------------------------------------------------------

    def propagate_failures(self) -> int:
        """
        Mark any PENDING task as BLOCKED if one or more of its
        dependencies has FAILED or is BLOCKED (transitively).

        Returns the number of tasks newly marked BLOCKED in this pass.
        """
        failed_or_blocked = {
            t.task_id for t in self._graph.tasks
            if t.status in (TaskStatus.FAILED, TaskStatus.BLOCKED)
        }
        newly_blocked = 0
        changed = True

        # Iterate until stable (transitive propagation across chains)
        while changed:
            changed = False
            for task in self._graph.tasks:
                if task.status != TaskStatus.PENDING:
                    continue
                if any(dep in failed_or_blocked for dep in task.depends_on):
                    task.status = TaskStatus.BLOCKED
                    task.error = task.error or "Blocked: upstream dependency failed."
                    failed_or_blocked.add(task.task_id)
                    newly_blocked += 1
                    changed = True

        if newly_blocked:
            self._stats.blocked_tasks += newly_blocked
            log.info("TaskRuntime: propagated failure to %d blocked task(s).", newly_blocked)
        return newly_blocked

    def unblock(self, task_id: str) -> bool:
        """
        Manually un-block a task (e.g. after a successful Replanner
        intervention on its blocking dependency). Returns True if the
        task was BLOCKED and is now PENDING again.
        """
        task = self._graph.get_task(task_id)
        if task is None or task.status != TaskStatus.BLOCKED:
            return False
        task.status = TaskStatus.PENDING
        task.error = ""
        return True

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------

    def topological_batches(self) -> list[list[str]]:
        """
        Compute the full batch structure (task ids grouped by dependency
        depth) without mutating task state — useful for visualisation
        and the Plan Critic.
        """
        remaining = {t.task_id: set(t.depends_on) for t in self._graph.tasks}
        batches: list[list[str]] = []
        done: set[str] = set()

        while remaining:
            ready = [tid for tid, deps in remaining.items() if deps <= done]
            if not ready:
                # Cycle or dangling dependency — stop to avoid infinite loop
                log.warning("TaskRuntime: topological_batches stalled (cycle or dangling deps).")
                break
            batches.append(ready)
            done.update(ready)
            for tid in ready:
                del remaining[tid]

        return batches

    @property
    def stats(self) -> DAGRuntimeStats:
        return self._stats

    @property
    def is_complete(self) -> bool:
        return self._graph.is_complete

    @property
    def has_unrecoverable_blocks(self) -> bool:
        """True if any task is permanently BLOCKED with no PENDING tasks left to run."""
        any_blocked = any(t.status == TaskStatus.BLOCKED for t in self._graph.tasks)
        any_runnable = self.has_runnable_work()
        return any_blocked and not any_runnable and not self._graph.is_complete
