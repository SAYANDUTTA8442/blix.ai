"""
Dynamic Replanning Engine — Blix v0.3.6  (Upgrade 1)

The biggest jump toward autonomy: instead of "plan once, execute forever",
the agent can revise its plan mid-execution when a task fails persistently.

    Plan → Execute → Observe → Replan → Continue

Example (from spec)
--------------------
    Goal: Build RAG System
    Step 1: Use Chroma
    Observation: Chroma unavailable
    Replanner: Switch to FAISS
    Continue

Trigger conditions
-------------------
The ``AgentExecutor`` calls ``Replanner.should_replan()`` when a task
exhausts its retries (the v0.3.5 "skip" decision point). If a replan is
warranted, ``Replanner.replan()`` produces a patched ``TaskGraph``:

1. The failed task is replaced with an alternative-tool variant (if a
   different tool can plausibly handle it), OR
2. The failed task is decomposed into smaller sub-steps, OR
3. As a last resort, the task is dropped and downstream tasks are
   re-checked for feasibility without it.

Every replan is recorded as a structured failure (via ``FailureMemory``)
so repeated failures of the *same* kind don't trigger infinite replan
loops — ``max_replans_per_task`` bounds this.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agents.types import Task, TaskGraph, TaskStatus
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Replan strategies
# ---------------------------------------------------------------------------


class ReplanStrategy(str, Enum):
    SWITCH_TOOL = "switch_tool"
    DECOMPOSE = "decompose"
    DROP_TASK = "drop_task"
    NO_ACTION = "no_action"


@dataclass
class ReplanResult:
    """Outcome of one replanning attempt."""

    strategy: ReplanStrategy
    modified_task_ids: list[str] = field(default_factory=list)
    new_task_ids: list[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "modified_task_ids": self.modified_task_ids,
            "new_task_ids": self.new_task_ids,
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Tool alternatives table — heuristic substitution map
# ---------------------------------------------------------------------------

# When a tool fails, what's a plausible alternative for the same kind of work?
_TOOL_ALTERNATIVES: dict[str, list[str]] = {
    "web_search": ["memory_search", "llm"],
    "memory_search": ["web_search", "llm"],
    "python_tool": ["llm"],
    "synthesis": ["llm"],
    "reasoning": ["memory_search", "llm"],
    "file_tool": ["llm"],
}


# ---------------------------------------------------------------------------
# Replanner
# ---------------------------------------------------------------------------


class Replanner:
    """
    Decides whether and how to revise a ``TaskGraph`` mid-execution.

    Parameters
    ----------
    tool_registry:
        Optional ``ToolRegistry`` — used to confirm alternative tools exist.
    failure_memory:
        Optional ``FailureMemory`` — records each replan trigger and
        looks up known fixes from prior runs.
    tool_reliability:
        Optional ``ToolReliabilityRegistry`` — used to rank alternative
        tools by historical reliability rather than a fixed heuristic order.
    max_replans_per_task:
        Hard cap on how many times a single task may be replanned, to
        prevent infinite substitution loops.
    """

    def __init__(
        self,
        tool_registry: Optional[object] = None,
        failure_memory: Optional[object] = None,
        tool_reliability: Optional[object] = None,
        max_replans_per_task: int = 2,
    ) -> None:
        self._registry = tool_registry
        self._failures = failure_memory
        self._reliability = tool_reliability
        self._max_replans = max_replans_per_task

    # ------------------------------------------------------------------
    # Trigger decision
    # ------------------------------------------------------------------

    def should_replan(self, task: Task, graph: TaskGraph) -> bool:
        """
        Decide whether a failed task warrants replanning rather than
        simply being marked FAILED.

        Triggers when:
        - the task has failed and hasn't exceeded max_replans_per_task, AND
        - an alternative tool exists OR the task can plausibly be decomposed
        """
        replan_count = task.metadata.get("replan_count", 0)
        if replan_count >= self._max_replans:
            return False
        if task.status != TaskStatus.FAILED:
            return False
        return True

    # ------------------------------------------------------------------
    # Core replanning
    # ------------------------------------------------------------------

    def replan(self, task: Task, graph: TaskGraph, failure_reason: str = "") -> ReplanResult:
        """
        Revise the plan to recover from ``task``'s failure.

        Mutates ``graph`` in place (or ``task``'s fields) and returns a
        ``ReplanResult`` describing what was done.
        """
        # Record the failure for future planning, before attempting recovery
        if self._failures is not None:
            self._failures.record(  # type: ignore[union-attr]
                task_title=task.title, tool=task.tool_hint or "unknown",
                failure=failure_reason or task.error, goal=graph.goal,
            )

        replan_count = task.metadata.get("replan_count", 0)

        # Strategy 1: switch tool
        alt_tool = self._find_alternative_tool(task)
        if alt_tool:
            return self._apply_switch_tool(task, alt_tool, replan_count)

        # Strategy 2: decompose into smaller steps
        if self._can_decompose(task):
            return self._apply_decompose(task, graph, replan_count)

        # Strategy 3: drop the task and let downstream tasks proceed without it
        return self._apply_drop_task(task, graph, replan_count)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _find_alternative_tool(self, task: Task) -> Optional[str]:
        current_tool = task.tool_hint
        if current_tool is None:
            return None

        candidates = _TOOL_ALTERNATIVES.get(current_tool, [])
        # Filter out tools already tried for this task
        tried = set(task.metadata.get("tried_tools", [current_tool]))
        candidates = [c for c in candidates if c not in tried]

        if not candidates:
            return None

        # Confirm the candidate is actually registered
        if self._registry is not None:
            registered = {t.name for t in self._registry.list_tools()}  # type: ignore[union-attr]
            candidates = [c for c in candidates if c in registered]

        if not candidates:
            return None

        # Rank by reliability if available
        if self._reliability is not None:
            ranked = self._reliability.rank_tools_by_reliability(candidates)  # type: ignore[union-attr]
            return ranked[0][0] if ranked else candidates[0]

        return candidates[0]

    def _apply_switch_tool(self, task: Task, alt_tool: str, replan_count: int) -> ReplanResult:
        old_tool = task.tool_hint
        tried = set(task.metadata.get("tried_tools", [old_tool] if old_tool else []))
        tried.add(alt_tool)

        task.tool_hint = alt_tool
        task.status = TaskStatus.PENDING
        task.attempts = 0  # fresh attempts budget with the new tool
        task.metadata["tried_tools"] = list(tried)
        task.metadata["replan_count"] = replan_count + 1
        task.error = ""

        explanation = f"Switched tool from '{old_tool}' to '{alt_tool}' after failure."
        log.info("Replanner: %s (task='%s')", explanation, task.title)
        return ReplanResult(
            strategy=ReplanStrategy.SWITCH_TOOL,
            modified_task_ids=[task.task_id],
            explanation=explanation,
        )

    def _can_decompose(self, task: Task) -> bool:
        """Heuristic: only decompose tasks that haven't already been decomposed."""
        return not task.metadata.get("decomposed", False) and len(task.description) > 20

    def _apply_decompose(self, task: Task, graph: TaskGraph, replan_count: int) -> ReplanResult:
        """
        Split a failed task into two smaller sequential sub-tasks using the
        same tool, on the theory that a smaller, more specific request is
        more likely to succeed.
        """
        sub1 = Task(
            title=f"{task.title} (part 1)",
            description=f"Gather initial information for: {task.description}",
            tool_hint=task.tool_hint,
            depends_on=list(task.depends_on),
        )
        sub2 = Task(
            title=f"{task.title} (part 2)",
            description=f"Complete the remaining work for: {task.description}",
            tool_hint=task.tool_hint,
            depends_on=[sub1.task_id],
        )
        sub1.metadata["decomposed"] = True
        sub2.metadata["decomposed"] = True

        # Re-point anything that depended on the original failed task to sub2
        for other in graph.tasks:
            if task.task_id in other.depends_on:
                other.depends_on = [d for d in other.depends_on if d != task.task_id] + [sub2.task_id]

        graph.tasks = [t for t in graph.tasks if t.task_id != task.task_id]
        graph.add_task(sub1)
        graph.add_task(sub2)

        explanation = f"Decomposed failed task '{task.title}' into two sub-steps."
        log.info("Replanner: %s", explanation)
        return ReplanResult(
            strategy=ReplanStrategy.DECOMPOSE,
            modified_task_ids=[task.task_id],
            new_task_ids=[sub1.task_id, sub2.task_id],
            explanation=explanation,
        )

    def _apply_drop_task(self, task: Task, graph: TaskGraph, replan_count: int) -> ReplanResult:
        """
        Last resort: mark the task permanently skipped and remove it from
        other tasks' dependency lists so the rest of the plan can proceed.
        """
        task.status = TaskStatus.SKIPPED
        task.metadata["replan_count"] = replan_count + 1

        for other in graph.tasks:
            if task.task_id in other.depends_on:
                other.depends_on = [d for d in other.depends_on if d != task.task_id]

        explanation = f"Dropped unrecoverable task '{task.title}'; downstream tasks unblocked."
        log.info("Replanner: %s", explanation)
        return ReplanResult(
            strategy=ReplanStrategy.DROP_TASK,
            modified_task_ids=[task.task_id],
            explanation=explanation,
        )
