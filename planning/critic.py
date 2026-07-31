"""
Plan Critic — Blix v0.3.6  (Upgrade 6)

"Think before acting." Validates a ``TaskGraph`` BEFORE execution starts,
catching:

* Missing steps (heuristic: goal mentions a need the plan never addresses)
* Circular dependencies (a depends on b depends on a)
* Impossible actions (task references a tool that doesn't exist in the registry)
* Risky actions (tasks that would use a historically unreliable tool, or
  match a known ``FailureMemory`` pattern)

The Critic does not block execution outright — it annotates the
``TaskGraph`` with ``CriticIssue`` objects and a verdict
(``approved`` / ``approved_with_warnings`` / ``rejected``), leaving the
final call to the caller (typically the ``AgentExecutor``, which can
choose to run anyway, ask for confirmation, or trigger a replan first).

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agents.types import Task, TaskGraph
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Issue model
# ---------------------------------------------------------------------------


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CriticIssue:
    """One issue found by the Plan Critic."""

    severity: IssueSeverity
    category: str          # "circular_dependency" | "missing_tool" | "risky_tool" | "missing_step" | "known_failure"
    message: str
    task_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "task_id": self.task_id,
        }


class PlanVerdict(str, Enum):
    APPROVED = "approved"
    APPROVED_WITH_WARNINGS = "approved_with_warnings"
    REJECTED = "rejected"


@dataclass
class CritiqueReport:
    """Full critique result for one TaskGraph."""

    verdict: PlanVerdict
    issues: list[CriticIssue] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(i.severity == IssueSeverity.CRITICAL for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == IssueSeverity.WARNING for i in self.issues)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "issue_count": len(self.issues),
            "issues": [i.to_dict() for i in self.issues],
        }


# ---------------------------------------------------------------------------
# Plan Critic
# ---------------------------------------------------------------------------

# Heuristic "need" → keyword signals that a step should exist somewhere in the plan
_NEED_SIGNALS: dict[str, tuple[str, ...]] = {
    "verification": ("test", "verify", "check", "validate"),
    "persistence": ("save", "store", "persist", "write"),
}


class PlanCritic:
    """
    Validates a ``TaskGraph`` before execution.

    Parameters
    ----------
    tool_registry:
        Optional ``ToolRegistry`` — used to flag tasks referencing
        unregistered tools.
    tool_reliability:
        Optional ``ToolReliabilityRegistry`` — used to flag risky
        (historically unreliable) tool choices.
    failure_memory:
        Optional ``FailureMemory`` — used to flag tasks matching known
        past failures.
    reliability_warn_threshold:
        Tools with success_rate below this trigger a WARNING (only if
        the registry has enough confident samples).
    """

    def __init__(
        self,
        tool_registry: Optional[object] = None,
        tool_reliability: Optional[object] = None,
        failure_memory: Optional[object] = None,
        reliability_warn_threshold: float = 0.4,
    ) -> None:
        self._registry = tool_registry
        self._reliability = tool_reliability
        self._failures = failure_memory
        self._warn_threshold = reliability_warn_threshold

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def critique(self, graph: TaskGraph) -> CritiqueReport:
        """Run all checks and return a ``CritiqueReport``."""
        issues: list[CriticIssue] = []

        issues.extend(self._check_circular_dependencies(graph))
        issues.extend(self._check_missing_tools(graph))
        issues.extend(self._check_risky_tools(graph))
        issues.extend(self._check_known_failures(graph))
        issues.extend(self._check_missing_steps(graph))
        issues.extend(self._check_unreachable_tasks(graph))

        if any(i.severity == IssueSeverity.CRITICAL for i in issues):
            verdict = PlanVerdict.REJECTED
        elif any(i.severity == IssueSeverity.WARNING for i in issues):
            verdict = PlanVerdict.APPROVED_WITH_WARNINGS
        else:
            verdict = PlanVerdict.APPROVED

        report = CritiqueReport(verdict=verdict, issues=issues)
        log.info(
            "PlanCritic: verdict=%s issues=%d (critical=%d, warning=%d)",
            verdict.value, len(issues),
            sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL),
            sum(1 for i in issues if i.severity == IssueSeverity.WARNING),
        )
        return report

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _check_circular_dependencies(self, graph: TaskGraph) -> list[CriticIssue]:
        """Detect cycles in the dependency graph via DFS."""
        issues: list[CriticIssue] = []
        adjacency = {t.task_id: t.depends_on for t in graph.tasks}

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in adjacency}

        def visit(tid: str, path: list[str]) -> Optional[list[str]]:
            color[tid] = GRAY
            for dep in adjacency.get(tid, []):
                if dep not in adjacency:
                    continue  # dangling dependency, handled elsewhere
                if color.get(dep, WHITE) == GRAY:
                    return path + [dep]
                if color.get(dep, WHITE) == WHITE:
                    cycle = visit(dep, path + [dep])
                    if cycle:
                        return cycle
            color[tid] = BLACK
            return None

        seen_cycles: set[frozenset] = set()
        for tid in adjacency:
            if color[tid] == WHITE:
                cycle = visit(tid, [tid])
                if cycle:
                    key = frozenset(cycle)
                    if key not in seen_cycles:
                        seen_cycles.add(key)
                        issues.append(CriticIssue(
                            severity=IssueSeverity.CRITICAL,
                            category="circular_dependency",
                            message=f"Circular dependency detected: {' → '.join(cycle)}",
                            task_id=cycle[0],
                        ))
        return issues

    def _check_missing_tools(self, graph: TaskGraph) -> list[CriticIssue]:
        """Flag tasks whose tool_hint references an unregistered tool."""
        if self._registry is None:
            return []
        issues: list[CriticIssue] = []
        registered = {t.name for t in self._registry.list_tools()}  # type: ignore[union-attr]
        for task in graph.tasks:
            if task.tool_hint and task.tool_hint not in registered:
                issues.append(CriticIssue(
                    severity=IssueSeverity.CRITICAL,
                    category="missing_tool",
                    message=f"Task '{task.title}' references unregistered tool '{task.tool_hint}'.",
                    task_id=task.task_id,
                ))
        return issues

    def _check_risky_tools(self, graph: TaskGraph) -> list[CriticIssue]:
        """Flag tasks using historically unreliable tools."""
        if self._reliability is None:
            return []
        issues: list[CriticIssue] = []
        for task in graph.tasks:
            if not task.tool_hint:
                continue
            if not self._reliability.is_confident(task.tool_hint):  # type: ignore[union-attr]
                continue
            rate = self._reliability.success_rate(task.tool_hint)  # type: ignore[union-attr]
            if rate < self._warn_threshold:
                issues.append(CriticIssue(
                    severity=IssueSeverity.WARNING,
                    category="risky_tool",
                    message=(
                        f"Task '{task.title}' uses '{task.tool_hint}' "
                        f"(success_rate={rate:.2f}, historically unreliable)."
                    ),
                    task_id=task.task_id,
                ))
        return issues

    def _check_known_failures(self, graph: TaskGraph) -> list[CriticIssue]:
        """Flag tasks matching known FailureMemory patterns."""
        if self._failures is None:
            return []
        issues: list[CriticIssue] = []
        for task in graph.tasks:
            matches = self._failures.similar_failures(task.title, task.tool_hint)  # type: ignore[union-attr]
            if matches:
                top = matches[0]
                msg = f"Task '{task.title}' resembles a past failure: {top.failure}"
                if top.fix:
                    msg += f" (known fix: {top.fix})"
                issues.append(CriticIssue(
                    severity=IssueSeverity.WARNING,
                    category="known_failure",
                    message=msg,
                    task_id=task.task_id,
                ))
        return issues

    def _check_missing_steps(self, graph: TaskGraph) -> list[CriticIssue]:
        """
        Heuristic: if the goal text implies a need (verification,
        persistence) but no task addresses it, flag as INFO.
        """
        issues: list[CriticIssue] = []
        goal_lower = graph.goal.lower()
        all_text = " ".join((t.title + " " + t.description).lower() for t in graph.tasks)

        for need, signals in _NEED_SIGNALS.items():
            goal_implies = any(s in goal_lower for s in signals)
            plan_addresses = any(s in all_text for s in signals)
            if goal_implies and not plan_addresses:
                issues.append(CriticIssue(
                    severity=IssueSeverity.INFO,
                    category="missing_step",
                    message=f"Goal implies a need for '{need}' but no task addresses it.",
                ))
        return issues

    def _check_unreachable_tasks(self, graph: TaskGraph) -> list[CriticIssue]:
        """Flag tasks depending on a task_id that doesn't exist in the graph."""
        issues: list[CriticIssue] = []
        all_ids = {t.task_id for t in graph.tasks}
        for task in graph.tasks:
            dangling = [d for d in task.depends_on if d not in all_ids]
            if dangling:
                issues.append(CriticIssue(
                    severity=IssueSeverity.CRITICAL,
                    category="dangling_dependency",
                    message=f"Task '{task.title}' depends on nonexistent task id(s): {dangling}",
                    task_id=task.task_id,
                ))
        return issues
