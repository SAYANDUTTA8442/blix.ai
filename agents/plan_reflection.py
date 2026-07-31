"""
Plan Reflection — Blix v0.3.6  (Upgrade 8)

v0.3.5's ``ReflectionLoop`` reflects on individual TASK results. This
module adds REFLECTION ON THE PLAN AS A WHOLE — the questions a human
project lead would ask after a run finishes (or stalls):

    Why did the plan fail?
    Which step caused the failure?
    Can future plans improve?

This is where actual learning starts: ``PlanReflection`` produces a
``PlanReflectionReport`` that feeds directly into ``FailureMemory``
(Upgrade 4) and can inform the next ``Planner.plan()`` call for a
similar goal (by checking ``FailureMemory`` / ``ToolReliabilityRegistry``
before decomposing).

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agents.types import TaskGraph, TaskStatus
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------


@dataclass
class PlanReflectionReport:
    """
    Reflection on a completed (or stalled) agent run, at the plan level.

    Fields
    ------
    success:
        Whether the plan ultimately completed successfully.
    root_cause:
        Best-guess explanation of why the plan failed (None if it succeeded).
    failure_task_id:
        The task_id that first caused the cascading failure, if any.
    bottleneck_tool:
        The tool name most associated with failures in this run, if any.
    improvement_suggestions:
        Concrete suggestions for future plans tackling a similar goal.
    lessons:
        Short natural-language lessons, suitable for storage as Insights
        (v0.3.2 ``ReflectionEngine``) or ``FailureMemory`` fixes.
    """

    success: bool
    root_cause: Optional[str] = None
    failure_task_id: Optional[str] = None
    bottleneck_tool: Optional[str] = None
    improvement_suggestions: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "root_cause": self.root_cause,
            "failure_task_id": self.failure_task_id,
            "bottleneck_tool": self.bottleneck_tool,
            "improvement_suggestions": self.improvement_suggestions,
            "lessons": self.lessons,
        }

    def summary(self) -> str:
        if self.success:
            return "Plan completed successfully. " + " ".join(self.lessons)
        parts = [f"Plan failed: {self.root_cause or 'unknown cause'}."]
        if self.bottleneck_tool:
            parts.append(f"Bottleneck tool: {self.bottleneck_tool}.")
        if self.improvement_suggestions:
            parts.append("Suggestions: " + "; ".join(self.improvement_suggestions))
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Plan Reflection engine
# ---------------------------------------------------------------------------


class PlanReflection:
    """
    Analyses a completed/stalled ``TaskGraph`` (plus run history) to
    produce a plan-level reflection.

    Parameters
    ----------
    failure_memory:
        Optional ``FailureMemory`` — lessons are persisted here as fixes
        once a successful replan strategy is identified.
    reflection_engine:
        Optional v0.3.2 ``ReflectionEngine`` — lessons are also stored
        as PROJECT-scope insights for long-term visibility.
    """

    def __init__(
        self,
        failure_memory: Optional[object] = None,
        reflection_engine: Optional[object] = None,
    ) -> None:
        self._failures = failure_memory
        self._reflection_engine = reflection_engine

    def reflect(
        self,
        graph: TaskGraph,
        history: list[dict],
        replan_count: int = 0,
    ) -> PlanReflectionReport:
        """
        Produce a ``PlanReflectionReport`` for the given graph + run history.

        Parameters
        ----------
        graph:
            The (final) ``TaskGraph`` after execution.
        history:
            The step-by-step decision log from ``AgentExecutor.run()``
            (list of dicts with task_id/tool/decision/quality/note).
        replan_count:
            How many times the Replanner intervened during this run.
        """
        success = graph.is_complete and not graph.has_failures

        if success:
            lessons = self._extract_success_lessons(graph, history, replan_count)
            report = PlanReflectionReport(success=True, lessons=lessons)
            self._persist(graph, report)
            return report

        # Failure analysis
        failure_task = self._find_first_failure(graph)
        bottleneck_tool = self._find_bottleneck_tool(history)
        root_cause = self._explain_root_cause(failure_task, history)
        suggestions = self._generate_suggestions(failure_task, bottleneck_tool, replan_count)

        report = PlanReflectionReport(
            success=False,
            root_cause=root_cause,
            failure_task_id=failure_task.task_id if failure_task else None,
            bottleneck_tool=bottleneck_tool,
            improvement_suggestions=suggestions,
            lessons=[root_cause] if root_cause else [],
        )
        self._persist(graph, report)
        log.info("PlanReflection: %s", report.summary())
        return report

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _find_first_failure(self, graph: TaskGraph):
        failed = [t for t in graph.tasks if t.status == TaskStatus.FAILED]
        if not failed:
            return None
        # Earliest-created failed task is the most likely root cause
        return sorted(failed, key=lambda t: t.created_at)[0]

    def _find_bottleneck_tool(self, history: list[dict]) -> Optional[str]:
        """Find the tool with the most 'retry' or failed decisions in history."""
        tool_failure_counts: dict[str, int] = {}
        for h in history:
            if h.get("decision") in ("retry", "skip"):
                tool = h.get("tool", "")
                if tool:
                    tool_failure_counts[tool] = tool_failure_counts.get(tool, 0) + 1
        if not tool_failure_counts:
            return None
        return max(tool_failure_counts, key=lambda t: tool_failure_counts[t])

    def _explain_root_cause(self, failure_task, history: list[dict]) -> Optional[str]:
        if failure_task is None:
            return "Plan stalled with no ready tasks (likely a dependency or scheduling issue)."
        cause = f"Task '{failure_task.title}' failed"
        if failure_task.error:
            cause += f": {failure_task.error}"
        return cause

    def _generate_suggestions(
        self, failure_task, bottleneck_tool: Optional[str], replan_count: int,
    ) -> list[str]:
        suggestions: list[str] = []
        if bottleneck_tool:
            suggestions.append(
                f"Consider preferring an alternative to '{bottleneck_tool}' for similar tasks."
            )
        if failure_task is not None:
            suggestions.append(
                f"Break down '{failure_task.title}' into smaller, more specific sub-tasks."
            )
        if replan_count == 0:
            suggestions.append("Enable replanning for this goal type to allow tool substitution.")
        if not suggestions:
            suggestions.append("Insufficient information to suggest a specific improvement.")
        return suggestions

    def _extract_success_lessons(
        self, graph: TaskGraph, history: list[dict], replan_count: int,
    ) -> list[str]:
        lessons = []
        if replan_count > 0:
            lessons.append(f"Plan succeeded after {replan_count} replan(s) — substitution strategy worked.")
        retries = sum(1 for h in history if h.get("decision") == "retry")
        if retries > 0:
            lessons.append(f"Plan required {retries} retr(y/ies) but ultimately succeeded.")
        if not lessons:
            lessons.append("Plan executed cleanly with no retries or replans.")
        return lessons

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, graph: TaskGraph, report: PlanReflectionReport) -> None:
        if self._reflection_engine is not None:
            try:
                from reflection.reflection_engine import ReflectionScope
                self._reflection_engine.reflect(  # type: ignore[union-attr]
                    ReflectionScope.PROJECT,
                    f"agent_plan_{graph.graph_id}",
                    report.summary(),
                )
            except Exception as exc:
                log.debug("PlanReflection: reflection_engine persist failed (%s)", exc)

        if not report.success and self._failures is not None and report.failure_task_id:
            task = graph.get_task(report.failure_task_id)
            if task is not None and report.improvement_suggestions:
                try:
                    self._failures.record_fix(  # type: ignore[union-attr]
                        task.title, task.tool_hint or "unknown",
                        report.improvement_suggestions[0],
                    )
                except Exception as exc:
                    log.debug("PlanReflection: failure_memory persist failed (%s)", exc)
