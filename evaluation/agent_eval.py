"""
Agent Evaluation — Blix v0.3.5  (Module 10)

Measures agent execution quality across the full closed cognitive loop:

    Task Success Rate    — fraction of tasks completed successfully
    Tool Accuracy        — did the Tool Selection Engine pick correctly?
    Planning Accuracy    — did the planned TaskGraph match expectations?
    Execution Cost       — total tokens/tool-calls consumed
    Execution Time       — wall-clock duration
    Reflection Gain       — quality improvement from retry → final result

Extends ``ReasoningEvaluator`` (v0.3.4) so the full evaluation tower
(memory → knowledge → graph → reasoning → agent) is available from a
single ``AgentEvaluator`` instance, and is re-exported via ``blix_eval``.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agents.executor import AgentRunResult
from agents.types import Task, TaskGraph, TaskStatus
from evaluation.reasoning import ReasoningEvaluator


# ---------------------------------------------------------------------------
# Agent evaluation case
# ---------------------------------------------------------------------------


@dataclass
class AgentEvalCase:
    """
    One evaluation instance for agent execution.

    Fields
    ------
    case_id:
        Unique identifier.
    goal:
        Natural-language goal given to the agent.
    expected_tool_for_task:
        Mapping of task title → expected tool name (for tool accuracy).
    expected_task_count:
        Expected number of planned tasks (for planning accuracy), or
        a (min, max) range.
    expected_domain:
        Expected domain classification from GoalParser.
    """

    case_id: str = ""
    goal: str = ""
    expected_tool_for_task: dict = field(default_factory=dict)
    expected_task_count: Optional[tuple] = None
    expected_domain: str = ""


# ---------------------------------------------------------------------------
# Agent Evaluator
# ---------------------------------------------------------------------------


class AgentEvaluator(ReasoningEvaluator):
    """
    Extends ``ReasoningEvaluator`` with v0.3.5 agent execution metrics.
    """

    # ------------------------------------------------------------------
    # Task Success Rate
    # ------------------------------------------------------------------

    @staticmethod
    def task_success_rate(graph: TaskGraph) -> float:
        """Fraction of tasks in COMPLETED state (out of all non-pending tasks)."""
        terminal = [t for t in graph.tasks if t.status != TaskStatus.PENDING]
        if not terminal:
            return 0.0
        completed = sum(1 for t in terminal if t.status == TaskStatus.COMPLETED)
        return completed / len(terminal)

    @staticmethod
    def run_success_rate(results: list[AgentRunResult]) -> float:
        """Fraction of agent runs that completed successfully (no failures, fully done)."""
        if not results:
            return 0.0
        return sum(1 for r in results if r.success) / len(results)

    # ------------------------------------------------------------------
    # Tool Accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def tool_accuracy(
        graph: TaskGraph,
        expected_tool_for_task: dict,
    ) -> float:
        """
        Fraction of tasks where the tool actually used (recorded in
        ``task.metadata['last_tool']`` or inferred from ``tool_hint``)
        matches the expected tool.

        ``expected_tool_for_task``: dict mapping task title → expected tool name.
        """
        if not expected_tool_for_task:
            return 1.0
        total = 0
        correct = 0
        for task in graph.tasks:
            if task.title not in expected_tool_for_task:
                continue
            total += 1
            expected = expected_tool_for_task[task.title]
            actual = task.metadata.get("last_tool") or task.tool_hint
            if actual == expected:
                correct += 1
        return correct / total if total else 1.0

    # ------------------------------------------------------------------
    # Planning Accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def planning_accuracy(
        graph: TaskGraph,
        expected_task_count: Optional[tuple] = None,
        expected_domain: str = "",
        actual_domain: str = "",
    ) -> float:
        """
        How well did the plan match expectations?

        Combines:
        - task count within expected (min, max) range → 0.6 weight
        - domain classification match → 0.4 weight

        If only one signal is provided, it gets full weight.
        """
        scores: list[tuple[float, float]] = []

        if expected_task_count is not None:
            lo, hi = expected_task_count
            in_range = lo <= len(graph.tasks) <= hi
            scores.append((1.0 if in_range else 0.0, 0.6))

        if expected_domain:
            match = 1.0 if expected_domain.lower() == actual_domain.lower() else 0.0
            scores.append((match, 0.4))

        if not scores:
            return 1.0

        total_weight = sum(w for _, w in scores)
        return round(sum(s * w for s, w in scores) / total_weight, 4)

    # ------------------------------------------------------------------
    # Execution Cost / Time
    # ------------------------------------------------------------------

    @staticmethod
    def execution_cost(history: list[dict]) -> dict[str, float]:
        """
        Aggregate cost metrics from an ``AgentRunResult.history`` list.

        Returns
        -------
        dict with: total_steps, tool_calls, retries
        """
        total_steps = len(history)
        retries = sum(1 for h in history if h.get("decision") == "retry")
        tool_calls = sum(1 for h in history if h.get("tool"))
        return {
            "total_steps": total_steps,
            "tool_calls": tool_calls,
            "retries": retries,
        }

    @staticmethod
    def execution_time(result: AgentRunResult) -> float:
        """Wall-clock duration in seconds for one agent run."""
        return result.duration_secs

    @staticmethod
    def mean_execution_time(results: list[AgentRunResult]) -> float:
        if not results:
            return 0.0
        return sum(r.duration_secs for r in results) / len(results)

    # ------------------------------------------------------------------
    # Reflection Gain
    # ------------------------------------------------------------------

    @staticmethod
    def reflection_gain(history: list[dict]) -> float:
        """
        Quality improvement attributable to retries.

        For tasks that were retried, compares the quality score of the
        first attempt vs. the final accepted attempt. Returns the mean
        improvement (can be negative if retries made things worse).

        Returns 0.0 if no retries occurred.
        """
        by_task: dict[str, list[float]] = {}
        for h in history:
            tid = h.get("task_id")
            if tid is None:
                continue
            by_task.setdefault(tid, []).append(h.get("quality", 0.0))

        gains: list[float] = []
        for qualities in by_task.values():
            if len(qualities) >= 2:
                gains.append(qualities[-1] - qualities[0])

        if not gains:
            return 0.0
        return round(sum(gains) / len(gains), 4)

    @staticmethod
    def retry_effectiveness(history: list[dict]) -> float:
        """
        Fraction of retried tasks that eventually succeeded (decision == 'accept'
        on a later attempt after at least one 'retry').
        """
        by_task: dict[str, list[str]] = {}
        for h in history:
            tid = h.get("task_id")
            if tid is None:
                continue
            by_task.setdefault(tid, []).append(h.get("decision", ""))

        retried_tasks = [decisions for decisions in by_task.values() if "retry" in decisions]
        if not retried_tasks:
            return 1.0  # no retries needed — vacuously perfect
        succeeded = sum(1 for decisions in retried_tasks if decisions[-1] == "accept")
        return succeeded / len(retried_tasks)

    # ------------------------------------------------------------------
    # Combined agent evaluation pass
    # ------------------------------------------------------------------

    def evaluate_agent_run(
        self,
        result: AgentRunResult,
        case: Optional[AgentEvalCase] = None,
        actual_domain: str = "",
    ) -> dict[str, float]:
        """
        Run all applicable agent metrics for a single ``AgentRunResult``.

        Returns a summary dict.
        """
        metrics: dict[str, float] = {
            "task_success_rate": self.task_success_rate(result.graph),
            "execution_time_secs": self.execution_time(result),
        }
        cost = self.execution_cost(result.history)
        metrics.update({f"cost_{k}": float(v) for k, v in cost.items()})
        metrics["reflection_gain"] = self.reflection_gain(result.history)
        metrics["retry_effectiveness"] = self.retry_effectiveness(result.history)

        if case is not None:
            if case.expected_tool_for_task:
                metrics["tool_accuracy"] = self.tool_accuracy(
                    result.graph, case.expected_tool_for_task
                )
            if case.expected_task_count or case.expected_domain:
                metrics["planning_accuracy"] = self.planning_accuracy(
                    result.graph,
                    expected_task_count=case.expected_task_count,
                    expected_domain=case.expected_domain,
                    actual_domain=actual_domain,
                )

        return metrics

    def evaluate_agent_batch(
        self,
        results: list[AgentRunResult],
    ) -> dict[str, float]:
        """Aggregate metrics across multiple agent runs."""
        if not results:
            return {}
        return {
            "run_success_rate": self.run_success_rate(results),
            "mean_execution_time_secs": self.mean_execution_time(results),
            "mean_task_success_rate": sum(
                self.task_success_rate(r.graph) for r in results
            ) / len(results),
        }
