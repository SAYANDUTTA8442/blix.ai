"""
Agent Benchmark Suite — Blix v0.3.6  (Upgrade 9)

Without benchmarks, agent progress is unknown. ``AdaptiveAgentEvaluator``
extends ``AgentEvaluator`` (v0.3.5) with the metrics needed to measure
whether v0.3.6's adaptive loop is actually working:

    Task Success Rate       — already in AgentEvaluator (v0.3.5)
    Verification Accuracy   — NEW — did VerificationEngine correctly
                                gate bad results?
    Replanning Success      — NEW — when the Replanner intervened, did
                                the run go on to succeed?
    Recovery Rate           — NEW — fraction of FAILED tasks (graph-level)
                                that were ultimately worked around (via
                                replan or otherwise) vs. left as
                                permanent failures
    Execution Cost           — already in AgentEvaluator (v0.3.5)
    Tool Efficiency           — NEW — mean reliability-weighted cost per
                                successful task

This completes the full evaluation tower:
    MemoryEvaluator → ExtendedMemoryEvaluator → CognitiveEvaluator
        → ReasoningEvaluator → AgentEvaluator → AdaptiveAgentEvaluator

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agents.executor import AgentRunResult
from agents.types import TaskGraph, TaskStatus
from evaluation.agent_eval import AgentEvalCase, AgentEvaluator
from verification.verifier import VerificationReport


# ---------------------------------------------------------------------------
# Benchmark case
# ---------------------------------------------------------------------------


@dataclass
class AgentBenchmarkCase(AgentEvalCase):
    """
    Extends ``AgentEvalCase`` with v0.3.6 adaptive-loop expectations.

    Fields
    ------
    expect_verification_failures:
        Number of tasks in this case expected to fail verification at
        least once (i.e. the benchmark deliberately includes a flawed
        output to confirm the Verifier catches it).
    expect_replan:
        Whether this case is expected to require at least one replan
        to succeed.
    """

    expect_verification_failures: int = 0
    expect_replan: bool = False


# ---------------------------------------------------------------------------
# Adaptive Agent Evaluator
# ---------------------------------------------------------------------------


class AdaptiveAgentEvaluator(AgentEvaluator):
    """
    Extends ``AgentEvaluator`` with v0.3.6 adaptive-loop metrics.
    """

    # ------------------------------------------------------------------
    # Verification Accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def verification_accuracy(
        verification_reports: list[VerificationReport],
        expected_pass: list[bool],
    ) -> float:
        """
        Fraction of verification reports whose pass/fail outcome matched
        the expected ground truth.

        ``expected_pass[i]`` corresponds to ``verification_reports[i]``.
        """
        if not verification_reports or len(verification_reports) != len(expected_pass):
            return 0.0
        correct = sum(
            1 for report, expected in zip(verification_reports, expected_pass)
            if report.passed == expected
        )
        return correct / len(verification_reports)

    @staticmethod
    def verification_catch_rate(results: list[AgentRunResult]) -> float:
        """
        Fraction of runs where at least one task's history shows a
        verification-triggered retry (i.e. the verifier caught something
        the Observation layer alone would have accepted).

        Approximated via history notes mentioning "Verif" since
        ``AgentRunResult.history`` stores compact dicts, not full
        ``VerificationReport`` objects.
        """
        if not results:
            return 0.0
        caught = sum(
            1 for r in results
            if any("verif" in h.get("note", "").lower() for h in r.history)
        )
        return caught / len(results)

    # ------------------------------------------------------------------
    # Replanning Success
    # ------------------------------------------------------------------

    @staticmethod
    def replanning_success_rate(results: list[AgentRunResult]) -> float:
        """
        Of all runs that triggered at least one replan, what fraction
        ultimately succeeded?
        """
        replanned = [r for r in results if r.replan_count > 0]
        if not replanned:
            return 1.0  # no replans needed — vacuously perfect
        return sum(1 for r in replanned if r.success) / len(replanned)

    @staticmethod
    def mean_replans_per_run(results: list[AgentRunResult]) -> float:
        if not results:
            return 0.0
        return sum(r.replan_count for r in results) / len(results)

    # ------------------------------------------------------------------
    # Recovery Rate
    # ------------------------------------------------------------------

    @staticmethod
    def recovery_rate(graph: TaskGraph) -> float:
        """
        Fraction of tasks that experienced at least one failure
        (tracked via ``task.metadata['replan_count']`` or attempts > 1)
        but ended up COMPLETED rather than permanently FAILED/SKIPPED.

        Distinct from ``replanning_success_rate``: this measures
        task-level recovery within a single graph, not run-level outcome.
        """
        struggled = [
            t for t in graph.tasks
            if t.attempts > 1 or t.metadata.get("replan_count", 0) > 0
        ]
        if not struggled:
            return 1.0  # nothing struggled — vacuously perfect
        recovered = sum(1 for t in struggled if t.status == TaskStatus.COMPLETED)
        return recovered / len(struggled)

    @staticmethod
    def batch_recovery_rate(results: list[AgentRunResult]) -> float:
        rates = [AdaptiveAgentEvaluator.recovery_rate(r.graph) for r in results]
        return sum(rates) / len(rates) if rates else 1.0

    # ------------------------------------------------------------------
    # Tool Efficiency
    # ------------------------------------------------------------------

    @staticmethod
    def tool_efficiency(result: AgentRunResult) -> float:
        """
        Successful tasks per tool call — penalises runs that needed many
        retries/replans to get the same amount of completed work done.

        Returns 0 if no tool calls were made.
        """
        cost = AdaptiveAgentEvaluator.execution_cost(result.history)
        if cost["tool_calls"] == 0:
            return 0.0
        return round(result.completed_tasks / cost["tool_calls"], 4)

    @staticmethod
    def mean_tool_efficiency(results: list[AgentRunResult]) -> float:
        if not results:
            return 0.0
        scores = [AdaptiveAgentEvaluator.tool_efficiency(r) for r in results]
        return round(sum(scores) / len(scores), 4)

    # ------------------------------------------------------------------
    # Combined benchmark pass
    # ------------------------------------------------------------------

    def benchmark_run(
        self,
        result: AgentRunResult,
        case: Optional[AgentBenchmarkCase] = None,
        actual_domain: str = "",
    ) -> dict[str, float]:
        """Full v0.3.6 benchmark metrics for one run, building on v0.3.5's evaluate_agent_run."""
        metrics = self.evaluate_agent_run(result, case=case, actual_domain=actual_domain)
        metrics["recovery_rate"] = self.recovery_rate(result.graph)
        metrics["tool_efficiency"] = self.tool_efficiency(result)
        metrics["replan_count"] = float(result.replan_count)
        return metrics

    def benchmark_batch(self, results: list[AgentRunResult]) -> dict[str, float]:
        """Aggregate v0.3.6 benchmark metrics across multiple runs."""
        if not results:
            return {}
        batch = self.evaluate_agent_batch(results)
        batch["replanning_success_rate"] = self.replanning_success_rate(results)
        batch["mean_replans_per_run"] = self.mean_replans_per_run(results)
        batch["batch_recovery_rate"] = self.batch_recovery_rate(results)
        batch["mean_tool_efficiency"] = self.mean_tool_efficiency(results)
        batch["verification_catch_rate"] = self.verification_catch_rate(results)
        return batch
