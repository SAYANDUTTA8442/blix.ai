"""
Agent Executor — Blix v0.3.5  (Module 3)

The closed cognitive loop:

    while not finished:
        think()           — select next task + choose tool
        execute()         — run the tool
        observe()         — interpret the result
        reflect()         — evaluate + decide (accept/retry/skip)
        update_memory()   — persist important results
        tick()            — advance working memory

This is the core agentic architecture. The Executor coordinates:
    * ``TaskGraph``       — which tasks exist and their dependencies
    * ``ToolRegistry``    — which tool to use for each task
    * ``WorkingMemory``   — short-term state between tasks
    * ``ObservationLayer`` — structured result interpretation
    * ``ReflectionLoop``  — quality evaluation and improvement
    * ``MilestoneTracker`` — sync to GoalTracker

Python 3.10 compatible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from agents.observation import ObservationLayer
from agents.plan_reflection import PlanReflection, PlanReflectionReport
from agents.reflection_loop import ReflectionDecision, ReflectionLoop
from agents.state import AgentState, ToolReliabilityStats as _ToolReliabilityStats
from agents.types import ExecutionHistoryEntry, ExecutionStatus, Task, TaskGraph, TaskStatus
from agents.working_memory import WorkingMemory
from planning.critic import CritiqueReport, PlanCritic, PlanVerdict
from planning.replanner import Replanner
from tools.registry import ToolRegistry
from verification.verifier import VerificationEngine, VerificationReport
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Execution run result
# ---------------------------------------------------------------------------


@dataclass
class AgentRunResult:
    """
    Summary of one complete agent execution run.

    Returned by ``AgentExecutor.run()``.
    """

    goal: str
    graph: TaskGraph
    completed_tasks: int = 0
    failed_tasks: int = 0
    skipped_tasks: int = 0
    total_steps: int = 0
    success: bool = False
    final_output: str = ""
    history: list[dict] = field(default_factory=list)
    duration_secs: float = 0.0
    # v0.3.6 additions
    replan_count: int = 0
    critique: Optional[dict] = None
    plan_reflection: Optional[dict] = None
    agent_state: Optional[dict] = None
    aborted_by_critic: bool = False

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "graph_id": self.graph.graph_id,
            "progress": self.graph.progress,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "skipped_tasks": self.skipped_tasks,
            "total_steps": self.total_steps,
            "success": self.success,
            "final_output": self.final_output[:500],
            "duration_secs": round(self.duration_secs, 2),
            "task_summary": self.graph.status_summary(),
            "replan_count": self.replan_count,
            "critique": self.critique,
            "plan_reflection": self.plan_reflection,
            "agent_state": self.agent_state,
            "aborted_by_critic": self.aborted_by_critic,
        }


# ---------------------------------------------------------------------------
# Executor configuration
# ---------------------------------------------------------------------------


@dataclass
class ExecutorConfig:
    """Tunable parameters for the AgentExecutor."""

    max_steps: int = 50             # hard limit on total steps per run
    max_task_retries: int = 2       # max retries per task (also in ReflectionLoop)
    step_delay_secs: float = 0.0    # optional throttle between steps
    require_tool_confirmation: bool = False  # prompt human for dangerous tools
    verbose: bool = False           # log step-by-step progress
    # v0.3.6 additions
    abort_on_critic_rejection: bool = True   # stop before executing if PlanCritic rejects
    enable_verification: bool = True          # run VerificationEngine before reflection
    enable_replanning: bool = True            # call Replanner on skip decisions


# ---------------------------------------------------------------------------
# Agent Executor
# ---------------------------------------------------------------------------


class AgentExecutor:
    """
    Orchestrates the full agent execution loop for a ``TaskGraph``.

    Parameters
    ----------
    tool_registry:
        ``ToolRegistry`` with all available tools.
    working_memory:
        ``WorkingMemory`` instance for this execution.
    observation_layer:
        ``ObservationLayer`` for result interpretation.
    reflection_loop:
        ``ReflectionLoop`` for quality evaluation.
    milestone_tracker:
        Optional ``MilestoneTracker`` — syncs to GoalTracker.
    config:
        ``ExecutorConfig`` tuning parameters.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        working_memory: WorkingMemory,
        observation_layer: ObservationLayer,
        reflection_loop: ReflectionLoop,
        milestone_tracker: Optional[object] = None,
        config: Optional[ExecutorConfig] = None,
        # v0.3.6 additions — all optional, default to None (v0.3.5 behavior)
        plan_critic: Optional[PlanCritic] = None,
        verification_engine: Optional[VerificationEngine] = None,
        replanner: Optional[Replanner] = None,
        plan_reflection: Optional[PlanReflection] = None,
    ) -> None:
        self._registry = tool_registry
        self._wm = working_memory
        self._obs = observation_layer
        self._reflect = reflection_loop
        self._tracker = milestone_tracker
        self._cfg = config or ExecutorConfig()
        self._critic = plan_critic
        self._verifier = verification_engine
        self._replanner = replanner
        self._plan_reflection = plan_reflection

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        graph: TaskGraph,
        goal_id: Optional[str] = None,
    ) -> AgentRunResult:
        """
        Execute the full ``TaskGraph``.

        Parameters
        ----------
        graph:
            The planned ``TaskGraph`` to execute.
        goal_id:
            Optional GoalTracker goal_id for milestone tracking.

        Returns
        -------
        ``AgentRunResult`` with full run summary.
        """
        t_start = time.monotonic()
        result = AgentRunResult(goal=graph.goal, graph=graph)
        step = 0

        log.info("AgentExecutor: starting run for goal='%s' (%d tasks)", graph.goal[:60], len(graph.tasks))

        # ── v0.3.6: AgentState — unified cognitive state for this run ──
        state = AgentState(goal=graph.goal)
        state.set_plan(graph, is_replan=False)

        # ── v0.3.6: PlanCritic — think before acting ────────────────────
        if self._critic is not None:
            critique = self._critic.critique(graph)
            result.critique = critique.to_dict()
            if critique.verdict == PlanVerdict.REJECTED and self._cfg.abort_on_critic_rejection:
                log.warning(
                    "AgentExecutor: plan rejected by PlanCritic (%d critical issue(s)); aborting run.",
                    sum(1 for i in critique.issues if i.severity.value == "critical"),
                )
                result.total_steps = 0
                result.success = False
                result.aborted_by_critic = True
                result.duration_secs = time.monotonic() - t_start
                result.final_output = "Run aborted: plan rejected by PlanCritic.\n" + critique.to_dict().__repr__()
                result.agent_state = state.to_dict()
                return result

        while not graph.is_complete and step < self._cfg.max_steps:
            step += 1

            # ── THINK: choose next task ──────────────────────────────
            task = graph.next_task()
            if task is None:
                if graph.has_failures:
                    log.warning("AgentExecutor: no ready tasks but graph has failures. Stopping.")
                    break
                # All tasks done
                break

            self._wm.set_current_task(task.task_id, task.title)
            task.status = TaskStatus.IN_PROGRESS
            task.attempts += 1

            if self._cfg.verbose:
                log.info("AgentExecutor [step %d]: task='%s' tool_hint=%s",
                         step, task.title, task.tool_hint)

            # ── THINK: select tool ───────────────────────────────────
            tool = self._registry.select_tool(task)
            if tool is None:
                log.warning("AgentExecutor: no tool found for task '%s'; skipping.", task.title)
                task.status = TaskStatus.SKIPPED
                result.skipped_tasks += 1
                self._wm.tick()
                continue

            # ── EXECUTE ──────────────────────────────────────────────
            context = self._wm.snapshot()
            if self._cfg.step_delay_secs > 0:
                time.sleep(self._cfg.step_delay_secs)

            exec_result = tool.execute(task, context)

            # ── OBSERVE ──────────────────────────────────────────────
            observation = self._obs.observe(exec_result)
            state.record_observation(observation)
            state.cost.record_call(
                duration_secs=exec_result.duration_ms / 1000.0,
                tokens=exec_result.tokens_used,
            )

            # ── VERIFY (v0.3.6) ──────────────────────────────────────
            verification_report: Optional[VerificationReport] = None
            if self._verifier is not None and self._cfg.enable_verification:
                verification_report = self._verifier.verify(task, exec_result)
                if not verification_report.passed and observation.success:
                    # Output looked fine to the Observation layer but failed
                    # structural verification — downgrade to force a retry.
                    observation.success = False
                    observation.quality_score = min(observation.quality_score, 0.2)
                    observation.retry_suggested = True
                    if not observation.retry_hint:
                        observation.retry_hint = verification_report.summary()

            # ── REFLECT ──────────────────────────────────────────────
            decision = self._reflect.reflect(task, observation, goal=graph.goal)

            # Persist result into working memory
            if observation.success:
                self._wm.set_task_output(task.task_id, exec_result.output)
                # Also store as a named key for downstream tasks
                self._wm.set(
                    f"task_{task.task_id}_facts",
                    observation.extracted_facts,
                    task_id=task.task_id,
                )

            # ── UPDATE STATE ─────────────────────────────────────────
            if decision.should_retry():
                # Inject retry hint into task metadata and re-queue
                task.status = TaskStatus.PENDING
                task.metadata["retry_hint"] = decision.retry_hint
                if task.tool_hint is None and observation.tool_name:
                    task.metadata["last_tool"] = observation.tool_name
                log.info("AgentExecutor: retrying task '%s' (hint: %s)",
                         task.title, decision.retry_hint[:80])
                state.cost.record_call(is_retry=True)

            elif decision.should_skip():
                # v0.3.6: try to replan instead of treating this as terminal.
                task.mark_failed(f"Failed after {task.attempts} attempt(s): {decision.note}")
                replanned = False
                if (
                    self._replanner is not None
                    and self._cfg.enable_replanning
                    and self._replanner.should_replan(task, graph)
                ):
                    replan_result = self._replanner.replan(task, graph, failure_reason=task.error)
                    state.set_plan(graph, is_replan=True)
                    result.replan_count += 1
                    decision.note += f" | Replanned: {replan_result.explanation}"
                    if task.status != TaskStatus.SKIPPED:
                        # SWITCH_TOOL or DECOMPOSE — task (or its replacement) is runnable again
                        replanned = True
                        state.record_failure(task.task_id, {
                            "task": task.title, "tool": observation.tool_name,
                            "failure": task.error, "strategy": replan_result.strategy.value,
                        })
                    else:
                        # DROP_TASK — permanently gone
                        result.failed_tasks += 1
                        state.record_failure(task.task_id, {
                            "task": task.title, "tool": observation.tool_name, "failure": task.error,
                        })
                if not replanned and task.status != TaskStatus.SKIPPED:
                    result.failed_tasks += 1
                    state.record_failure(task.task_id, {
                        "task": task.title, "tool": observation.tool_name, "failure": task.error,
                    })
                log.info("AgentExecutor: task '%s' failed (replanned=%s)", task.title, replanned)

            else:
                # Accept
                task.mark_completed(exec_result.output[:500])
                result.completed_tasks += 1
                state.record_completion(task.task_id)
                log.info("AgentExecutor: completed task '%s' (quality=%.2f)",
                         task.title, decision.quality_score)

            tool_name_for_state = exec_result.tool_name
            rel_stats = state.tool_reliability.setdefault(
                tool_name_for_state, _ToolReliabilityStats(tool_name_for_state)
            )
            rel_stats.record(observation.success)
            state.update_confidence(self._estimate_confidence(graph, state))

            # Append to run history
            result.history.append({
                "step": step,
                "task_id": task.task_id,
                "task_title": task.title,
                "tool": exec_result.tool_name,
                "decision": decision.action,
                "quality": round(decision.quality_score, 3),
                "note": decision.note[:120],
            })

            # Sync milestones
            if self._tracker is not None and goal_id is not None:
                try:
                    self._tracker.sync(goal_id, graph)  # type: ignore[union-attr]
                except Exception as exc:
                    log.debug("MilestoneTracker sync failed: %s", exc)

            self._wm.tick()

        # ── FINALISE ─────────────────────────────────────────────────
        result.total_steps = step
        result.success = not graph.has_failures and graph.is_complete
        result.duration_secs = time.monotonic() - t_start
        result.final_output = self._build_final_output(graph)
        state.cost.execution_time_secs = result.duration_secs

        # ── v0.3.6: PlanReflection — reflect on the whole plan ──────────
        if self._plan_reflection is not None:
            plan_report = self._plan_reflection.reflect(
                graph, result.history, replan_count=result.replan_count,
            )
            result.plan_reflection = plan_report.to_dict()

        result.agent_state = state.to_dict()

        log.info(
            "AgentExecutor: run complete — success=%s completed=%d failed=%d "
            "skipped=%d steps=%d duration=%.1fs replans=%d",
            result.success, result.completed_tasks, result.failed_tasks,
            result.skipped_tasks, result.total_steps, result.duration_secs,
            result.replan_count,
        )
        return result

    def _estimate_confidence(self, graph: TaskGraph, state: AgentState) -> float:
        """
        Heuristic running confidence: blends plan progress with the
        mean observed quality of recent observations.
        """
        progress_frac = graph.progress / 100.0
        recent = state.recent_observations(5)
        if recent:
            mean_quality = sum(o.quality_score for o in recent) / len(recent)
        else:
            mean_quality = 0.5
        return round(0.6 * progress_frac + 0.4 * mean_quality, 3)

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    def _build_final_output(self, graph: TaskGraph) -> str:
        """Assemble final output from all completed task results."""
        parts: list[str] = [f"Goal: {graph.goal}", ""]
        for task in graph.tasks:
            if task.status == TaskStatus.COMPLETED and task.result:
                parts.append(f"## {task.title}")
                parts.append(task.result[:400])
                parts.append("")
        if not any(t.status == TaskStatus.COMPLETED for t in graph.tasks):
            parts.append("No tasks were completed successfully.")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# AgentSession — combines Planner + Executor for a single goal execution
# ---------------------------------------------------------------------------


class AgentSession:
    """
    High-level facade for running an agent on a natural-language goal.

    Combines:
        Planner → TaskGraph
        AgentExecutor → AgentRunResult

    Parameters
    ----------
    planner:
        ``Planner`` instance from ``planning.planner``.
    executor:
        ``AgentExecutor`` instance.
    goal_tracker:
        Optional v0.3.2 ``GoalTracker`` for milestone tracking.
    """

    def __init__(
        self,
        planner: object,
        executor: AgentExecutor,
        goal_tracker: Optional[object] = None,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._gt = goal_tracker
        self._sessions: list[AgentRunResult] = []

    def run(self, goal_text: str) -> AgentRunResult:
        """
        Plan and execute a natural-language goal end-to-end.

        1. Parse + decompose into TaskGraph
        2. Create GoalTracker entry (if available)
        3. Execute the TaskGraph
        4. Return AgentRunResult
        """
        parsed_goal, graph = self._planner.plan(goal_text)  # type: ignore[union-attr]

        goal_id: Optional[str] = None
        if self._gt is not None:
            try:
                from planning.planner import MilestoneTracker
                tracker = MilestoneTracker(self._gt)
                goal_id = tracker.create_goal_from_graph(graph)
            except Exception as exc:
                log.debug("AgentSession: GoalTracker integration failed (%s)", exc)

        result = self._executor.run(graph, goal_id=goal_id)
        self._sessions.append(result)
        return result

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    def recent_sessions(self, n: int = 5) -> list[AgentRunResult]:
        return self._sessions[-n:]
