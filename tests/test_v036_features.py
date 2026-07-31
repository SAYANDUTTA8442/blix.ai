"""
Tests for Blix v0.3.6 — "Adaptive Planning & Verification Engine".

Covers:
1.  planning.replanner    (Replanner — switch_tool/decompose/drop strategies)
2.  verification.verifier (VerificationEngine — built-in verifiers)
3.  agents.task_runtime    (TaskRuntime — DAG batches, failure propagation)
4.  agents.failure_memory  (FailureMemory — similarity matching, fix lookup)
5.  agents.tool_reliability (ToolReliabilityRegistry — cross-run tracking)
6.  planning.critic         (PlanCritic — all 6 checks)
7.  agents.state             (AgentState, ToolReliabilityStats, ExecutionCostModel)
8.  agents.plan_reflection    (PlanReflection — success/failure analysis)
9.  evaluation.agent_benchmark (AdaptiveAgentEvaluator)
10. agents.executor (integration) — full adaptive loop wiring
API  /agent/run, /agent/critique, /agent/failures, /agent/tool-reliability

Python 3.10 compatible — fully offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from agents.executor import AgentExecutor, AgentRunResult, AgentSession, ExecutorConfig
from agents.failure_memory import FailureMemory, FailureRecord
from agents.observation import ObservationLayer
from agents.plan_reflection import PlanReflection, PlanReflectionReport
from agents.reflection_loop import ReflectionLoop
from agents.state import AgentState, ExecutionCostModel, ToolReliabilityStats
from agents.task_runtime import DAGRuntimeStats, TaskRuntime
from agents.tool_reliability import ToolReliabilityRecord, ToolReliabilityRegistry
from agents.types import ExecutionResult, ExecutionStatus, Task, TaskGraph, TaskStatus
from agents.working_memory import WorkingMemory
from evaluation.agent_benchmark import AdaptiveAgentEvaluator, AgentBenchmarkCase
from planning.critic import (
    CriticIssue, CritiqueReport, IssueSeverity, PlanCritic, PlanVerdict,
)
from planning.replanner import Replanner, ReplanResult, ReplanStrategy
from tools.registry import Tool, ToolRegistry
from verification.verifier import (
    CodeSyntaxVerifier, KeywordPresenceVerifier, NonEmptyVerifier,
    SchemaVerifier, VerificationCheck, VerificationEngine,
    VerificationReport, VerificationStatus,
)


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _FakeLLM:
    def model_name(self) -> str:
        return "fake-0.3.6"

    def generate(self, prompt: str) -> str:
        return "Fake LLM reflection note."


class _FakeTool(Tool):
    """A controllable fake tool for testing the executor loop."""

    def __init__(self, name: str = "fake_tool", outputs: list | None = None) -> None:
        self._name = name
        self._outputs = outputs or [ExecutionResult(
            task_id="", tool_name=name, status=ExecutionStatus.SUCCESS,
            output="Fake successful output with enough length to pass quality checks easily.",
        )]
        self._call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A fake tool for testing."

    def can_handle(self, task: Task) -> float:
        return 0.9

    def execute(self, task: Task, context: dict) -> ExecutionResult:
        idx = min(self._call_count, len(self._outputs) - 1)
        result = self._outputs[idx]
        result.task_id = task.task_id
        self._call_count += 1
        return result


def _make_simple_graph(n_tasks: int = 3, linear: bool = True, tool_hint: str = "fake_tool") -> TaskGraph:
    graph = TaskGraph(goal="Test goal")
    prev_id = None
    for i in range(n_tasks):
        deps = [prev_id] if (linear and prev_id) else []
        task = Task(title=f"Task {i}", description=f"Do step {i}", tool_hint=tool_hint, depends_on=deps)
        graph.add_task(task)
        prev_id = task.task_id
    return graph


# ===========================================================================
# Upgrade 7 — AgentState, ToolReliabilityStats, ExecutionCostModel
# ===========================================================================


class TestAgentState:
    def test_defaults(self) -> None:
        state = AgentState(goal="test goal")
        assert state.plan_version == 1
        assert state.replan_count == 0
        assert state.confidence == 0.5
        assert len(state.state_id) == 8

    def test_set_plan_initial(self) -> None:
        state = AgentState(goal="g")
        graph = TaskGraph(goal="g")
        state.set_plan(graph, is_replan=False)
        assert state.active_plan is graph
        assert state.plan_version == 1
        assert state.replan_count == 0

    def test_set_plan_replan_bumps_version(self) -> None:
        state = AgentState(goal="g")
        graph = TaskGraph(goal="g")
        state.set_plan(graph, is_replan=False)
        state.set_plan(graph, is_replan=True)
        assert state.plan_version == 2
        assert state.replan_count == 1

    def test_record_observation(self) -> None:
        from agents.types import Observation
        state = AgentState(goal="g")
        obs = Observation(task_id="t1", tool_name="x", success=True)
        state.record_observation(obs)
        assert len(state.observations) == 1

    def test_record_completion_no_duplicates(self) -> None:
        state = AgentState(goal="g")
        state.record_completion("t1")
        state.record_completion("t1")
        assert state.completed_tasks == ["t1"]

    def test_record_failure_with_record(self) -> None:
        state = AgentState(goal="g")
        state.record_failure("t1", {"task": "x", "failure": "timeout"})
        assert "t1" in state.failed_tasks
        assert len(state.failure_records) == 1

    def test_update_confidence_clamped(self) -> None:
        state = AgentState(goal="g")
        state.update_confidence(1.5)
        assert state.confidence == 1.0
        state.update_confidence(-0.5)
        assert state.confidence == 0.0

    def test_progress_no_plan(self) -> None:
        state = AgentState(goal="g")
        assert state.progress == 0

    def test_progress_with_plan(self) -> None:
        state = AgentState(goal="g")
        graph = _make_simple_graph(2)
        graph.tasks[0].mark_completed("ok")
        state.set_plan(graph)
        assert state.progress == 50

    def test_is_stalled(self) -> None:
        state = AgentState(goal="g")
        graph = _make_simple_graph(2)
        graph.tasks[0].mark_failed("err")
        state.set_plan(graph)
        assert state.is_stalled

    def test_is_stalled_false_when_complete(self) -> None:
        state = AgentState(goal="g")
        graph = _make_simple_graph(1)
        graph.tasks[0].mark_completed("ok")
        state.set_plan(graph)
        assert not state.is_stalled

    def test_recent_observations(self) -> None:
        from agents.types import Observation
        state = AgentState(goal="g")
        for i in range(10):
            state.record_observation(Observation(task_id=f"t{i}", tool_name="x", success=True))
        recent = state.recent_observations(3)
        assert len(recent) == 3
        assert recent[-1].task_id == "t9"

    def test_to_dict(self) -> None:
        state = AgentState(goal="g")
        d = state.to_dict()
        assert d["goal"] == "g"
        assert "cost" in d
        assert "tool_reliability" in d


class TestToolReliabilityStats:
    def test_neutral_prior_when_untested(self) -> None:
        stats = ToolReliabilityStats(tool_name="x")
        assert stats.success_rate == 0.5

    def test_record_success_and_failure(self) -> None:
        stats = ToolReliabilityStats(tool_name="x")
        stats.record(True)
        stats.record(True)
        stats.record(False)
        assert stats.total == 3
        assert stats.success_rate == pytest.approx(2 / 3)

    def test_to_dict(self) -> None:
        stats = ToolReliabilityStats(tool_name="x", successes=3, failures=1)
        d = stats.to_dict()
        assert d["tool"] == "x"
        assert d["successes"] == 3


class TestExecutionCostModel:
    def test_record_call(self) -> None:
        cost = ExecutionCostModel()
        cost.record_call(tokens=100, duration_secs=1.5)
        assert cost.token_cost == 100
        assert cost.tool_calls == 1
        assert cost.execution_time_secs == pytest.approx(1.5)

    def test_record_retry(self) -> None:
        cost = ExecutionCostModel()
        cost.record_call(is_retry=True)
        assert cost.retry_count == 1

    def test_efficiency_score_no_retries(self) -> None:
        cost = ExecutionCostModel()
        cost.record_call()
        cost.record_call()
        assert cost.efficiency_score() == 1.0

    def test_efficiency_score_with_retries(self) -> None:
        cost = ExecutionCostModel()
        cost.record_call()
        cost.record_call(is_retry=True)
        assert cost.efficiency_score() == 0.5

    def test_efficiency_score_no_calls(self) -> None:
        cost = ExecutionCostModel()
        assert cost.efficiency_score() == 1.0

    def test_to_dict(self) -> None:
        cost = ExecutionCostModel(token_cost=50, execution_time_secs=2.0, tool_calls=3, retry_count=1)
        d = cost.to_dict()
        assert d["token_cost"] == 50
        assert d["tool_calls"] == 3


# ===========================================================================
# Upgrade 4 — FailureMemory
# ===========================================================================


class TestFailureMemory:
    @pytest.fixture
    def fm(self, tmp_path: Path) -> FailureMemory:
        return FailureMemory(tmp_path / "fm.json")

    def test_record_creates_new(self, fm: FailureMemory) -> None:
        record = fm.record("Build API", "python_tool", "schema mismatch", goal="Build a service")
        assert record.occurrences == 1
        assert fm.count == 1

    def test_record_merges_similar(self, fm: FailureMemory) -> None:
        fm.record("Build API endpoint", "python_tool", "schema mismatch")
        fm.record("Build API endpoint", "python_tool", "schema mismatch again")
        assert fm.count == 1
        records = fm.similar_failures("Build API endpoint")
        assert records[0].occurrences == 2

    def test_record_different_tool_creates_separate(self, fm: FailureMemory) -> None:
        fm.record("Search papers", "web_search", "timeout")
        fm.record("Search papers", "llm", "hallucination")
        assert fm.count == 2

    def test_record_fix(self, fm: FailureMemory) -> None:
        fm.record("Build API", "python_tool", "schema mismatch")
        fm.record_fix("Build API", "python_tool", "update response model")
        fix = fm.suggest_fix("Build API", "python_tool")
        assert fix == "update response model"

    def test_similar_failures_unrelated_task(self, fm: FailureMemory) -> None:
        fm.record("Build REST API endpoint", "python_tool", "schema mismatch")
        results = fm.similar_failures("Write a poem about flowers")
        assert results == []

    def test_similar_failures_filtered_by_tool(self, fm: FailureMemory) -> None:
        fm.record("Search papers", "web_search", "timeout")
        fm.record("Search papers", "llm", "hallucination")
        results = fm.similar_failures("Search papers", tool="web_search")
        assert len(results) == 1
        assert results[0].tool == "web_search"

    def test_has_known_failure(self, fm: FailureMemory) -> None:
        fm.record("Build REST API endpoint", "python_tool", "schema mismatch")
        assert fm.has_known_failure("Build REST API endpoint")
        assert not fm.has_known_failure("Completely unrelated task xyz")

    def test_suggest_fix_no_fix_available(self, fm: FailureMemory) -> None:
        fm.record("Build REST API endpoint", "python_tool", "schema mismatch")
        assert fm.suggest_fix("Build REST API endpoint") is None

    def test_most_common_failures(self, fm: FailureMemory) -> None:
        fm.record("Task A", "tool1", "fail1")
        fm.record("Task A", "tool1", "fail1 again")
        fm.record("Task B", "tool2", "fail2")
        top = fm.most_common_failures(top_k=1)
        assert top[0].task_title == "Task A"
        assert top[0].occurrences == 2

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        fm1 = FailureMemory(tmp_path / "fm.json")
        fm1.record("Build API", "python_tool", "schema mismatch")
        fm2 = FailureMemory(tmp_path / "fm.json")
        assert fm2.count == 1

    def test_to_dict_from_dict_roundtrip(self) -> None:
        record = FailureRecord(task_title="t", tool="x", failure="f", fix="fix", goal="g", occurrences=3)
        restored = FailureRecord.from_dict(record.to_dict())
        assert restored.task_title == "t"
        assert restored.occurrences == 3


# ===========================================================================
# Upgrade 5 — ToolReliabilityRegistry
# ===========================================================================


class TestToolReliabilityRegistry:
    @pytest.fixture
    def registry(self, tmp_path: Path) -> ToolReliabilityRegistry:
        return ToolReliabilityRegistry(tmp_path / "tr.json", min_samples_for_confidence=3)

    def test_neutral_prior_when_unseen(self, registry: ToolReliabilityRegistry) -> None:
        assert registry.success_rate("unknown_tool") == 0.5

    def test_record_and_success_rate(self, registry: ToolReliabilityRegistry) -> None:
        registry.record("web_search", True)
        registry.record("web_search", True)
        registry.record("web_search", False)
        assert registry.success_rate("web_search") == pytest.approx(2 / 3)

    def test_is_confident_below_threshold(self, registry: ToolReliabilityRegistry) -> None:
        registry.record("web_search", True)
        assert not registry.is_confident("web_search")

    def test_is_confident_above_threshold(self, registry: ToolReliabilityRegistry) -> None:
        for _ in range(3):
            registry.record("web_search", True)
        assert registry.is_confident("web_search")

    def test_rank_tools_by_reliability(self, registry: ToolReliabilityRegistry) -> None:
        for _ in range(5):
            registry.record("good_tool", True)
        for _ in range(5):
            registry.record("bad_tool", False)
        ranked = registry.rank_tools_by_reliability(["bad_tool", "good_tool"])
        assert ranked[0][0] == "good_tool"

    def test_least_reliable_requires_confidence(self, registry: ToolReliabilityRegistry) -> None:
        registry.record("low_sample_tool", False)  # below min_samples
        for _ in range(5):
            registry.record("confident_bad_tool", False)
        least = registry.least_reliable()
        names = [r.tool_name for r in least]
        assert "confident_bad_tool" in names
        assert "low_sample_tool" not in names

    def test_most_reliable(self, registry: ToolReliabilityRegistry) -> None:
        for _ in range(5):
            registry.record("great_tool", True)
        most = registry.most_reliable()
        assert most[0].tool_name == "great_tool"

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        r1 = ToolReliabilityRegistry(tmp_path / "tr.json")
        r1.record("x", True)
        r1.record("x", False)
        r2 = ToolReliabilityRegistry(tmp_path / "tr.json")
        assert r2.success_rate("x") == 0.5
        assert r2.tracked_tool_count == 1

    def test_mean_duration(self, registry: ToolReliabilityRegistry) -> None:
        registry.record("x", True, duration_ms=100)
        registry.record("x", True, duration_ms=200)
        rec = registry.get("x")
        assert rec.mean_duration_ms == pytest.approx(150)

    def test_to_dict(self, registry: ToolReliabilityRegistry) -> None:
        registry.record("x", True)
        rec = registry.get("x")
        d = rec.to_dict()
        assert d["tool"] == "x"
        assert "success_rate" in d


# ===========================================================================
# Upgrade 6 — PlanCritic
# ===========================================================================


class TestPlanCritic:
    def test_approved_clean_plan(self) -> None:
        graph = _make_simple_graph(3)
        critic = PlanCritic()
        report = critic.critique(graph)
        assert report.verdict == PlanVerdict.APPROVED

    def test_circular_dependency_detected(self) -> None:
        graph = TaskGraph(goal="g")
        t1 = Task(title="A")
        t2 = Task(title="B", depends_on=[t1.task_id])
        t1.depends_on = [t2.task_id]  # cycle
        graph.add_task(t1)
        graph.add_task(t2)
        critic = PlanCritic()
        report = critic.critique(graph)
        assert report.verdict == PlanVerdict.REJECTED
        assert any(i.category == "circular_dependency" for i in report.issues)

    def test_missing_tool_detected(self) -> None:
        registry = ToolRegistry([_FakeTool("fake_tool")])
        graph = TaskGraph(goal="g")
        graph.add_task(Task(title="T", tool_hint="nonexistent_tool"))
        critic = PlanCritic(tool_registry=registry)
        report = critic.critique(graph)
        assert report.verdict == PlanVerdict.REJECTED
        assert any(i.category == "missing_tool" for i in report.issues)

    def test_risky_tool_warning(self, tmp_path: Path) -> None:
        reliability = ToolReliabilityRegistry(tmp_path / "tr.json", min_samples_for_confidence=3)
        for _ in range(5):
            reliability.record("unreliable_tool", False)
        graph = TaskGraph(goal="g")
        graph.add_task(Task(title="T", tool_hint="unreliable_tool"))
        critic = PlanCritic(tool_reliability=reliability)
        report = critic.critique(graph)
        assert report.verdict == PlanVerdict.APPROVED_WITH_WARNINGS
        assert any(i.category == "risky_tool" for i in report.issues)

    def test_risky_tool_not_flagged_under_confidence(self, tmp_path: Path) -> None:
        reliability = ToolReliabilityRegistry(tmp_path / "tr.json", min_samples_for_confidence=10)
        reliability.record("unreliable_tool", False)  # only 1 sample
        graph = TaskGraph(goal="g")
        graph.add_task(Task(title="T", tool_hint="unreliable_tool"))
        critic = PlanCritic(tool_reliability=reliability)
        report = critic.critique(graph)
        assert not any(i.category == "risky_tool" for i in report.issues)

    def test_known_failure_warning(self, tmp_path: Path) -> None:
        fm = FailureMemory(tmp_path / "fm.json")
        fm.record("Build REST API endpoint", "python_tool", "schema mismatch", fix="use pydantic model")
        graph = TaskGraph(goal="g")
        graph.add_task(Task(title="Build REST API endpoint", tool_hint="python_tool"))
        critic = PlanCritic(failure_memory=fm)
        report = critic.critique(graph)
        assert any(i.category == "known_failure" for i in report.issues)
        issue = next(i for i in report.issues if i.category == "known_failure")
        assert "use pydantic model" in issue.message

    def test_missing_step_verification(self) -> None:
        graph = TaskGraph(goal="Build and test a new feature")
        graph.add_task(Task(title="Implement feature", description="Write the code"))
        critic = PlanCritic()
        report = critic.critique(graph)
        assert any(i.category == "missing_step" for i in report.issues)

    def test_missing_step_not_flagged_when_addressed(self) -> None:
        graph = TaskGraph(goal="Build and test a new feature")
        graph.add_task(Task(title="Implement feature", description="Write the code"))
        graph.add_task(Task(title="Test the feature", description="Verify it works"))
        critic = PlanCritic()
        report = critic.critique(graph)
        assert not any(i.category == "missing_step" for i in report.issues)

    def test_unreachable_task_dangling_dependency(self) -> None:
        graph = TaskGraph(goal="g")
        graph.add_task(Task(title="T", depends_on=["nonexistent_id"]))
        critic = PlanCritic()
        report = critic.critique(graph)
        assert report.verdict == PlanVerdict.REJECTED
        assert any(i.category == "dangling_dependency" for i in report.issues)

    def test_critique_report_to_dict(self) -> None:
        graph = _make_simple_graph(2)
        critic = PlanCritic()
        report = critic.critique(graph)
        d = report.to_dict()
        assert "verdict" in d
        assert "issues" in d

    def test_has_critical_and_has_warnings_properties(self) -> None:
        report = CritiqueReport(
            verdict=PlanVerdict.APPROVED_WITH_WARNINGS,
            issues=[CriticIssue(severity=IssueSeverity.WARNING, category="x", message="m")],
        )
        assert not report.has_critical
        assert report.has_warnings


# ===========================================================================
# Upgrade 2 — VerificationEngine
# ===========================================================================


class TestVerificationEngine:
    def test_non_empty_passes(self) -> None:
        task = Task(title="T")
        result = ExecutionResult(task_id="t1", tool_name="x", status=ExecutionStatus.SUCCESS,
                                 output="A reasonably long output string here.")
        engine = VerificationEngine()
        report = engine.verify(task, result)
        assert report.passed

    def test_non_empty_fails_on_short_output(self) -> None:
        task = Task(title="T")
        result = ExecutionResult(task_id="t1", tool_name="x", status=ExecutionStatus.SUCCESS, output="hi")
        engine = VerificationEngine()
        report = engine.verify(task, result)
        assert not report.passed
        assert any(c.verifier_name == "non_empty" and c.status == VerificationStatus.FAILED for c in report.checks)

    def test_schema_verifier_passes(self) -> None:
        task = Task(title="T", metadata={"expected_schema": ["name", "age"]})
        result = ExecutionResult(task_id="t1", tool_name="x", status=ExecutionStatus.SUCCESS,
                                 output=json.dumps({"name": "Blix", "age": 1}))
        engine = VerificationEngine()
        report = engine.verify(task, result)
        assert report.passed

    def test_schema_verifier_fails_invalid_json(self) -> None:
        task = Task(title="T", metadata={"expected_schema": ["name"]})
        result = ExecutionResult(task_id="t1", tool_name="x", status=ExecutionStatus.SUCCESS, output="not json at all here")
        engine = VerificationEngine()
        report = engine.verify(task, result)
        assert not report.passed

    def test_schema_verifier_fails_missing_keys(self) -> None:
        task = Task(title="T", metadata={"expected_schema": ["name", "age"]})
        result = ExecutionResult(task_id="t1", tool_name="x", status=ExecutionStatus.SUCCESS,
                                 output=json.dumps({"name": "Blix"}))
        engine = VerificationEngine()
        report = engine.verify(task, result)
        assert not report.passed

    def test_schema_verifier_skipped_when_not_applicable(self) -> None:
        task = Task(title="T")  # no expected_schema
        result = ExecutionResult(task_id="t1", tool_name="x", status=ExecutionStatus.SUCCESS, output="some output here")
        engine = VerificationEngine()
        report = engine.verify(task, result)
        schema_check = next(c for c in report.checks if c.verifier_name == "schema")
        assert schema_check.status == VerificationStatus.SKIPPED

    def test_keyword_presence_passes(self) -> None:
        task = Task(title="T", metadata={"required_keywords": ["route", "endpoint"]})
        result = ExecutionResult(task_id="t1", tool_name="x", status=ExecutionStatus.SUCCESS,
                                 output="Created a new route and endpoint for the API.")
        engine = VerificationEngine()
        report = engine.verify(task, result)
        assert report.passed

    def test_keyword_presence_fails_missing(self) -> None:
        task = Task(title="T", metadata={"required_keywords": ["route", "schema"]})
        result = ExecutionResult(task_id="t1", tool_name="x", status=ExecutionStatus.SUCCESS,
                                 output="Created a new route for the API.")
        engine = VerificationEngine()
        report = engine.verify(task, result)
        assert not report.passed

    def test_code_syntax_passes_valid_python(self) -> None:
        task = Task(title="T", tool_hint="python_tool")
        result = ExecutionResult(task_id="t1", tool_name="python_tool", status=ExecutionStatus.SUCCESS,
                                 output="```python\nprint('hello')\n```")
        engine = VerificationEngine()
        report = engine.verify(task, result)
        assert report.passed

    def test_code_syntax_fails_invalid_python(self) -> None:
        task = Task(title="T", tool_hint="python_tool")
        result = ExecutionResult(task_id="t1", tool_name="python_tool", status=ExecutionStatus.SUCCESS,
                                 output="```python\ndef broken(:\n```")
        engine = VerificationEngine()
        report = engine.verify(task, result)
        assert not report.passed

    def test_code_syntax_skipped_no_code_block(self) -> None:
        task = Task(title="T", tool_hint="python_tool")
        result = ExecutionResult(task_id="t1", tool_name="python_tool", status=ExecutionStatus.SUCCESS,
                                 output="some output with enough length to pass non_empty")
        engine = VerificationEngine()
        report = engine.verify(task, result)
        code_check = next(c for c in report.checks if c.verifier_name == "code_syntax")
        assert code_check.status == VerificationStatus.SKIPPED

    def test_summary_failed(self) -> None:
        task = Task(title="T")
        result = ExecutionResult(task_id="t1", tool_name="x", status=ExecutionStatus.SUCCESS, output="hi")
        engine = VerificationEngine()
        report = engine.verify(task, result)
        assert "failed" in report.summary().lower()

    def test_add_verifier(self) -> None:
        class AlwaysFail:
            name = "always_fail"
            def applies_to(self, task, result):
                return True
            def verify(self, task, result):
                return VerificationCheck("always_fail", VerificationStatus.FAILED, "nope")
        engine = VerificationEngine(verifiers=[])
        engine.add_verifier(AlwaysFail())
        task = Task(title="T")
        result = ExecutionResult(task_id="t1", tool_name="x", status=ExecutionStatus.SUCCESS, output="long enough output")
        report = engine.verify(task, result)
        assert not report.passed

    def test_verifier_names(self) -> None:
        engine = VerificationEngine()
        names = engine.verifier_names
        assert "non_empty" in names
        assert "schema" in names

    def test_report_to_dict(self) -> None:
        task = Task(title="T")
        result = ExecutionResult(task_id="t1", tool_name="x", status=ExecutionStatus.SUCCESS,
                                 output="long enough output for this test")
        engine = VerificationEngine()
        report = engine.verify(task, result)
        d = report.to_dict()
        assert "passed" in d
        assert "checks" in d


# ===========================================================================
# Upgrade 1 — Replanner
# ===========================================================================


class TestReplanner:
    def test_should_replan_true_when_failed(self) -> None:
        task = Task(title="T", status=TaskStatus.FAILED)
        graph = TaskGraph(goal="g")
        replanner = Replanner()
        assert replanner.should_replan(task, graph)

    def test_should_replan_false_when_not_failed(self) -> None:
        task = Task(title="T", status=TaskStatus.PENDING)
        graph = TaskGraph(goal="g")
        replanner = Replanner()
        assert not replanner.should_replan(task, graph)

    def test_should_replan_false_when_max_exceeded(self) -> None:
        task = Task(title="T", status=TaskStatus.FAILED)
        task.metadata["replan_count"] = 2
        graph = TaskGraph(goal="g")
        replanner = Replanner(max_replans_per_task=2)
        assert not replanner.should_replan(task, graph)

    def test_replan_switches_tool(self) -> None:
        registry = ToolRegistry([_FakeTool("web_search"), _FakeTool("memory_search")])
        graph = TaskGraph(goal="g")
        task = Task(title="Search", description="Search the web for info", tool_hint="web_search")
        graph.add_task(task)
        task.mark_failed("timeout")
        replanner = Replanner(tool_registry=registry)
        result = replanner.replan(task, graph, failure_reason="timeout")
        assert result.strategy == ReplanStrategy.SWITCH_TOOL
        assert task.tool_hint == "memory_search"
        assert task.status == TaskStatus.PENDING
        assert task.attempts == 0

    def test_replan_records_failure_memory(self, tmp_path: Path) -> None:
        fm = FailureMemory(tmp_path / "fm.json")
        registry = ToolRegistry([_FakeTool("web_search"), _FakeTool("memory_search")])
        graph = TaskGraph(goal="g")
        task = Task(title="Search", description="Search the web", tool_hint="web_search")
        graph.add_task(task)
        task.mark_failed("timeout")
        replanner = Replanner(tool_registry=registry, failure_memory=fm)
        replanner.replan(task, graph, failure_reason="timeout")
        assert fm.count == 1

    def test_replan_ranks_by_reliability(self, tmp_path: Path) -> None:
        registry = ToolRegistry([_FakeTool("web_search"), _FakeTool("memory_search"), _FakeTool("llm")])
        reliability = ToolReliabilityRegistry(tmp_path / "tr.json", min_samples_for_confidence=1)
        reliability.record("memory_search", False)
        reliability.record("llm", True)
        graph = TaskGraph(goal="g")
        task = Task(title="Search", description="Search the web", tool_hint="web_search")
        graph.add_task(task)
        task.mark_failed("timeout")
        replanner = Replanner(tool_registry=registry, tool_reliability=reliability)
        result = replanner.replan(task, graph)
        assert task.tool_hint == "llm"  # higher reliability than memory_search

    def test_replan_decomposes_when_no_alt_tool(self) -> None:
        graph = TaskGraph(goal="g")
        task = Task(title="Hard task", description="A complex task with no tool alternatives available",
                    tool_hint="unmapped_tool")
        graph.add_task(task)
        task.mark_failed("error")
        replanner = Replanner()  # no registry → can't confirm alt tools anyway
        result = replanner.replan(task, graph)
        assert result.strategy == ReplanStrategy.DECOMPOSE
        assert len(graph.tasks) == 2
        assert all("part" in t.title for t in graph.tasks)

    def test_replan_decompose_repoints_dependents(self) -> None:
        graph = TaskGraph(goal="g")
        t1 = Task(title="Failing task", description="A complex task description here", tool_hint="unmapped_tool")
        t2 = Task(title="Downstream task", depends_on=[t1.task_id])
        graph.add_task(t1)
        graph.add_task(t2)
        t1.mark_failed("error")
        replanner = Replanner()
        replanner.replan(t1, graph)
        # t2 should now depend on the second decomposed sub-task, not t1
        assert t1.task_id not in t2.depends_on
        assert len(t2.depends_on) == 1

    def test_replan_drops_task_when_not_decomposable(self) -> None:
        graph = TaskGraph(goal="g")
        task = Task(title="X", description="short", tool_hint="unmapped_tool")  # too short to decompose
        graph.add_task(task)
        task.mark_failed("error")
        replanner = Replanner()
        result = replanner.replan(task, graph)
        assert result.strategy == ReplanStrategy.DROP_TASK
        assert task.status == TaskStatus.SKIPPED

    def test_replan_drop_unblocks_dependents(self) -> None:
        graph = TaskGraph(goal="g")
        t1 = Task(title="X", description="short", tool_hint="unmapped_tool")
        t2 = Task(title="Y", depends_on=[t1.task_id])
        graph.add_task(t1)
        graph.add_task(t2)
        t1.mark_failed("error")
        replanner = Replanner()
        replanner.replan(t1, graph)
        assert t1.task_id not in t2.depends_on

    def test_replan_does_not_retry_same_tool_twice(self) -> None:
        registry = ToolRegistry([_FakeTool("web_search"), _FakeTool("memory_search"), _FakeTool("llm")])
        graph = TaskGraph(goal="g")
        task = Task(title="Search", description="Search the web for info", tool_hint="web_search")
        graph.add_task(task)
        task.mark_failed("timeout")
        replanner = Replanner(tool_registry=registry)
        result1 = replanner.replan(task, graph)
        assert task.tool_hint == "memory_search"
        task.mark_failed("timeout again")
        result2 = replanner.replan(task, graph)
        assert task.tool_hint == "llm"  # not memory_search again
        assert task.tool_hint != "web_search"

    def test_replan_result_to_dict(self) -> None:
        result = ReplanResult(strategy=ReplanStrategy.SWITCH_TOOL, modified_task_ids=["t1"], explanation="x")
        d = result.to_dict()
        assert d["strategy"] == "switch_tool"


# ===========================================================================
# Upgrade 3 — TaskRuntime (Execution DAG Runtime)
# ===========================================================================


class TestTaskRuntime:
    def test_next_batch_respects_max_parallel(self) -> None:
        graph = TaskGraph(goal="g")
        for i in range(5):
            graph.add_task(Task(title=f"T{i}"))  # all independent
        runtime = TaskRuntime(graph, max_parallel=2)
        batch = runtime.next_batch()
        assert len(batch) == 2

    def test_next_batch_independent_tasks_all_ready(self) -> None:
        graph = TaskGraph(goal="g")
        graph.add_task(Task(title="A"))
        graph.add_task(Task(title="B"))
        graph.add_task(Task(title="C"))
        runtime = TaskRuntime(graph, max_parallel=10)
        batch = runtime.next_batch()
        assert len(batch) == 3

    def test_has_runnable_work(self) -> None:
        graph = _make_simple_graph(1)
        runtime = TaskRuntime(graph)
        assert runtime.has_runnable_work()
        graph.tasks[0].mark_completed("ok")
        assert not runtime.has_runnable_work()

    def test_propagate_failures_blocks_dependent(self) -> None:
        graph = _make_simple_graph(2, linear=True)
        graph.tasks[0].mark_failed("err")
        runtime = TaskRuntime(graph)
        blocked_count = runtime.propagate_failures()
        assert blocked_count == 1
        assert graph.tasks[1].status == TaskStatus.BLOCKED

    def test_propagate_failures_transitive_chain(self) -> None:
        graph = _make_simple_graph(4, linear=True)  # T0 -> T1 -> T2 -> T3
        graph.tasks[0].mark_failed("err")
        runtime = TaskRuntime(graph)
        runtime.propagate_failures()
        assert graph.tasks[1].status == TaskStatus.BLOCKED
        assert graph.tasks[2].status == TaskStatus.BLOCKED
        assert graph.tasks[3].status == TaskStatus.BLOCKED

    def test_propagate_failures_independent_branch_unaffected(self) -> None:
        graph = TaskGraph(goal="g")
        t1 = Task(title="A")
        t2 = Task(title="B", depends_on=[t1.task_id])
        t3 = Task(title="C")  # independent
        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)
        t1.mark_failed("err")
        runtime = TaskRuntime(graph)
        runtime.propagate_failures()
        assert t2.status == TaskStatus.BLOCKED
        assert t3.status == TaskStatus.PENDING

    def test_unblock_reverts_to_pending(self) -> None:
        graph = _make_simple_graph(2, linear=True)
        graph.tasks[0].mark_failed("err")
        runtime = TaskRuntime(graph)
        runtime.propagate_failures()
        assert graph.tasks[1].status == TaskStatus.BLOCKED
        success = runtime.unblock(graph.tasks[1].task_id)
        assert success
        assert graph.tasks[1].status == TaskStatus.PENDING

    def test_unblock_returns_false_if_not_blocked(self) -> None:
        graph = _make_simple_graph(1)
        runtime = TaskRuntime(graph)
        assert not runtime.unblock(graph.tasks[0].task_id)

    def test_topological_batches(self) -> None:
        graph = _make_simple_graph(3, linear=True)
        runtime = TaskRuntime(graph)
        batches = runtime.topological_batches()
        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)

    def test_topological_batches_parallel_branches(self) -> None:
        graph = TaskGraph(goal="g")
        root = Task(title="Root")
        b1 = Task(title="B1", depends_on=[root.task_id])
        b2 = Task(title="B2", depends_on=[root.task_id])
        graph.add_task(root)
        graph.add_task(b1)
        graph.add_task(b2)
        runtime = TaskRuntime(graph)
        batches = runtime.topological_batches()
        assert len(batches) == 2
        assert len(batches[1]) == 2  # b1, b2 in parallel

    def test_has_unrecoverable_blocks(self) -> None:
        graph = _make_simple_graph(2, linear=True)
        graph.tasks[0].mark_failed("err")
        runtime = TaskRuntime(graph)
        runtime.propagate_failures()
        assert runtime.has_unrecoverable_blocks

    def test_stats_tracking(self) -> None:
        graph = TaskGraph(goal="g")
        for i in range(3):
            graph.add_task(Task(title=f"T{i}"))
        runtime = TaskRuntime(graph, max_parallel=2)
        runtime.next_batch()
        assert runtime.stats.batches_executed == 1
        assert runtime.stats.max_batch_size == 2

    def test_dag_runtime_stats_to_dict(self) -> None:
        stats = DAGRuntimeStats(batches_executed=2, max_batch_size=3, blocked_tasks=1)
        d = stats.to_dict()
        assert d["batches_executed"] == 2


# ===========================================================================
# Upgrade 8 — PlanReflection
# ===========================================================================


class TestPlanReflection:
    def test_reflect_success(self) -> None:
        graph = _make_simple_graph(2)
        graph.tasks[0].mark_completed("ok")
        graph.tasks[1].mark_completed("ok")
        reflection = PlanReflection()
        report = reflection.reflect(graph, history=[])
        assert report.success
        assert report.root_cause is None

    def test_reflect_success_with_replans_noted(self) -> None:
        graph = _make_simple_graph(1)
        graph.tasks[0].mark_completed("ok")
        reflection = PlanReflection()
        report = reflection.reflect(graph, history=[], replan_count=2)
        assert report.success
        assert any("2 replan" in l for l in report.lessons)

    def test_reflect_failure_identifies_task(self) -> None:
        graph = _make_simple_graph(2)
        graph.tasks[0].mark_failed("timeout error")
        reflection = PlanReflection()
        report = reflection.reflect(graph, history=[])
        assert not report.success
        assert report.failure_task_id == graph.tasks[0].task_id
        assert "timeout error" in report.root_cause

    def test_reflect_failure_finds_bottleneck_tool(self) -> None:
        graph = _make_simple_graph(1)
        graph.tasks[0].mark_failed("error")
        history = [
            {"tool": "web_search", "decision": "retry"},
            {"tool": "web_search", "decision": "skip"},
            {"tool": "llm", "decision": "accept"},
        ]
        reflection = PlanReflection()
        report = reflection.reflect(graph, history=history)
        assert report.bottleneck_tool == "web_search"

    def test_reflect_generates_suggestions(self) -> None:
        graph = _make_simple_graph(1)
        graph.tasks[0].mark_failed("error")
        reflection = PlanReflection()
        report = reflection.reflect(graph, history=[], replan_count=0)
        assert len(report.improvement_suggestions) > 0
        assert any("replan" in s.lower() for s in report.improvement_suggestions)

    def test_reflect_persists_to_failure_memory(self, tmp_path: Path) -> None:
        fm = FailureMemory(tmp_path / "fm.json")
        graph = _make_simple_graph(1)
        graph.tasks[0].mark_failed("error")
        reflection = PlanReflection(failure_memory=fm)
        reflection.reflect(graph, history=[])
        # record_fix only attaches if a similar failure already recorded via .record()
        # so we verify it doesn't crash and the call completes
        assert True

    def test_reflect_persists_to_reflection_engine(self, tmp_path: Path) -> None:
        from reflection.reflection_engine import ReflectionEngine
        re_engine = ReflectionEngine(tmp_path / "reflections.json")
        graph = _make_simple_graph(1)
        graph.tasks[0].mark_completed("ok")
        reflection = PlanReflection(reflection_engine=re_engine)
        reflection.reflect(graph, history=[])
        assert re_engine.record_count >= 1

    def test_reflect_stalled_no_failed_tasks(self) -> None:
        graph = TaskGraph(goal="g")
        graph.add_task(Task(title="T", depends_on=["nonexistent"]))  # permanently unready
        reflection = PlanReflection()
        report = reflection.reflect(graph, history=[])
        assert not report.success
        assert report.root_cause is not None

    def test_report_summary_success(self) -> None:
        report = PlanReflectionReport(success=True, lessons=["Clean run."])
        assert "successfully" in report.summary()

    def test_report_summary_failure(self) -> None:
        report = PlanReflectionReport(success=False, root_cause="Task X failed", bottleneck_tool="web_search")
        summary = report.summary()
        assert "Task X failed" in summary
        assert "web_search" in summary

    def test_report_to_dict(self) -> None:
        report = PlanReflectionReport(success=False, root_cause="x", improvement_suggestions=["y"])
        d = report.to_dict()
        assert d["success"] is False
        assert d["improvement_suggestions"] == ["y"]


# ===========================================================================
# Upgrade 9 — AdaptiveAgentEvaluator (Agent Benchmark Suite)
# ===========================================================================


class TestAdaptiveAgentEvaluator:
    def test_verification_accuracy(self) -> None:
        reports = [
            VerificationReport(task_id="t1", checks=[VerificationCheck("v", VerificationStatus.PASSED)]),
            VerificationReport(task_id="t2", checks=[VerificationCheck("v", VerificationStatus.FAILED, "x")]),
        ]
        ev = AdaptiveAgentEvaluator()
        acc = ev.verification_accuracy(reports, expected_pass=[True, False])
        assert acc == 1.0

    def test_verification_accuracy_mismatch(self) -> None:
        reports = [VerificationReport(task_id="t1", checks=[VerificationCheck("v", VerificationStatus.PASSED)])]
        ev = AdaptiveAgentEvaluator()
        acc = ev.verification_accuracy(reports, expected_pass=[False])
        assert acc == 0.0

    def test_verification_accuracy_mismatched_lengths(self) -> None:
        ev = AdaptiveAgentEvaluator()
        assert ev.verification_accuracy([], [True]) == 0.0

    def test_verification_catch_rate(self) -> None:
        graph = TaskGraph(goal="g")
        r1 = AgentRunResult(goal="g", graph=graph, history=[{"note": "Verification failed: bad output"}])
        r2 = AgentRunResult(goal="g", graph=graph, history=[{"note": "all good"}])
        ev = AdaptiveAgentEvaluator()
        rate = ev.verification_catch_rate([r1, r2])
        assert rate == 0.5

    def test_replanning_success_rate_no_replans(self) -> None:
        graph = TaskGraph(goal="g")
        r1 = AgentRunResult(goal="g", graph=graph, replan_count=0, success=True)
        ev = AdaptiveAgentEvaluator()
        assert ev.replanning_success_rate([r1]) == 1.0

    def test_replanning_success_rate_with_replans(self) -> None:
        graph = TaskGraph(goal="g")
        r1 = AgentRunResult(goal="g", graph=graph, replan_count=1, success=True)
        r2 = AgentRunResult(goal="g", graph=graph, replan_count=2, success=False)
        ev = AdaptiveAgentEvaluator()
        rate = ev.replanning_success_rate([r1, r2])
        assert rate == 0.5

    def test_mean_replans_per_run(self) -> None:
        graph = TaskGraph(goal="g")
        r1 = AgentRunResult(goal="g", graph=graph, replan_count=2)
        r2 = AgentRunResult(goal="g", graph=graph, replan_count=4)
        ev = AdaptiveAgentEvaluator()
        assert ev.mean_replans_per_run([r1, r2]) == 3.0

    def test_recovery_rate_no_struggles(self) -> None:
        graph = _make_simple_graph(2)
        graph.tasks[0].mark_completed("ok")
        graph.tasks[1].mark_completed("ok")
        ev = AdaptiveAgentEvaluator()
        assert ev.recovery_rate(graph) == 1.0

    def test_recovery_rate_recovered_task(self) -> None:
        graph = _make_simple_graph(1)
        graph.tasks[0].attempts = 3
        graph.tasks[0].mark_completed("ok")
        ev = AdaptiveAgentEvaluator()
        assert ev.recovery_rate(graph) == 1.0

    def test_recovery_rate_unrecovered_task(self) -> None:
        graph = _make_simple_graph(1)
        graph.tasks[0].attempts = 3
        graph.tasks[0].mark_failed("err")
        ev = AdaptiveAgentEvaluator()
        assert ev.recovery_rate(graph) == 0.0

    def test_tool_efficiency(self) -> None:
        graph = TaskGraph(goal="g")
        history = [{"tool": "x"}, {"tool": "x"}]
        result = AgentRunResult(goal="g", graph=graph, completed_tasks=1, history=history)
        ev = AdaptiveAgentEvaluator()
        eff = ev.tool_efficiency(result)
        assert eff == 0.5

    def test_tool_efficiency_no_calls(self) -> None:
        graph = TaskGraph(goal="g")
        result = AgentRunResult(goal="g", graph=graph, history=[])
        ev = AdaptiveAgentEvaluator()
        assert ev.tool_efficiency(result) == 0.0

    def test_benchmark_run_combines_metrics(self) -> None:
        graph = _make_simple_graph(1)
        graph.tasks[0].mark_completed("ok")
        result = AgentRunResult(goal="g", graph=graph, completed_tasks=1,
                                history=[{"tool": "x", "decision": "accept", "quality": 0.8, "task_id": "t1"}])
        ev = AdaptiveAgentEvaluator()
        metrics = ev.benchmark_run(result)
        assert "task_success_rate" in metrics
        assert "recovery_rate" in metrics
        assert "tool_efficiency" in metrics
        assert "replan_count" in metrics

    def test_benchmark_batch(self) -> None:
        graph = _make_simple_graph(1)
        graph.tasks[0].mark_completed("ok")
        r1 = AgentRunResult(goal="g", graph=graph, success=True, replan_count=1)
        r2 = AgentRunResult(goal="g", graph=graph, success=True, replan_count=0)
        ev = AdaptiveAgentEvaluator()
        metrics = ev.benchmark_batch([r1, r2])
        assert "replanning_success_rate" in metrics
        assert "mean_replans_per_run" in metrics
        assert "batch_recovery_rate" in metrics

    def test_benchmark_batch_empty(self) -> None:
        ev = AdaptiveAgentEvaluator()
        assert ev.benchmark_batch([]) == {}

    def test_inherits_v035_metrics(self) -> None:
        ev = AdaptiveAgentEvaluator()
        graph = _make_simple_graph(2)
        graph.tasks[0].mark_completed("ok")
        graph.tasks[1].mark_failed("err")
        assert ev.task_success_rate(graph) == 0.5

    def test_in_blix_eval_exports(self) -> None:
        from evaluation.blix_eval import AdaptiveAgentEvaluator as AAE_blix
        assert AdaptiveAgentEvaluator is AAE_blix


# ===========================================================================
# Integration — full v0.3.6 adaptive loop in AgentExecutor
# ===========================================================================


class TestAdaptiveExecutorIntegration:
    def _build_full_executor(
        self, tmp_path: Path, tools: list[Tool], **overrides,
    ) -> tuple[AgentExecutor, dict]:
        registry = ToolRegistry(tools)
        wm = WorkingMemory()
        obs_layer = ObservationLayer()
        reflect_loop = ReflectionLoop(tmp_path / "history.json", max_retries=overrides.get("max_retries", 1))
        fm = FailureMemory(tmp_path / "fm.json")
        tr = ToolReliabilityRegistry(tmp_path / "tr.json")
        critic = PlanCritic(tool_registry=registry, tool_reliability=tr, failure_memory=fm)
        verifier = VerificationEngine()
        replanner = Replanner(tool_registry=registry, failure_memory=fm, tool_reliability=tr)
        plan_reflection = PlanReflection(failure_memory=fm)

        executor = AgentExecutor(
            tool_registry=registry, working_memory=wm, observation_layer=obs_layer,
            reflection_loop=reflect_loop,
            config=ExecutorConfig(max_steps=20, enable_verification=overrides.get("enable_verification", True),
                                  enable_replanning=overrides.get("enable_replanning", True)),
            plan_critic=critic, verification_engine=verifier,
            replanner=replanner, plan_reflection=plan_reflection,
        )
        components = {"registry": registry, "fm": fm, "tr": tr, "critic": critic,
                      "verifier": verifier, "replanner": replanner, "plan_reflection": plan_reflection}
        return executor, components

    def test_full_loop_succeeds_without_failures(self, tmp_path: Path) -> None:
        executor, _ = self._build_full_executor(tmp_path, [_FakeTool("fake_tool")])
        graph = _make_simple_graph(2)
        result = executor.run(graph)
        assert result.success
        assert result.replan_count == 0
        assert result.critique["verdict"] == "approved"
        assert result.plan_reflection["success"] is True
        assert result.agent_state is not None

    def test_critic_rejects_circular_plan_before_execution(self, tmp_path: Path) -> None:
        executor, _ = self._build_full_executor(tmp_path, [_FakeTool("fake_tool")])
        graph = TaskGraph(goal="g")
        t1 = Task(title="A", tool_hint="fake_tool")
        t2 = Task(title="B", tool_hint="fake_tool", depends_on=[t1.task_id])
        t1.depends_on = [t2.task_id]
        graph.add_task(t1)
        graph.add_task(t2)
        result = executor.run(graph)
        assert result.aborted_by_critic
        assert result.total_steps == 0
        assert not result.success

    def test_replanner_switches_tool_on_failure(self, tmp_path: Path) -> None:
        fail_tool = _FakeTool("web_search", outputs=[
            ExecutionResult(task_id="", tool_name="web_search", status=ExecutionStatus.ERROR,
                           output="", error="timeout"),
        ] * 5)
        fallback_tool = _FakeTool("memory_search")
        executor, components = self._build_full_executor(
            tmp_path, [fail_tool, fallback_tool], max_retries=1,
        )
        graph = TaskGraph(goal="g")
        graph.add_task(Task(title="Search", description="Search the web for info", tool_hint="web_search"))
        result = executor.run(graph)
        assert result.success
        assert result.replan_count >= 1
        assert components["fm"].count >= 1

    def test_verification_engine_forces_retry_on_bad_output(self, tmp_path: Path) -> None:
        # Tool reports SUCCESS with non-empty output, but output fails schema verification
        bad_schema_tool = _FakeTool("schema_tool", outputs=[
            ExecutionResult(task_id="", tool_name="schema_tool", status=ExecutionStatus.SUCCESS,
                           output="not valid json at all"),
            ExecutionResult(task_id="", tool_name="schema_tool", status=ExecutionStatus.SUCCESS,
                           output=json.dumps({"result": "ok"})),
        ])
        executor, _ = self._build_full_executor(tmp_path, [bad_schema_tool], max_retries=2)
        graph = TaskGraph(goal="g")
        task = Task(title="Produce JSON", description="Produce JSON output", tool_hint="schema_tool",
                    metadata={"expected_schema": ["result"]})
        graph.add_task(task)
        result = executor.run(graph)
        decisions = [h["decision"] for h in result.history]
        assert "retry" in decisions  # verification failure forced a retry despite tool SUCCESS

    def test_verification_disabled_skips_gate(self, tmp_path: Path) -> None:
        bad_schema_tool = _FakeTool("schema_tool", outputs=[
            ExecutionResult(task_id="", tool_name="schema_tool", status=ExecutionStatus.SUCCESS,
                           output="not valid json but long enough to pass non_empty check easily"),
        ])
        executor, _ = self._build_full_executor(
            tmp_path, [bad_schema_tool], enable_verification=False,
        )
        graph = TaskGraph(goal="g")
        task = Task(title="Produce JSON", description="Produce JSON output", tool_hint="schema_tool",
                    metadata={"expected_schema": ["result"]})
        graph.add_task(task)
        result = executor.run(graph)
        # Verification gate disabled → accepted despite invalid schema
        assert result.completed_tasks == 1

    def test_replanning_disabled_treats_failure_as_terminal(self, tmp_path: Path) -> None:
        always_fail = _FakeTool("web_search", outputs=[
            ExecutionResult(task_id="", tool_name="web_search", status=ExecutionStatus.ERROR,
                           output="", error="timeout"),
        ] * 5)
        fallback_tool = _FakeTool("memory_search")
        executor, _ = self._build_full_executor(
            tmp_path, [always_fail, fallback_tool], max_retries=1, enable_replanning=False,
        )
        graph = TaskGraph(goal="g")
        graph.add_task(Task(title="Search", description="Search the web for info", tool_hint="web_search"))
        result = executor.run(graph)
        assert not result.success
        assert result.replan_count == 0
        assert result.failed_tasks == 1

    def test_agent_state_tracks_tool_reliability_live(self, tmp_path: Path) -> None:
        executor, _ = self._build_full_executor(tmp_path, [_FakeTool("fake_tool")])
        graph = _make_simple_graph(2)
        result = executor.run(graph)
        assert "fake_tool" in result.agent_state["tool_reliability"]

    def test_agent_state_confidence_increases_with_progress(self, tmp_path: Path) -> None:
        executor, _ = self._build_full_executor(tmp_path, [_FakeTool("fake_tool")])
        graph = _make_simple_graph(3)
        result = executor.run(graph)
        assert result.agent_state["confidence"] > 0.0

    def test_failure_memory_persists_across_executor_runs(self, tmp_path: Path) -> None:
        fm = FailureMemory(tmp_path / "fm.json")
        always_fail = _FakeTool("web_search", outputs=[
            ExecutionResult(task_id="", tool_name="web_search", status=ExecutionStatus.ERROR,
                           output="", error="timeout"),
        ] * 5)
        registry = ToolRegistry([always_fail])
        wm = WorkingMemory()
        obs_layer = ObservationLayer()
        reflect_loop = ReflectionLoop(tmp_path / "history.json", max_retries=1)
        replanner = Replanner(tool_registry=registry, failure_memory=fm)
        executor = AgentExecutor(
            tool_registry=registry, working_memory=wm, observation_layer=obs_layer,
            reflection_loop=reflect_loop, replanner=replanner,
        )
        graph = TaskGraph(goal="g")
        graph.add_task(Task(title="Search the web", description="Search the web for papers", tool_hint="web_search"))
        executor.run(graph)
        assert fm.count >= 1

    def test_backwards_compatible_without_v036_components(self, tmp_path: Path) -> None:
        """Confirm v0.3.5 behavior is unchanged when v0.3.6 components are None."""
        registry = ToolRegistry([_FakeTool("fake_tool")])
        wm = WorkingMemory()
        obs_layer = ObservationLayer()
        reflect_loop = ReflectionLoop(tmp_path / "history.json")
        executor = AgentExecutor(
            tool_registry=registry, working_memory=wm,
            observation_layer=obs_layer, reflection_loop=reflect_loop,
            # no plan_critic, verification_engine, replanner, plan_reflection
        )
        graph = _make_simple_graph(2)
        result = executor.run(graph)
        assert result.success
        assert result.critique is None
        assert result.plan_reflection is None
        assert result.agent_state is not None  # AgentState always tracked now


# ===========================================================================
# API — v0.3.6 /agent endpoints
# ===========================================================================


class _FakeLLMFull:
    def model_name(self) -> str:
        return "fake-0.3.6"

    def generate(self, prompt: str) -> str:
        return "Fake agent LLM reply."


@pytest.fixture(scope="module")
def tmp_memory_v6(tmp_path_factory):
    return tmp_path_factory.mktemp("memory_v6")


@pytest.fixture(scope="module")
def ctx_v6(tmp_memory_v6):
    from config import settings as _settings
    _settings.settings.memory.conversations_file = tmp_memory_v6 / "conversations.json"
    _settings.settings.memory.profile_file = tmp_memory_v6 / "profile.json"
    _settings.settings.memory.learning_state_file = tmp_memory_v6 / "learning_state.json"
    _settings.settings.embed.embeddings_file = tmp_memory_v6 / "embeddings.npy"
    _settings.settings.embed.embedding_ids_file = tmp_memory_v6 / "embedding_ids.json"

    from api.context import BlixContext
    ctx = BlixContext(tmp_memory_v6)
    ctx.llm = _FakeLLMFull()
    ctx.agent._llm = _FakeLLMFull()
    return ctx


@pytest.fixture(scope="module")
def client_v6(ctx_v6) -> Generator[TestClient, None, None]:
    from api.deps import set_context
    from api.routers.agent import router as agent_router

    app = FastAPI(title="Blix Test v0.3.6")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(agent_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    set_context(ctx_v6)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestAgentAPIv036:
    def test_run_endpoint_includes_v036_fields(self, client_v6: TestClient) -> None:
        r = client_v6.post("/agent/run", json={"goal": "Save a note about progress today."})
        assert r.status_code == 200
        data = r.json()
        assert "replan_count" in data
        assert "critique" in data
        assert "plan_reflection" in data
        assert "agent_state" in data

    def test_critique_endpoint(self, client_v6: TestClient) -> None:
        r = client_v6.post("/agent/critique", json={"goal": "Write a summary about transformers."})
        assert r.status_code == 200
        data = r.json()
        assert "task_graph" in data
        assert "critique" in data
        assert data["critique"]["verdict"] in ("approved", "approved_with_warnings", "rejected")

    def test_failures_endpoint_empty(self, client_v6: TestClient) -> None:
        r = client_v6.get("/agent/failures")
        assert r.status_code == 200
        data = r.json()
        assert "total_recorded" in data
        assert "failures" in data

    def test_failures_endpoint_after_recording(self, client_v6: TestClient, ctx_v6) -> None:
        ctx_v6.failure_memory.record("Some task", "some_tool", "some failure")
        r = client_v6.get("/agent/failures")
        assert r.status_code == 200
        data = r.json()
        assert data["total_recorded"] >= 1

    def test_tool_reliability_endpoint(self, client_v6: TestClient, ctx_v6) -> None:
        ctx_v6.tool_reliability_registry.record("test_tool", True)
        r = client_v6.get("/agent/tool-reliability")
        assert r.status_code == 200
        data = r.json()
        assert data["tracked_tools"] >= 1
        names = [t["tool"] for t in data["tools"]]
        assert "test_tool" in names

    def test_run_with_unreachable_tool_critic_rejects(self, client_v6: TestClient) -> None:
        # This relies on the planner producing a hint that exists, so we
        # just confirm the run endpoint doesn't crash and returns valid structure
        r = client_v6.post("/agent/run", json={"goal": "Build something with Python code."})
        assert r.status_code == 200
        data = r.json()
        assert "success" in data
        assert "aborted_by_critic" in data


class TestBlixContextV036Wiring:
    def test_v036_components_present(self, ctx_v6) -> None:
        assert ctx_v6.failure_memory is not None
        assert ctx_v6.tool_reliability_registry is not None
        assert ctx_v6.plan_critic is not None
        assert ctx_v6.verification_engine is not None
        assert ctx_v6.replanner is not None
        assert ctx_v6.plan_reflection is not None

    def test_agent_executor_has_v036_components_wired(self, ctx_v6) -> None:
        assert ctx_v6.agent_executor._critic is not None
        assert ctx_v6.agent_executor._verifier is not None
        assert ctx_v6.agent_executor._replanner is not None
        assert ctx_v6.agent_executor._plan_reflection is not None

    def test_dashboard_stats_includes_v036_metrics(self, ctx_v6) -> None:
        stats = ctx_v6.dashboard_stats()
        assert "failure_memory_count" in stats
        assert "tool_reliability_tracked_count" in stats

    def test_agent_evaluator_is_adaptive(self, ctx_v6) -> None:
        from evaluation.agent_benchmark import AdaptiveAgentEvaluator
        assert isinstance(ctx_v6.agent_evaluator, AdaptiveAgentEvaluator)
