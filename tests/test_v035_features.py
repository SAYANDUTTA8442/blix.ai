"""
Tests for Blix v0.3.5 — "Agent Execution Framework".

Covers:
1.  agents.types          (Task, TaskGraph, ExecutionResult, Observation)
2.  tools.registry         (Tool base, ToolRegistry, all concrete tools)
3.  planning.planner       (GoalParser, TaskDecomposer, Planner, MilestoneTracker)
4.  agents.working_memory  (WorkingMemory TTL eviction)
5.  agents.observation     (ObservationLayer quality scoring)
6.  agents.reflection_loop (ReflectionLoop accept/retry/skip decisions)
7.  agents.executor        (AgentExecutor full loop, AgentSession)
8.  evaluation.agent_eval  (AgentEvaluator metrics)
9.  API                    (/agent/run, /agent/plan, /agent/history, /agent/tools)

All tests run fully offline: WebSearchTool network calls are expected to
fail in the sandboxed environment (api.duckduckgo.com is not in the
allowed domain list) — we test its error-handling path, not live results.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from agents.executor import AgentExecutor, AgentRunResult, AgentSession, ExecutorConfig
from agents.observation import ObservationLayer
from agents.reflection_loop import ReflectionDecision, ReflectionLoop
from agents.types import (
    ExecutionHistoryEntry, ExecutionResult, ExecutionStatus,
    Observation, Task, TaskGraph, TaskStatus, WorkingMemoryEntry,
)
from agents.working_memory import WorkingMemory
from evaluation.agent_eval import AgentEvalCase, AgentEvaluator
from planning.planner import (
    GoalParser, MilestoneTracker, ParsedGoal, Planner, TaskDecomposer,
)
from tools.registry import (
    FileTool, LLMTool, MemorySearchTool, MemoryWriteTool, PythonTool,
    ReasoningTool, SynthesisTool, Tool, ToolMatch, ToolRegistry, WebSearchTool,
)


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _FakeLLM:
    def model_name(self) -> str:
        return "fake-0.3.5"

    def generate(self, prompt: str) -> str:
        return "Fake LLM response."


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


def _make_simple_graph(n_tasks: int = 3, linear: bool = True) -> TaskGraph:
    graph = TaskGraph(goal="Test goal")
    prev_id = None
    for i in range(n_tasks):
        deps = [prev_id] if (linear and prev_id) else []
        task = Task(title=f"Task {i}", description=f"Do step {i}", tool_hint="fake_tool", depends_on=deps)
        graph.add_task(task)
        prev_id = task.task_id
    return graph


# ===========================================================================
# Module: agents.types
# ===========================================================================


class TestAgentTypes:
    def test_task_defaults(self) -> None:
        task = Task(title="Test")
        assert task.status == TaskStatus.PENDING
        assert task.attempts == 0
        assert len(task.task_id) == 8

    def test_task_is_ready_no_deps(self) -> None:
        task = Task(title="T")
        assert task.is_ready(set())

    def test_task_is_ready_with_deps(self) -> None:
        task = Task(title="T", depends_on=["abc123"])
        assert not task.is_ready(set())
        assert task.is_ready({"abc123"})

    def test_task_mark_completed(self) -> None:
        task = Task(title="T")
        task.mark_completed("done")
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "done"
        assert task.completed_at is not None

    def test_task_mark_failed(self) -> None:
        task = Task(title="T")
        task.mark_failed("error msg")
        assert task.status == TaskStatus.FAILED
        assert task.error == "error msg"

    def test_task_to_dict(self) -> None:
        task = Task(title="T", description="desc")
        d = task.to_dict()
        assert d["title"] == "T"
        assert d["status"] == "pending"

    def test_taskgraph_add_and_get(self) -> None:
        graph = TaskGraph(goal="g")
        task = Task(title="T")
        graph.add_task(task)
        assert graph.get_task(task.task_id) is task
        assert graph.get_task("nonexistent") is None

    def test_taskgraph_ready_tasks_respects_deps(self) -> None:
        graph = TaskGraph(goal="g")
        t1 = Task(title="T1")
        t2 = Task(title="T2", depends_on=[t1.task_id])
        graph.add_task(t1)
        graph.add_task(t2)
        ready = graph.ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == t1.task_id

    def test_taskgraph_next_task_after_completion(self) -> None:
        graph = TaskGraph(goal="g")
        t1 = Task(title="T1")
        t2 = Task(title="T2", depends_on=[t1.task_id])
        graph.add_task(t1)
        graph.add_task(t2)
        t1.mark_completed("ok")
        nxt = graph.next_task()
        assert nxt.task_id == t2.task_id

    def test_taskgraph_is_complete(self) -> None:
        graph = TaskGraph(goal="g")
        t1 = Task(title="T1")
        graph.add_task(t1)
        assert not graph.is_complete
        t1.mark_completed("ok")
        assert graph.is_complete

    def test_taskgraph_has_failures(self) -> None:
        graph = TaskGraph(goal="g")
        t1 = Task(title="T1")
        graph.add_task(t1)
        t1.mark_failed("err")
        assert graph.has_failures

    def test_taskgraph_progress(self) -> None:
        graph = TaskGraph(goal="g")
        t1, t2 = Task(title="T1"), Task(title="T2")
        graph.add_task(t1)
        graph.add_task(t2)
        assert graph.progress == 0
        t1.mark_completed("ok")
        assert graph.progress == 50

    def test_taskgraph_status_summary(self) -> None:
        graph = TaskGraph(goal="g")
        t1 = Task(title="T1")
        graph.add_task(t1)
        t1.mark_completed("ok")
        summary = graph.status_summary()
        assert summary["completed"] == 1
        assert summary["pending"] == 0

    def test_execution_result_is_success(self) -> None:
        r = ExecutionResult(task_id="x", tool_name="t", status=ExecutionStatus.SUCCESS)
        assert r.is_success()
        r2 = ExecutionResult(task_id="x", tool_name="t", status=ExecutionStatus.FAILURE)
        assert not r2.is_success()

    def test_working_memory_entry_expiry(self) -> None:
        entry = WorkingMemoryEntry(key="k", value="v", ttl_steps=2, age_steps=2)
        assert entry.is_expired()
        entry2 = WorkingMemoryEntry(key="k", value="v", ttl_steps=5, age_steps=1)
        assert not entry2.is_expired()

    def test_execution_history_entry_to_dict(self) -> None:
        entry = ExecutionHistoryEntry(goal="g", task_id="t1", tool="web_search", success=True)
        d = entry.to_dict()
        assert d["goal"] == "g"
        assert d["success"] is True


# ===========================================================================
# Module: tools.registry
# ===========================================================================


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry([_FakeTool("a"), _FakeTool("b")])
        assert registry.get("a") is not None
        assert registry.get("nonexistent") is None
        assert len(registry.list_tools()) == 2

    def test_schema(self) -> None:
        registry = ToolRegistry([_FakeTool("a")])
        schema = registry.schema()
        assert schema[0]["name"] == "a"

    def test_select_tool_by_hint(self) -> None:
        registry = ToolRegistry([_FakeTool("a"), _FakeTool("b")])
        task = Task(title="T", tool_hint="b")
        selected = registry.select_tool(task)
        assert selected.name == "b"

    def test_select_tool_by_can_handle(self) -> None:
        class HighScoreTool(_FakeTool):
            def can_handle(self, task: Task) -> float:
                return 0.95
        class LowScoreTool(_FakeTool):
            def can_handle(self, task: Task) -> float:
                return 0.05
        registry = ToolRegistry([LowScoreTool("low"), HighScoreTool("high")])
        task = Task(title="T")  # no hint
        selected = registry.select_tool(task)
        assert selected.name == "high"

    def test_select_tool_none_if_low_scores(self) -> None:
        class ZeroTool(_FakeTool):
            def can_handle(self, task: Task) -> float:
                return 0.0
        registry = ToolRegistry([ZeroTool("z")])
        task = Task(title="T")
        assert registry.select_tool(task) is None

    def test_rank_tools(self) -> None:
        registry = ToolRegistry([_FakeTool("a"), _FakeTool("b")])
        task = Task(title="T")
        ranked = registry.rank_tools(task)
        assert len(ranked) == 2
        assert all(isinstance(m, ToolMatch) for m in ranked)


class TestConcreteTools:
    def test_memory_search_tool(self, tmp_path: Path) -> None:
        from core.memory_manager import MemoryManager
        from core.semantic_retriever import SemanticRetriever
        from core.embedding_store import EmbeddingStore
        from core.memory_retriever import MemoryRetriever

        mm = MemoryManager(
            conversations_file=tmp_path / "conv.json",
            profile_file=tmp_path / "profile.json",
            learning_state_file=tmp_path / "ls.json",
        )
        mm.add_memory("What is attention?", "Attention is a mechanism in transformers.")
        embed_store = EmbeddingStore(
            embed_model_name="all-MiniLM-L6-v2",
            embeddings_file=tmp_path / "emb.npy",
            ids_file=tmp_path / "ids.json",
        )
        legacy = MemoryRetriever()
        retriever = SemanticRetriever(embedding_store=embed_store, legacy_retriever=legacy)

        tool = MemorySearchTool(mm, retriever)
        task = Task(title="Recall attention info", description="attention mechanism")
        result = tool.execute(task, {})
        assert result.status in (ExecutionStatus.SUCCESS,)

    def test_memory_write_tool(self, tmp_path: Path) -> None:
        from core.memory_manager import MemoryManager
        mm = MemoryManager(
            conversations_file=tmp_path / "conv.json",
            profile_file=tmp_path / "profile.json",
            learning_state_file=tmp_path / "ls.json",
        )
        tool = MemoryWriteTool(mm)
        task = Task(title="Save fact", description="Important fact to remember.")
        result = tool.execute(task, {})
        assert result.status == ExecutionStatus.SUCCESS
        assert "#" in result.output

    def test_web_search_tool_handles_network_failure_gracefully(self) -> None:
        # api.duckduckgo.com is not in the sandbox's allowed domains list,
        # so this should fail gracefully rather than raising.
        tool = WebSearchTool()
        task = Task(title="Search test", description="latest RAG papers")
        result = tool.execute(task, {})
        # Either succeeds (if network allowed) or fails gracefully
        assert result.status in (ExecutionStatus.SUCCESS, ExecutionStatus.FAILURE)
        assert result.tool_name == "web_search"
        # Should never raise — error captured in result
        if result.status == ExecutionStatus.FAILURE:
            assert result.error != "" or "failed" in result.output.lower()

    def test_file_tool_write_and_read(self, tmp_path: Path) -> None:
        tool = FileTool(tmp_path / "workspace")
        write_task = Task(title="Write file", metadata={"op": "write", "filename": "test.txt", "content": "Hello agent!"})
        write_result = tool.execute(write_task, {})
        assert write_result.status == ExecutionStatus.SUCCESS

        read_task = Task(title="Read file", metadata={"op": "read", "filename": "test.txt"})
        read_result = tool.execute(read_task, {})
        assert "Hello agent!" in read_result.output

    def test_file_tool_read_missing_file(self, tmp_path: Path) -> None:
        tool = FileTool(tmp_path / "workspace")
        task = Task(title="Read missing", metadata={"op": "read", "filename": "missing.txt"})
        result = tool.execute(task, {})
        assert result.status == ExecutionStatus.FAILURE

    def test_file_tool_list(self, tmp_path: Path) -> None:
        tool = FileTool(tmp_path / "workspace")
        tool.execute(Task(title="w", metadata={"op": "write", "filename": "a.txt", "content": "x"}), {})
        result = tool.execute(Task(title="list", metadata={"op": "list"}), {})
        assert "a.txt" in result.output

    def test_file_tool_path_escape_blocked(self, tmp_path: Path) -> None:
        tool = FileTool(tmp_path / "workspace")
        task = Task(title="Escape", metadata={"op": "read", "filename": "../../etc/passwd"})
        result = tool.execute(task, {})
        assert result.status == ExecutionStatus.ERROR

    def test_python_tool_executes_safe_code(self) -> None:
        tool = PythonTool()
        task = Task(title="Compute", metadata={"code": "print(2 + 2)"})
        result = tool.execute(task, {})
        assert result.status == ExecutionStatus.SUCCESS
        assert "4" in result.output

    def test_python_tool_extracts_code_block(self) -> None:
        tool = PythonTool()
        task = Task(title="Compute", description="```python\nprint('from block')\n```")
        result = tool.execute(task, {})
        assert result.status == ExecutionStatus.SUCCESS
        assert "from block" in result.output

    def test_python_tool_handles_errors_gracefully(self) -> None:
        tool = PythonTool()
        task = Task(title="Bad code", metadata={"code": "1 / 0"})
        result = tool.execute(task, {})
        assert result.status == ExecutionStatus.FAILURE
        assert "ZeroDivisionError" in result.error

    def test_python_tool_no_code_provided(self) -> None:
        tool = PythonTool()
        task = Task(title="No code", description="just a description")
        result = tool.execute(task, {})
        assert result.status == ExecutionStatus.ERROR

    def test_python_tool_blocks_unsafe_builtins(self) -> None:
        tool = PythonTool()
        # 'open' and '__import__' should not be in safe builtins
        task = Task(title="Unsafe", metadata={"code": "open('/etc/passwd')"})
        result = tool.execute(task, {})
        assert result.status == ExecutionStatus.FAILURE  # NameError caught

    def test_synthesis_tool(self, tmp_path: Path) -> None:
        from knowledge.synthesis import KnowledgeSynthesisEngine
        engine = KnowledgeSynthesisEngine(tmp_path / "reports.json")
        tool = SynthesisTool(engine)
        task = Task(title="Synthesize findings")
        context = {"finding_1": "Transformers use self-attention mechanisms for sequence processing."}
        result = tool.execute(task, context)
        assert result.status == ExecutionStatus.SUCCESS

    def test_synthesis_tool_no_context(self, tmp_path: Path) -> None:
        from knowledge.synthesis import KnowledgeSynthesisEngine
        engine = KnowledgeSynthesisEngine(tmp_path / "reports.json")
        tool = SynthesisTool(engine)
        task = Task(title="Synthesize")
        result = tool.execute(task, {})
        assert result.status == ExecutionStatus.FAILURE

    def test_reasoning_tool(self, tmp_path: Path) -> None:
        from core.memory_graph import MemoryGraph, EntityKind, RelationKind
        from core.cognitive_query_engine import CognitiveQueryEngine
        g = MemoryGraph(tmp_path / "graph.json")
        g.upsert_relation("Blix", EntityKind.PROJECT, RelationKind.USES, "FastAPI", EntityKind.SKILL)
        cqe = CognitiveQueryEngine(g)
        tool = ReasoningTool(cqe)
        task = Task(title="Query graph", description="What does Blix use?")
        result = tool.execute(task, {})
        assert result.status == ExecutionStatus.SUCCESS
        assert "FastAPI" in result.output

    def test_llm_tool(self) -> None:
        tool = LLMTool(_FakeLLM())
        task = Task(title="Generate text", description="Write something.")
        result = tool.execute(task, {})
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output == "Fake LLM response."

    def test_llm_tool_injects_context(self) -> None:
        class CapturingLLM:
            def __init__(self):
                self.last_prompt = ""
            def generate(self, prompt):
                self.last_prompt = prompt
                return "ok"
        llm = CapturingLLM()
        tool = LLMTool(llm)
        task = Task(title="T", description="Do X")
        tool.execute(task, {"key1": "important context value"})
        assert "important context value" in llm.last_prompt


# ===========================================================================
# Module: planning.planner
# ===========================================================================


class TestGoalParser:
    def test_heuristic_domain_detection_research(self) -> None:
        parser = GoalParser()
        result = parser.parse("Create a research paper on memory systems.")
        assert result.domain == "research"

    def test_heuristic_domain_detection_coding(self) -> None:
        parser = GoalParser()
        result = parser.parse("Build a RAG system using Python.")
        assert result.domain == "coding"

    def test_heuristic_domain_detection_writing(self) -> None:
        parser = GoalParser()
        result = parser.parse("Write an essay about transformers.")
        assert result.domain == "writing"

    def test_heuristic_complexity_simple(self) -> None:
        parser = GoalParser()
        result = parser.parse("Just give me a quick summary.")
        assert result.complexity == "simple"

    def test_heuristic_complexity_complex(self) -> None:
        parser = GoalParser()
        result = parser.parse("Give me a comprehensive and detailed analysis.")
        assert result.complexity == "complex"

    def test_requires_web_detection(self) -> None:
        parser = GoalParser()
        result = parser.parse("Search the web for the latest RAG papers.")
        assert result.requires_web

    def test_requires_code_detection(self) -> None:
        parser = GoalParser()
        result = parser.parse("Implement a Python function for this.")
        assert result.requires_code

    def test_title_extraction(self) -> None:
        parser = GoalParser()
        result = parser.parse("Build a RAG system. It should be fast.")
        assert "Build a RAG system" in result.title

    def test_llm_parse(self) -> None:
        class FakeLLM:
            def generate(self, prompt):
                return json.dumps({
                    "title": "Build RAG",
                    "description": "Build a RAG system",
                    "domain": "coding",
                    "complexity": "complex",
                    "estimated_tasks": 5,
                    "requires_web": False,
                    "requires_code": True,
                    "requires_files": False,
                })
            def model_name(self):
                return "fake"

        parser = GoalParser(llm=FakeLLM())
        result = parser.parse("Build a RAG system")
        assert result.title == "Build RAG"
        assert result.estimated_tasks == 5

    def test_llm_parse_failure_falls_back(self) -> None:
        class BadLLM:
            def generate(self, prompt):
                return "not json"
            def model_name(self):
                return "fake"

        parser = GoalParser(llm=BadLLM())
        result = parser.parse("Build a RAG system with Python.")
        assert result.domain == "coding"  # heuristic fallback worked


class TestTaskDecomposer:
    def test_heuristic_decompose_research(self) -> None:
        parsed = ParsedGoal(raw_input="research goal", title="Research Goal", description="d", domain="research")
        decomposer = TaskDecomposer()
        graph = decomposer.decompose(parsed)
        assert len(graph.tasks) > 0
        assert graph.goal == "research goal"

    def test_heuristic_decompose_coding(self) -> None:
        parsed = ParsedGoal(raw_input="code goal", title="Code Goal", description="d", domain="coding")
        decomposer = TaskDecomposer()
        graph = decomposer.decompose(parsed)
        assert any(t.tool_hint == "python_tool" for t in graph.tasks)

    def test_decompose_creates_valid_dependencies(self) -> None:
        parsed = ParsedGoal(raw_input="g", title="G", description="d", domain="research")
        decomposer = TaskDecomposer()
        graph = decomposer.decompose(parsed)
        all_ids = {t.task_id for t in graph.tasks}
        for task in graph.tasks:
            for dep in task.depends_on:
                assert dep in all_ids

    def test_decompose_unknown_domain_uses_general(self) -> None:
        parsed = ParsedGoal(raw_input="g", title="G", description="d", domain="unknown_domain")
        decomposer = TaskDecomposer()
        graph = decomposer.decompose(parsed)
        assert len(graph.tasks) > 0

    def test_llm_decompose(self) -> None:
        class FakeLLM:
            def generate(self, prompt):
                return json.dumps([
                    {"title": "Step 1", "description": "First step", "depends_on": [], "tool_hint": "web_search"},
                    {"title": "Step 2", "description": "Second step", "depends_on": [0], "tool_hint": "llm"},
                ])
            def model_name(self):
                return "fake"

        parsed = ParsedGoal(raw_input="g", title="G", description="d", domain="research")
        decomposer = TaskDecomposer(llm=FakeLLM())
        graph = decomposer.decompose(parsed)
        assert len(graph.tasks) == 2
        assert graph.tasks[1].depends_on == [graph.tasks[0].task_id]

    def test_llm_decompose_failure_falls_back(self) -> None:
        class BadLLM:
            def generate(self, prompt):
                return "not json"
            def model_name(self):
                return "fake"

        parsed = ParsedGoal(raw_input="g", title="G", description="d", domain="coding")
        decomposer = TaskDecomposer(llm=BadLLM())
        graph = decomposer.decompose(parsed)
        assert len(graph.tasks) > 0  # heuristic fallback


class TestPlanner:
    def test_plan_combines_parser_and_decomposer(self) -> None:
        planner = Planner()
        parsed, graph = planner.plan("Build a RAG system with Python.")
        assert parsed.domain == "coding"
        assert len(graph.tasks) > 0
        assert graph.goal == "Build a RAG system with Python."


class TestMilestoneTracker:
    def test_create_goal_from_graph(self, tmp_path: Path) -> None:
        from reflection.goal_tracker import GoalTracker
        gt = GoalTracker(tmp_path / "goals.json")
        graph = _make_simple_graph(3)
        tracker = MilestoneTracker(gt)
        goal_id = tracker.create_goal_from_graph(graph)
        goal = gt.get(goal_id)
        assert goal is not None
        assert len(goal.milestones) == 3

    def test_sync_marks_completed_milestones(self, tmp_path: Path) -> None:
        from reflection.goal_tracker import GoalTracker, ItemStatus
        gt = GoalTracker(tmp_path / "goals.json")
        graph = _make_simple_graph(2)
        tracker = MilestoneTracker(gt)
        goal_id = tracker.create_goal_from_graph(graph)

        graph.tasks[0].mark_completed("done")
        tracker.sync(goal_id, graph)

        goal = gt.get(goal_id)
        completed_milestones = [m for m in goal.milestones if m.status == ItemStatus.DONE]
        assert len(completed_milestones) == 1

    def test_update_blockers_for_failed_tasks(self, tmp_path: Path) -> None:
        from reflection.goal_tracker import GoalTracker
        gt = GoalTracker(tmp_path / "goals.json")
        graph = _make_simple_graph(2)
        tracker = MilestoneTracker(gt)
        goal_id = tracker.create_goal_from_graph(graph)

        graph.tasks[0].mark_failed("Something went wrong")
        tracker.update_blockers(goal_id, graph)

        goal = gt.get(goal_id)
        assert len(goal.active_blockers) == 1


# ===========================================================================
# Module: agents.working_memory
# ===========================================================================


class TestWorkingMemory:
    def test_set_and_get(self) -> None:
        wm = WorkingMemory()
        wm.set("key1", "value1")
        assert wm.get("key1") == "value1"

    def test_get_missing_returns_default(self) -> None:
        wm = WorkingMemory()
        assert wm.get("missing", "default") == "default"

    def test_has(self) -> None:
        wm = WorkingMemory()
        wm.set("key1", "value1")
        assert wm.has("key1")
        assert not wm.has("missing")

    def test_delete(self) -> None:
        wm = WorkingMemory()
        wm.set("key1", "value1")
        wm.delete("key1")
        assert not wm.has("key1")

    def test_clear(self) -> None:
        wm = WorkingMemory()
        wm.set("key1", "value1")
        wm.clear()
        assert wm.size == 0
        assert wm.step == 0

    def test_ttl_eviction_on_tick(self) -> None:
        wm = WorkingMemory(default_ttl=2)
        wm.set("key1", "value1")
        wm.tick()  # age=1
        assert wm.has("key1")
        wm.tick()  # age=2, ttl=2 → expired
        assert not wm.has("key1")

    def test_max_entries_evicts_oldest(self) -> None:
        wm = WorkingMemory(max_entries=2)
        wm.set("key1", "value1")
        wm.set("key2", "value2")
        wm.set("key3", "value3")  # should evict key1
        assert wm.size <= 2

    def test_task_output_helpers(self) -> None:
        wm = WorkingMemory()
        wm.set_task_output("task1", "output text")
        assert wm.get_task_output("task1") == "output text"

    def test_current_task_helpers(self) -> None:
        wm = WorkingMemory()
        wm.set_current_task("task1", "Title")
        assert wm.get_current_task_id() == "task1"

    def test_snapshot(self) -> None:
        wm = WorkingMemory()
        wm.set("a", 1)
        wm.set("b", 2)
        snap = wm.snapshot()
        assert snap == {"a": 1, "b": 2}

    def test_snapshot_excludes_expired(self) -> None:
        wm = WorkingMemory(default_ttl=1)
        wm.set("a", 1)
        wm.tick()  # age=1, expired
        snap = wm.snapshot()
        assert "a" not in snap

    def test_summary(self) -> None:
        wm = WorkingMemory()
        wm.set("a", 1)
        s = wm.summary()
        assert "1" in s


# ===========================================================================
# Module: agents.observation
# ===========================================================================


class TestObservationLayer:
    def test_observe_success(self) -> None:
        layer = ObservationLayer()
        result = ExecutionResult(
            task_id="t1", tool_name="web_search", status=ExecutionStatus.SUCCESS,
            output="Found relevant results about transformers and attention mechanisms in depth.",
        )
        obs = layer.observe(result)
        assert obs.success
        assert obs.quality_score > 0.3

    def test_observe_failure(self) -> None:
        layer = ObservationLayer()
        result = ExecutionResult(
            task_id="t1", tool_name="web_search", status=ExecutionStatus.ERROR,
            output="", error="Connection timeout",
        )
        obs = layer.observe(result)
        assert not obs.success
        assert obs.quality_score == 0.0
        assert obs.retry_suggested

    def test_observe_extracts_bullet_facts(self) -> None:
        layer = ObservationLayer()
        result = ExecutionResult(
            task_id="t1", tool_name="web_search", status=ExecutionStatus.SUCCESS,
            output="Key findings:\n- Attention improves recall\n- Graphs reduce drift\n- Hierarchies help",
        )
        obs = layer.observe(result)
        assert len(obs.extracted_facts) >= 2

    def test_observe_extracts_sentence_facts_no_bullets(self) -> None:
        layer = ObservationLayer()
        result = ExecutionResult(
            task_id="t1", tool_name="llm", status=ExecutionStatus.SUCCESS,
            output="This is the first important sentence about the topic. This is the second one with more detail.",
        )
        obs = layer.observe(result)
        assert len(obs.extracted_facts) >= 1

    def test_retry_suggested_on_timeout(self) -> None:
        layer = ObservationLayer()
        result = ExecutionResult(
            task_id="t1", tool_name="web_search", status=ExecutionStatus.FAILURE,
            output="", error="Request timeout after 10s",
        )
        obs = layer.observe(result)
        assert obs.retry_suggested
        assert "timeout" in obs.retry_hint.lower() or "shorter" in obs.retry_hint.lower()

    def test_retry_suggested_on_no_results(self) -> None:
        layer = ObservationLayer()
        result = ExecutionResult(
            task_id="t1", tool_name="web_search", status=ExecutionStatus.SUCCESS,
            output="No results found for this query.",
        )
        obs = layer.observe(result)
        assert obs.retry_suggested

    def test_no_retry_on_good_result(self) -> None:
        layer = ObservationLayer()
        result = ExecutionResult(
            task_id="t1", tool_name="web_search", status=ExecutionStatus.SUCCESS,
            output="Found excellent comprehensive results with detailed information about the successful search query and generated content.",
        )
        obs = layer.observe(result)
        assert not obs.retry_suggested

    def test_batch_observe(self) -> None:
        layer = ObservationLayer()
        results = [
            ExecutionResult(task_id="t1", tool_name="x", status=ExecutionStatus.SUCCESS, output="Good result here with enough text."),
            ExecutionResult(task_id="t2", tool_name="x", status=ExecutionStatus.ERROR, output="", error="failed"),
        ]
        observations = layer.batch_observe(results)
        assert len(observations) == 2
        assert observations[0].success
        assert not observations[1].success

    def test_summary_includes_tool_name(self) -> None:
        layer = ObservationLayer()
        result = ExecutionResult(
            task_id="t1", tool_name="my_special_tool", status=ExecutionStatus.SUCCESS,
            output="Some output text here.",
        )
        obs = layer.observe(result)
        assert "my_special_tool" in obs.summary


# ===========================================================================
# Module: agents.reflection_loop
# ===========================================================================


class TestReflectionLoop:
    @pytest.fixture
    def loop(self, tmp_path: Path) -> ReflectionLoop:
        return ReflectionLoop(tmp_path / "history.json", max_retries=2)

    def test_accept_high_quality_success(self, loop: ReflectionLoop) -> None:
        task = Task(title="T")
        obs = Observation(task_id="t1", tool_name="x", success=True, quality_score=0.8)
        decision = loop.reflect(task, obs, goal="test goal")
        assert decision.action == "accept"

    def test_retry_low_quality_with_attempts_left(self, loop: ReflectionLoop) -> None:
        task = Task(title="T")
        task.attempts = 1
        obs = Observation(task_id="t1", tool_name="x", success=False, quality_score=0.1,
                          retry_suggested=True, retry_hint="try again")
        decision = loop.reflect(task, obs, goal="test goal")
        assert decision.action == "retry"
        assert decision.retry_hint == "try again"

    def test_skip_after_max_retries(self, loop: ReflectionLoop) -> None:
        task = Task(title="T")
        task.attempts = 3  # exceeds max_retries=2
        obs = Observation(task_id="t1", tool_name="x", success=False, quality_score=0.1,
                          retry_suggested=True)
        decision = loop.reflect(task, obs, goal="test goal")
        assert decision.action == "skip"

    def test_history_persisted(self, loop: ReflectionLoop) -> None:
        task = Task(title="T")
        obs = Observation(task_id="t1", tool_name="x", success=True, quality_score=0.8)
        loop.reflect(task, obs, goal="test goal")
        assert loop.history_count == 1

    def test_history_persistence_roundtrip(self, tmp_path: Path) -> None:
        loop1 = ReflectionLoop(tmp_path / "h.json")
        task = Task(title="T")
        obs = Observation(task_id="t1", tool_name="x", success=True, quality_score=0.8)
        loop1.reflect(task, obs, goal="g")

        loop2 = ReflectionLoop(tmp_path / "h.json")
        assert loop2.history_count == 1

    def test_success_rate(self, loop: ReflectionLoop) -> None:
        task = Task(title="T")
        obs_good = Observation(task_id="t1", tool_name="x", success=True, quality_score=0.8)
        obs_bad = Observation(task_id="t2", tool_name="x", success=False, quality_score=0.1)
        loop.reflect(task, obs_good, goal="g")
        loop.reflect(task, obs_bad, goal="g")
        assert loop.success_rate() == 0.5

    def test_mean_quality(self, loop: ReflectionLoop) -> None:
        task = Task(title="T")
        obs1 = Observation(task_id="t1", tool_name="x", success=True, quality_score=0.8)
        obs2 = Observation(task_id="t2", tool_name="x", success=True, quality_score=0.4)
        loop.reflect(task, obs1, goal="g")
        loop.reflect(task, obs2, goal="g")
        assert loop.mean_quality() == pytest.approx(0.6)

    def test_get_history_filter_by_goal(self, loop: ReflectionLoop) -> None:
        task = Task(title="T")
        obs = Observation(task_id="t1", tool_name="x", success=True, quality_score=0.8)
        loop.reflect(task, obs, goal="goal alpha")
        loop.reflect(task, obs, goal="goal beta")
        results = loop.get_history(goal="alpha")
        assert len(results) == 1

    def test_update_reflection_engine_called(self, tmp_path: Path) -> None:
        from reflection.reflection_engine import ReflectionEngine
        re_engine = ReflectionEngine(tmp_path / "reflections.json")
        loop = ReflectionLoop(tmp_path / "history.json", reflection_engine=re_engine)
        task = Task(title="T")
        obs = Observation(task_id="t1", tool_name="x", success=True, quality_score=0.8)
        loop.reflect(task, obs, goal="g")
        assert re_engine.record_count >= 1

    def test_update_memory_on_high_quality_accept(self, tmp_path: Path) -> None:
        from core.memory_manager import MemoryManager
        mm = MemoryManager(
            conversations_file=tmp_path / "conv.json",
            profile_file=tmp_path / "profile.json",
            learning_state_file=tmp_path / "ls.json",
        )
        loop = ReflectionLoop(tmp_path / "history.json", memory_manager=mm)
        task = Task(title="T")
        obs = Observation(task_id="t1", tool_name="x", success=True, quality_score=0.9,
                          summary="High quality output worth remembering for later.")
        loop.reflect(task, obs, goal="g")
        assert mm.memory_count() >= 1

    def test_llm_note_generation(self, tmp_path: Path) -> None:
        loop = ReflectionLoop(tmp_path / "history.json", llm=_FakeLLM())
        task = Task(title="T")
        obs = Observation(task_id="t1", tool_name="x", success=True, quality_score=0.8)
        decision = loop.reflect(task, obs, goal="g")
        assert decision.note == "Fake LLM response."

    def test_reflection_decision_helpers(self) -> None:
        d1 = ReflectionDecision(action="retry")
        assert d1.should_retry()
        assert not d1.should_skip()
        d2 = ReflectionDecision(action="skip")
        assert d2.should_skip()


# ===========================================================================
# Module: agents.executor
# ===========================================================================


class TestAgentExecutor:
    def _build_executor(self, tmp_path: Path, tools: list[Tool]) -> AgentExecutor:
        registry = ToolRegistry(tools)
        wm = WorkingMemory()
        obs_layer = ObservationLayer()
        reflect_loop = ReflectionLoop(tmp_path / "history.json", max_retries=2)
        return AgentExecutor(
            tool_registry=registry, working_memory=wm,
            observation_layer=obs_layer, reflection_loop=reflect_loop,
            config=ExecutorConfig(max_steps=20),
        )

    def test_run_completes_all_tasks(self, tmp_path: Path) -> None:
        executor = self._build_executor(tmp_path, [_FakeTool("fake_tool")])
        graph = _make_simple_graph(3)
        result = executor.run(graph)
        assert result.completed_tasks == 3
        assert result.success
        assert graph.is_complete

    def test_run_respects_dependencies(self, tmp_path: Path) -> None:
        executor = self._build_executor(tmp_path, [_FakeTool("fake_tool")])
        graph = _make_simple_graph(3, linear=True)
        result = executor.run(graph)
        # Tasks should complete in dependency order
        completed_order = [h["task_id"] for h in result.history if h["decision"] == "accept"]
        task_ids_in_order = [t.task_id for t in graph.tasks]
        assert completed_order == task_ids_in_order

    def test_run_no_tool_found_skips_task(self, tmp_path: Path) -> None:
        executor = self._build_executor(tmp_path, [])  # no tools registered
        graph = _make_simple_graph(1)
        result = executor.run(graph)
        assert result.skipped_tasks == 1

    def test_run_retries_on_failure_then_succeeds(self, tmp_path: Path) -> None:
        fail_then_succeed = _FakeTool("fake_tool", outputs=[
            ExecutionResult(task_id="", tool_name="fake_tool", status=ExecutionStatus.FAILURE,
                           output="", error="timeout occurred"),
            ExecutionResult(task_id="", tool_name="fake_tool", status=ExecutionStatus.SUCCESS,
                           output="Now it succeeded with plenty of good detailed output content."),
        ])
        executor = self._build_executor(tmp_path, [fail_then_succeed])
        graph = _make_simple_graph(1)
        result = executor.run(graph)
        assert result.completed_tasks == 1
        # Should show a retry then accept in history
        decisions = [h["decision"] for h in result.history]
        assert "retry" in decisions
        assert "accept" in decisions

    def test_run_skips_after_max_retries_exhausted(self, tmp_path: Path) -> None:
        always_fail = _FakeTool("fake_tool", outputs=[
            ExecutionResult(task_id="", tool_name="fake_tool", status=ExecutionStatus.ERROR,
                           output="", error="persistent timeout"),
        ] * 5)
        executor = self._build_executor(tmp_path, [always_fail])
        graph = _make_simple_graph(1)
        result = executor.run(graph)
        assert result.failed_tasks == 1
        assert not result.success

    def test_run_respects_max_steps(self, tmp_path: Path) -> None:
        always_fail = _FakeTool("fake_tool", outputs=[
            ExecutionResult(task_id="", tool_name="fake_tool", status=ExecutionStatus.ERROR,
                           output="", error="timeout"),
        ] * 100)
        registry = ToolRegistry([always_fail])
        wm = WorkingMemory()
        obs_layer = ObservationLayer()
        reflect_loop = ReflectionLoop(tmp_path / "history.json", max_retries=100)
        executor = AgentExecutor(
            tool_registry=registry, working_memory=wm,
            observation_layer=obs_layer, reflection_loop=reflect_loop,
            config=ExecutorConfig(max_steps=5),
        )
        graph = _make_simple_graph(1)
        result = executor.run(graph)
        assert result.total_steps <= 5

    def test_final_output_assembled_from_completed_tasks(self, tmp_path: Path) -> None:
        executor = self._build_executor(tmp_path, [_FakeTool("fake_tool")])
        graph = _make_simple_graph(2)
        result = executor.run(graph)
        assert "Task 0" in result.final_output
        assert "Task 1" in result.final_output

    def test_working_memory_stores_task_outputs(self, tmp_path: Path) -> None:
        registry = ToolRegistry([_FakeTool("fake_tool")])
        wm = WorkingMemory()
        obs_layer = ObservationLayer()
        reflect_loop = ReflectionLoop(tmp_path / "history.json")
        executor = AgentExecutor(
            tool_registry=registry, working_memory=wm,
            observation_layer=obs_layer, reflection_loop=reflect_loop,
        )
        graph = _make_simple_graph(1)
        executor.run(graph)
        task_id = graph.tasks[0].task_id
        assert wm.get_task_output(task_id) is not None

    def test_milestone_tracker_synced_during_run(self, tmp_path: Path) -> None:
        from reflection.goal_tracker import GoalTracker
        from planning.planner import MilestoneTracker

        gt = GoalTracker(tmp_path / "goals.json")
        graph = _make_simple_graph(2)
        tracker = MilestoneTracker(gt)
        goal_id = tracker.create_goal_from_graph(graph)

        registry = ToolRegistry([_FakeTool("fake_tool")])
        wm = WorkingMemory()
        obs_layer = ObservationLayer()
        reflect_loop = ReflectionLoop(tmp_path / "history.json")
        executor = AgentExecutor(
            tool_registry=registry, working_memory=wm,
            observation_layer=obs_layer, reflection_loop=reflect_loop,
            milestone_tracker=tracker,
        )
        executor.run(graph, goal_id=goal_id)

        from reflection.goal_tracker import ItemStatus
        goal = gt.get(goal_id)
        completed = [m for m in goal.milestones if m.status == ItemStatus.DONE]
        assert len(completed) == 2

    def test_to_dict(self, tmp_path: Path) -> None:
        executor = self._build_executor(tmp_path, [_FakeTool("fake_tool")])
        graph = _make_simple_graph(1)
        result = executor.run(graph)
        d = result.to_dict()
        assert "goal" in d
        assert "task_summary" in d


class TestAgentSession:
    def test_run_plans_and_executes(self, tmp_path: Path) -> None:
        registry = ToolRegistry([_FakeTool("fake_tool"), _FakeTool("web_search"),
                                  _FakeTool("llm"), _FakeTool("synthesis")])
        wm = WorkingMemory()
        obs_layer = ObservationLayer()
        reflect_loop = ReflectionLoop(tmp_path / "history.json")
        executor = AgentExecutor(
            tool_registry=registry, working_memory=wm,
            observation_layer=obs_layer, reflection_loop=reflect_loop,
        )
        planner = Planner()
        session = AgentSession(planner=planner, executor=executor)
        result = session.run("Build a Python script for data analysis.")
        assert isinstance(result, AgentRunResult)
        assert session.session_count == 1

    def test_session_with_goal_tracker_integration(self, tmp_path: Path) -> None:
        from reflection.goal_tracker import GoalTracker
        gt = GoalTracker(tmp_path / "goals.json")
        registry = ToolRegistry([_FakeTool("fake_tool")])
        wm = WorkingMemory()
        obs_layer = ObservationLayer()
        reflect_loop = ReflectionLoop(tmp_path / "history.json")
        executor = AgentExecutor(
            tool_registry=registry, working_memory=wm,
            observation_layer=obs_layer, reflection_loop=reflect_loop,
        )
        planner = Planner()
        session = AgentSession(planner=planner, executor=executor, goal_tracker=gt)
        session.run("Write a summary about transformers.")
        assert gt.count >= 1

    def test_recent_sessions(self, tmp_path: Path) -> None:
        registry = ToolRegistry([_FakeTool("fake_tool")])
        wm = WorkingMemory()
        obs_layer = ObservationLayer()
        reflect_loop = ReflectionLoop(tmp_path / "history.json")
        executor = AgentExecutor(
            tool_registry=registry, working_memory=wm,
            observation_layer=obs_layer, reflection_loop=reflect_loop,
        )
        planner = Planner()
        session = AgentSession(planner=planner, executor=executor)
        session.run("Test goal 1")
        session.run("Test goal 2")
        recent = session.recent_sessions(1)
        assert len(recent) == 1


# ===========================================================================
# Module: evaluation.agent_eval
# ===========================================================================


class TestAgentEvaluator:
    def test_task_success_rate(self) -> None:
        graph = _make_simple_graph(2)
        graph.tasks[0].mark_completed("ok")
        graph.tasks[1].mark_failed("err")
        ev = AgentEvaluator()
        rate = ev.task_success_rate(graph)
        assert rate == 0.5

    def test_task_success_rate_no_terminal_tasks(self) -> None:
        graph = _make_simple_graph(2)  # all pending
        ev = AgentEvaluator()
        assert ev.task_success_rate(graph) == 0.0

    def test_run_success_rate(self) -> None:
        graph1 = TaskGraph(goal="g1")
        graph2 = TaskGraph(goal="g2")
        r1 = AgentRunResult(goal="g1", graph=graph1, success=True)
        r2 = AgentRunResult(goal="g2", graph=graph2, success=False)
        ev = AgentEvaluator()
        assert ev.run_success_rate([r1, r2]) == 0.5

    def test_run_success_rate_empty(self) -> None:
        ev = AgentEvaluator()
        assert ev.run_success_rate([]) == 0.0

    def test_tool_accuracy(self) -> None:
        graph = _make_simple_graph(2)
        graph.tasks[0].metadata["last_tool"] = "web_search"
        graph.tasks[1].metadata["last_tool"] = "python_tool"
        ev = AgentEvaluator()
        expected = {"Task 0": "web_search", "Task 1": "llm"}  # second is wrong
        acc = ev.tool_accuracy(graph, expected)
        assert acc == 0.5

    def test_tool_accuracy_empty_expected(self) -> None:
        graph = _make_simple_graph(1)
        ev = AgentEvaluator()
        assert ev.tool_accuracy(graph, {}) == 1.0

    def test_planning_accuracy_task_count_in_range(self) -> None:
        graph = _make_simple_graph(4)
        ev = AgentEvaluator()
        acc = ev.planning_accuracy(graph, expected_task_count=(3, 5))
        assert acc == 1.0

    def test_planning_accuracy_task_count_out_of_range(self) -> None:
        graph = _make_simple_graph(10)
        ev = AgentEvaluator()
        acc = ev.planning_accuracy(graph, expected_task_count=(3, 5))
        assert acc == 0.0

    def test_planning_accuracy_domain_match(self) -> None:
        graph = _make_simple_graph(3)
        ev = AgentEvaluator()
        acc = ev.planning_accuracy(graph, expected_domain="coding", actual_domain="coding")
        assert acc == 1.0

    def test_planning_accuracy_combined(self) -> None:
        graph = _make_simple_graph(4)
        ev = AgentEvaluator()
        acc = ev.planning_accuracy(
            graph, expected_task_count=(3, 5),
            expected_domain="coding", actual_domain="research",
        )
        # task count matches (weight 0.6), domain doesn't (weight 0.4) → 0.6
        assert acc == pytest.approx(0.6)

    def test_execution_cost(self) -> None:
        history = [
            {"decision": "accept", "tool": "web_search"},
            {"decision": "retry", "tool": "web_search"},
            {"decision": "accept", "tool": "llm"},
        ]
        ev = AgentEvaluator()
        cost = ev.execution_cost(history)
        assert cost["total_steps"] == 3
        assert cost["retries"] == 1
        assert cost["tool_calls"] == 3

    def test_execution_time(self) -> None:
        graph = TaskGraph(goal="g")
        result = AgentRunResult(goal="g", graph=graph, duration_secs=5.5)
        ev = AgentEvaluator()
        assert ev.execution_time(result) == 5.5

    def test_mean_execution_time(self) -> None:
        graph = TaskGraph(goal="g")
        r1 = AgentRunResult(goal="g", graph=graph, duration_secs=2.0)
        r2 = AgentRunResult(goal="g", graph=graph, duration_secs=4.0)
        ev = AgentEvaluator()
        assert ev.mean_execution_time([r1, r2]) == 3.0

    def test_reflection_gain_positive(self) -> None:
        history = [
            {"task_id": "t1", "quality": 0.2, "decision": "retry"},
            {"task_id": "t1", "quality": 0.8, "decision": "accept"},
        ]
        ev = AgentEvaluator()
        gain = ev.reflection_gain(history)
        assert gain == pytest.approx(0.6)

    def test_reflection_gain_no_retries(self) -> None:
        history = [{"task_id": "t1", "quality": 0.8, "decision": "accept"}]
        ev = AgentEvaluator()
        assert ev.reflection_gain(history) == 0.0

    def test_retry_effectiveness_success(self) -> None:
        history = [
            {"task_id": "t1", "decision": "retry"},
            {"task_id": "t1", "decision": "accept"},
        ]
        ev = AgentEvaluator()
        assert ev.retry_effectiveness(history) == 1.0

    def test_retry_effectiveness_failure(self) -> None:
        history = [
            {"task_id": "t1", "decision": "retry"},
            {"task_id": "t1", "decision": "skip"},
        ]
        ev = AgentEvaluator()
        assert ev.retry_effectiveness(history) == 0.0

    def test_retry_effectiveness_no_retries_vacuous(self) -> None:
        history = [{"task_id": "t1", "decision": "accept"}]
        ev = AgentEvaluator()
        assert ev.retry_effectiveness(history) == 1.0

    def test_evaluate_agent_run_combines_metrics(self) -> None:
        graph = _make_simple_graph(2)
        graph.tasks[0].mark_completed("ok")
        graph.tasks[1].mark_completed("ok")
        result = AgentRunResult(goal="g", graph=graph, duration_secs=1.5,
                                history=[{"task_id": "t1", "quality": 0.8, "decision": "accept", "tool": "x"}])
        ev = AgentEvaluator()
        metrics = ev.evaluate_agent_run(result)
        assert "task_success_rate" in metrics
        assert "execution_time_secs" in metrics
        assert "reflection_gain" in metrics

    def test_evaluate_agent_run_with_case(self) -> None:
        graph = _make_simple_graph(3)
        result = AgentRunResult(goal="g", graph=graph)
        case = AgentEvalCase(case_id="c1", expected_task_count=(2, 4), expected_domain="coding")
        ev = AgentEvaluator()
        metrics = ev.evaluate_agent_run(result, case=case, actual_domain="coding")
        assert "planning_accuracy" in metrics
        assert metrics["planning_accuracy"] == 1.0

    def test_evaluate_agent_batch(self) -> None:
        graph = _make_simple_graph(2)
        graph.tasks[0].mark_completed("ok")
        graph.tasks[1].mark_completed("ok")
        r1 = AgentRunResult(goal="g1", graph=graph, success=True, duration_secs=1.0)
        r2 = AgentRunResult(goal="g2", graph=graph, success=True, duration_secs=2.0)
        ev = AgentEvaluator()
        batch_metrics = ev.evaluate_agent_batch([r1, r2])
        assert batch_metrics["run_success_rate"] == 1.0
        assert batch_metrics["mean_execution_time_secs"] == 1.5

    def test_evaluate_agent_batch_empty(self) -> None:
        ev = AgentEvaluator()
        assert ev.evaluate_agent_batch([]) == {}

    def test_agent_evaluator_inherits_reasoning_metrics(self) -> None:
        ev = AgentEvaluator()
        # Should have access to v0.3.4 ReasoningEvaluator methods
        assert ev.reasoning_accuracy(["A"], ["A"]) == 1.0

    def test_agent_evaluator_in_blix_eval(self) -> None:
        from evaluation.blix_eval import AgentEvaluator as AE_from_blix
        assert AgentEvaluator is AE_from_blix


# ===========================================================================
# API — /agent endpoints
# ===========================================================================


class _FakeLLMFull:
    def model_name(self) -> str:
        return "fake-0.3.5"

    def generate(self, prompt: str) -> str:
        return "Fake agent LLM reply."


@pytest.fixture(scope="module")
def tmp_memory_v5(tmp_path_factory):
    return tmp_path_factory.mktemp("memory_v5")


@pytest.fixture(scope="module")
def ctx_v5(tmp_memory_v5):
    from config import settings as _settings
    _settings.settings.memory.conversations_file = tmp_memory_v5 / "conversations.json"
    _settings.settings.memory.profile_file = tmp_memory_v5 / "profile.json"
    _settings.settings.memory.learning_state_file = tmp_memory_v5 / "learning_state.json"
    _settings.settings.embed.embeddings_file = tmp_memory_v5 / "embeddings.npy"
    _settings.settings.embed.embedding_ids_file = tmp_memory_v5 / "embedding_ids.json"

    from api.context import BlixContext
    ctx = BlixContext(tmp_memory_v5)
    ctx.llm = _FakeLLMFull()
    ctx.agent._llm = _FakeLLMFull()
    return ctx


@pytest.fixture(scope="module")
def client_v5(ctx_v5) -> Generator[TestClient, None, None]:
    from api.deps import set_context
    from api.routers.agent import router as agent_router

    app = FastAPI(title="Blix Test v0.3.5")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(agent_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    set_context(ctx_v5)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestAgentAPI:
    def test_plan_endpoint(self, client_v5: TestClient) -> None:
        r = client_v5.post("/agent/plan", json={"goal": "Write a summary about transformers."})
        assert r.status_code == 200
        data = r.json()
        assert "parsed_goal" in data
        assert "task_graph" in data
        assert len(data["task_graph"]["tasks"]) > 0

    def test_plan_endpoint_empty_goal_rejected(self, client_v5: TestClient) -> None:
        r = client_v5.post("/agent/plan", json={"goal": ""})
        assert r.status_code == 422

    def test_run_endpoint(self, client_v5: TestClient) -> None:
        r = client_v5.post("/agent/run", json={"goal": "Save a note about today's progress."})
        assert r.status_code == 200
        data = r.json()
        assert "goal" in data
        assert "task_summary" in data
        assert "history" in data

    def test_history_endpoint(self, client_v5: TestClient) -> None:
        # Ensure at least one run happened
        client_v5.post("/agent/run", json={"goal": "Quick test task."})
        r = client_v5.get("/agent/history")
        assert r.status_code == 200
        data = r.json()
        assert "entries" in data
        assert "success_rate" in data

    def test_history_filter_by_goal(self, client_v5: TestClient) -> None:
        client_v5.post("/agent/run", json={"goal": "Unique searchable goal xyz123."})
        r = client_v5.get("/agent/history?goal=xyz123")
        assert r.status_code == 200
        data = r.json()
        assert all("xyz123" in e["goal"].lower() for e in data["entries"]) or data["total"] == 0

    def test_sessions_endpoint(self, client_v5: TestClient) -> None:
        client_v5.post("/agent/run", json={"goal": "Another test goal."})
        r = client_v5.get("/agent/sessions?limit=3")
        assert r.status_code == 200
        data = r.json()
        assert "sessions" in data
        assert "total_sessions" in data
        assert data["total_sessions"] >= 1

    def test_tools_endpoint(self, client_v5: TestClient) -> None:
        r = client_v5.get("/agent/tools")
        assert r.status_code == 200
        data = r.json()
        assert "tools" in data
        tool_names = {t["name"] for t in data["tools"]}
        assert "web_search" in tool_names
        assert "python_tool" in tool_names
        assert "memory_search" in tool_names

    def test_run_goal_too_long_rejected(self, client_v5: TestClient) -> None:
        r = client_v5.post("/agent/run", json={"goal": "x" * 3000})
        assert r.status_code == 422


class TestBlixContextAgentWiring:
    def test_agent_components_present(self, ctx_v5) -> None:
        assert ctx_v5.tool_registry is not None
        assert ctx_v5.planner is not None
        assert ctx_v5.agent_executor is not None
        assert ctx_v5.agent_session is not None
        assert ctx_v5.agent_evaluator is not None
        assert ctx_v5.reflection_loop is not None

    def test_dashboard_stats_includes_agent_metrics(self, ctx_v5) -> None:
        stats = ctx_v5.dashboard_stats()
        assert "agent_sessions" in stats
        assert "execution_history_count" in stats
        assert "agent_success_rate" in stats

    def test_tool_registry_has_all_tools(self, ctx_v5) -> None:
        names = {t.name for t in ctx_v5.tool_registry.list_tools()}
        expected = {"memory_search", "memory_write", "web_search", "file_tool",
                   "python_tool", "synthesis", "reasoning", "llm"}
        assert expected.issubset(names)
