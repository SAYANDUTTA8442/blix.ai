# Blix v0.3.5 — Agent Execution Framework

> Upgrade from v0.3.4. The biggest architectural jump in the Blix series:
> Blix transforms from a cognitive **reasoning** system into an
> action-oriented **agent** capable of planning, executing, observing,
> reflecting on, and learning from tasks.
>
> This closes the cognitive loop:
>
> ```
> Goal → Plan → Act → Observe → Reflect → Learn
> ```
>
> No breaking changes. All new modules are additive and follow the
> established dependency-injection pattern.

---

## Mission

```
v0.3.4 (Reasoning Layer):
    User Goal → Memory → Knowledge Graph → Reasoning → Answer
    Blix can explain how to build a RAG system, but cannot build it.

v0.3.5 (Execution Layer):
    User Goal → Planner → Task Graph → Agent Executor → Tool Calls
              → Observation → Reflection → Memory Update
    Blix can now work on goals.
```

---

## New Packages

```
planning/
└── planner.py           Modules 1+2 — GoalParser, TaskDecomposer, Planner, MilestoneTracker

agents/
├── types.py              Shared types — Task, TaskGraph, ExecutionResult, Observation, ...
├── working_memory.py      Module 8  — WorkingMemory (TTL-based short-term store)
├── observation.py         Module 6  — ObservationLayer
├── reflection_loop.py     Module 7  — ReflectionLoop (accept/retry/skip)
└── executor.py            Module 3  — AgentExecutor, AgentSession (the closed loop)

tools/
└── registry.py            Modules 4+5 — Tool, ToolRegistry, all concrete tools

evaluation/
└── agent_eval.py          Module 10 — AgentEvaluator

api/routers/
└── agent.py               API — /agent/run, /agent/plan, /agent/history, /agent/sessions, /agent/tools
```

---

## Module 1 — Planner (`planning/planner.py`)

```python
from planning.planner import Planner

planner = Planner(llm=llm)
parsed_goal, task_graph = planner.plan("Create a research paper on memory systems.")

parsed_goal.domain        # "research"
parsed_goal.complexity    # "medium"
task_graph.tasks          # [Literature Review, Related Work, Method Design, Evaluation, Paper Draft]
```

`GoalParser` extracts structured intent (domain, complexity, requires_web/code/files)
via heuristic regex pattern matching or LLM. `TaskDecomposer` converts the parsed
goal into a `TaskGraph` using domain-specific templates (research, coding, writing,
analysis, general) or LLM-based decomposition.

---

## Module 2 — Task Decomposer

```python
{
  "goal": "Build RAG System",
  "subtasks": [
    {"title": "Select Embedding Model", "tool_hint": "llm", "depends_on": []},
    {"title": "Build Vector Store", "tool_hint": "python_tool", "depends_on": [0]},
    {"title": "Create Retriever", "tool_hint": "python_tool", "depends_on": [1]},
    {"title": "Connect LLM", "tool_hint": "llm", "depends_on": [2]},
    {"title": "Evaluate", "tool_hint": "synthesis", "depends_on": [3]}
  ]
}
```

This becomes the `TaskGraph` execution DAG, with `depends_on` enforcing topological
execution order via `TaskGraph.ready_tasks()` / `next_task()`.

---

## Module 3 — Agent Executor (`agents/executor.py`)

**The biggest architectural jump.** `AgentExecutor.run()` implements the full loop:

```python
while not graph.is_complete:
    task = graph.next_task()              # THINK: pick ready task
    tool = registry.select_tool(task)     # THINK: choose tool
    result = tool.execute(task, context)  # EXECUTE
    observation = obs_layer.observe(result)  # OBSERVE
    decision = reflect_loop.reflect(task, observation)  # REFLECT
    # accept → mark COMPLETED; retry → re-queue with hint; skip → mark FAILED
    working_memory.tick()                 # UPDATE
```

`AgentSession` is the high-level facade combining `Planner` + `AgentExecutor`:

```python
from agents.executor import AgentSession

session = AgentSession(planner=planner, executor=executor, goal_tracker=goal_tracker)
result = session.run("Analyze latest RAG research.")

result.completed_tasks   # 4
result.success           # True
result.final_output      # assembled markdown summary
result.history           # step-by-step decision log
```

---

## Module 4 — Tool Registry (`tools/registry.py`)

```python
from tools.registry import ToolRegistry, WebSearchTool, PythonTool, FileTool, ...

registry = ToolRegistry([
    MemorySearchTool(mm, retriever),
    MemoryWriteTool(mm),
    WebSearchTool(),           # DuckDuckGo Instant Answer API, no key required
    FileTool(workspace_dir),   # sandboxed read/write/list
    PythonTool(),              # sandboxed exec() with whitelisted builtins
    SynthesisTool(synthesis_engine),
    ReasoningTool(cognitive_query_engine),
    LLMTool(llm),
])
```

Every tool implements:

```python
class Tool(ABC):
    name: str
    description: str
    requires_confirmation: bool = False
    def can_handle(self, task: Task) -> float: ...   # 0-1 confidence score
    def execute(self, task: Task, context: dict) -> ExecutionResult: ...
```

`PythonTool` sandboxes execution: whitelisted safe builtins only (no `open`,
`__import__`, `exec`, `eval` available inside the snippet), 5-second conceptual
timeout, stdout capture. `FileTool` enforces workspace path containment (blocks
`../` escapes). `WebSearchTool` degrades gracefully on network failure (sandboxed
environments without internet egress get a clean `FAILURE` result, not a crash).

---

## Module 5 — Tool Selection Engine

```python
tool = registry.select_tool(task)
```

Priority:
1. **Explicit hint** — `task.tool_hint` set by the planner (e.g. `"web_search"`)
2. **Confidence ranking** — `can_handle(task)` scored across all tools, highest wins
3. **None** if no tool scores above 0.1 → task is skipped, not silently misrouted

```python
# "Latest RAG papers?"      → web_search (can_handle: 0.85)
# "Summarize PDF"           → file_tool / document processing
# "Plot benchmark results"  → python_tool (can_handle: 0.9)
```

`registry.rank_tools(task)` exposes the full ranked list for debugging/explainability.

---

## Module 6 — Observation Layer (`agents/observation.py`)

```python
observation = obs_layer.observe(execution_result)

observation.success           # bool
observation.summary           # "[web_search] Found relevant results... (842 chars total)"
observation.extracted_facts   # bullet-point or sentence-extracted facts
observation.quality_score     # 0-1 heuristic (length + keyword signals)
observation.retry_suggested   # True if timeout/no-results/rate-limit patterns matched
observation.retry_hint        # "Retry with a broader or different query."
```

Failure example from the spec:

```python
ExecutionResult(status=ExecutionStatus.ERROR, error="timeout")
→ Observation(success=False, quality_score=0.0, retry_suggested=True,
              retry_hint="Retry with a shorter timeout or simpler query.")
```

---

## Module 7 — Reflection Loop (`agents/reflection_loop.py`)

```python
decision = reflect_loop.reflect(task, observation, goal="Build RAG system")

decision.action       # "accept" | "retry" | "skip"
decision.retry_hint    # passed back to executor for the next attempt
decision.note          # "Task output was low quality. Retrying: broader query."
```

Example from the spec — search query returned poor results:

```
Action  → web_search("RAG")
Result  → 0 results
Evaluation → quality_score=0.1, retry_suggested=True
Improvement → retry_hint="Retry with a broader or different query."
Next attempt → web_search("retrieval augmented generation 2026") → succeeds
```

Every reflection is persisted to `execution_history.json` (Module 9) and feeds
into the v0.3.2 `ReflectionEngine` as a session-scope insight. High-quality
accepted results (`quality ≥ 0.5`) are also written to long-term memory.

---

## Module 8 — Working Memory (`agents/working_memory.py`)

```python
from agents.working_memory import WorkingMemory

wm = WorkingMemory(max_entries=50, default_ttl=20)
wm.set("search_results", results, ttl=10)
wm.set_task_output(task_id, output)
wm.tick()    # advance step counter, evict expired entries
context = wm.snapshot()   # flat dict injected into tool.execute(task, context)
```

TTL-based eviction (steps, not wall-clock) keeps the architecture deterministic
and testable. This mirrors the working-memory pattern in OpenAI Deep Research /
Claude Research / Manus-style architectures, scoped per-execution and discarded
when the agent finishes (long-term facts persist via `MemoryWriteTool` or
`ReflectionLoop.update_memory()` instead).

---

## Module 9 — Execution History

Every action becomes a persisted record:

```json
{
  "goal": "Analyze latest RAG research",
  "task_id": "a3f9c1d2",
  "task_title": "Search for relevant information",
  "tool": "web_search",
  "result_summary": "[web_search] Found relevant results about RAG architectures...",
  "success": true,
  "quality_score": 0.82,
  "reflection_note": "Task 'Search for relevant information' completed via web_search with high quality (0.82).",
  "executed_at": "2026-06-17T..."
}
```

Stored in `memory/execution_history.json` (capped at the last 500 entries),
queryable via `ReflectionLoop.get_history(goal=..., limit=...)` and the
`GET /agent/history` API endpoint. This is the substrate for "learn from
previous executions" in future versions.

---

## Module 10 — Agent Evaluation (`evaluation/agent_eval.py`)

```python
from blix_eval import AgentEvaluator, AgentEvalCase

ev = AgentEvaluator()   # extends ReasoningEvaluator (v0.3.4) extends ... (full tower)

metrics = ev.evaluate_agent_run(result, case=AgentEvalCase(
    expected_task_count=(3, 6), expected_domain="research",
))
```

| Metric | Method |
|---|---|
| Task Success Rate | `task_success_rate(graph)` |
| Tool Accuracy | `tool_accuracy(graph, expected_tool_for_task)` |
| Planning Accuracy | `planning_accuracy(graph, expected_task_count, expected_domain)` |
| Execution Cost | `execution_cost(history)` → steps, tool_calls, retries |
| Execution Time | `execution_time(result)` / `mean_execution_time(results)` |
| Reflection Gain | `reflection_gain(history)` → quality delta from first→final attempt |
| Retry Effectiveness | `retry_effectiveness(history)` → fraction of retries that succeeded |

`AgentEvaluator` extends `ReasoningEvaluator` (v0.3.4), so the **full evaluation
tower** — memory, knowledge, graph reasoning, and now agent execution — is
available from one class, re-exported through `blix_eval`.

---

## API Endpoints (v0.3.5 additions)

| Method | Path | Description |
|---|---|---|
| POST | `/agent/run` | Plan + execute a goal end-to-end |
| POST | `/agent/plan` | Preview the TaskGraph without executing |
| GET | `/agent/history` | Recent execution history entries |
| GET | `/agent/sessions` | Recent agent run summaries |
| GET | `/agent/tools` | List registered tools and their schemas |

### Example: `/agent/run`

```json
POST /agent/run
{ "goal": "Analyze latest RAG research." }

→ 200 OK
{
  "goal": "Analyze latest RAG research.",
  "graph_id": "a1b2c3d4",
  "progress": 100,
  "completed_tasks": 4,
  "failed_tasks": 0,
  "skipped_tasks": 0,
  "total_steps": 4,
  "success": true,
  "final_output": "## Search for relevant information\n...\n## Synthesise knowledge report\n...",
  "duration_secs": 3.4,
  "task_summary": {"pending": 0, "completed": 4, "failed": 0, ...},
  "history": [
    {"step": 1, "task_title": "Search for relevant information", "tool": "web_search", "decision": "accept", "quality": 0.78},
    ...
  ]
}
```

---

## Architecture After v0.3.5

```
Memory Layer        (v0.2/v0.3 — episodic/semantic/procedural, lifecycle, clusters)
      ↓
Knowledge Layer      (v0.3.1/v0.3.2 — facts, documents, media, synthesis)
      ↓
Reasoning Layer       (v0.3.4 — CognitiveQueryEngine, multi-hop, explainability)
      ↓
Planning Layer         (v0.3.5 NEW — GoalParser, TaskDecomposer, Planner)
      ↓
Execution Layer         (v0.3.5 NEW — AgentExecutor, ToolRegistry, WorkingMemory)
      ↓
Reflection Layer          (v0.3.5 NEW — ObservationLayer, ReflectionLoop, ExecutionHistory)
      ↓
API Layer                  (v0.3.3+ — FastAPI, now with /agent/*)
```

---

## Test Coverage

```
tests/test_v03_features.py      75 tests
tests/test_v031_features.py    118 tests
tests/test_v032_features.py    129 tests
tests/test_v033_features.py     76 tests
tests/test_v034_features.py    116 tests
tests/test_v035_features.py    140 tests  ← NEW
tests/test_memory_manager.py    ~60 tests
tests/test_semantic_retriever   ~40 tests
tests/test_tutor_agent.py        17 tests
──────────────────────────────────────────
Total                           790 tests  all passing
```

```bash
python -m pytest tests/ -q
# 790 passed
```

WebSearchTool tests verify graceful failure handling rather than live network
results, since sandboxed evaluation environments may not have egress to
`api.duckduckgo.com`. PythonTool tests cover both successful execution and
safe failure on disallowed operations (e.g. `open()` raises `NameError`
inside the sandbox, captured as a clean `FAILURE` result).

---

## What's Deliberately NOT Built Yet

Per the spec's guidance, v0.3.5 stops at a single-agent closed loop:

- Multi-agent collaboration / agent-to-agent communication
- Self-modifying agents (agents that rewrite their own tool definitions)
- Long-horizon autonomy (multi-day/multi-session goal persistence beyond
  what `GoalTracker` already provides)
- Browser automation tool

These are natural v0.4+ extensions once the closed loop here — Goal → Plan →
Act → Observe → Reflect → Learn — has been validated against the v0.3.5
evaluation suite in real usage.

---

## Migration from v0.3.4

No breaking changes. New files only:

```
planning/planner.py
agents/types.py
agents/working_memory.py
agents/observation.py
agents/reflection_loop.py
agents/executor.py
tools/registry.py
evaluation/agent_eval.py
api/routers/agent.py
```

`BlixContext` gains: `tool_registry`, `planner`, `milestone_tracker`,
`agent_working_memory`, `observation_layer`, `reflection_loop`,
`agent_executor`, `agent_session`, `agent_evaluator`, `agent_workspace` (Path).

New storage files: `memory/execution_history.json`, `memory/agent_workspace/`
(sandboxed file tool workspace).

`blix_eval` now also exports `AgentEvaluator` and `AgentEvalCase`.
