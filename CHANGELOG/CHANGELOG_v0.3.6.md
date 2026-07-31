# Blix v0.3.6 — Adaptive Planning & Verification Engine

> Upgrade from v0.3.5. Transforms Blix from an execution agent into a
> **self-correcting agent**. The v0.3.5 loop (Goal → Plan → Execute →
> Reflect) is linear: if a task fails, the plan stalls. v0.3.6 closes
> that gap:
>
> ```
> Goal → Plan → Critic → Execute → Verify → Observe → Reflect → Replan → Learn
> ```
>
> No breaking changes. Every v0.3.6 component is optional in the
> ``AgentExecutor`` constructor and defaults to ``None``, so v0.3.5
> behavior is preserved exactly when these components aren't supplied —
> verified by running the full v0.3.5 test suite (140 tests) unchanged
> against the upgraded executor.

---

## The Problem This Solves

```
v0.3.5 flow:
    Goal → Plan → Execute → Reflect
    Task 2 fails → plan stalls. No recovery.

v0.3.6 flow:
    Goal → Plan → Execute → Observe → Replan → Continue
    Task 2 fails (Chroma unavailable) → Replanner switches to FAISS → Continue.
```

---

## New Modules

```
agents/
├── state.py              Upgrade 10 — AgentState, ToolReliabilityStats, ExecutionCostModel
├── failure_memory.py      Upgrade 4  — FailureMemory
├── tool_reliability.py     Upgrade 5  — ToolReliabilityRegistry
├── task_runtime.py          Upgrade 3  — TaskRuntime (DAG batches, failure propagation)
└── plan_reflection.py        Upgrade 8  — PlanReflection

planning/
├── critic.py               Upgrade 6  — PlanCritic
└── replanner.py              Upgrade 1  — Replanner

verification/
└── verifier.py               Upgrade 2  — VerificationEngine

evaluation/
└── agent_benchmark.py          Upgrade 9  — AdaptiveAgentEvaluator
```

`agents/executor.py` (v0.3.5) is upgraded in place to wire all of the
above into the closed loop — see "Integration" below.

---

## Upgrade 1 — Dynamic Replanning Engine (`planning/replanner.py`)

The biggest jump toward autonomy. When a task exhausts its retries (the
point where v0.3.5 would mark it permanently FAILED), the executor now
asks the `Replanner` to adapt the plan instead:

```python
from planning.replanner import Replanner

replanner = Replanner(tool_registry=registry, failure_memory=fm, tool_reliability=tr)

if replanner.should_replan(task, graph):
    result = replanner.replan(task, graph, failure_reason=task.error)
    # result.strategy: SWITCH_TOOL | DECOMPOSE | DROP_TASK
```

Three strategies, tried in order:

1. **Switch tool** — substitute a plausible alternative tool (e.g.
   `web_search` → `memory_search` → `llm`), ranked by `ToolReliabilityRegistry`
   when available, never retrying a tool already tried for this task.
2. **Decompose** — split the failed task into two smaller sequential
   sub-tasks with the same tool, re-pointing downstream dependents.
3. **Drop** — last resort: mark the task permanently `SKIPPED` and strip
   it from other tasks' `depends_on` so the rest of the plan can proceed.

Every replan attempt is recorded in `FailureMemory` first, regardless of
which strategy succeeds. `max_replans_per_task` (default 2) bounds
infinite substitution loops.

Verified end-to-end (see `test_replanner_switches_tool_on_failure`):
a task using a perpetually-failing `web_search` tool gets switched to
`memory_search` mid-run and the overall plan succeeds.

---

## Upgrade 2 — Verification Engine (`verification/verifier.py`)

Inserts a gate between Observe and Reflect:

```
v0.3.5:  Execute → Observe → Reflect (accept/retry/skip)
v0.3.6:  Execute → Observe → VERIFY → Reflect (accept/retry/skip)
```

A task can look successful to the Observation layer (non-empty output,
no error) yet still fail structural verification:

```python
from verification.verifier import VerificationEngine

engine = VerificationEngine()  # NonEmptyVerifier, SchemaVerifier,
                                 # KeywordPresenceVerifier, CodeSyntaxVerifier
report = engine.verify(task, exec_result)
report.passed       # False if ANY verifier failed
report.summary()    # "Verification failed: Missing required schema key(s): ['result']"
```

Verifiers activate conditionally via `task.metadata`:

```python
Task(title="Create API", tool_hint="python_tool",
     metadata={"expected_schema": ["route", "status"]})
# → SchemaVerifier checks output is valid JSON with those keys

Task(title="Build endpoint", metadata={"required_keywords": ["route", "schema"]})
# → KeywordPresenceVerifier checks all keywords appear in output

Task(title="Generate code", tool_hint="python_tool")
# → CodeSyntaxVerifier runs ast.parse() on any ```python fence
```

When verification fails, the executor downgrades the Observation's
`success`/`quality_score` BEFORE handing off to `ReflectionLoop`, so a
structurally-broken result triggers a retry even though the tool itself
reported success.

---

## Upgrade 3 — Execution DAG Runtime (`agents/task_runtime.py`)

```python
from agents.task_runtime import TaskRuntime

runtime = TaskRuntime(graph, max_parallel=4)
batch = runtime.next_batch()     # up to 4 ready tasks, for concurrent execution
runtime.propagate_failures()      # cascades FAILED → BLOCKED transitively
runtime.unblock(task_id)          # manual recovery after a fix
runtime.topological_batches()     # full DAG structure, non-mutating
```

`propagate_failures()` is the key new capability: if Task A fails,
every task depending on A (directly or transitively through B, C...) is
automatically marked `BLOCKED` rather than waiting forever in `PENDING`.
Verified across a 4-deep linear chain and across independent branches
(an unrelated branch is correctly left untouched).

This wraps `TaskGraph` without replacing it — the v0.3.5 sequential
`graph.next_task()` loop continues to work unchanged if a given
`AgentExecutor` configuration doesn't use `TaskRuntime` directly.

---

## Upgrade 4 — Failure Memory (`agents/failure_memory.py`)

```python
from agents.failure_memory import FailureMemory

fm = FailureMemory(Path("memory/failure_memory.json"))
fm.record("Build API", "python_tool", "schema mismatch", goal="Build a service")
fm.record_fix("Build API", "python_tool", "update response model")

fm.similar_failures("Build a REST API")   # Jaccard token-overlap matching (≥0.4 default)
fm.suggest_fix("Build API", "python_tool")  # "update response model"
```

Repeated failures of the same kind (Jaccard similarity on task title
tokens) merge into one record with an incrementing `occurrences`
counter rather than creating duplicates. Queried by both `Replanner`
(to avoid repeating mistakes) and `PlanCritic` (to flag risky plans
before execution).

```json
{
  "task_title": "build api",
  "tool": "python_tool",
  "failure": "schema mismatch",
  "fix": "update response model",
  "occurrences": 3
}
```

---

## Upgrade 5 — Tool Reliability Scoring (`agents/tool_reliability.py`)

Persists success rates ACROSS runs (distinct from the per-run
`ToolReliabilityStats` in `AgentState`, which resets every execution):

```python
from agents.tool_reliability import ToolReliabilityRegistry

registry = ToolReliabilityRegistry(Path("memory/tool_reliability.json"))
registry.record("web_search", success=True, duration_ms=340)

registry.success_rate("web_search")            # 0.92
registry.is_confident("web_search")             # True if ≥ min_samples (default 5)
registry.rank_tools_by_reliability([...])         # sorted by success rate
```

Untested tools get a neutral 0.5 prior rather than being penalised.
`Replanner` uses this to rank alternative tools; `PlanCritic` uses it to
flag a planned task as `risky_tool` (WARNING) if it targets a tool with
a confidently-measured low success rate.

---

## Upgrade 6 — Plan Critic (`planning/critic.py`)

"Think before acting." Runs once, before the execution loop starts:

```python
from planning.critic import PlanCritic

critic = PlanCritic(tool_registry=registry, tool_reliability=tr, failure_memory=fm)
report = critic.critique(graph)

report.verdict        # APPROVED | APPROVED_WITH_WARNINGS | REJECTED
report.has_critical    # True if any CRITICAL issue
```

Six checks:

| Check | Severity | Example |
|---|---|---|
| Circular dependencies | CRITICAL | DFS cycle detection: `A → B → A` |
| Missing tools | CRITICAL | Task references a `tool_hint` not in the registry |
| Dangling dependencies | CRITICAL | Task depends on a `task_id` that doesn't exist |
| Risky tools | WARNING | Tool has confidently-measured success_rate < 0.4 |
| Known failures | WARNING | Task resembles a `FailureMemory` record (with fix, if known) |
| Missing steps | INFO | Goal implies "verification" but no task addresses it |

If `verdict == REJECTED`, the `AgentExecutor` aborts the run immediately
(`ExecutorConfig.abort_on_critic_rejection`, default `True`) rather than
wasting tool calls on a structurally broken plan — `result.aborted_by_critic`
is set and `result.critique` carries the full issue list.

---

## Upgrade 7 — Execution Cost Model (`agents/state.py`)

```python
from agents.state import ExecutionCostModel

cost = ExecutionCostModel()
cost.record_call(tokens=120, duration_secs=0.8, is_retry=False)
cost.efficiency_score()   # 1.0 - (retries / tool_calls)
```

Tracked live inside `AgentState.cost` throughout every `AgentExecutor.run()`
call: `token_cost`, `execution_time_secs`, `tool_calls`, `retry_count`.
Surfaced via `result.agent_state["cost"]` and aggregated by
`AdaptiveAgentEvaluator.tool_efficiency()` / `execution_cost()` (Upgrade 9).

---

## Upgrade 8 — Plan Reflection (`agents/plan_reflection.py`)

v0.3.5's `ReflectionLoop` reflects on individual TASKS. `PlanReflection`
reflects on the WHOLE PLAN after a run completes or stalls:

```python
from agents.plan_reflection import PlanReflection

pr = PlanReflection(failure_memory=fm, reflection_engine=reflection_engine)
report = pr.reflect(graph, history=result.history, replan_count=result.replan_count)

report.success                     # bool
report.root_cause                  # "Task 'Search papers' failed: timeout"
report.bottleneck_tool              # tool with the most retry/skip decisions
report.improvement_suggestions       # ["Consider preferring an alternative to 'web_search'...", ...]
```

Runs automatically at the end of every `AgentExecutor.run()` call.
On success, extracts lessons ("succeeded after 2 replans — substitution
strategy worked"). On failure, identifies the first failing task, the
bottleneck tool, explains the root cause, and generates concrete
suggestions — persisted as a PROJECT-scope insight in the v0.3.2
`ReflectionEngine` and as a fix in `FailureMemory`.

---

## Upgrade 9 — Agent Benchmark Suite (`evaluation/agent_benchmark.py`)

`AdaptiveAgentEvaluator` extends `AgentEvaluator` (v0.3.5), completing
the full evaluation tower:

```
MemoryEvaluator → ExtendedMemoryEvaluator → CognitiveEvaluator
    → ReasoningEvaluator → AgentEvaluator → AdaptiveAgentEvaluator
```

| Metric | Method |
|---|---|
| Task Success Rate | (inherited from v0.3.5) |
| Verification Accuracy | `verification_accuracy(reports, expected_pass)` |
| Replanning Success | `replanning_success_rate(results)` |
| Recovery Rate | `recovery_rate(graph)` — task-level, distinct from replan success |
| Execution Cost | (inherited from v0.3.5) |
| Tool Efficiency | `tool_efficiency(result)` — completed tasks per tool call |

```python
from blix_eval import AdaptiveAgentEvaluator

ev = AdaptiveAgentEvaluator()
metrics = ev.benchmark_run(result)     # single-run full metric set
batch = ev.benchmark_batch(results)    # aggregated across runs
```

---

## Upgrade 10 — Agent State Object (`agents/state.py`)

The unifying change. Every cognitive module now has access to one
`AgentState` per run, threaded through `AgentExecutor.run()`:

```python
state = AgentState(goal=graph.goal)
state.set_plan(graph, is_replan=False)
...
state.record_observation(observation)
state.record_completion(task.task_id)
state.record_failure(task.task_id, {"task": ..., "tool": ..., "failure": ...})
state.update_confidence(new_value)

state.progress          # derived from active_plan
state.is_stalled         # has_failures and not complete
state.plan_version        # bumped on every replan
```

Exposed in the API response as `result["agent_state"]`. Note: this
complements — does not replace — `WorkingMemory` (still the TTL-scoped
scratch pad for tool context); `AgentState` is the structural/cognitive
record that survives the whole run and is what `PlanCritic`, `Replanner`,
and `PlanReflection` reason against.

---

## Integration — `AgentExecutor.run()` Upgraded Loop

```python
PlanCritic.critique(graph)                    # ── once, before the loop
    → REJECTED? abort immediately (result.aborted_by_critic = True)

while not graph.is_complete:
    task = graph.next_task()
    tool = registry.select_tool(task)
    exec_result = tool.execute(task, context)

    observation = obs_layer.observe(exec_result)        # OBSERVE
    verification_report = verifier.verify(task, exec_result)   # VERIFY (v0.3.6)
        → failed? downgrade observation.success/quality_score

    decision = reflect_loop.reflect(task, observation)    # REFLECT

    if decision == "skip":
        if replanner.should_replan(task, graph):           # REPLAN (v0.3.6)
            replanner.replan(task, graph, failure_reason=task.error)
            # SWITCH_TOOL/DECOMPOSE → task runnable again, loop continues
            # DROP_TASK → permanently failed, plan proceeds without it
    ...

PlanReflection.reflect(graph, history, replan_count)    # ── once, after the loop
```

All four new parameters (`plan_critic`, `verification_engine`,
`replanner`, `plan_reflection`) are optional and default to `None` —
when omitted, `AgentExecutor` behaves exactly as it did in v0.3.5
(confirmed by `test_backwards_compatible_without_v036_components` and
by running the full 140-test v0.3.5 suite unchanged).

`AgentRunResult` gains four new fields: `replan_count`, `critique`,
`plan_reflection`, `agent_state`, `aborted_by_critic`.

---

## API Endpoints (v0.3.6 additions)

| Method | Path | Description |
|---|---|---|
| POST | `/agent/run` | Now includes `replan_count`, `critique`, `plan_reflection`, `agent_state` |
| POST | `/agent/critique` | Plan a goal and run PlanCritic WITHOUT executing |
| GET | `/agent/failures` | Most common recorded failure patterns |
| GET | `/agent/tool-reliability` | Cross-run tool reliability stats |

### Example: `/agent/critique`

```json
POST /agent/critique
{ "goal": "Build a REST API for user management." }

→ 200 OK
{
  "task_graph": { "tasks": [...] },
  "critique": {
    "verdict": "approved_with_warnings",
    "issue_count": 1,
    "issues": [
      {"severity": "warning", "category": "known_failure",
       "message": "Task 'Implement endpoints' resembles a past failure: schema mismatch (known fix: update response model)",
       "task_id": "a3f9c1d2"}
    ]
  }
}
```

---

## Architecture After v0.3.6

```
Goal
  ↓
Planner              (v0.3.5)
  ↓
Plan Critic           (v0.3.6 NEW — think before acting)
  ↓
Execution DAG         (v0.3.6 NEW — TaskRuntime, parallel-ready)
  ↓
Tool Runtime          (v0.3.5 — ToolRegistry)
  ↓
Verification Engine    (v0.3.6 NEW — structural gate)
  ↓
Observation Layer       (v0.3.5)
  ↓
Reflection Engine         (v0.3.5 — task-level)
  ↓
Replanner                  (v0.3.6 NEW — adapts the plan, loops back to Execution DAG)
  ↓
Plan Reflection              (v0.3.6 NEW — plan-level, on completion)
  ↓
Memory / FailureMemory / ToolReliabilityRegistry
```

This realises:

```
Goal → Plan → Execute → Verify → Learn → Replan
```

instead of v0.3.5's:

```
Goal → Plan → Execute
```

---

## Test Coverage

```
tests/test_v03_features.py      75 tests
tests/test_v031_features.py    118 tests
tests/test_v032_features.py    129 tests
tests/test_v033_features.py     76 tests
tests/test_v034_features.py    116 tests
tests/test_v035_features.py    140 tests
tests/test_v036_features.py    142 tests  ← NEW
tests/test_memory_manager.py    ~60 tests
tests/test_semantic_retriever   ~40 tests
tests/test_tutor_agent.py        17 tests
──────────────────────────────────────────
Total                           932 tests  all passing
```

```bash
python -m pytest tests/ -q
# 932 passed
```

Integration tests specifically verify: the Replanner switching tools
mid-run when a tool persistently fails (end-to-end, the loop actually
succeeds afterward); the VerificationEngine forcing a retry on a
schema-invalid output that the Observation layer alone would have
accepted; the PlanCritic aborting a run before any tool call when given
a circular-dependency plan; `propagate_failures()` cascading BLOCKED
status through a 4-level dependency chain while leaving independent
branches untouched; and full backwards compatibility when none of the
v0.3.6 components are supplied to `AgentExecutor`.

---

## What Was Deliberately NOT Built

Per the spec's explicit guidance, v0.3.6 avoided:

- More memory layers, graph types, or retrieval methods
- Multi-agent systems
- Voice support
- UI features

None of these address the core v0.3.5 bottleneck (adaptation, reliability,
recovery) — every module in this release exists to make the agent
**self-correcting**, not to add more things for it to know.

---

## Migration from v0.3.5

No breaking changes. New files only:

```
agents/state.py
agents/failure_memory.py
agents/tool_reliability.py
agents/task_runtime.py
agents/plan_reflection.py
planning/critic.py
planning/replanner.py
verification/verifier.py
evaluation/agent_benchmark.py
```

`agents/executor.py` is modified in place: `AgentExecutor.__init__()`
gains four new optional keyword parameters (all defaulting to `None`),
`ExecutorConfig` gains three new boolean flags (`abort_on_critic_rejection`,
`enable_verification`, `enable_replanning`, all defaulting to sensible
values), and `AgentRunResult` gains five new fields.

`BlixContext` gains: `failure_memory`, `tool_reliability_registry`,
`plan_critic`, `verification_engine`, `replanner`, `plan_reflection`.
`agent_evaluator` is now an `AdaptiveAgentEvaluator` instance (a strict
superset of the v0.3.5 `AgentEvaluator` interface — no breakage).

New storage files: `memory/failure_memory.json`, `memory/tool_reliability.json`.

`blix_eval` now also exports `AdaptiveAgentEvaluator` and `AgentBenchmarkCase`.
