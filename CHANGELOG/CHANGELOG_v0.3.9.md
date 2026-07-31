# Blix v0.3.9 — "Global Workspace"

## Problem Statement

Through v0.3.8, Blix had hierarchical memory, a knowledge graph,
reflection, planning, execution, beliefs, state tracking, truth
maintenance, meta-cognition, a self-model, and procedural memory — all
genuinely intelligent subsystems. But they were **independent**:

- Memory, Planner, Reflection, and SelfModel each operated in their
  own world. A discovery in one (e.g. the Planner hitting a failure)
  had no general mechanism to reach the others (Reflection, Self
  Model, Belief Layer, Failure Memory) — only the specific wiring
  someone happened to add in `BlixContext`.
- Nothing decided what deserved attention. Every memory, belief, and
  plan step was treated as equally important, all the time.
- There was no shared "working stage" — no single place holding
  "what's currently being thought about" that every subsystem could
  read from.
- Retrieval considered only the literal query, never the live
  cognitive context (active goal, current focus).
- There was no way to pause a long-running goal and pick it back up
  later in roughly the same cognitive state.

v0.3.9 is the largest architectural shift since the project began. It
does **not** add more memory, more tools, or more planners — it
introduces a Global Workspace (per Baars' Global Workspace Theory,
with influences from LIDA, SOAR, ACT-R, and Blackboard Systems) so
that previously-isolated subsystems become a coordinated cognitive
system.

---

## New Packages & Modules

| Package | Module | Purpose |
|---------|--------|---------|
| `events/` | `event_types.py` | Typed cognitive event vocabulary |
| `events/` | `event_bus.py` | Synchronous pub/sub dispatch |
| `events/` | `event_store.py` | Persisted event audit log |
| `workspace/` | `attention_manager.py` | Weighted attention scoring + gated entry |
| `workspace/` | `broadcast_bus.py` | GWT-style cross-subsystem broadcast |
| `workspace/` | `workspace_memory.py` | The working stage itself |
| `workspace/` | `global_workspace.py` | Orchestrates attention → entry → broadcast |
| `workspace/` | `snapshot.py` | Capture/restore for task suspension |
| `workspace/` | `inner_dialogue.py` | Multi-voice commentary pass |
| `specialists/` | `base.py`, `memory_specialist.py`, `planning_specialist.py`, `reflection_specialist.py`, `verification_specialist.py`, `consensus.py` | Internal Specialists + Consensus |
| `retrieval/` | `active_attention_retriever.py` | Context-aware retrieval |
| `evaluation/` | `workspace_metrics.py`, `attention_metrics.py`, `coordination_metrics.py` | Coordination-quality evaluation |

Plus: `api/routers/workspace.py` — 8 new endpoints.

---

## 4. Cognitive Event Bus

Everything important can now become a typed event:

```python
EventType.TASK_COMPLETED, EventType.FAILURE, EventType.BELIEF_UPDATED,
EventType.STATE_CHANGED, EventType.REFLECTION_GENERATED, EventType.PLAN_CREATED,
EventType.CONFIDENCE_CHANGED, EventType.STRATEGY_SWITCHED, EventType.WORKSPACE_BROADCAST
```

```python
event_bus.subscribe(EventType.FAILURE, on_failure_handler)
event_bus.publish(failure_event("planner", "Step 1", "tool timeout"))
```

`EventBus` is deliberately a synchronous, single-process, in-memory
dispatcher (no async, no external broker) — that's the right
complexity level for Blix today. `EventStore` persists a capped
(5,000-event) audit log separately, so the bus doesn't need to know
how/whether events are stored.

## 1–3. Global Workspace, Attention, Broadcast

```
Global Workspace
                      ↓
     --------------------------------
     ↓       ↓        ↓       ↓
 Memory Planner Reflection SelfModel
     ↓       ↓        ↓       ↓
             Broadcast
```

```python
global_workspace.submit_candidate(AttentionCandidate(
    ref_id="failure_1", source="planner", content_summary="Step 1 failed",
    relevance=0.9, urgency=0.9, novelty=0.8, confidence=0.7,
))
result = global_workspace.run_cycle(active_goal="fix the bug")
# result.entered      -> items that won attention and entered the workspace
# result.broadcasts_sent -> count of cross-subsystem notifications sent
```

`AttentionManager` implements the spec's literal formula:

```python
attention_score = 0.4*relevance + 0.3*urgency + 0.2*novelty + 0.1*confidence
```

Only candidates above `entry_threshold` (default 0.5), capped at
`capacity` (default 7, modeling limited-capacity working memory),
actually enter `WorkspaceMemory`. Every entry triggers a broadcast via
`BroadcastBus`, which wraps `EventBus` with subsystem-registration
sugar and a broadcast log used by `evaluation.coordination_metrics`.

I verified the full cycle end-to-end: a high-attention candidate
entered and was received by a registered listener; a low-attention
candidate was correctly rejected and never broadcast.

## 5. Internal Specialists

```
Workspace
  ↓
Specialists
  ↓
Consensus
```

Four specialists wrap existing subsystems behind one uniform
interface (`consult(topic, **context) -> SpecialistOpinion`):
`MemorySpecialist` (belief lookup), `PlanningSpecialist`
(`PlanQualityEvaluator`), `ReflectionSpecialist` (`FailureMemory`),
`VerificationSpecialist` (`VerificationEngine`). None of them
duplicate logic — each is a thin adapter.

```python
result = specialist_consensus.decide("Step 1", graph=graph, critique=critique)
# result.majority_verdict -> "supports" | "opposes" | "uncertain" | "no_opinion"
# result.agreement_ratio    -> fraction of opinionated specialists agreeing
# result.is_contested          -> True if agreement_ratio < 2/3
```

Specialists with nothing to contribute return `no_opinion`, which is
excluded from the majority calculation but preserved in the full
opinion list for transparency — this is the beginning of a
Society-of-Mind pattern: independent opinions, aggregated afterward,
rather than a single reasoning stream.

## 6. Active Attention Retrieval

```python
active_attention_retriever.retrieve(
    memories, query,
    current_workspace=workspace_memory,
    active_goal="ship the feature",
    attention_focus=workspace_memory.attention_focus,
)
```

Wraps the existing `MemoryRetriever` (v0.2, unmodified — its
`retrieve(memories, query)` signature still works exactly as before
for every existing caller) and adds a re-ranking pass that blends base
retrieval order (50%) with goal-text alignment (25%) and
workspace-content alignment (25%). I stress-tested this against a case
where the base retriever's literal-query ranking and the live
workspace context disagreed — the workspace-aligned memory correctly
moved to the top, while supplying no context at all reproduces the
base retriever's exact ordering (full backward compatibility).

## 7. Workspace Snapshotting

```python
snapshot = workspace_snapshots.capture(
    global_workspace, important_beliefs=["b1"],
    current_plan_graph_id="g1", current_plan_summary="2-step plan",
    current_failures=["timeout on step 1"],
)
# ... later, possibly in a new process ...
workspace_snapshots.restore(snapshot.snapshot_id, global_workspace)
```

Captures active goal, attention focus, and workspace item summaries —
references and summaries, not full copies of every subsystem's data
(beliefs/plans still live in their own stores). This is what makes
task suspension and resumption possible.

## 8. Internal Dialogue

```python
inner_dialogue.register_voice("Planner", planner_voice(strategy_manager, ref_key))
inner_dialogue.register_voice("Critic", critic_voice(plan_quality_score))
inner_dialogue.register_voice("Self Model", self_model_voice(self_model, "math"))
inner_dialogue.register_voice("Reflection", reflection_voice(failure_memory))

transcript = inner_dialogue.run("how to proceed")
print(transcript.as_text())
```

```
Planner:
Need strategy. Repeated failures suggest the current approach should change.

Self Model:
Math capability high (0.90).

Reflection:
Previous failures indicate caution around: timeout error
```

I verified this against the spec's literal example output format.
Voices that have nothing to say (e.g. Critic with no plan score
available) are silently skipped rather than padding the transcript —
each voice is a thin adapter over an existing module, not new
reasoning logic.

## 9. Cognitive Coordination Metrics

`CoordinationMetrics` extends `MetacognitionMetrics` (v0.3.8),
completing the evaluation tower:

```
... → AdaptiveAgentEvaluator → StateMetrics → MetacognitionMetrics → CoordinationMetrics
```

```python
coordination_metrics.run_coordination_bench(consensus_results, broadcast_bus)
# {"consensus_convergence_rate": ..., "mean_agreement_ratio": ...,
#  "no_opinion_rate": ..., "subsystem_participation_rate": ..., "isolation_rate": ...}
```

`subsystem_participation_rate` returns 0.0 (not a misleadingly perfect
1.0) when no subsystems are registered — nothing CAN participate, so
that's correctly scored as zero coordination, not perfect coordination.

---

## Architecture After v0.3.9

```
Observe
  ↓
Working Memory
  ↓
Beliefs
  ↓
State Tracker
  ↓
Truth Manager
  ↓
Planner
  ↓
Executor
  ↓
Reflection
  ↓
Meta-Cognitive Controller
  ↓
Global Workspace        ← NEW
  ↓
Attention                ← NEW
  ↓
Broadcast                 ← NEW
  ↓
Internal Specialists       ← NEW
  ↓
Adaptation
```

---

## API — `/workspace`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/workspace/state` | Current workspace contents |
| POST | `/workspace/submit` | Submit a candidate for the next attention cycle |
| POST | `/workspace/cycle` | Run one attention → entry → broadcast cycle |
| GET | `/workspace/snapshots` | List saved workspace snapshots |
| POST | `/workspace/snapshots` | Capture a new snapshot of current state |
| POST | `/workspace/snapshots/{id}/restore` | Restore a snapshot's goal/focus context |
| POST | `/workspace/specialists/consult` | Poll all specialists + consensus for a topic |
| POST | `/workspace/inner-dialogue` | Run an inner-dialogue pass over a topic |

`dashboard_stats()` gained 4 new keys: `workspace_cycle_count`,
`workspace_snapshots_stored`, `cognitive_events_logged`,
`broadcasts_sent`.

---

## Test Coverage

| Module | Tests |
|--------|-------|
| Event Types / Bus / Store | 22 |
| Attention Manager | 10 |
| Broadcast Bus | 8 |
| Workspace Memory | 9 |
| Global Workspace | 9 |
| Memory Specialist | 4 |
| Planning Specialist | 3 |
| Reflection Specialist | 3 |
| Verification Specialist | 3 |
| Specialist Consensus | 6 |
| Active Attention Retriever | 5 |
| Workspace Snapshot | 10 |
| Inner Dialogue | 13 |
| Workspace Metrics | 4 |
| Attention Metrics | 5 |
| Coordination Metrics | 9 |
| BlixContext integration | 5 |
| `/workspace` API | 9 |
| **v0.3.9 total** | **137** |
| **Full project total** | **1374** (1237 prior + 137 new), all passing |

---

## Migration Notes

No breaking changes. `MemoryRetriever.retrieve(memories, query)`
remains entirely unchanged — `ActiveAttentionRetriever` wraps it
rather than modifying it, and falls back to identical base-retriever
ordering whenever no workspace context is supplied. Every new
component takes its dependencies as optional constructor parameters.

Existing v0.3.0–v0.3.8 modules (`PlanCritic`, `Replanner`,
`ToolReliabilityRegistry`, `FailureMemory`, `ReflectionEngine`,
`PlanQualityEvaluator`, `VerificationEngine`, `SelfModel`, etc.) are
**unmodified** — v0.3.9 composes and wraps them rather than changing
their behavior. The full pre-existing 1237-test suite passes
unchanged.
