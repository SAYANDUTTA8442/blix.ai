# Blix v0.3.8 — "Meta-Cognitive Layer"

## Problem Statement

Through v0.3.7, Blix could remember (Memory), reason about what's true
right now vs. historically (State Tracking & Truth Maintenance), and
adapt mid-execution when a plan ran into trouble (PlanCritic,
Replanner, ToolReliabilityRegistry). But it had no model of **itself**:

- It had no persistent answer to "what am I actually good at?" — every
  domain was treated as equally capable, every time.
- "Confidence" was an implicit, scattered notion (a critic verdict
  here, a tool-reliability number there) rather than a first-class
  quantity attached uniformly to beliefs, plans, tools, and answers.
- Reasoning strategy never changed based on the situation — the same
  approach was used whether a goal was trivial or highly complex,
  whether confidence was high or low, whether something had just
  failed twice in a row.
- Successful task sequences were thrown away after each run instead of
  being distilled into reusable skills.
- Reflection happened per-run (`PlanReflection`, v0.3.6) but nothing
  looked **across** runs to notice recurring process problems like
  "we keep replanning" or "confidence keeps coming in low."

v0.3.8 is explicitly **not** another memory or retrieval upgrade. It
is the layer that lets Blix evaluate itself, understand its own
capabilities, and adapt its behavior accordingly — laying the
groundwork for v0.4.x (Global Workspace, attention, internal
multi-agent systems, world models).

---

## New Modules

| # | Module | Purpose |
|---|--------|---------|
| 1 | `metacognition/controller.py` | Monitor → Detect → Adapt orchestration |
| 2 | `metacognition/self_model.py` | Persistent capabilities/weaknesses/strengths/known-limits |
| 3a | `metacognition/confidence_manager.py` | Generic namespaced confidence store |
| 3b | `reasoning/confidence_reasoner.py` | Derives plan/tool/answer confidence from signals |
| 4 | `metacognition/strategy_manager.py` | Chooses DIRECT / TREE_OF_THOUGHT / CRITIC_FIRST / DECOMPOSE_FURTHER |
| 5 | `metacognition/capability_tracker.py` | Per-domain accuracy tracking, syncs into Self Model |
| 6 | `memory/procedural_memory.py` | Distills successful sequences into reusable `Skill`s |
| 7 | `planning/plan_evaluator.py` | Complexity / risk / confidence / dependency-density / expected-success scoring |
| 8 | `agents/execution_feedback.py` | Routes run outcomes to FailureMemory + CapabilityTracker |
| 9 | `reflection/meta_reflection.py` | Cross-run pattern detection → behavior-change insights |
| 10a | `evaluation/confidence_metrics.py` | Calibration: Brier score, ECE, over/under-confidence rate |
| 10b | `evaluation/capability_metrics.py` | Self-awareness gap between believed and actual capability |
| 10c | `evaluation/metacognition_metrics.py` | Detection accuracy, adaptation responsiveness/effectiveness |

Plus: `api/routers/metacognition.py` — 7 new endpoints.

---

## 1. Meta-Cognitive Controller

The orchestration layer. Composes everything below it rather than
re-implementing monitoring or adaptation:

```python
controller = MetaCognitiveController(
    plan_evaluator=plan_evaluator,
    confidence_reasoner=confidence_reasoner,
    strategy_manager=strategy_manager,
)
report, decision = controller.run_cycle("task_ref", graph=graph, critique=critique)
# report.issues -> [CognitiveIssue.LOW_CONFIDENCE]
# decision.action -> AdaptationAction.CHANGE_STRATEGY
```

Detection priority on `adapt()`: `REPEATED_FAILURES` (→ `REPLAN` or
`CHANGE_STRATEGY`, depending on whether the strategy manager picked
`DECOMPOSE_FURTHER`) > `HALLUCINATION_RISK` (→ `FLAG_FOR_REVIEW`) >
`LOW_BELIEF_CONSISTENCY` (→ `FLAG_FOR_REVIEW`) > `LOW_CONFIDENCE` (→
`CHANGE_STRATEGY`) > nothing.

Every monitored signal is optional — a turn with no plan graph, no
retrieval, and no belief-consistency check still produces a valid
(empty) report rather than erroring.

## 2. Self Model

```python
self_model.set_capability("coding", 0.93)
self_model.set_capability("legal_reasoning", 0.52)

self_model.is_weak_in("legal_reasoning")   # True  (< 0.6)
self_model.is_strong_in("coding")            # True  (>= 0.85)
self_model.low_capability_domains()             # ["legal_reasoning"]
```

Weaknesses/strengths lists are kept in sync automatically as scores
cross the 0.6 / 0.85 thresholds. `known_limits` and `preferences` are
free-form, manually-set fields for things that aren't naturally a
0-1 score (e.g. "Cannot verify facts after training cutoff").

## 3. Confidence System

`ConfidenceManager` is a generic namespaced store — any caller
registers `(namespace, ref_id) -> score`:

```python
confidence_manager.set("belief", "b_1234", 0.91)
confidence_manager.set("plan", graph.graph_id, 0.74)
confidence_manager.set("tool", "web_search", 0.83)
confidence_manager.reinforce("belief", "b_1234", 0.05)   # 0.91 -> 0.96
```

`ConfidenceReasoner` is the computation layer that **derives** these
scores rather than storing arbitrary ones:

```python
estimate = confidence_reasoner.plan_confidence(graph, critique)
# blends critic_verdict (50%) + tool_reliability (30%) + size_penalty (20%)
estimate.score        # 0.85
estimate.factors       # {"critic_verdict": 1.0, "tool_reliability": 0.5, "size_penalty": 1.0}
```

Confidence now genuinely propagates: `PlanQualityEvaluator` calls
`ConfidenceReasoner.plan_confidence()` rather than re-deriving it, and
`MetaCognitiveController` reads that same number to decide whether to
flag `LOW_CONFIDENCE`.

## 4. Strategy Manager

```python
strategy_manager.decide("task_ref", confidence=0.3)
# -> StrategyDecision(strategy=CRITIC_FIRST, reason="Confidence 0.30 is below threshold 0.50...")

strategy_manager.record_failure("task_ref")
strategy_manager.record_failure("task_ref")
strategy_manager.decide("task_ref", confidence=0.9)
# -> StrategyDecision(strategy=DECOMPOSE_FURTHER, ...)   # repeated failure takes priority
```

Decision priority: `repeated_failure` > `low_confidence` >
`high_complexity` > `DIRECT`. This module decides **which** strategy a
situation calls for; it does not implement Tree-of-Thought reasoning
itself (out of scope for this release) — that's left as a
strategy-consumer for a future version.

## 5. Capability Tracker

```python
capability_tracker.record_outcome("coding", success=True)
capability_tracker.record_outcome("coding", success=True)
capability_tracker.record_outcome("coding", success=False)
capability_tracker.accuracy("coding")          # 0.667

capability_tracker.sync_to_self_model(self_model)   # pushes confidently-measured domains into SelfModel
```

Neutral 0.5 prior when untested (consistent with v0.3.6's
`ToolReliabilityRegistry` convention). Only domains with
`min_samples_for_confidence` (default 5) observations sync into the
Self Model, so it doesn't get noisy single-observation scores.

## 6. Procedural Memory

```python
procedural_memory.learn_from_success(
    goal="Research the latest developments in transformer architectures",
    steps=["retrieve_documents", "summarize", "extract_insights", "update_knowledge"],
    name="research_analysis",
)
# Skill(name="research_analysis", steps=[...], use_count=1, success_count=1)

procedural_memory.find_matching_skill("Research recent papers on transformers")
# -> matching Skill via Jaccard similarity, if above threshold (default 0.4)
```

Reusing a matched skill on a future similar goal reinforces it
(`use_count`/`success_count` increment) rather than creating a
duplicate. This is intentionally a simple token-overlap matcher, not a
new planner — `suggest_steps()` returns a template the Planner *could*
use, but actually building/executing the plan remains v0.3.5's job.

## 7. Plan Quality Evaluator

Sits between Critic and Executor:

```
Planner → Critic → Plan Evaluator → Executor
```

```python
score = plan_evaluator.evaluate(graph, critique)
# PlanQualityScore(complexity=0.25, risk=0.0, confidence=0.85,
#                   dependency_density=1.0, expected_success=0.8125)
```

`expected_success = confidence - 0.3*risk - 0.15*complexity`, blending
an optimistic signal (derived confidence) against two pessimistic ones
(risk from critic issues + unreliable tools, complexity from step
count).

## 8. Execution Feedback Loop

```python
execution_feedback.record_run_result(agent_run_result, domain="coding")
```

Iterates every completed/failed/skipped task in an `AgentRunResult`,
infers domain via keyword heuristics if not given, and fans the
outcome out to `FailureMemory` (on failure) and `CapabilityTracker`
(every outcome) — without `AgentExecutor` needing direct knowledge of
metacognition internals.

## 9. Meta-Reflection

```python
insights = meta_reflection.analyze_runs(run_summaries)
# [BehaviorChangeInsight(
#     pattern="frequent_replanning",
#     observation="Frequent replanning observed (mean 3.0 replans/run across 3 run(s)).",
#     suggested_change="Current planning strategy is too shallow — consider decomposing
#                        plans further upfront or invoking the critic earlier.",
# )]
```

Detects three cross-run patterns: frequent replanning, frequently low
confidence, and repeated tool bottlenecks. Persists insights into the
existing v0.3.2 `ReflectionEngine` under `ReflectionScope.BEHAVIOR`
(via `reflect()`) rather than maintaining a separate store — behavior
insights show up alongside every other kind of reflective insight.

## 10. Evaluation Suite

```python
ConfidenceMetrics.brier_score(cases)                  # calibration quality, 0=perfect
ConfidenceMetrics.expected_calibration_error(cases)      # weighted bucket-gap summary
ConfidenceMetrics.overconfidence_rate(cases)               # high-confidence-but-wrong rate

CapabilityMetrics.self_awareness_gaps(self_model, tracker)   # believed vs. actual per domain
CapabilityMetrics.self_awareness_score(gaps)                   # 1 - mean_abs_gap

MetacognitionMetrics().run_metacognition_bench(cases, decisions)
# {"issue_detection_accuracy": ..., "false_alarm_rate": ..., "missed_detection_rate": ...,
#  "adaptation_responsiveness": ..., "adaptation_effectiveness": ..., "strategy_switch_rate": ...}
```

`MetacognitionMetrics` extends `StateMetrics`, completing the
evaluation tower:

```
MemoryEvaluator → ExtendedMemoryEvaluator → CognitiveEvaluator →
ReasoningEvaluator → AgentEvaluator → AdaptiveAgentEvaluator →
StateMetrics → MetacognitionMetrics
```

`adaptation_effectiveness()` deliberately **excludes** cases with
unknown outcome (`outcome_improved=None`) rather than counting them as
failures, since "we don't know yet" isn't evidence of ineffectiveness.

---

## Architecture After v0.3.8

```
Observe
  ↓
Working Memory
  ↓
Belief Layer
  ↓
State Tracker
  ↓
Truth Manager
  ↓
Planner
  ↓
Critic
  ↓
Plan Evaluator        ← NEW
  ↓
Executor
  ↓
Execution Feedback    ← NEW
  ↓
Reflection
  ↓
Meta-Reflection       ← NEW
  ↓
Meta-Cognitive Controller   ← NEW
  ↓
Self Model            ← NEW
  ↓
Adaptation
```

---

## API — `/metacognition`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/metacognition/self-model` | Current capabilities/weaknesses/strengths snapshot |
| GET | `/metacognition/capabilities` | Raw per-domain accuracy track record |
| GET | `/metacognition/confidence/{namespace}` | Confidence records, optional `?threshold=` filter |
| POST | `/metacognition/strategy/decide` | Get a strategy decision for `{ref_key, confidence}` |
| GET | `/metacognition/skills` | List all learned procedural skills |
| POST | `/metacognition/skills/match` | Find best-matching skill for `{goal}` |
| GET | `/metacognition/behavior-insights` | Recent cross-run behavior-change insights |

`dashboard_stats()` gained 4 new keys: `tracked_capabilities`,
`learned_skills`, `confidence_records_tracked`,
`execution_feedback_entries`.

---

## Test Coverage

| Module | Tests |
|--------|-------|
| Self Model | 15 |
| Confidence Manager | 12 |
| Confidence Reasoner | 12 |
| Strategy Manager | 10 |
| Capability Tracker | 12 |
| Procedural Memory | 13 |
| Plan Quality Evaluator | 9 |
| Execution Feedback Loop | 13 |
| Meta-Reflection | 10 |
| Meta-Cognitive Controller | 10 |
| Confidence Metrics | 9 |
| Capability Metrics | 9 |
| Metacognition Metrics | 10 |
| BlixContext integration | 5 |
| `/metacognition` API | 12 |
| **v0.3.8 total** | **163** |
| **Full project total** | **1237** (1074 prior + 163 new), all passing |

---

## Migration Notes

No breaking changes. Every new component is additive and wired via
optional constructor parameters (default `None`) on existing classes —
`PlanQualityEvaluator`, `ExecutionFeedbackLoop`, `MetaReflectionEngine`,
and `MetaCognitiveController` all degrade gracefully when their
optional dependencies aren't supplied (e.g. `ConfidenceReasoner` with
no `ToolReliabilityRegistry` falls back to neutral 0.5 tool-reliability
factors rather than erroring).

Existing v0.3.0–v0.3.7 modules (`PlanCritic`, `Replanner`,
`ToolReliabilityRegistry`, `FailureMemory`, `ReflectionEngine`,
`StateMetrics`, etc.) are **unmodified** — v0.3.8 composes them rather
than changing their behavior. The full pre-existing 1074-test suite
passes unchanged.
