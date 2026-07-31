# Blix v0.3.16 — Adaptive Dual Memory Architecture (ADMA)

## Overview

v0.3.16 transforms Blix from a memory-augmented system into a continuously
learning cognitive architecture. The central innovation is that memory no
longer stores only knowledge — it now stores **policies**.

Static prompting is replaced by a dynamic prompt compiler. Retrieval weights
and planner configurations are no longer fixed — they are selected by a
contextual bandit at runtime and updated from observable outcomes.

## Research Contribution

> **ADMA: Adaptive Dual Memory Architecture for Continual Learning and
> Personalized Cognitive AI**

Core contributions:
1. **Dual Memory** — System Memory (operational knowledge) + User Memory (personalisation)
2. **Policy Memory** — policies stored as learnable bandit arms, not static config
3. **Adaptive Prompt Compiler** — dynamic prompt assembly from active policy configs
4. **Online Policy Learning** — contextual bandits with Thompson sampling (no RLHF)
5. **Reward Engine** — 15 observable reward signals (8 system, 7 user)
6. **Adaptive Retrieval** — policy-driven HybridRetriever weight selection
7. **Adaptive Planning** — policy-driven BeamSearchPlanner configuration
8. **Ablation Framework v3** — dependency injection replacing broken env-flag mechanism

---

## New Modules

### `policy/` (new package)

```
policy/
  __init__.py
  models.py        PolicyRecord, PolicyVersion, RewardSignal, PolicyDomain,
                   PolicyType (15 types), RewardType (15 types)
  store.py         PolicyStore — SQLite-backed (policy.db, WAL mode)
                   Tables: policies, policy_versions, reward_log
  reward.py        SystemRewardEngine (8 reward functions)
                   UserRewardEngine (7 reward functions)
                   RewardEngine (unified facade with dispatch)
  learner.py       PolicyLearner — contextual bandit, Thompson sampling
                   _default_policies() — 15 default arms across 5 types
  optimizer.py     PolicyOptimizer — decay, aging, retirement, mutation, rollback
  compiler.py      PolicySelector, PolicyCompiler, CompiledPrompt
  adaptive.py      AdaptiveRetriever, AdaptivePlanner
  ablation_v3.py   AblationConfig, AblationV3Runner, AblationV3Report,
                   stub implementations for all 7 injectable components
```

### `memory/system/` and `memory/user/` (new packages)

```
memory/
  system/
    __init__.py
    system_memory.py   SystemMemory — operational knowledge domain
  user/
    __init__.py
    user_memory.py     UserMemory — per-user personalisation domain
  manager.py           MemoryManager — unified routing + merging + dedup
```

---

## Architecture

```
Experience
    │
    ├─────────────────────────────────────┐
    │ System Experience                   │ User Experience
    │                                     │
    ▼                                     ▼
SystemMemory                         UserMemory
    │                                     │
    │  (both backed by HGSHM v0.3.15)    │
    │                                     │
    ▼                                     ▼
System Policies ──── PolicyLearner ──── User Policies
(bandit arms)         (Thompson          (bandit arms)
                       sampling)
    │                                     │
    └──────────────┬──────────────────────┘
                   │
            PolicyCompiler
          (dynamic prompting)
                   │
             Planner + Workspace
                   │
                  LLM
```

---

## Policy Learning Algorithm

**Contextual Bandits with Thompson Sampling over Beta distributions.**

- Each `PolicyRecord` maintains Beta(α, β) state
- Arm selection: draw from Beta(α, β) for each arm; pick highest draw
- Update rule: reward ≥ 0.5 → α += reward (fractional); reward < 0.5 → β += (1-reward)
- Temporal decay: α → 1 + (α-1)×0.995 per observation (half-life ≈ 139 observations)
- Convergence: detected when confidence spread over last 5 snapshots < 0.02
- Rollback: triggered when recent mean drops > 0.10 vs older mean

This is provably optimal for explore/exploit in the bandit setting (sublinear regret).
No ML library required — pure Python, interpretable, symbolically verifiable.

---

## Default Policy Arms

15 default policies installed on fresh Blix, all starting Beta(1,1) = 50% confidence:

| Type                | Arms                                         |
|---------------------|----------------------------------------------|
| RETRIEVAL_WEIGHTS   | balanced, semantic_heavy, graph_heavy        |
| PLANNER_CONFIG      | conservative, balanced, aggressive           |
| REASONING_STRATEGY  | direct, stepwise, exhaustive                 |
| ANSWER_STYLE        | concise, balanced, verbose                   |
| DIFFICULTY_LEVEL    | easy, medium, hard                           |

---

## Memory Domains

### SystemMemory
- `store_workflow(description, success, latency_ms)` — workflow traces
- `store_benchmark_result(name, score, n_cases)` — benchmark history
- `store_failure_pattern(pattern, resolution, frequency)` — failure memory
- `store_principle(statement, confidence)` — operational principles
- `store_api_knowledge(topic, content)` — API / algorithm reference
- `recall(query, top_k)` → MemoryContext
- `benchmark_history(name)` → list[MemoryNode]
- `recent_failures()` → list[MemoryNode]

All nodes tagged `["system_memory"]`.

### UserMemory
- `store_preference(category, preference, strength)` — user preferences
- `store_interaction(query, response_accepted, correction)` — interaction history
- `store_goal(goal, priority)` — long-term goals
- `store_learning_progress(topic, understood)` — what user knows
- `store_project(name, description, stack)` — project context
- `record_correction(original, correction, severity)` — explicit corrections
- `recall(query, top_k)` → MemoryContext
- `cold_start_profile()` — quick check: is this user new?
- `preferences(category)`, `goals()`, `corrections()` — typed retrieval

All nodes tagged `["user_memory", "user:{user_id}"]` for strict isolation.

### MemoryManager
- `query(query, user_id, include_system, include_user, include_general)` → RoutedContext
- `get_user_memory(user_id)` — cached UserMemory instances
- `store_system(text)`, `store_user(user_id, text)` — convenience writes
- Deduplicates results across domains by node_id (keeps highest score)

---

## Ablation Framework v3

Replaces env-flag mechanism (v2) with true dependency injection.

**v2 problem:** Benchmark cases called component APIs directly without
checking `is_ablated()`, so ablating a component produced zero delta.

**v3 fix:** Each condition replaces the actual component with a stub
implementation at injection time. The stub is absent — not just flagged
as absent.

Stub implementations:
- `_NullPolicyLearner` — all selects return None
- `_NullPolicySelector` — all configs return fallback defaults
- `_NullRewardEngine` — all dispatches are no-ops
- `_FixedWeightRetriever` — uniform weights, no adaptation
- `_FixedConfigPlanner` — conservative fixed config

8 predefined conditions:
1. `full_system` — baseline (all components enabled)
2. `without_policy_learning`
3. `without_reward_engine`
4. `without_user_memory`
5. `without_system_memory`
6. `without_adaptive_retrieval`
7. `without_adaptive_planning`
8. `without_policy_compiler`

Report output: delta_score, pass_rate delta, Cohen's d, impact level
(CRITICAL / HIGH / MEDIUM / LOW), JSON and CSV export.

---

## Test Suite

| File                      | Tests | Coverage                                    |
|---------------------------|-------|---------------------------------------------|
| test_v0313_features.py    |    82 | v0.3.7–v0.3.13 cognitive stack              |
| test_gap_fixes.py         |    21 | 12 architectural gap fixes                  |
| test_v0315_hgshm.py       |    91 | HGSHM — 30 modules, all layers             |
| test_v0316_adma.py        |   121 | ADMA — policy, memory, compiler, ablation  |
| **Total**                 | **315** | **All passing ✓**                         |

v0.3.16 test breakdown (121 tests):

| Class                   | Tests | What is tested                               |
|-------------------------|-------|----------------------------------------------|
| TestPolicyRecord        |    14 | confidence, update, decay, Thompson, CI, snap |
| TestRewardSignal        |     3 | clamping, is_positive, serialisation         |
| TestPolicyStore         |     8 | CRUD, versions, rollback, reward log, stats  |
| TestRewardEngine        |    11 | system rewards ×6, user rewards ×3, dispatch |
| TestPolicyLearner       |    13 | select, observe, broadcast, curve, rollback  |
| TestPolicyOptimizer     |     6 | decay, retire, mutant, convergence, cycle    |
| TestPolicySelector      |     6 | system/user select, weights, config, fallback |
| TestPolicyCompiler      |     8 | compile, active_policies, verbosity, memory  |
| TestSystemMemory        |     9 | all store methods, recall, history, stats    |
| TestUserMemory          |    16 | all store methods, retrieval, isolation, cold-start |
| TestMemoryManager       |     7 | routing, dedup, cache, latency, to_context   |
| TestADMAIntegration     |     7 | end-to-end policy learning + memory pipeline |
| TestAblationV3          |     8 | conditions, stubs, run, report, JSON, CSV    |
| TestADMAStress          |     4 | 1000 obs, convergence high/low, multi-user  |

---

## Bug Fixes

- `PolicyVersion` field naming: `beta` (no underscore) vs `PolicyRecord.beta_`
  (trailing underscore to avoid shadowing `math.beta`). Fixed `PolicyStore.save_version()`
  to map `"beta_": d["beta"]` for SQL binding, and `get_history()`/`rollback()` to
  remap `beta_` column back to `beta` field on read.
- `MemoryType.KNOWLEDGE` does not exist — changed to `MemoryType.FACT` in
  `SystemMemory.store_api_knowledge()`.
- `SystemMemory.store_principle()` was calling `hgshm.add_principle()` which doesn't
  propagate the `system_memory` tag. Fixed to call `hgshm.remember()` directly with
  explicit tags.
- `PolicyVersion.beta_` reference in `PolicyStore.rollback()` and
  `PolicyLearner.learning_curve()` — fixed to `.beta`.

---

## Backward Compatibility

All 194 v0.3.15 tests pass unchanged. HGSHM, BeliefStore, CauseGraph, and
PrincipleStore shims are unmodified. v0.3.16 is purely additive:

- New package: `policy/`
- New packages: `memory/system/`, `memory/user/`
- New module: `memory/manager.py`
- New database: `policy.db` (co-located with `hgshm.db`)

No existing API was removed or modified.

---

## Known Limitations

1. **Benchmark coverage for ADMA benchmarks deferred.** The 400+ benchmark
   expansion (system learning, user learning, prompt compiler, routing, policy)
   is planned for v0.3.17. The ablation framework uses a 4-test minimal suite
   when blix_eval is not extended.
2. **Adaptive retrieval weight injection.** `AdaptiveRetriever` attempts to
   set `hgshm.hybrid_retriever._weights` — this accesses a private attribute.
   A proper `HybridRetriever.set_weights()` API is planned.
3. **Single-process concurrency.** PolicyStore uses SQLite WAL mode but doesn't
   implement locking for multi-process PolicyLearner instances.
4. **Cold-start quality.** With Beta(1,1) uniform prior, the first few arm
   selections are effectively random. Performance on the first 5–10 interactions
   is equivalent to a randomly selected policy.

---

## Upgrade Path from v0.3.15

No migration required. Add ADMA to an existing Blix install:

```python
from pathlib import Path
from memory.hybrid.hgshm import HGSHM
from memory.system.system_memory import SystemMemory
from memory.user.user_memory import UserMemory
from memory.manager import MemoryManager
from policy.store import PolicyStore
from policy.learner import PolicyLearner
from policy.reward import RewardEngine
from policy.compiler import PolicySelector, PolicyCompiler

memory_dir = Path("./blix_data")
hgshm   = HGSHM(memory_dir)
store   = PolicyStore(memory_dir)
learner = PolicyLearner(store)
learner.register_defaults()          # install 15 default policy arms

engine   = RewardEngine(learner)
selector = PolicySelector(learner)
compiler = PolicyCompiler(selector)
sys_mem  = SystemMemory(hgshm)
mgr      = MemoryManager(hgshm, sys_mem)

# On each task:
ctx    = mgr.query(task, user_id="alice")
prompt = compiler.compile(task, user_id="alice",
                          memory_context=ctx.to_memory_context())

# After each task, observe the outcome:
engine.on_benchmark(score, benchmark_name)
engine.on_answer_accepted(accepted=True, user_id="alice")
```
