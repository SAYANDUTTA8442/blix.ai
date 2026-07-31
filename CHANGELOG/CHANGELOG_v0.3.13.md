# Blix v0.3.13 — "Curiosity + Active Experimentation"

## Status: Final Major Architectural Release

Per design intent, v0.3.13 is the last major architectural version
before entering the evaluation/benchmarking/paper phase. After this
release, the bottleneck shifts from architecture to evidence.

---

## What this adds

The previous stack could answer:
- What happened? (memory, v0.1-v0.2)
- Is it true? (truth maintenance, v0.3.7)
- Why did it happen? (causality, v0.3.11)
- What will happen? (prediction/search, v0.3.10-v0.3.12)

v0.3.13 adds the final cognitive loop:

    Decision -> Uncertainty -> Curiosity -> Experiment -> Observation -> Knowledge

---

## Philosophy (locked before implementation)

No RL. No intrinsic rewards. No self-play. No world models. No PPO.
No AlphaZero. Keep it symbolic. This is structured curiosity: a scan
over existing cognitive state that surfaces what deserves investigation,
then a science-like pipeline to test it.

---

## New Modules

**knowledge/knowledge_gap_tracker.py** — KnowledgeGapTracker
Discovers and persists KnowledgeGap(domain, severity, uncertainty,
evidence_count) records from three real sources: SelfModel capability
scores below threshold, FailureMemory recurring failures, and CauseGraph
low-confidence edges. Answers "what don't I know / where am I weak / what
needs exploration." Feeds CuriosityEngine (UNKNOWN_DOMAIN trigger) and
SelfModelStore.knowledge_gaps() live property.

**curiosity/curiosity_engine.py** — CuriosityEngine
Five symbolic triggers, all drawing from existing v0.3.x infrastructure:
  - LOW_CONFIDENCE: beliefs with confidence below threshold
  - CONTRADICTION: conflicting belief pairs (BeliefStore.find_conflicting_candidates)
  - SPARSE_EVIDENCE: beliefs with fewer than N observations
  - FREQUENT_FAILURES: FailureMemory domains with ≥3 occurrences
  - UNKNOWN_DOMAIN: KnowledgeGapTracker gaps needing exploration
Produces CuriositySignal(target, reason, novelty, uncertainty,
expected_information_gain) ranked by priority_score =
0.4*uncertainty + 0.4*expected_information_gain + 0.2*novelty.

**hypothesis/hypothesis_manager.py** — HypothesisManager
Full lifecycle: PENDING -> SUPPORTED/REJECTED/UNKNOWN. Distinct from
BeliefStore.add_hypothesis() (v0.3.11, a single staging slot) — this
tracks multiple evidence observations over time, explicit rejection,
and experiment linkage. When a hypothesis reaches SUPPORTED, it is
promoted to an OBSERVED belief via the established
add_hypothesis()/confirm_observation() pipeline. Rejected hypotheses
with multiple evidence pieces are surfaced by MetaCausalReflection
for cross-run pattern analysis.

**experiments/experiment_planner.py** — ExperimentPlanner
Converts CuriositySignals and Hypotheses into structured
Experiment(hypothesis_id, actions, expected_result, success_criteria)
objects. record_outcome() feeds results directly into
HypothesisManager.add_evidence(), which promotes to BeliefStore if
the support threshold is crossed. Per locked architecture: confirmed
experiment outcomes can call confirm_observation() directly, no extra
gating required, since a real experimental observation IS an observed
fact (unlike counterfactuals). plan_from_signal() generates default
actions/criteria for each trigger type.

---

## Extensions (existing modules, additive only)

**reflection_engine.py** — reflect_on_curiosity(target, hypothesis, outcome, learned)
Uses ReflectionScope.LEARNING (already existed) to record "why was I
curious / did I learn from this?"

**principle_synthesizer.py** — synthesize_from_experiment(experiment)
Third principle source alongside CauseGraph edges and failure clusters.
synthesize_all() gains an optional experiments= parameter.

**memory/future_memory.py** — record_experiment(), resolve_experiment(), experiments()
Stores experiment expected outcomes alongside predictions using the
existing ExpectedState pattern — no new persistence layer.

**causality/meta_causal_reflection.py** — which_hypotheses_failed_repeatedly()
Cross-run aggregate: which hypotheses were repeatedly rejected despite
multiple evidence pieces? Flags systematic knowledge gaps.

**metacognition/self_model.py** — knowledge_gaps(tracker=None)
Live query returning KnowledgeGapTracker.gaps() when a tracker is
provided, falling back to SelfModel.weaknesses as a rough proxy.
No new persistence (per locked "no new memory subsystems" principle
from v0.3.12).

---

## Full pipeline confirmed

Low confidence
  ↓ CuriosityEngine (LOW_CONFIDENCE trigger)
CuriositySignal
  ↓ HypothesisManager.propose()
Hypothesis (PENDING)
  ↓ ExperimentPlanner.plan_from_signal()
Experiment (PLANNED)
  ↓ ExperimentPlanner.record_outcome()
Evidence accumulated in HypothesisManager
  ↓ (confidence >= support_threshold)
SUPPORTED -> BeliefStore.confirm_observation()
  ↓
OBSERVED Belief
  ↓ PrincipleSynthesizer.synthesize_from_experiment()
Principle

All tested end-to-end in TestExperimentPlanner.test_successful_outcome_feeds_hypothesis_toward_support
and test_synthesize_all_with_experiments.

---

## API — /curiosity

| Method | Path | Purpose |
|--------|------|---------|
| GET | /curiosity/signals | Ranked curiosity signals from current state |
| POST | /curiosity/hypotheses | Propose a hypothesis |
| GET | /curiosity/hypotheses | List hypotheses (filterable by status) |
| POST | /curiosity/hypotheses/{id}/evidence | Add evidence |
| POST | /curiosity/experiments | Plan an experiment |
| POST | /curiosity/experiments/from-signal | Plan from curiosity signal |
| GET | /curiosity/experiments | List experiments |
| POST | /curiosity/experiments/{id}/outcome | Record outcome |
| GET | /curiosity/knowledge-gaps | List knowledge gaps |
| POST | /curiosity/knowledge-gaps/discover | Discover from self-model/failures/cause-graph |

dashboard_stats() gained: knowledge_gaps, pending_hypotheses, experiments_planned.

---

## Test Coverage

| Module | Tests |
|--------|-------|
| KnowledgeGapTracker | 14 |
| CuriosityEngine | 9 |
| HypothesisManager | 11 |
| ExperimentPlanner | 12 |
| Extensions (5 modules) | 13 |
| BlixContext integration | 7 |
| /curiosity API | 16 |
| v0.3.13 total | 82 |
| Full project total | 1729 (1647 prior + 82 new), all passing |

No test failures on first run.

---

## What comes next

Architecture is no longer the bottleneck.

v0.3.13 completes the cognitive stack:
Experience -> Memory -> Truth -> Reflection -> Metacognition ->
Global Workspace -> Prediction -> Cause -> Counterfactuals ->
Search -> Curiosity -> Hypotheses -> Experiments -> Knowledge

The next phase: benchmarks, baselines, ablations, statistics, papers.
