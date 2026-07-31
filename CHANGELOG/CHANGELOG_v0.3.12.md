# Blix v0.3.12 — "Imagination + Search"

## Problem Statement

After v0.3.11 (Causality), Blix could ask "why did it happen?" and
explore what would have happened under counterfactual alternatives.
What it couldn't do: think multiple steps ahead, search over candidate
action sequences, and choose based on that search. v0.3.12 closes that
gap — the largest remaining between Blix and DeepMind world-model systems.

Transition: Cause → Prediction → Imagination → Search → Decision.

---

## Scope, honestly stated

Before any code was written, 10 spec items were reviewed against what
already existed in v0.3.10-v0.3.11. Five items overlapped significantly
with existing infrastructure:

- Item 1 (Scenario Simulator) → already covered by CounterfactualScenarioEngine
- Item 3 (Imagination Engine) → same concept as CounterfactualScenarioEngine, different name
- Item 5 (World State Object) → already LatentState (v0.3.10)
- Item 6+10 (Prediction Memory + Evaluator) → FutureMemoryStore + ConfidenceMetrics already existed, just needed wiring
- Item 9 (Scenario Ranking Network) → ScenarioRanker already existed

Per the "no more memory subsystems" constraint, the final scope was:

**3 new modules:**
1. simulation/trajectory_graph.py — TrajectoryGraph, StateNode, ActionEdge, Trajectory, TrajectoryBuilder
2. planning/beam_search.py — BeamSearchPlanner (spec's highest-ROI item)
3. planning/search_critic.py — SearchCritic, DecisionExplanation

**2 extensions (no new top-level storage):**
4. causality/counterfactual_engine.py — explore_with_trajectories() added to CounterfactualScenarioEngine
5. evaluation/prediction_evaluator.py — PredictionEvaluator wiring FutureMemoryStore + ConfidenceMetrics, adds prediction_drift()

---

## New Modules

**simulation/trajectory_graph.py**

Upgrades single-hop cause-effect modeling to multi-step futures as
first-class objects:

    State0 -> Action -> State1 -> Action -> State2

StateNode wraps LatentState with depth tracking.
ActionEdge connects StateNodes.
Trajectory is an ordered chain: depth=len(edges), nodes=len(edges)+1.
TrajectoryBuilder builds incrementally; TrajectoryGraph holds multiple
in-memory for comparison.

Deliberately not persisted — trajectories are transient imagined
futures regenerated each planning pass, consistent with the "no new
memory subsystems" constraint. EpistemicStatus: PREDICTED or
COUNTERFACTUAL, never OBSERVED. No memory.beliefs import anywhere.

**planning/beam_search.py**

The spec's identified highest-ROI item. Goal -> generate candidates ->
evaluate trajectories -> choose best, using ValueNetwork (v0.3.10) to
score/prune top-K beams at each depth. Explicit scope: beam search
only, not MCTS/DreamerV3/MuZero, matching the design spec's
"Avoid implementing full MuZero" constraint.

Key design detail: _clone_and_step() branches a TrajectoryBuilder by
replaying prior actions into a fresh builder rather than mutating shared
state — necessary for correct beam branching.

**planning/search_critic.py**

Post-hoc explainability over a BeamSearchResult. Issues use the same
IssueSeverity vocabulary (INFO/WARNING/CRITICAL) as PlanCritic (v0.3.6)
rather than inventing a parallel taxonomy. Four issue categories:
thin_margin (winner nearly indifferent from runner-ups), shallow_search
(depth=1), high_risk_swing (max-min risk along trajectory ≥ 0.4),
untrained_value_network (honest flag when cold-starting).

---

## Extensions

**causality/counterfactual_engine.py**

Added explore_with_trajectories() — same ranking as explore(), but each
CounterfactualResult.trajectory is now populated with a real
TrajectoryBuilder.step() chain: current_state -> alternative_name ->
resulting_state. CounterfactualResult gained an optional trajectory
field (defaults None for backward compatibility).

Structural safeguard re-verified intact: still zero memory.beliefs
references after the Trajectory import was added. Covered by a
source-inspection test.

**evaluation/prediction_evaluator.py**

Thin adapter wiring FutureMemoryStore (v0.3.10) + ConfidenceMetrics
(v0.3.8) into one coherent reporting surface:
- calibration_report() — Brier score, ECE, over/under-confidence, per-bucket
- calibration_for_subject() — scoped by prediction subject
- prediction_drift() — genuinely new metric: split resolved predictions
  chronologically, compare earlier vs. recent Brier scores

Spec example confirmed: predicted 0.8 success, actual failure ->
Brier score 0.64, overconfidence_rate 1.0, exactly as expected.

---

## Two test failures caught during testing

1. test_no_actions_returns_none_trajectory — beam search with no
   action generator returns the initial start beam (depth 0), not None.
   The test was wrong about the expected behavior; fixed to assert
   best_trajectory.depth == 0 rather than is None.

2. test_clone_and_step_independence — _clone_and_step() requires an
   explicit value_delta argument; the test was missing it. Fixed.

Both were genuine test-specification errors caught on first run.

---

## API — /search

| Method | Path | Purpose |
|--------|------|---------|
| POST | /search/beam | Beam search from start state toward goal |
| POST | /search/explain | Explain a beam search decision (SearchCritic) |
| POST | /search/counterfactual/trajectories | Counterfactual exploration with full trajectory objects |
| GET | /search/predictions/calibration | Full calibration report (Brier/ECE/over-under-confidence) |
| GET | /search/predictions/drift | Calibration drift over time |
| GET | /search/predictions/calibration/{subject} | Calibration scoped to one subject |

dashboard_stats() gained: active_trajectories, resolved_predictions.

---

## Test Coverage

| Module | Tests |
|--------|-------|
| TrajectoryGraph | 14 |
| BeamSearchPlanner | 9 |
| SearchCritic | 8 |
| CounterfactualEngine extension | 7 |
| PredictionEvaluator | 9 |
| BlixContext integration | 6 |
| /search API | 10 |
| v0.3.12 total | 63 |
| Full project total | 1647 (1584 prior + 63 new), all passing |

---

## Migration Notes

No breaking changes. CounterfactualResult.trajectory defaults to None —
every existing caller of explore()/best() sees identical behavior.
All prior v0.3.10-v0.3.11 infrastructure (LatentState, ValueNetwork,
ScenarioRanker, CounterfactualScenarioEngine, FutureMemoryStore,
ConfidenceMetrics) is extended, not replaced. The full 1584-test
prior suite passes unchanged.
