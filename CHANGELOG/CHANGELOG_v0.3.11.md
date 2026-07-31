# Blix v0.3.11 — "Causal Cognition"

## Problem Statement

Through v0.3.10, Blix had memory, truth maintenance, a coordinated
Global Workspace, meta-cognition, and learned prediction — genuinely
broad cognitive infrastructure. But none of it answered why. Beliefs
tracked what was currently true; the world model predicted what would
happen next; nothing represented why one thing led to another, or let
Blix generalize "the cause behind repeated failure X" into a reusable
principle that future planning could draw on.

v0.3.11 closes that gap with a Causal Cognition layer — but only after
an unusual amount of upfront negotiation on scope, sequencing, and
safety, because "causality" is a word that can mean very different
(and very differently risky) engineering commitments. The final locked
architecture, sequencing, and safeguards below were agreed before any
code was written.

---

## The locked architecture

Phase 1 — Foundations (real, symbolic, no ML-honesty tension):
CauseGraph, BeliefDependencyGraph, CausalMemory

Phase 2 — Synthesis:
PrincipleSynthesizer, PrincipleGraph (with Principle as a first-class
dataclass, never a bare string)

Phase 3 — Reflection & Strategy, operating over PRINCIPLES, not raw
failures:
CausalReflection, MetaCausalReflection, StrategyEvolution

Phase 4 — Lightweight counterfactuals:
CounterfactualScenarioEngine (a single merged module — the
originally-separate "Imagination Engine" and "Scenario Tree Search"
ideas were redundant and were explicitly merged before implementation)

This ordering matters: principles are synthesized from causal evidence
before reflection and strategy consume them, so prescriptive
reflection and strategy evolution always cite an actual generalized
principle (or a raw CauseGraph edge) — never an ungrounded heuristic.

---

## Explicit scope reduction

Per locked design constraint, v0.3.11 does not include: Pearl
structural causal models, do-calculus, Bayesian networks, MCTS, world
simulators, or theory of mind. Every "causal" claim in this release is
evidence-counted correlational structure with causal-sounding labels
(CAUSES/ENABLES/BLOCKS assigned to an observed co-occurrence pattern)
— not validated causal inference. confidence on a CauseEdge means "how
often did we see trigger and effect co-occur," not "P(effect |
do(trigger))." This is stated directly in every relevant module's
docstring, not just here.

---

## Three architectural constraints, locked before implementation

1. Principles are first-class objects, never strings.

```python
@dataclass
class Principle:
    id: str
    statement: str
    confidence: float
    evidence_count: int
    supporting_causes: list[str]      # CauseEdge.edge_id references
    supporting_failures: list[str]    # FailureCluster identifiers
    status: EpistemicStatus = PRINCIPLE
```

PrincipleGraph, CausalReflection, and StrategyEvolution all operate on
Principle objects by reference (principle.id), never on bare text.

2. CauseGraph is typed, never an untyped string relation.

```python
@dataclass
class CauseEdge:
    trigger: str
    effect: str
    relation: CauseRelation   # CAUSES | INCREASES | DECREASES | ENABLES | BLOCKS
    confidence: float
    evidence_count: int
```

3. Counterfactual outputs must never enter the belief system
automatically.

This is the most important safeguard in the release, and it's enforced
structurally, not just by convention:

- causality/counterfactual_engine.py contains zero references to
  memory.beliefs — no import, no BeliefStore symbol anywhere in the
  file. This is verified with a source-inspection test
  (TestCounterfactualSafeguard.test_module_has_no_belief_store_import)
  that fails the build if anyone later adds that import.
- The only path from a counterfactual estimate toward a trusted belief
  is two separate, explicit calls: BeliefStore.add_hypothesis() (stages
  it as EpistemicStatus.HYPOTHESIS — not yet trusted) and, only after
  real confirming observation, BeliefStore.confirm_observation(). There
  is no single function anywhere in the codebase that can take a
  CounterfactualResult and silently turn it into an OBSERVED belief.

---

## Epistemic typing, everywhere — including in API responses

```python
class EpistemicStatus(str, Enum):
    OBSERVED = "observed"
    DERIVED = "derived"
    PREDICTED = "predicted"
    COUNTERFACTUAL = "counterfactual"
    PRINCIPLE = "principle"
    HYPOTHESIS = "hypothesis"
```

Every CounterfactualResult carries confidence, evidence_count (always
0 — a counterfactual has no direct evidence of its own, by
definition), basis, epistemic_status, and validated_causally=False —
and these fields appear directly in /causality/counterfactual/explore
JSON responses, not just in docstrings, per explicit requirement.

memory.beliefs.Belief gained a new epistemic_status field (additive,
defaults to OBSERVED — every pre-v0.3.11 caller's assumption is
preserved exactly), orthogonal to the existing status (TruthStatus,
v0.3.7's "is this currently true" axis).

---

## New Modules

| Phase | Module | Purpose |
|-------|--------|---------|
| shared | causality/epistemic_status.py | EpistemicStatus vocabulary |
| 1 | causality/cause_graph.py | Typed CauseEdge graph, evidence-counted |
| 1 | causality/belief_dependency_graph.py | supports/weakens DAG + damped confidence propagation |
| 1 | causality/causal_memory.py | Flat trigger->effect recall store |
| 2 | causality/principle.py | First-class Principle dataclass + store |
| 2 | causality/principle_graph.py | supports DAG over Principles |
| 2 | causality/principle_synthesizer.py | Mines CauseGraph/FailureClusterer -> Principle |
| 3 | causality/causal_reflection.py | Prescriptive, principle-grounded reflection (extends MetaReflectionEngine) |
| 3 | causality/meta_causal_reflection.py | Aggregate causal queries ("why do I repeatedly fail in X") |
| 3 | metacognition/strategy_evolution.py | Explainable, cause-cited strategy recommendations |
| 4 | causality/counterfactual_engine.py | Lightweight what-if ranking (merged Imagination + Scenario Tree) |

Plus: api/routers/causality.py — 12 new endpoints. memory/beliefs.py
gained epistemic_status, add_hypothesis(), confirm_observation(), and
a public persist() hook (all additive).

---

## Verified examples (matching the design spec literally)

Every module was tested against the spec's own worked examples, not
just synthetic data:

- CauseGraph: "no evaluation" BLOCKS "reliable optimization",
  "benchmarks" ENABLES "fast iteration" — exact match.
- BeliefDependencyGraph: A weakens (-0.3) -> B drops 0.8->0.56 -> C
  drops 0.8->0.704 (damped) — confirmed the falloff-with-hops behavior works.
- PrincipleGraph: "Always evaluate before optimizing" -> "Reliable
  optimization" -> "Faster iteration", propagating a +0.15 confidence
  boost through both hops — exact match.
- CausalReflection: produced "Relevant principle... / Alternative
  strategy: Tree-of-Thought / Estimated success: 0.37" — the
  prescriptive format from the spec, honestly tagged PREDICTED.
- MetaCausalReflection: answered all three named questions ("why do I
  repeatedly fail in research tasks", "what causes low confidence",
  "which strategies cause success") correctly — including catching and
  fixing a real bug where "increases low confidence" was initially
  miscounted as success-related due to a naive keyword match on
  "confidence" without checking for the negating qualifier "low".
- CounterfactualScenarioEngine: "What if Tree-of-Thought had been
  used?" ranked above "stayed Direct" with full epistemic metadata
  (confidence=0.25 honestly reflecting the untrained value network,
  validated_causally=False).

---

## API — /causality

| Method | Path | Purpose |
|--------|------|---------|
| POST | /causality/cause/record | Record a cause-effect observation |
| GET | /causality/cause/effects/{trigger} | Known effects of a trigger |
| GET | /causality/cause/causes/{effect} | Known causes of an effect |
| POST | /causality/belief-graph/dependency | Declare a supports/weakens belief dependency |
| POST | /causality/belief-graph/propagate | Propagate a confidence change |
| GET | /causality/principles | List synthesized principles |
| POST | /causality/principles/synthesize | Synthesize from current evidence |
| POST | /causality/reflect/causal | Prescriptive reflection on a failed topic |
| GET | /causality/reflect/why | Why do I repeatedly fail in <domain>? |
| GET | /causality/reflect/causes-of | What causes <effect>? |
| POST | /causality/strategy/evolve | Propose an explainable strategy change |
| POST | /causality/counterfactual/explore | Rank what-if alternatives |

dashboard_stats() gained 5 new keys: cause_graph_edges,
belief_dependency_edges, causal_memories, principles_synthesized,
principle_graph_edges.

---

## Test Coverage

| Module | Tests |
|--------|-------|
| EpistemicStatus | 8 |
| CauseGraph | 13 |
| Belief HYPOTHESIS gating | 8 |
| BeliefDependencyGraph | 8 |
| CausalMemory | 9 |
| Principle / PrincipleStore | 7 |
| PrincipleGraph | 3 |
| PrincipleSynthesizer | 5 |
| CausalReflection | 5 |
| MetaCausalReflection | 7 |
| StrategyEvolution | 8 |
| CounterfactualScenarioEngine | 10 |
| Counterfactual safeguard (structural) | 4 |
| BlixContext integration | 5 |
| /causality API | 13 |
| v0.3.11 total | 111 |
| Full project total | 1584 (1473 prior + 111 new), all passing |

One real bug was caught and fixed during testing: a naive keyword match
in which_strategies_cause_success() initially counted "increases low
confidence" as success-related (because "confidence" matched), fixed
by excluding effects carrying negative qualifiers ("low", "poor",
"reduced", etc.) despite a nominal keyword hit.

---

## Migration Notes

No breaking changes. Belief's new epistemic_status field defaults to
OBSERVED, exactly matching every pre-v0.3.11 caller's implicit
assumption. BeliefStore.add_or_reinforce()'s new epistemic_status
parameter defaults the same way. BeliefStore.persist() is a new public
method exposing existing save logic; the private _save() it wraps is
unchanged. MetaReflectionEngine, FailureClusterer, ValueNetwork,
ScenarioRanker, and StrategySelectorNetwork are all unmodified —
v0.3.11 composes and extends them rather than changing their behavior.
The full pre-existing 1473-test suite passes unchanged.
