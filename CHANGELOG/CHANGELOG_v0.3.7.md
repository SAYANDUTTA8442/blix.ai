# Blix v0.3.7 — Temporal State Tracking & Truth Maintenance

> Upgrade from v0.3.6. Blix stops being merely an adaptive agent and
> starts maintaining an evolving model of reality. Exactly three new
> cognitive capabilities, nothing else:
>
> ```
> State Tracking  +  Truth Maintenance  +  Belief Evolution
> ```
>
> No new agent features, planners, tools, UI, or multi-agent systems.
> No breaking changes — every v0.3.7 component is additive.

---

## The Problems This Solves

```
1. Contradictions handled incorrectly
   Before: Python and Rust both become competing memories.
   After:  Python (Historical) → Rust (Current), tracked as a transition.

2. Retrieval ignores temporal truth
   Before: score = semantic + recency + importance
   After:  score = semantic + recency + importance + state_relevance + belief_confidence

3. Graph is static
   Before: Sayan --works_on--> Blix  (no evolution)
   After:  Sayan --uses--> Python [2024] --uses--> PyTorch [2025] --uses--> Rust [2026]

4. Reflection lacks evolution
   Before: "current interests", "current goals"
   After:  Interest / Skill / Project / Identity Evolution narratives

5. Query engine lacks history
   Before: "What is my favorite language?"
   After:  + "What was it in 2024?" + "How has it evolved?"
           + "When did Blix adopt FastAPI?" + "What changed last month?"
```

---

## New Modules

```
core/
├── state_tracker.py            Item 1  — StateTracker, StateSnapshot
├── state_transition.py          Item 2  — StateTransitionEngine, StateTransition
├── truth_manager.py               Item 3  — TruthManager, TruthStatus
└── contradiction_resolver.py        Item 5  — ContradictionResolver (4 cases)

memory/
└── beliefs.py                       Item 4  — BeliefStore, Belief

retrieval/
└── temporal_retriever.py             Item 6  — TemporalRetriever (5-component score)

graph/
└── temporal_graph.py                  Item 7  — TemporalGraph, TemporalEdge

reasoning/
└── temporal_query.py                   Item 8  — TemporalQueryEngine

reflection/
└── state_reflection.py                  Item 9  — StateReflectionEngine

evaluation/
└── state_metrics.py                       Item 10 — StateMetrics (StateBench-lite)

api/routers/
└── temporal.py                              API   — /temporal/* endpoints
```

---

## Item 3 — TruthStatus (the foundational fix)

```python
from core.truth_manager import TruthStatus

class TruthStatus(str, Enum):
    ACTIVE       # currently believed true
    SUPERSEDED   # was true, explicitly replaced
    HISTORICAL   # was true, ended naturally, no conflict
    CONFLICTING  # unresolved — evidence on multiple sides
    ARCHIVED     # manually retired
```

`TruthManager` owns four operations against any tracked id (a belief id
or a state-snapshot id — the manager is storage-agnostic):

```python
truth_manager.replace(old_id, new_id)    # old → SUPERSEDED, new → ACTIVE
truth_manager.merge(id_a, id_b)           # collapse near-duplicates into one survivor
truth_manager.archive(record_id)           # retire without declaring a winner
truth_manager.resolve(record_id, status)    # direct status assignment
```

---

## Items 1+2 — State Tracking & Transitions

```python
from core.state_tracker import StateTracker
from core.state_transition import StateTransitionEngine

tracker = StateTracker(Path("memory/state_snapshots.json"))
engine = StateTransitionEngine(tracker, Path("memory/state_transitions.json"))

engine.transition("sayan", "favorite_language", "Python")   # initial assignment
engine.transition("sayan", "favorite_language", "Rust")      # → CLOSES Python snapshot, OPENS Rust

tracker.current("sayan", "favorite_language").value    # "Rust"
tracker.at_time("sayan", "favorite_language", "2024-12-01T00:00:00").value  # "Python"
tracker.history("sayan", "favorite_language")            # [Python(closed), Rust(active)]
```

Reasserting the SAME value reinforces confidence rather than creating a
spurious transition:

```python
engine.transition("sayan", "favorite_language", "Rust", confidence=0.6)
# No new StateTransition recorded — existing snapshot's confidence bumped instead.
```

Each ``StateSnapshot`` is `(entity, attribute, value, start_time, end_time, confidence)`
exactly as the spec's schema specifies.

---

## Item 5 — Contradiction Resolver (the core bug fix)

Classifies every contradiction into one of four cases instead of
winner-take-all:

```python
from core.contradiction_resolver import ContradictionResolver, ContradictionCase

resolver = ContradictionResolver(truth_manager, belief_store)

# Replacement — Delhi → Kolkata
resolver.classify("I moved to Delhi", "I now live in Kolkata", value_a="Delhi", value_b="Kolkata")
# → ContradictionCase.REPLACEMENT

# Parallel Truth — Python AND Rust
resolver.classify("I use Python for data work", "I also use Rust for systems")
# → ContradictionCase.PARALLEL_TRUTH

# Merge — AI / Artificial Intelligence
resolver.classify("AI", "Artificial Intelligence", value_a="AI", value_b="Artificial Intelligence")
# → ContradictionCase.MERGE

# Conflict — no markers, genuinely ambiguous
resolver.classify("I live in Mumbai", "I live in Chennai", value_a="Mumbai", value_b="Chennai")
# → ContradictionCase.CONFLICT
```

`resolve()` classifies AND applies the right `TruthManager` operation in
one call. For genuine `CONFLICT`s, `try_resolve_conflict()` re-evaluates
once enough evidence accumulates on one side (requiring BOTH more
distinct sources AND higher confidence to avoid flip-flopping on noisy
single observations):

```python
resolver.try_resolve_conflict(
    "claim_mumbai", "claim_chennai",
    evidence_count_a=1, evidence_count_b=5,
    source_count_a=1, source_count_b=4,
    confidence_a=0.3, confidence_b=0.9,
)
# → promotes the conflict to REPLACEMENT once Chennai clearly dominates
```

---

## Item 4 — Belief Store

```python
from memory.beliefs import Belief, BeliefStore

store = BeliefStore(Path("memory/beliefs.json"))
belief = store.add_or_reinforce("User's favorite language is Rust", confidence=0.7, source_memory_id=42)
# Re-observing a similar statement reinforces rather than duplicating:
store.add_or_reinforce("User's favorite language is Rust", source_memory_id=51)
# belief.evidence_count == 2, belief.source_count == 2, confidence bumped
```

`Belief(statement, confidence, evidence_count, source_count, status)`
exactly matches the spec's schema — `status` is a cached `TruthStatus`
owned by `TruthManager` for convenient filtering.

---

## Item 6 — Temporal Retriever (fixes "old facts compete equally")

```python
from retrieval.temporal_retriever import TemporalRetriever

retriever = TemporalRetriever(state_tracker, truth_manager, belief_store)

score = retriever.score(
    memory_id=14, semantic=0.8, recency=0.5, importance=0.6,
    entity="sayan", attribute="favorite_language", snapshot_id=snap.snapshot_id,
)
# final_score = semantic*0.35 + recency*0.2 + importance*0.25
#             + state_relevance*0.12 + belief_confidence*0.08
```

`state_relevance` scales by the TruthStatus of the snapshot the memory
supports: `ACTIVE`→1.0, `CONFLICTING`→0.5, `HISTORICAL`→0.2,
`SUPERSEDED`/`ARCHIVED`→0.0. Verified: two memories with IDENTICAL
semantic/recency/importance scores but different TruthStatus now
produce different final rankings — the bug from the spec is fixed.

---

## Item 7 — Temporal Graph

```python
from graph.temporal_graph import TemporalGraph

tgraph = TemporalGraph(Path("memory/temporal_graph.json"))
tgraph.add_relation("Blix", "uses", "Python", timestamp="2024-01-01T00:00:00")
tgraph.add_relation("Blix", "uses", "PyTorch", timestamp="2025-01-01T00:00:00")  # closes Python edge
tgraph.add_relation("Blix", "uses", "Rust", timestamp="2026-01-01T00:00:00")      # closes PyTorch edge

tgraph.evolution("Blix", "uses")            # [Python, PyTorch, Rust] chronological
tgraph.relations_at_time("Blix", "2024-06-01T00:00:00", relation="uses")  # [Python]
tgraph.current_relations("Blix", relation="uses")   # [Rust]
```

For naturally multi-valued relations (e.g. "knows"), pass
`close_previous=False` so multiple active edges can coexist without
artificially closing each other.

---

## Item 8 — Temporal Query Engine (all five query types from the spec)

```python
from reasoning.temporal_query import TemporalQueryEngine

qe = TemporalQueryEngine(state_tracker, state_transitions, temporal_graph, default_entity="sayan")

qe.query("What was my favorite language in 2024?").answer   # "Python"
qe.query("How has my favorite language evolved?").answer     # "Python → PyTorch → Rust"
qe.query("When did sayan adopt Rust?").answer                  # "2026-01-01T00:00:00"
qe.query("What changed during the last 365 days?").answer       # "...: 'PyTorch' → 'Rust'"
qe.query("What is my favorite language?").answer                 # "Rust"  (current — fallback pattern)
```

Five distinct query types — `historical_year`, `evolution`,
`transition`, `recent_changes`, `current` — each with its own
explanation string for transparency.

---

## Item 9 — State Reflection Engine

```python
from reflection.state_reflection import StateReflectionEngine

state_reflection = StateReflectionEngine(state_transitions)
report = state_reflection.generate("sayan")

report.skill_evolution      # [EvolutionEntry(attribute="favorite_language", chain=["Python","Rust"], ...)]
report.interest_evolution     # research_focus, favorite_topic, ...
report.project_evolution        # current_project, project_status, ...
report.identity_evolution         # city, role, affiliation, ...

report.summary()
# "Skills: favorite language evolved through 1 change(s): Python → Rust"
```

Deliberately built as a thin synthesis layer over `StateTransitionEngine`
history — no new memory storage, exactly as instructed. The
attribute-to-dimension mapping is overridable per deployment.

`recent_shifts(entity, days=30)` returns a flat, dimension-tagged list
of everything that changed in a time window, for dashboards/digests.

---

## Item 10 — StateBench-lite (`evaluation/state_metrics.py`)

`StateMetrics` extends `AdaptiveAgentEvaluator` (v0.3.6), completing the
evaluation tower:

```
MemoryEvaluator → ... → AgentEvaluator → AdaptiveAgentEvaluator → StateMetrics
```

| Metric | Method |
|---|---|
| Current State Accuracy | `current_state_accuracy(tracker, cases)` |
| Historical State Accuracy | `historical_state_accuracy(tracker, cases)` |
| Transition Accuracy | `transition_accuracy(tracker, cases)` — with time tolerance |
| State Hallucination Rate | `state_hallucination_rate(predicted, ground_truth_exists)` |
| Belief Drift | `belief_drift(confidence_before, confidence_after)` |
| Truth Consistency | `truth_consistency(truth_manager, pairs, tracker)` |

```python
from blix_eval import StateMetrics, StateAccuracyCase, TransitionAccuracyCase

metrics = StateMetrics()
results = metrics.run_statebench(tracker, truth_manager, state_cases, transition_cases)
# {"current_state_accuracy": 1.0, "historical_state_accuracy": 1.0,
#  "transition_accuracy": 1.0, "truth_consistency": 1.0, ...}
```

`truth_consistency` specifically catches the failure mode where a
CLOSED snapshot is incorrectly still marked `ACTIVE` in `TruthManager`
— verified by a dedicated test that forces this inconsistency and
confirms the metric drops to 0.0.

---

## API Endpoints (v0.3.7 additions)

| Method | Path | Description |
|---|---|---|
| GET | `/temporal/state/{entity}/{attribute}` | Current tracked value + truth status |
| GET | `/temporal/state/{entity}/{attribute}/history` | Full chronological history |
| GET | `/temporal/state/{entity}/{attribute}/at?timestamp=` | Value at a point in time |
| POST | `/temporal/query` | Natural-language temporal query |
| GET | `/temporal/evolution/{entity}` | Full 4-dimension evolution report |
| GET | `/temporal/evolution/{entity}/recent?days=` | Recent cross-dimension shifts |
| GET | `/temporal/beliefs?status=` | List beliefs, optionally filtered by TruthStatus |
| GET | `/temporal/truth/{record_id}` | TruthRecord status + history |
| POST | `/temporal/resolve` | Classify + resolve a contradiction pair |

### Example: `/temporal/query`

```json
POST /temporal/query
{ "query": "How has my favorite language evolved?" }

→ 200 OK
{
  "query": "How has my favorite language evolved?",
  "query_type": "evolution",
  "answer": "Python → PyTorch → Rust",
  "timeline": [ {"value": "Python", "is_active": false, ...}, ... ],
  "explanation": "sayan's favorite language evolved through 3 stage(s): Python → PyTorch → Rust."
}
```

---

## Architecture After v0.3.7

```
Memory Layer
   ↓
Knowledge Layer
   ↓
Reasoning Layer
   ↓
Planning Layer
   ↓
Execution Layer
   ↓
Verification & Replanning Layer      (v0.3.6)
   ↓
Temporal State Layer                  (v0.3.7 NEW)
   ├── StateTracker / StateTransitionEngine
   ├── TruthManager / ContradictionResolver
   ├── BeliefStore
   ├── TemporalGraph
   ├── TemporalRetriever
   ├── TemporalQueryEngine
   └── StateReflectionEngine
   ↓
API Layer
```

Blix is now a **Temporal Cognitive System**: it doesn't just know
things and act on goals — it maintains a coherent, time-aware model of
what it believes, why, and how that's changed.

---

## Test Coverage

```
tests/test_v03_features.py      75 tests
tests/test_v031_features.py    118 tests
tests/test_v032_features.py    129 tests
tests/test_v033_features.py     76 tests
tests/test_v034_features.py    116 tests
tests/test_v035_features.py    140 tests
tests/test_v036_features.py    142 tests
tests/test_v037_features.py    142 tests  ← NEW
tests/test_memory_manager.py    ~60 tests
tests/test_semantic_retriever   ~40 tests
tests/test_tutor_agent.py        17 tests
──────────────────────────────────────────
Total                          1074 tests  all passing
```

```bash
python -m pytest tests/ -q
# 1074 passed
```

Integration tests specifically verify: the literal Python→Rust scenario
from the spec end-to-end (transition recorded, old snapshot closed,
reinforcement vs. transition distinguished correctly); all four
contradiction cases classified and resolved correctly against the
spec's own examples (Delhi→Kolkata, Python+Rust, AI/Artificial
Intelligence, Mumbai/Chennai); all five temporal query types; and a
forced truth-consistency violation correctly dropping the metric to 0.0.

---

## What Was Deliberately NOT Built

Per the spec's explicit scope boundary:

- No multi-agent architecture, new planners, or execution tools
- No voice, browser automation, or UI improvements
- No additional memory layers beyond what's needed for state/truth/belief
- No reinforcement learning or world model

This release exists for exactly three capabilities — State Tracking,
Truth Maintenance, and Belief Evolution — and nothing else.

---

## Migration from v0.3.6

No breaking changes. New files only:

```
core/state_tracker.py
core/state_transition.py
core/truth_manager.py
core/contradiction_resolver.py
memory/beliefs.py
retrieval/temporal_retriever.py
graph/temporal_graph.py
reasoning/temporal_query.py
reflection/state_reflection.py
evaluation/state_metrics.py
api/routers/temporal.py
```

`BlixContext` gains: `state_tracker`, `state_transitions`, `truth_manager`,
`belief_store`, `contradiction_resolver`, `temporal_graph`,
`temporal_retriever`, `temporal_query_engine`, `state_reflection`,
`state_metrics`.

New storage files: `memory/state_snapshots.json`, `memory/state_transitions.json`,
`memory/truth_records.json`, `memory/beliefs.json`, `memory/temporal_graph.json`.

`blix_eval` now also exports `StateMetrics`, `StateAccuracyCase`, and
`TransitionAccuracyCase`.
