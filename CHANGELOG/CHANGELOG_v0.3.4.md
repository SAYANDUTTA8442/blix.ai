# Blix v0.3.4 — Knowledge Graph Reasoning & Cognitive Queries

> Upgrade from v0.3.3. Enables Blix to **reason over knowledge** instead
> of merely retrieving it. The system can now answer structured questions
> by traversing the knowledge graph, infer transitive relationships across
> multiple hops, and attach a full evidence chain to every answer.
>
> No breaking changes. All new modules are additive and follow the
> established dependency-injection pattern.

---

## Theme

```
v0.3.0 → Memory Architecture
v0.3.1 → Knowledge System
v0.3.2 → Reflection System
v0.3.3 → API Platform
v0.3.4 → Graph Reasoning    ← this release
```

---

## New Modules

```
core/
├── cognitive_query_engine.py  Features 1+2 — CognitiveQueryEngine
└── explainability.py           Feature 6  — ExplainabilityEngine

knowledge/
└── research_assistant.py       Feature 5  — ResearchAssistant

reflection/
└── mql_v2.py                   Feature 3  — MQLv2Engine (expression queries)

evaluation/
└── reasoning.py                Feature 7  — ReasoningEvaluator

api/routers/
└── reasoning_research.py      API        — /reason + /research endpoints
```

---

## Feature 1 — Graph Reasoning Engine

**File:** `core/cognitive_query_engine.py` → `CognitiveQueryEngine`

```python
from core.cognitive_query_engine import CognitiveQueryEngine

cqe = CognitiveQueryEngine(graph=graph, reasoner=graph_reasoner)
result = cqe.query("What does Blix use?")
# result.answer   → ["FastAPI", "ChromaDB", "Transformers"]
# result.trace    → ReasoningTrace with steps, confidence, source_memory_ids
```

Answers natural-language queries by pattern-matching the query against
14 pre-defined templates and dispatching a directed graph traversal:

| Query template | Traversal |
|---|---|
| "What does X use?" | Outgoing `uses` edges from X |
| "What does X work on?" | Outgoing `works_on` edges from X |
| "What are X's goals?" | Outgoing `goal_is` edges from X |
| "Who works on X?" | Inverse traversal: nodes pointing to X |
| "Who collaborates with X?" | Inverse `collaborates_with` edges |
| "What does X study?" | Outgoing `studies_at` edges from X |
| "What does X …?" | Generic outgoing traversal (fallback) |

All queries return a `QueryResult{answer, trace, raw_nodes}`. The trace
carries the full `ReasoningTrace{steps, source_memory_ids, confidence, explanation}`.

---

## Feature 2 — Multi-Hop Inference

Three inference modes on `CognitiveQueryEngine`:

### `multi_hop_query(start, end)` — connecting-path inference

```python
result = cqe.multi_hop_query("Sayan", "FastAPI")
# Finds: Sayan → works_on → Blix → uses → FastAPI
# result.answer → ["Blix"]  (the connecting intermediate)
```

BFS up to `max_depth` hops, collecting all intermediate nodes.

### `infer_transitive(entity, relation, depth)` — transitive closure

```python
result = cqe.infer_transitive("Blix", "uses", depth=2)
# depth=1: FastAPI, ChromaDB, Transformers
# depth=2: everything those use in turn
```

Follows only edges of the specified relation type, producing the full
transitive closure up to `depth` hops.

Both return `QueryResult` with a full `ReasoningTrace`.

---

## Feature 3 — MQL v2 (Expression-Style Queries)

**File:** `reflection/mql_v2.py` → `MQLv2Engine`

Extends the v0.3.2 `"show ..."` commands with expression-style queries:

```
memories where topics contains "transformers"
memories where importance >= 0.7
facts about "attention"
facts min_confidence = 0.8
insights last_30_days
insights category = "trend"
goals status = active
goals priority <= 2
goals project = "Blix"
graph neighbours "Sayan"
graph path "Sayan" to "FastAPI"
query "What does Blix use?"          ← Feature 1 cognitive query
infer "Blix" via "uses" depth 2      ← Feature 2 transitive inference
multihop "Sayan" to "FastAPI"        ← Feature 2 multi-hop
```

`MQLv2Engine` transparently falls back to the v0.3.2 `MQLEngine`
"show ..." commands, so no existing MQL usage breaks.

```python
from reflection.mql_v2 import MQLv2Engine

mql = MQLv2Engine(
    memory_manager=mm, consolidation_engine=ce,
    goal_tracker=gt, graph=graph,
    graph_reasoner=reasoner, cognitive_query_engine=cqe,
    # + all v0.3.2 components for "show ..." fallback
)
result = mql.run('query "What does Blix use?"')
result.trace   # full reasoning trace
```

The `is_mql_command()` check now also detects expression-style prefixes:
`memories`, `facts`, `insights`, `goals`, `graph`, `query`, `infer`, `multihop`.

---

## Feature 5 — Research Assistant Mode

**File:** `knowledge/research_assistant.py` → `ResearchAssistant`

```
Paper / Technical Doc
    ↓ DocumentProcessor (v0.3.2)
    ↓ ResearchAssistant.process()
    ↓ ResearchNotes{
          summary, methodology,
          key_findings, limitations, future_work,
          related_concepts, entities, related_topics,
          confidence
      }
    ↓ ConsolidationEngine    ← key findings → canonical facts
    ↓ MemoryGraph            ← entities → graph nodes/edges
    ↓ KnowledgeSynthesisEngine ← notes → synthesis source
```

```python
from knowledge.research_assistant import ResearchAssistant

ra = ResearchAssistant(
    notes_file=Path("memory/research_notes.json"),
    llm=llm, consolidation_engine=ce, graph=graph,
)
notes = ra.process(processed_doc)
notes.summary          # "2-3 sentence high-level summary"
notes.methodology      # "We use hierarchical compression."
notes.key_findings     # ["Attention improves recall by 20%."]
notes.limitations      # ["Only tested on small datasets."]
notes.future_work      # ["Test on multilingual data."]

ra.search("attention")          # text-based search over all notes
ra.get("doc_001")               # retrieve by doc_id
```

API endpoint: `GET /research`, `GET /research/{doc_id}`, `GET /research/search?q=...`

---

## Feature 6 — Explainability Layer

**File:** `core/explainability.py` → `ExplainabilityEngine`, `ExplainedResponse`

Every Blix answer can now be annotated with a full evidence chain:

```python
from core.explainability import ExplainabilityEngine

explain = ExplainabilityEngine(
    memory_manager=mm, retriever=retriever,
    consolidation_engine=ce, reflection_engine=re,
    graph=graph, graph_reasoner=reasoner,
)

result = explain.explain(
    answer="FastAPI, ChromaDB",
    query="What does Blix use?",
    reasoning_trace=cqe_trace,
)

print(result.explain_str())
# Answer: FastAPI, ChromaDB
# Overall confidence: 0.87
# Total evidence sources: 4
#
# Memory evidence:
#   - Memory #14: "Integrated ChromaDB as the embedding store" (rel=0.85)
#
# Canonical facts:
#   - Fact fact_3: "Blix uses FastAPI for its API layer." (conf=0.88, n=12)
#
# Graph evidence:
#   - Graph path: Blix →[uses]→ FastAPI (conf=0.95)
#   - Graph path: Blix →[uses]→ ChromaDB (conf=0.90)
#
# Reasoning path:
#   Blix →[uses]→ FastAPI  (confidence=0.95)
#   Blix →[uses]→ ChromaDB (confidence=0.90)
```

`ExplainedResponse.overall_confidence` is a weighted mean:
- Memory: weight 1.0
- Canonical facts: weight 1.5 (highest — most verified)
- Graph: weight 1.2
- Insights: weight 0.8

API endpoint: `GET /reason/explain?q=...&answer=...`

---

## Feature 7 — Evaluation Framework v2

**File:** `evaluation/reasoning.py` → `ReasoningEvaluator`

Extends `CognitiveEvaluator` (v0.3.3) with reasoning-specific metrics:

| Metric | Description |
|---|---|
| `reasoning_accuracy` | Fraction of expected answers found (substring, case-insensitive) |
| `reasoning_precision` | Fraction of predicted answers that are correct (anti-hallucination) |
| `graph_coverage` | Fraction of expected entities+edges present in the graph |
| `path_accuracy` | Whether predicted hop count matches expected (±tolerance) |
| `inference_recall` | Fraction of expected transitive nodes found |
| `explainability_score` | Completeness of evidence chain (0–1) |

```python
from blix_eval import ReasoningEvaluator, ReasoningCase

ev = ReasoningEvaluator()
results = ev.evaluate_reasoning(
    cases=[
        ReasoningCase(
            case_id="c1",
            query="What does Blix use?",
            expected_answers=["FastAPI", "ChromaDB"],
        ),
    ],
    query_fn=cqe.query,
    graph=graph,
    expected_graph_entities=["Sayan", "Blix", "FastAPI"],
    expected_graph_edges=[("Sayan", "works_on", "Blix")],
)
# results: {"reasoning_accuracy": 1.0, "graph_coverage": 1.0, ...}
```

`ReasoningCase` also supports multi-hop (`expected_intermediates`),
transitive inference (`expected_transitive_nodes`), and path hop checks.

---

## API Endpoints (v0.3.4 additions)

| Method | Path | Description |
|---|---|---|
| POST | `/reason/query` | Natural-language cognitive graph query |
| POST | `/reason/multihop` | Multi-hop path query (start → ? → end) |
| POST | `/reason/infer` | Transitive inference (entity, relation, depth) |
| GET | `/reason/explain` | Full evidence chain for a query |
| GET | `/research` | List all research notes |
| GET | `/research/search?q=` | Search research notes |
| GET | `/research/{doc_id}` | Get notes by doc id |

### Example: `/reason/query`

```json
POST /reason/query
{
  "query": "What does Blix use?",
  "explain": true
}

→ 200 OK
{
  "query": "What does Blix use?",
  "answer": ["FastAPI", "ChromaDB", "Transformers"],
  "is_empty": false,
  "trace": {
    "steps": [
      {"from": "Blix", "relation": "uses", "to": "FastAPI", "confidence": 0.95},
      {"from": "Blix", "relation": "uses", "to": "ChromaDB", "confidence": 0.90}
    ],
    "confidence": 0.9,
    "explanation": "Traversed outgoing 'uses' edges from 'Blix'. Found 3 result(s)."
  },
  "explanation": { ... }
}
```

---

## Architecture After v0.3.4

```
User / API Client
      │
      ├── /chat               → TutorAgent (LLM + memory)
      ├── /reason/query       → CognitiveQueryEngine
      │       │
      │       ▼
      │   Knowledge Graph
      │       │
      │       ▼
      │   ReasoningTrace ──► ExplainabilityEngine
      │                               │
      │               ┌───────────────┼────────────────┐
      │           MemoryEvidence  FactEvidence  GraphEvidence
      │
      ├── /research           → ResearchAssistant
      │       │
      │       ▼
      │   ResearchNotes
      │       ↓        ↓          ↓
      │   MemoryGraph  Facts  Synthesis
      │
      └── /chat/mql (v2)     → MQLv2Engine
              │
    ┌─────────┴──────────┐
    Expression queries   "show …" fallback (v0.3.2)
```

---

## Test Coverage

```
tests/test_v03_features.py      75 tests
tests/test_v031_features.py    118 tests
tests/test_v032_features.py    129 tests
tests/test_v033_features.py     76 tests
tests/test_v034_features.py    116 tests  ← NEW
tests/test_memory_manager.py    ~60 tests
tests/test_semantic_retriever   ~40 tests
tests/test_tutor_agent.py        17 tests
──────────────────────────────────────────
Total                           650 tests  all passing
```

```bash
python -m pytest tests/ -q
# 650 passed
```

---

## What NOT to Build Yet

Consistent with the spec guidance, the following are still deferred to v0.4:

- Autonomous agents / agentic loops
- Multi-agent frameworks
- Browser agents / tool agents
- Self-modifying agents
- Planner / executor separation

**v0.4 will build on the solid reasoning foundation of v0.3.4** to add
structured planning — once reasoning is proven correct by the v0.3.4
evaluation suite.

---

## Migration from v0.3.3

No breaking changes. New files only:

```
core/cognitive_query_engine.py
core/explainability.py
knowledge/research_assistant.py
reflection/mql_v2.py
evaluation/reasoning.py
api/routers/reasoning_research.py
```

`BlixContext` gains three new attributes: `cognitive_query_engine`,
`explainability_engine`, `research_assistant`. The MQL engine is
transparently upgraded from `MQLEngine` to `MQLv2Engine` (backwards
compatible: all "show ..." commands still work).

`blix_eval` now also exports `ReasoningEvaluator` and `ReasoningCase`.
