# Blix v0.3.3 — Platformization & Knowledge Intelligence

> Upgrade from v0.3.2. Turns the Blix cognitive architecture into an
> **API-first platform**: a FastAPI backend exposes every subsystem as a
> REST endpoint with OpenAPI/Swagger docs, streaming responses, and async
> execution. The insight engine is upgraded from summaries to actionable
> intelligence with evidence and recommendations.
>
> **No breaking changes.** All new files are additive. The CLI `app.py`
> continues to work unchanged alongside the new API.

---

## Theme

```
v0.3.0 → Memory Architecture
v0.3.1 → Knowledge System
v0.3.2 → Reflection System
v0.3.3 → Knowledge Intelligence + API Platform   ← this release
```

---

## New Modules

```
api/
├── __init__.py
├── context.py            — BlixContext: unified dependency container
├── deps.py               — FastAPI dependency injection (get_context)
├── models.py             — All Pydantic request/response types
├── server.py             — FastAPI app factory + uvicorn entrypoint
└── routers/
    ├── __init__.py
    ├── chat.py           — POST /chat, POST /chat/stream, POST /chat/mql
    ├── memory.py         — GET/POST /memory, /memory/{id}, /memory/search
    ├── knowledge.py      — GET/POST /knowledge/facts, /knowledge/synthesize
    ├── reflection.py     — GET/POST /reflection/insights, /reflection/run
    ├── graph.py          — GET/POST /graph, /graph/nodes, /graph/path
    ├── documents.py      — POST /documents/upload, GET /documents
    └── stats_goals.py    — GET /stats, GET/POST /goals, /goals/{id}/...

reflection/
└── insight_engine.py     — InsightGenerationEngine (Feature 3 upgrade)
```

---

## Feature 1 — FastAPI Backend ⭐⭐⭐⭐⭐

```bash
# Start the API server
uvicorn api.server:app --reload --port 8000

# OpenAPI docs
open http://localhost:8000/docs

# Redoc
open http://localhost:8000/redoc
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/chat` | Single-turn chat; returns full reply |
| POST | `/chat/stream` | Streaming SSE chat |
| POST | `/chat/mql` | Memory Query Language command |
| GET | `/memory` | Paginated memory list |
| GET | `/memory/{id}` | Memory by id |
| GET | `/memory/search?q=...` | Semantic memory search |
| GET | `/memory/lifecycle` | Lifecycle state counts |
| POST | `/memory/{id}/compress` | Manually compress a memory |
| GET | `/knowledge/facts` | Canonical facts (filterable) |
| GET | `/knowledge/facts/strongest` | Top-k by confidence |
| POST | `/knowledge/synthesize` | Generate a knowledge report |
| GET | `/knowledge/reports` | List knowledge reports |
| GET | `/knowledge/reports/{id}` | Single report |
| GET | `/reflection/insights` | Recent reflection insights |
| GET | `/reflection/insights/actionable` | Actionable insights (v0.3.3) |
| POST | `/reflection/run` | Trigger a reflection pass |
| POST | `/reflection/insights/generate` | Full insight generation pass |
| GET | `/graph` | Full graph snapshot |
| GET | `/graph/nodes` | Node list (filterable by kind) |
| GET | `/graph/nodes/{id}` | Node detail + neighbours |
| GET | `/graph/path?from_id=&to_id=` | Shortest path |
| GET | `/graph/centrality` | Top nodes by degree centrality |
| POST | `/graph/relations` | Add/update a relation |
| POST | `/documents/upload` | Upload PDF/TXT/MD/DOCX/HTML |
| GET | `/documents` | List processed documents |
| GET | `/stats` | Dashboard statistics |
| GET | `/goals` | Goal list (filterable) |
| POST | `/goals` | Create goal |
| GET | `/goals/{id}` | Goal by id |
| PATCH | `/goals/{id}/progress` | Set progress override |
| POST | `/goals/{id}/blockers` | Add blocker |
| DELETE | `/goals/{id}/blockers` | Resolve blocker |
| POST | `/goals/{id}/milestones` | Add milestone |

### Architecture

```
BlixContext (api/context.py)
    — constructs once at startup (lifespan)
    — injected into all endpoints via FastAPI Depends
    — wraps every v0.3–v0.3.3 component
    — provides dashboard_stats() aggregate

FastAPI App (api/server.py)
    — create_app() factory for testability
    — CORS middleware (permissive for dev)
    — uvicorn-compatible module-level `app`
    — OpenAPI at /docs, Redoc at /redoc
```

### Streaming (SSE)

`POST /chat/stream` returns Server-Sent Events:

```
data: {"token": "This "}
data: {"token": "is a "}
data: {"token": "reply."}
data: {"done": true, "memory_id": 42}
```

Currently chunks the completed LLM response in 10-char pieces (same-latency
simulation). Replace with a native streaming provider in v0.4.

---

## Feature 3 — Insight Generation Engine (upgrade)

**File:** `reflection/insight_engine.py`

Upgrades reflection from "summary" to actionable intelligence:

```
v0.3.2: Insight{insight: str, confidence: float}
v0.3.3: ActionableInsight{insight, category, confidence, evidence, recommendation}
```

```python
from reflection.insight_engine import InsightGenerationEngine, InsightCategory

engine = InsightGenerationEngine(Path("memory/actionable_insights.json"), llm=llm)
insights = engine.generate_all(memories=all_memories, goals=goals, project_states=states)
```

Four analysis passes:

| Pass | `InsightCategory` | Example |
|---|---|---|
| `analyze_topic_trends()` | `research_interest` | "Most conversations involve AI systems." |
| `analyze_activity_trend()` | `trend` | "User's focus shifted from chatbots to memory systems." |
| `analyze_bottlenecks()` | `bottleneck` | "'evaluation framework' blocks 3 goals." |
| `analyze_project_patterns()` | `project_pattern` | "Project 'Blix' has accumulated 4 risks." |

Each `ActionableInsight` includes:
- `evidence` — supporting data points
- `recommendation` — concrete action (LLM-phrased if available, else templated)

`ActionableInsight.to_insight()` converts to v0.3.2 `Insight` for backward storage compat.

---

## BlixContext (unified wiring)

**File:** `api/context.py`

```python
from api.context import BlixContext

ctx = BlixContext.build()   # standard path (memory/ dir)
ctx = BlixContext(memory_dir=Path("/custom"))  # tests

ctx.agent        # TutorAgent
ctx.graph        # MemoryGraph
ctx.goals        # GoalTracker
ctx.reflection   # ReflectionEngine
ctx.insight_engine  # InsightGenerationEngine (NEW)
ctx.consolidation   # ConsolidationEngine
ctx.synthesis    # KnowledgeSynthesisEngine
ctx.mql          # MQLEngine
ctx.dashboard_stats()  # → dict with 19 counters
```

`BlixContext` is the single construction path shared by both the CLI
(`app.py`) and the API (`api/server.py`) — eliminating configuration drift.

---

## Dashboard Statistics

`GET /stats` (and `ctx.dashboard_stats()`) returns:

```json
{
  "memory_count": 142,
  "embedding_index_size": 142,
  "knowledge_facts": 37,
  "projects": 5,
  "projects_at_risk": 1,
  "graph_nodes": 89,
  "graph_edges": 112,
  "goals": 8,
  "active_goals": 3,
  "insights": 24,
  "reflection_records": 18,
  "knowledge_reports": 6,
  "semantic_clusters": 12,
  "lifecycle_state_counts": {"active": 120, "compressed": 18, "archived": 4, "deleted": 0},
  "contradictions_unresolved": 2,
  "session_count": 31,
  "daily_summaries": 7,
  "weekly_summaries": 2,
  "background": {"running": true, "queue_size": 0, "processed": 389, "failed": 0}
}
```

---

## Test Coverage

```
tests/test_v03_features.py      75 tests
tests/test_v031_features.py    118 tests
tests/test_v032_features.py    129 tests
tests/test_v033_features.py     76 tests  ← NEW
tests/test_memory_manager.py    ~60 tests
tests/test_semantic_retriever   ~40 tests
tests/test_tutor_agent.py        17 tests
──────────────────────────────────────────
Total                           534 tests  all passing
```

```bash
python -m pytest tests/ -q
# 534 passed
```

---

## Running the API

```bash
# Install API extras
pip install -e ".[api]"

# Start server (development)
uvicorn api.server:app --reload --port 8000

# Or run directly
python api/server.py

# Explore endpoints
open http://localhost:8000/docs
```

---

## Migration from v0.3.2

No breaking changes. Add these new files:
```
api/                  (entirely new)
reflection/insight_engine.py   (new)
```

`BlixContext` is optional — existing `app.py` code continues to work.
To adopt `BlixContext` in your own CLI code:

```python
# Before (v0.3.2 style):
from config.settings import settings
mm = MemoryManager(...)
graph = MemoryGraph(...)
# ... 20+ lines of manual wiring

# After (v0.3.3):
from api.context import BlixContext
ctx = BlixContext.build()
agent = ctx.agent
```
