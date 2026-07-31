# Blix — Cognitive AI Agent System

**v0.3.15 · Hybrid Graph-Based Semantic Hierarchical Memory**

Blix is a modular, symbolic cognitive agent architecture combining
persistent memory, causal reasoning, imagination, curiosity, and active
experimentation into a unified cognitive stack.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          BLIX v0.3.15                           │
├─────────────────────────────────────────────────────────────────┤
│  Cognitive Stack: Curiosity · Metacognition · Planning · Causal  │
├─────────────────────────────────────────────────────────────────┤
│                           HGSHM                                  │
│   GraphStore (hgshm.db) · VectorIndex (sqlite-vec, 256-dim)     │
│   HierarchyManager · ConsolidationEngine · ContextBuilder        │
│   HybridRetriever: 11-factor ranked fusion                       │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Pipeline (v0.3.15)

```
Query → Semantic · Vector · Graph Expansion · Temporal
     → Importance Ranking · Hierarchy · Contradictions · Causal
     → MemoryContext (typed object)
     → Reasoning
```

## Memory Hierarchy

```
Raw → Episode → Conversation → Session → Daily → Weekly
   → Monthly → Project → Concept → Principle → Knowledge → WorldModel
```
Compression is automatic. SUMMARISES edges preserve all source references.

## Quick Start

```python
from pathlib import Path
from memory.hybrid.hgshm import HGSHM

h = HGSHM(Path("memory/"))
h.believe("Deployment failed due to timeouts", confidence=0.85)
h.observe_cause("peak_traffic", "timeout_errors")
h.add_principle("Monitor timeouts before deployment windows")

ctx = h.recall("deployment failure root cause", top_k=10)
print(ctx.get_text_summary())
```

## Installation

```bash
pip install fastapi uvicorn httpx pydantic python-multipart sqlite-vec
```

## Graph Edge Types

supports · contradicts · causes · depends_on · part_of · derived_from
similar_to · explains · references · precedes · follows · belongs_to
requires · related_to · enables · blocks · summarises · instance_of

## Benchmarks

| Operation | Scale | Latency |
|-----------|-------|---------|
| Batch embedding | 100 texts | < 100ms |
| Vector ANN search | 200 nodes | < 10ms |
| Context assembly | 50 nodes | < 500ms |
| Consolidation | 100 nodes | < 200ms |

## Tests

194 passing (82 v0.3.13 + 21 gap fixes + 91 v0.3.15 HGSHM)

## Roadmap

v0.3.15 HGSHM → v0.4 Cognitive Kernel → v0.5 Adaptive Intelligence
→ v0.6 Autonomous Researcher → v1.0 Cognitive Operating System

## License

MIT
