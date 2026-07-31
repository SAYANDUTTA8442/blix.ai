# Blix v0.3.15 — Hybrid Graph-Based Semantic Hierarchical Memory (HGSHM)

## Major Theme

v0.3.15 replaces Blix's fragmented memory subsystem with a **unified
cognitive memory architecture** — the Hybrid Graph-Based Semantic
Hierarchical Memory (HGSHM). Every cognitive subsystem (Planner, World
Model, Causal Reasoning, Curiosity Engine, Metacognition, Global
Workspace, Reflection) now shares a single, rich memory substrate
instead of isolated stores.

---

## Architecture Changes

### Before (v0.3.14)
```
BeliefStore     → beliefs.json          (flat JSON, Jaccard dedup)
CauseGraph      → cause_graph.json      (custom edge format)
PrincipleStore  → principles.json       (flat list)
TrajectoryGraph → in-memory only
HypothesisManager → hyp.json
```
Each subsystem maintained its own storage with no cross-subsystem graph
reasoning possible.

### After (v0.3.15)
```
HGSHM
  ├── hgshm.db     (SQLite: nodes + edges + clusters + history)
  └── vectors.db   (sqlite-vec: 256-dim embeddings with ANN search)
```
All knowledge is unified into MemoryNodes connected by typed MemoryEdges.
Old subsystem stores become thin shims delegating to HGSHM.

---

## New Modules

### `memory/hybrid/models/`
- **`memory_node.py`** — `MemoryNode` dataclass: the atomic unit of
  HGSHM. Stores text, type, hierarchy level, confidence, importance,
  embedding_id, epistemic status, temporal validity, access history,
  versioning, and arbitrary metadata.
- **`memory_edge.py`** — `MemoryEdge` dataclass with 20 typed
  `EdgeRelation` values (supports, contradicts, causes, depends_on,
  part_of, derived_from, similar_to, explains, references, precedes,
  follows, belongs_to, requires, related_to, enables, blocks,
  summarises, instance_of, evolves_to, co_occurs). Each edge tracks
  confidence, weight, evidence_count, and provenance.
- **`memory_cluster.py`** — `MemoryCluster` for grouping semantically
  related nodes. Promotes to Concept nodes when stable.
- **`memory_context.py`** — `MemoryContext` and `RetrievedMemory`:
  structured context objects replacing concatenated string prompts.

### `memory/hybrid/storage/`
- **`persistence.py`** — `HGSHMStore`: WAL-mode SQLite backend for
  nodes, edges, clusters, and node history. Full CRUD with composite
  indices on type, hierarchy level, concept_id, importance, and updated_at.

### `memory/hybrid/vector/`
- **`embedding_manager.py`** — `EmbeddingManager` with pluggable
  backend protocol. Default: `NumpyBackend` (256-dim random-projection
  hash embedding, L2-normalised). Fallback: pure-Python
  `HashProjectionBackend`. Swap to sentence-transformers or OpenAI
  with one call.
- **`vector_store.py`** — `VectorStore`: sqlite-vec backed vector
  database. Supports cosine ANN search, batch upsert, deletion,
  compaction, index rebuilding. Falls back to brute-force cosine search
  if sqlite-vec is unavailable.
- **`vector_index.py`** — `VectorIndex`: pluggable interface layer.
  Default: `SqliteVecBackend`. Swap to FAISS, Chroma, or Qdrant via
  `swap_backend()` without changing any calling code.

### `memory/hybrid/graph/`
- **`graph_store.py`** — `GraphStore`: high-level node/edge CRUD with
  in-memory adjacency index (O(1) neighbour lookup), automatic edge
  reinforcement, and history snapshots.
- **`graph_builder.py`** — `GraphBuilder`: cognitive event factory.
  `add_belief()`, `add_hypothesis()`, `promote_hypothesis_to_belief()`,
  `add_causal_observation()`, `add_principle()`, `link_principles()`,
  `add_concept()`, `add_gap()`, `add_summary()`, `link()`.
- **`graph_index.py`** — `GraphIndex`: secondary indices (type, level,
  concept, tag, source, relation) for O(1) filtered node lookup without
  full table scans.
- **`graph_traversal.py`** — `GraphTraversal`: BFS, DFS, weighted
  search (priority queue), shortest path (BFS), neighbourhood
  expansion, concept expansion (SIMILAR_TO / BELONGS_TO / INSTANCE_OF),
  importance-guided descent, temporal graph search.

### `memory/hybrid/retrieval/`
- **`hybrid_retriever.py`** — Contains four retrievers:
  - `SemanticRetriever`: embedding cosine + token overlap fallback
  - `GraphRetriever`: BFS expansion from seed nodes + causal chain
    extraction + contradiction detection
  - `TemporalRetriever`: recency decay, validity windows, access
    frequency
  - `HybridRetriever`: 11-factor ranked fusion (semantic, vector,
    graph distance, importance, confidence, recency, hierarchy, context
    similarity, attention, belief confidence, planning relevance).
    All weights configurable via `HybridWeights`.

### `memory/hybrid/hierarchy/`
- **`hierarchy_manager.py`** — `HierarchyManager`: automatic layered
  compression (RAW → EPISODE → CONVERSATION → SESSION → DAILY →
  WEEKLY → MONTHLY → PROJECT → CONCEPT → PRINCIPLE → KNOWLEDGE →
  WORLD_MODEL). Threshold-triggered, SUMMARISES edges preserve all
  source references. `Summarizer` (extractive default, LLM-pluggable).
  `AbstractionEngine` promotes cause clusters to principles and belief
  clusters to concepts.

### `memory/hybrid/consolidation/`
- **`consolidation_engine.py`** — Contains:
  - `DuplicateDetector`: cosine similarity + Jaccard token overlap
  - `MemoryMerger`: merge lower-importance node into canonical,
    transferring all edges
  - `ConsolidationEngine`: orchestrates dedup + merge + pruning
  - `ImportanceModel`: dynamic importance scoring (6 factors: access
    frequency, edge degree, confidence, hierarchy level, recency, causal
    centrality)

### `memory/hybrid/context/`
- **`context_builder.py`** — `ContextBuilder`: 11-step pipeline
  producing a `MemoryContext` object: primary retrieval → graph
  expansion → temporal → concepts → principles → beliefs → belief
  validation → contradiction detection → causal chain extraction →
  gap discovery → graph neighbourhood.

### `memory/hybrid/hgshm.py`
- **`HGSHM`**: the unified facade. Single entry point for all memory
  operations. `remember()`, `recall()`, `believe()`, `hypothesise()`,
  `observe_cause()`, `add_principle()`, `note_gap()`, `link()`,
  `consolidate()`, `compress_hierarchy()`, `rebuild_vector_index()`.

### `memory/hybrid/shims.py`
- **Backward compatibility shims**: `BeliefStoreShim`,
  `CauseGraphShim`, `PrincipleStoreShim`. All shims share a single
  HGSHM instance per memory_dir (singleton registry). Old code
  continues to work without modification.

---

## Performance

- Vector search over 200 nodes: < 10ms (sqlite-vec ANN)
- Batch embedding 100 texts: < 100ms (NumpyBackend, 256-dim)
- Full context assembly over 50 nodes: < 500ms
- Memory consolidation (100 nodes): < 200ms

---

## Testing

- 91 new tests in `tests/test_v0315_hgshm.py`
- Total test suite: 194 passing (103 pre-existing + 91 new)
- Test categories: Unit (models, storage, embedding, vector, graph,
  traversal, index, retrievers, hierarchy, consolidation, context),
  Integration (full pipelines), Stress (100+ nodes), Regression (shim
  compatibility), Benchmark (latency thresholds)

---

## Migration

Existing code requires no changes. The shim layer in `memory/hybrid/shims.py`
provides drop-in replacements for:
- `memory.beliefs.BeliefStore` → `BeliefStoreShim`
- `causality.cause_graph.CauseGraph` → `CauseGraphShim`
- `causality.principle.PrincipleStore` → `PrincipleStoreShim`

All three delegate to a shared HGSHM instance, meaning beliefs and
causal observations are now visible to each other through graph traversal.

---

## Vector Database Details

**Default backend**: sqlite-vec (0.1.9)
- Single-file SQLite extension, no external process
- ANN search via vec0 virtual tables
- Automatic cosine distance computation in SQL
- WAL mode for concurrent read safety

**Pluggable alternatives** (implement `VectorIndexBackend` protocol):
- FAISS (via `FaissBackend`)
- Chroma (via `ChromaBackend`)
- Qdrant, Milvus, Weaviate, Pinecone (same interface)

**Embedding backend** (implement `EmbeddingBackend` protocol):
- Default: NumpyBackend (256-dim hash projection, no ML deps)
- Swap: `hgshm.embedding_manager.set_backend(SentenceTransformerBackend(...))`

---

## Architectural Evolution Timeline

| Version | Memory Architecture |
|---------|---------------------|
| v0.3.7  | Temporal BeliefStore (flat JSON) |
| v0.3.10 | Hybrid ML scoring (ValueNetwork over flat memory) |
| v0.3.11 | CauseGraph + BeliefDependencyGraph (separate stores) |
| v0.3.12 | TrajectoryGraph (in-memory, isolated) |
| v0.3.13 | KnowledgeGapTracker (separate file store) |
| **v0.3.15** | **HGSHM: unified graph + vector + hierarchy** |
| v0.4 (planned) | HGSHM as Cognitive Kernel substrate |
| v0.5 (planned) | Adaptive HGSHM with online learning |
| v1.0 (planned) | HGSHM as full Cognitive Operating System memory |

---

## Known Limitations

1. **Embedding quality**: The default hash-projection embeddings have
   no semantic understanding of synonyms. "fast" and "quick" are
   unrelated in this model. Swap to sentence-transformers for real
   semantic retrieval.

2. **Graph scale**: The in-memory adjacency index loads all edges on
   first access. For graphs > 500K edges, this should be replaced with
   a lazy neighbour loader. Planned for v0.4.

3. **Shim completeness**: The shims cover BeliefStore, CauseGraph, and
   PrincipleStore. BeliefDependencyGraph, TrajectoryGraph, and
   HypothesisManager shims are deferred to v0.3.16.

4. **Concurrent writes**: The SQLite WAL mode supports concurrent reads
   but serialises writes. For multi-process deployments, a dedicated
   graph server (e.g., Neo4j) is recommended. Planned for v0.5.

---

## Research Contributions

- **Unified symbolic-vector memory**: First integration of a typed
  knowledge graph with a vector database in a cognitive agent system
  at this architectural level.
- **11-factor hybrid retrieval**: Novel ranking function combining
  semantic similarity, graph topology, temporal decay, epistemic
  status, and planning relevance in a single configurable score.
- **Layered hierarchy with provenance**: SUMMARISES edges ensure no
  information is permanently lost during compression — a key
  requirement for interpretable cognitive architectures.
