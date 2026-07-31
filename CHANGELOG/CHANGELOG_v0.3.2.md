# Blix v0.3.2 — Reflection, Consolidation & Knowledge Processing

> Upgrade from v0.3.1. Transforms Blix from a memory architecture into a
> **Personal Cognitive Knowledge System**: it reflects on experience,
> consolidates information into stable facts, tracks goals, understands
> project trajectories, learns from documents and media, and lets the
> user inspect its mind directly via MQL.
>
> All v0.3.2 modules are additive and dependency-injected, following the
> v0.3/v0.3.1 pattern. No breaking changes to existing storage or interfaces.

---

## New Top-Level Packages

```
reflection/
├── reflection_engine.py       Feature 1 — Reflection Engine
├── consolidation_engine.py    Feature 2 — Memory Consolidation Engine
├── goal_tracker.py             Feature 3 — Goal Tracking System
├── project_intelligence.py     Feature 4 — Project Intelligence Engine
├── scheduler.py                 Feature 9 — Reflection Scheduler
└── mql.py                        Feature 10 — Memory Query Language

knowledge/
├── document_processor.py       Feature 6 — Document Processor
├── media_processor.py           Feature 7 — Media Processor
└── synthesis.py                  Feature 8 — Knowledge Synthesis Engine

evaluation/
├── cognitive.py                  Feature 5 — CognitiveEvaluator
└── blix_eval/__init__.py        Feature 5 — deliverable re-export package
```

---

## Feature 1 — Reflection Engine

**File:** `reflection/reflection_engine.py`

```
Memory → Reflection → Insights → Knowledge
```

```python
from reflection.reflection_engine import ReflectionEngine, ReflectionScope

engine = ReflectionEngine(Path("memory/reflections.json"), llm=llm)

engine.reflect_session(session_summary)     # SessionSummary from HierarchyManager
engine.reflect_daily(daily_summary)
engine.reflect_weekly(weekly_summary)
engine.reflect_project(project_summary)     # ProjectSummary from ProjectManager
engine.reflect_learning(learning_state)
engine.reflect_behavior(recent_memories)

engine.get_recent_insights(scope=ReflectionScope.WEEKLY, limit=5)
engine.get_insights_since(datetime(...))
```

Each call produces one or more `Insight` objects:

```json
{"insight": "User's primary focus has shifted from chatbot development to cognitive memory systems.", "confidence": 0.91}
```

With an LLM provider, insights are generated via a structured JSON prompt.
Without one, a heuristic keyword-frequency reflector produces a lower-confidence
generic insight — keeping the engine fully offline-testable.

---

## Feature 2 — Memory Consolidation Engine

**File:** `reflection/consolidation_engine.py`

```
100 similar memories → Canonical Fact
```

```python
from reflection.consolidation_engine import ConsolidationEngine

engine = ConsolidationEngine(Path("memory/canonical_facts.json"))
cf = engine.consolidate("User prefers PyTorch for AI development", source_memory_id=42, topic="ml")
# cf.confidence grows with each corroborating call:
#   confidence = 1 - (1 - base_confidence) ** evidence_count
```

```json
{"fact": "User prefers PyTorch for AI development", "confidence": 0.98, "evidence_count": 37}
```

- **Duplicate detection**: token-level Jaccard similarity (configurable threshold)
- **Merging**: shortest variant becomes canonical; all phrasings retained in `variants`
- **Confidence accumulation**: saturating formula caps at 0.99
- **Memory compression hook**: `consolidatable_memory_ids(min_evidence=3)` returns
  memory ids safe to hand to `MemoryLifecycleManager.compress()` (v0.3.1), always
  preserving the first/original source memory for provenance

---

## Feature 3 — Goal Tracking System

**File:** `reflection/goal_tracker.py`

New entities: `Goal`, `Milestone`, `Task`, `Blocker`. `Goal.progress` is
**computed** from milestone/task completion ratios (or overridden explicitly):

```python
from reflection.goal_tracker import GoalTracker, GoalStatus

gt = GoalTracker(Path("memory/goals.json"))
g = gt.create_goal("Build Blix v0.4", priority=1, related_project="Blix")
gt.add_milestone(g.goal_id, "Design phase")
gt.add_task(g.goal_id, "Write eval harness")
gt.complete_item(g.goal_id, "Design phase")
gt.add_blocker(g.goal_id, "evaluation framework")

gt.prioritized_goals()          # active goals sorted by priority then progress
g.to_summary_dict()
```

```json
{"goal": "Build Blix v0.4", "progress": 72, "status": "active", "blockers": ["evaluation framework"]}
```

A goal auto-transitions to `COMPLETED` when progress reaches 100%.

---

## Feature 4 — Project Intelligence Engine

**File:** `reflection/project_intelligence.py`

Layers a `ProjectState` (focus, priority, progress, risks, risk_level,
next_steps, related_memory_ids, related_goal_id) on top of v0.3's
`ProjectSummary` — fully additive, separate `project_intelligence.json`.

```python
from reflection.project_intelligence import ProjectIntelligenceEngine

pi = ProjectIntelligenceEngine(Path("memory/project_intelligence.json"), project_manager=pm)
pi.set_focus("Blix", "Reflection Engine")
pi.add_risk("Blix", "evaluation framework incomplete")
pi.sync_progress_from_goal("Blix", goal)        # pulls progress/blockers from GoalTracker
pi.link_memories("Blix", [101, 102, 103])

pi.project_report("Blix")        # merges ProjectState + ProjectSummary
pi.at_risk_projects()
```

```json
{"project": "Blix", "focus": "Reflection Engine", "progress": 68, "risk_level": "medium"}
```

Risk level is heuristic: 0 risks → low, 1-2 → medium, 3+ → high, with
escalation if stagnant (no update within `stagnation_days`, default 14).

---

## Feature 5 — Advanced Evaluation Framework (`blix_eval`)

**Files:** `evaluation/cognitive.py`, `evaluation/blix_eval/__init__.py`

The spec-mandated `blix_eval/` deliverable re-exports the full evaluation
stack (v0.3 → v0.3.2) under one namespace:

```python
from blix_eval import CognitiveEvaluator, HypothesisRegistry, EvalDataset

ce = CognitiveEvaluator()
ce.recall_at_k(retrieved, relevant, k=5)
ce.mean_reciprocal_rank(retrieved, relevant)
ce.forgetting_rate(lifecycle_manager)
ce.profile_stability(audit_entries, total_turns=120)
ce.project_accuracy(project_states, ground_truth)
ce.milestone_accuracy(goals, ground_truth_milestones)
ce.insight_accuracy(insights, ground_truth_insights)
ce.reflection_consistency(insights_run_a, insights_run_b)

ce.evaluate_cognitive(retrieval_results=..., lifecycle_manager=lm, ...)
```

| Category | Metrics |
|---|---|
| Retrieval | Recall@K, MRR, Precision@K (v0.3) |
| Memory | Retention Rate, Forgetting Rate, Memory Drift |
| Profile | Profile Accuracy (v0.3), Profile Stability |
| Projects | Project Accuracy, Milestone Accuracy |
| Reflection | Insight Accuracy, Reflection Consistency |

`CognitiveEvaluator` extends `ExtendedMemoryEvaluator` (v0.3.1) extends
`MemoryEvaluator` (v0.3) — full backwards compatibility.

---

## Feature 6 — Document Processor

**File:** `knowledge/document_processor.py`

```
Document → Chunking → Embedding (caller) → Fact Extraction → Graph Update (caller) → Memory Storage
```

Supports **PDF, TXT, MD, DOCX, HTML** (pdfplumber, python-docx, stdlib html.parser):

```python
from knowledge.document_processor import DocumentProcessor

proc = DocumentProcessor(llm=llm, chunk_size=1000, chunk_overlap=150)
doc = proc.process_file(Path("paper.pdf"))

doc.summary           # "2-3 sentence summary"
doc.key_findings       # ["Attention is all you need", ...]
doc.concepts           # ["attention", "transformers", ...]
doc.related_topics     # for SemanticClusterIndex / graph
doc.entities           # [("Transformer", "skill"), ...] for MemoryGraph.upsert_relation
doc.chunks             # DocumentChunk list, page-aware for PDFs
```

Without an LLM, heuristic frequency-based concept extraction and
first-sentence summaries keep the pipeline usable offline.

---

## Feature 7 — Media Processor

**File:** `knowledge/media_processor.py`

```python
from knowledge.media_processor import MediaProcessor, TranscriptionBackend

mp = MediaProcessor(llm=llm, transcription_backend=my_whisper_backend)
result = mp.process(Path("diagram.png"))   # ImageProcessor — fully functional
result = mp.process(Path("lecture.mp3"))   # AudioProcessor — needs backend
result = mp.process(Path("talk.mp4"))      # VideoProcessor — needs ffmpeg + backend
```

| Sub-processor | Status | Requirements |
|---|---|---|
| `ImageProcessor` | **Fully functional** | Pillow + pytesseract + tesseract-ocr binary. OCR text → heuristic or LLM diagram/scene analysis (objects, diagram_notes, topics). |
| `AudioProcessor` | **Pluggable** | Implement `TranscriptionBackend.transcribe(path) -> [(start,end,text),...]` with whisper/faster-whisper/cloud ASR. `NullTranscriptionBackend` (default) logs a warning and returns empty — keeps the module importable and testable without ML deps. |
| `VideoProcessor` | **Pluggable + ffmpeg** | Samples frames via ffmpeg → `ImageProcessor`; extracts audio track → `AudioProcessor`. Gracefully degrades with an explanatory summary if ffmpeg is absent. |

All three produce `ProcessedMedia`, convertible to `ProcessedDocument`
via `.to_processed_document()` for unified memory storage alongside
Feature 6 documents.

---

## Feature 8 — Knowledge Synthesis Engine

**File:** `knowledge/synthesis.py`

```
Memories + Projects + Documents + Media + Graph → Knowledge Report
```

```python
from knowledge.synthesis import KnowledgeSynthesisEngine, SynthesisSource

kse = KnowledgeSynthesisEngine(Path("memory/knowledge_reports.json"), llm=llm)

sources = (
    KnowledgeSynthesisEngine.from_memories(recent_memories)
    + KnowledgeSynthesisEngine.from_projects(project_states)
    + KnowledgeSynthesisEngine.from_documents(processed_docs)
    + KnowledgeSynthesisEngine.from_media(processed_media)
    + KnowledgeSynthesisEngine.from_graph_facts(graph_edges)
)
report = kse.synthesize(sources)
report.title, report.narrative, report.key_points, report.topics
```

LLM mode produces a synthesised narrative tying sources together;
heuristic mode aggregates topic frequencies and lists per-source snippets.

---

## Feature 9 — Reflection Scheduler

**File:** `reflection/scheduler.py`

```
Every session → Session Reflection
Every day     → Daily Reflection
Every week    → Weekly Reflection
Every month   → Deep Reflection
```

```python
from reflection.scheduler import ReflectionScheduler

sched = ReflectionScheduler(Path("memory/reflection_schedule.json"))

triggered = sched.run_due(
    on_session=lambda: reflection_engine.reflect_session(latest_session_summary),
    on_daily=lambda: reflection_engine.reflect_daily(daily_summary),
    on_weekly=lambda: reflection_engine.reflect_weekly(weekly_summary),
    on_monthly=lambda: kse.synthesize(all_recent_sources),
)
# triggered: subset of ["session", "daily", "weekly", "monthly"]
```

Each callback is wrapped in try/except (failure isolation, consistent with
`BackgroundProcessor`): one scope failing never blocks the others. Designed
to be called from `BackgroundProcessor.REGENERATE_SUMMARY` handler.

---

## Feature 10 — Memory Query Language (MQL)

**File:** `reflection/mql.py`

```python
from reflection.mql import MQLEngine

mql = MQLEngine(
    goal_tracker=gt, project_intelligence=pi, memory_manager=mm,
    retriever=retriever, reflection_engine=re_engine,
    consolidation_engine=ce, semantic_cluster_index=sci,
    contradiction_detector=cd, lifecycle_manager=lm,
)

mql.is_mql_command(user_input)   # True if input starts with "show "
result = mql.run("show active goals")
print(result.text)
```

Supported commands:

```
show active goals
show goals
show project <name>
show project risks
show memories about <topic>
show reflections this week | today | this month
show reflections
show strongest skills
show facts
show contradictions
show topic clusters
show memory lifecycle
```

Every command degrades gracefully: if a required component wasn't passed
to `MQLEngine`, the result explains what's missing instead of erroring —
so MQL is safe to wire in even for partially-configured deployments.

---

## Integration Sketch

```python
# app.py — additive wiring (all optional)
from reflection.reflection_engine import ReflectionEngine
from reflection.consolidation_engine import ConsolidationEngine
from reflection.goal_tracker import GoalTracker
from reflection.project_intelligence import ProjectIntelligenceEngine
from reflection.scheduler import ReflectionScheduler
from reflection.mql import MQLEngine
from knowledge.document_processor import DocumentProcessor
from knowledge.media_processor import MediaProcessor
from knowledge.synthesis import KnowledgeSynthesisEngine

reflection_engine = ReflectionEngine(MEMORY_DIR / "reflections.json", llm=llm)
consolidation = ConsolidationEngine(MEMORY_DIR / "canonical_facts.json")
goals = GoalTracker(MEMORY_DIR / "goals.json")
project_intel = ProjectIntelligenceEngine(MEMORY_DIR / "project_intelligence.json", project_manager=pm)
scheduler = ReflectionScheduler(MEMORY_DIR / "reflection_schedule.json")
doc_processor = DocumentProcessor(llm=llm)
media_processor = MediaProcessor(llm=llm)
synthesis = KnowledgeSynthesisEngine(MEMORY_DIR / "knowledge_reports.json", llm=llm)

mql = MQLEngine(
    goal_tracker=goals, project_intelligence=project_intel,
    memory_manager=mm, retriever=retriever,
    reflection_engine=reflection_engine, consolidation_engine=consolidation,
    semantic_cluster_index=cluster_index, contradiction_detector=contradiction_detector,
    lifecycle_manager=lifecycle_manager,
)

# In the CLI loop, before normal chat handling:
if mql.is_mql_command(user_input):
    print(mql.run(user_input).text)
    continue

# After each session ends:
scheduler.run_due(
    on_session=lambda: reflection_engine.reflect_session(hierarchy.get_latest_sessions(1)[0]),
    on_daily=lambda: reflection_engine.reflect_daily(hierarchy.get_daily(today)),
    on_weekly=lambda: reflection_engine.reflect_weekly(hierarchy.get_weekly(this_week)),
    on_monthly=lambda: synthesis.synthesize(
        KnowledgeSynthesisEngine.from_memories(recent_memories)
        + KnowledgeSynthesisEngine.from_projects(project_intel.list_all())
    ),
)
```

---

## Updated Architecture

```
User
 │
 ├── Chat System (TutorAgent, v0.3)
 │
 ├── Memory Layer (v0.3 / v0.3.1)
 │   ├── Episodic / Semantic / Procedural (memory_types.py)
 │   ├── Hierarchical (Raw→Session→Daily→Weekly→Project)
 │   ├── Semantic Clusters (topic-based)
 │   ├── Knowledge Graph + GraphReasoner
 │   ├── Lifecycle (active→compressed→archived→deleted)
 │   └── Project Memory (ProjectManager)
 │
 ├── Reflection Layer (v0.3.2, NEW)
 │   ├── ReflectionEngine        — Insights from memory
 │   ├── ConsolidationEngine      — Canonical Facts
 │   ├── GoalTracker               — Goals/Milestones/Tasks/Blockers
 │   ├── ProjectIntelligenceEngine — ProjectState (focus/risk/progress)
 │   └── ReflectionScheduler       — session/daily/weekly/monthly triggers
 │
 ├── Knowledge Layer (v0.3.2, NEW)
 │   ├── DocumentProcessor   — PDF/TXT/MD/DOCX/HTML
 │   ├── MediaProcessor       — Image (full) / Audio+Video (pluggable)
 │   ├── KnowledgeSynthesisEngine — multi-source Knowledge Reports
 │   └── FactVerifier (v0.3.1)
 │
 ├── Inspection Layer (v0.3.2, NEW)
 │   └── MQL — "show active goals", "show project risks", ...
 │
 └── Evaluation Layer (blix_eval)
     ├── Retrieval Metrics  — Precision/Recall/F1/Recall@K/MRR
     ├── Memory Metrics     — Retention/Forgetting/Drift
     ├── Profile Metrics    — Accuracy/Stability
     ├── Project Metrics    — Accuracy/Milestone Accuracy
     └── Reflection Metrics — Insight Accuracy/Consistency
```

---

## Test Coverage

```
tests/test_v03_features.py     75 tests
tests/test_v031_features.py   118 tests
tests/test_v032_features.py   129 tests   ← NEW (Features 1-10)
tests/test_memory_manager.py   ~60 tests
tests/test_semantic_retriever  ~40 tests
tests/test_tutor_agent.py       17 tests
──────────────────────────────────────────
Total                          458 tests  all passing
```

```bash
python -m pytest tests/ -q
# 458 passed
```

---

## New Optional Dependencies

```toml
[project.optional-dependencies]
documents = ["pdfplumber>=0.10.0", "python-docx>=1.0.0"]   # Feature 6
media     = ["Pillow>=10.0.0", "pytesseract>=0.3.10"]       # Feature 7 (image OCR)
```

Video/audio transcription requires a `TranscriptionBackend` implementation
(e.g. wrapping `openai-whisper` or `faster-whisper`) and `ffmpeg` on PATH —
deliberately left as an extension point rather than a hard dependency.

---

## Migration from v0.3.1

No breaking changes. Every v0.3.2 component is a new, independent module
with its own JSON storage file (`reflections.json`, `canonical_facts.json`,
`goals.json`, `project_intelligence.json`, `reflection_schedule.json`,
`knowledge_reports.json`). Nothing is imported by `TutorAgent` by default —
wire components into `app.py` as shown in the Integration Sketch above.
