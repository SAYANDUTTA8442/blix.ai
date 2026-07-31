<div align="center">


<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,40:0f172a,80:1e1b4b,100:4f46e5&height=140&section=header&text=Blix&fontSize=56&fontColor=ffffff&fontAlignY=58&fontAlign=50&desc=Cognitive%20Agent%20Architecture%20%C2%B7%20Memory%20%C2%B7%20Reasoning%20%C2%B7%20Recovery&descSize=14&descAlignY=80&descColor=a5b4fc&animation=fadeIn" width="100%"/>

</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-a5b4fc?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-4f46e5?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![sqlite-vec](https://img.shields.io/badge/sqlite--vec-Vector%20Search-0467df?style=flat-square&logo=sqlite&logoColor=white)](https://github.com/asg017/sqlite-vec)
[![Status](https://img.shields.io/badge/Status-Active%20Development-22c55e?style=flat-square)]()
[![Version](https://img.shields.io/badge/Version-v0.3.16-a5b4fc?style=flat-square)]()
[![Tests](https://img.shields.io/badge/Tests-1%2C962%20passing-22c55e?style=flat-square)]()

<br/>

*Built by [Sayan Dutta](https://sayandutta.netlify.app) · AI Researcher · IIT Patna*

</div>

---

> **Blix is not a chatbot wrapper.**
> It is a ground-up cognitive agent architecture exploring what it takes for an AI system to **remember, reason, verify, and adapt** in a principled way — combining hierarchical graph‑vector memory, causal reasoning, self‑reflective metacognition, and online policy learning, running entirely on local infrastructure.

---

## Table of Contents

- [Why Blix Exists](#-why-blix-exists)
- [What Makes Blix Different](#-what-makes-blix-different)
- [Architecture](#-architecture)
  - [HGSHM — Hybrid Graph-Based Semantic Hierarchical Memory](#-hgshm--hybrid-graph-based-semantic-hierarchical-memory)
  - [ADMA — Adaptive Dual Memory Architecture](#-adma--adaptive-dual-memory-architecture)
  - [Causal Reasoning & Truth Maintenance](#-causal-reasoning--truth-maintenance)
  - [Planning & Workspace Coordination](#-planning--workspace-coordination)
  - [Metacognition & Reflection](#-metacognition--reflection)
  - [Agents, Curiosity & Autonomous Research](#-agents-curiosity--autonomous-research)
- [StateBench](#-statebench)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Roadmap](#-roadmap)
- [Research Context](#-research-context)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🔍 Why Blix Exists

Most LLM applications today are stateless wrappers: they call an API, format a prompt, and return a response. Between sessions, they remember nothing. Within sessions, they accumulate a context window until it overflows. They have no explicit belief state, no way to detect when a belief is wrong, and no mechanism to *learn from their own behavior* without a training run.

**The problems Blix is built to solve:**

| Problem | What most systems do | What Blix does |
|---|---|---|
| Context loss between sessions | Forget everything | Persistent hierarchical memory (Raw → Session → Daily → Weekly → Project → Principle → World Model) |
| Contradictory knowledge | Accept any new claim silently | `TruthManager` + `ContradictionResolver` detect and repair inconsistencies |
| No model of belief | No explicit belief state | Confidence-scored belief store with a full revision audit trail |
| Static memory | Flat key-value or vector store | Fused graph + vector + hierarchical retrieval (`HybridRetriever`, 11-factor ranking) |
| Fixed heuristics | Hand-tuned config forever | Contextual-bandit **policy learning** adapts retrieval weights, planning aggressiveness, and prompt style from observed outcomes — no retraining |
| Failure is unrecoverable | Crash or hallucinate | Dedicated failure memory + principled replanning (`agents/failure_memory.py`, `planning/replanner.py`) |
| Cloud dependency | API calls required | Fully local via Ollama or HuggingFace Transformers |
| Monolithic design | Hard to extend or ablate | 30+ independently testable packages, dependency‑injected ablation harness (`policy/ablation_v3.py`) |

---

## ✦ What Makes Blix Different

```
Most AI assistants:    User → Prompt → LLM → Response
                                  (stateless, amnesiac)

Blix:                              User
                                     │
                              TutorAgent (orchestrator)
                                     │
                ┌────────────────────┼────────────────────────┐
                │                    │                        │
        HGSHM Retrieval      Metacognition Controller   Curiosity Engine
     (semantic·vector·graph        │                (gap tracking, hypotheses,
       ·temporal·importance)  Strategy Selector          experiment planning)
                │              (bandit-selected)              │
                └────────────────────┼────────────────────────┘
                                     │
                          Policy Compiler (ADMA)
                       dynamic prompt · retrieval weights
                                     │
                              Planner (beam search)
                                     │
                        Global Workspace + Broadcast Bus
                       ┌─────────────┼─────────────┐
                   Verifier      Critic/Evaluator   Specialists
                (evidence check)  (score / replan)  (consensus vote)
                                     │
                              Local LLM (Ollama /
                              HF Transformers)
                                     │
                        Response + Memory Consolidation
                                     │
                          Failure Memory (if needed)
                                     │
                                 Replanner
```

---

## 🏗️ Architecture

Blix has grown, over twenty versioned releases (v0.1 → v0.3.16), from a memory-augmented chat tutor into a full cognitive stack of **30+ cooperating subsystems**, each independently testable and injectable for ablation studies.

---

### 🧠 HGSHM — Hybrid Graph-Based Semantic Hierarchical Memory

The current foundation of Blix's memory (introduced v0.3.15, replacing the earlier flat FAISS index).

```
┌─────────────────────────────────────────────────────────────────┐
│                            HGSHM                                 │
├─────────────────────────────────────────────────────────────────┤
│  GraphStore (hgshm.db)  ·  VectorIndex (sqlite-vec, cosine ANN)  │
│  HierarchyManager  ·  ConsolidationEngine  ·  ContextBuilder     │
│  HybridRetriever — 11-factor ranked fusion                      │
└─────────────────────────────────────────────────────────────────┘

Raw → Episode → Conversation → Session → Daily → Weekly
   → Monthly → Project → Concept → Principle → Knowledge → WorldModel
```

- **Vector layer** — `sqlite-vec`-backed ANN search (`memory/hybrid/vector/`), with a pure-Python brute-force cosine fallback when the extension isn't installed.
- **Graph layer** — entity/relation storage with 18 edge types (`supports`, `contradicts`, `causes`, `depends_on`, `derived_from`, `enables`, `blocks`, `summarises`, …) and full traversal (`memory/hybrid/graph/`).
- **Hierarchy layer** — automatic compression from raw memories up through session → daily → weekly → project summaries; `SUMMARISES` edges preserve every source reference so nothing is silently lost.
- **Consolidation** — a background engine merges, dedupes, and rolls up memory nodes on a schedule (`consolidation_engine.py`).
- Query pipeline: `Semantic · Vector · Graph Expansion · Temporal → Importance Ranking → Hierarchy → Contradictions → Causal → typed MemoryContext → Reasoning`.

---

### ⚙️ ADMA — Adaptive Dual Memory Architecture

The newest and most significant addition (v0.3.16). Static configuration is replaced by **online policy learning** — Blix tunes its own retrieval/planning/prompting behavior from observed outcomes, without gradient updates or RLHF.

```
Experience
    │
    ├───────────────────────────────────────┐
    │ System Experience                     │ User Experience
    ▼                                       ▼
SystemMemory                           UserMemory
 (operational knowledge)          (per-user personalisation)
    │            both backed by HGSHM       │
    └──────────────────┬────────────────────┘
                        ▼
        PolicyLearner — Thompson Sampling over Beta(α, β) arms
                        │
                  PolicyCompiler
              (dynamic prompt assembly)
                        │
              Planner + Global Workspace
                        │
                       LLM
```

- **Dual memory** — `SystemMemory` (workflow traces, benchmark history, failure patterns, operational principles) and `UserMemory` (preferences, corrections, goals, learning progress) as separate domains, unified by `MemoryManager`.
- **Policy memory** — retrieval weights, planner aggressiveness, reasoning strategy, answer verbosity, and difficulty level are each modeled as a set of **bandit arms** (15 defaults across 5 policy types), not hardcoded constants.
- **Reward engine** — 15 observable reward signals (8 system-side, 7 user-side) drive arm updates.
- **Learning rule** — reward ≥ 0.5 increments α (fractionally), reward < 0.5 increments β; temporal decay pulls α → 1 + (α−1)×0.995 per observation (~139-observation half-life); automatic rollback if recent mean confidence drops >0.10 vs. the historical mean.
- **Ablation framework v3** — dependency injection replaces the old env-flag mechanism, so every one of the 7 injectable ADMA components can be swapped or stubbed for controlled experiments (`policy/ablation_v3.py`).

---

### ⚖️ Causal Reasoning & Truth Maintenance

Blix maintains an explicit causal and belief model, not just a similarity index.

- **`core/truth_manager.py` + `core/contradiction_resolver.py`** — detect direct contradictions between incoming claims and stored beliefs, resolve by confidence/recency, and propagate updates to dependent beliefs, with a full audit trail.
- **`causality/`** — `CauseGraph`, `CausalMemory`, `CounterfactualEngine`, `CausalReflection` / `MetaCausalReflection`, `BeliefDependencyGraph`, `PrincipleSynthesizer`, `EpistemicStatus` — a dedicated package for building and reflecting on cause‑effect structure, not just correlational retrieval.
- **`reasoning/`** — confidence modeling and temporal querying ("what did I believe about X at time T?").

---

### 🖥️ Planning & Workspace Coordination

The executive layer that decides what to do and whether to trust the result.

- **`planning/`** — `BeamSearchPlanner`, `Critic`, `SearchCritic`, `PlanEvaluator`, `Replanner` — multi-candidate plan search with scoring and revision, not single-shot generation.
- **`workspace/`** — `GlobalWorkspace`, `BroadcastBus`, `NeuralAttentionManager`, `InnerDialogue`, `WorkspaceMemory`, `AttentionManager` — a global-workspace-theory-inspired coordination layer where subsystems broadcast and compete for attention before a response commits.
- **`verification/`** — an independent verifier checks retrieved evidence and logical consistency before commit.
- **`specialists/`** — planning, memory, verification, and reflection specialists vote via a `consensus.py` module rather than a single component deciding alone.
- **`world_model/`** — `LatentWorldModel`, `ScenarioRanker`, `ValueNetwork` for forward simulation and ranking candidate futures; `simulation/trajectory_graph.py` tracks the resulting branches.

---

### 🪞 Metacognition & Reflection

Blix models *itself*, not just the world.

- **`metacognition/`** — `SelfModel`, `StrategyEvolution`, `StrategySelector`/`StrategyManager`, `ConfidenceManager`, `CapabilityTracker` — tracks what Blix is good at, evolves its own strategies, and calibrates confidence.
- **`reflection/`** — `ReflectionEngine`, `MetaReflection`, `StateReflection`, `InsightEngine`, `ConsolidationEngine`, `GoalTracker`, and a **Meta Query Language** (`mql.py`, `mql_v2.py`) for structured introspective queries over memory.
- **`evaluation/`** — a large internal benchmark harness (`agent_benchmark.py`, `capability_metrics.py`, `cognitive.py`, `coordination_metrics.py`, `metacognition_metrics.py`, `workspace_metrics.py`, `state_metrics.py`, `attention_metrics.py`, `research.py`) plus a CLI runner — this is where **StateBench** lives.

---

### 🤖 Agents, Curiosity & Autonomous Research

- **`agents/`** — `Executor`, `TaskRuntime`, `WorkingMemory`, `ExecutionFeedback`, `FailureMemory`, `ToolReliability`, `ToolSuccessPredictor`, `PlanReflection`, `ReflectionLoop` — a full agent execution loop with tool-outcome tracking, not just a plan-then-execute pipeline.
- **`curiosity/` + `hypothesis/` + `experiments/`** — `CuriosityEngine`, `HypothesisManager`, `ExperimentPlanner`, and `knowledge/knowledge_gap_tracker.py` drive autonomous, self-directed exploration: Blix can notice what it doesn't know and plan an experiment to find out.
- **`knowledge/`** — `DocumentProcessor` (PDF/TXT/MD/DOCX/HTML ingestion via `pdfplumber` + `python-docx`), `MediaProcessor` (OCR via `pytesseract` + `Pillow`, pluggable audio/video transcription), `ResearchAssistant`, `synthesis.py`.
- **`api/`** — a FastAPI surface (`Blix — Cognitive Knowledge Platform`) exposing 20 routers: chat, memory, knowledge, reflection, graph, documents, stats, goals, reasoning-research, agent(s), temporal, metacognition, workspace, ml, causality, search, curiosity, world_model, simulation, specialists.

---

## 📐 StateBench

**StateBench** is Blix's internal benchmark suite for evaluating cognitive subsystems independently, run through `evaluation/cli.py` and the `evaluation/` metrics modules described above.

| Benchmark area | What it evaluates |
|---|---|
| State tracking | Does Blix correctly maintain and update belief state across a multi-turn sequence? |
| Contradiction detection | Does the Truth Maintenance Engine catch planted contradictions? |
| Retrieval fidelity | Does `HybridRetriever` surface the most contextually relevant items across vector + graph + temporal signals? |
| Replan success rate | When the first plan fails, does replanning recover a correct response? |
| Temporal decay | Does memory correctly down-weight stale information over time? |
| Confidence calibration | Are confidence scores meaningful predictors of response accuracy? |
| Policy convergence (new, v0.3.16) | Do bandit arms converge to a stable, high-reward configuration, and does rollback trigger correctly on regression? |

**Full repository test suite: 1,962 tests, all passing** (verified against the v0.3.16 source tree). StateBench results specific to the v0.3.1 milestone are documented separately in the 41-page Minor Project I academic report.

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Language | Python 3.10+ | Core implementation |
| Deep Learning | PyTorch, HuggingFace Transformers | Local model inference and tokenization (`llm/transformers_provider.py`) |
| LLM Runtime | Ollama | Alternative local LLM backend (`llm/ollama_provider.py`) |
| Vector Search | sqlite-vec | Semantic memory ANN search, with pure-Python fallback |
| Embeddings | Sentence-Transformers | Semantic encoding for memory and graph |
| Graph + Hierarchy | Custom (`memory/hybrid/graph`, `memory/hybrid/hierarchy`) | Entity-relation storage, multi-hop traversal, rollup compression |
| Policy Learning | Custom Thompson-sampling bandits (`policy/`) | Online, gradient-free adaptation of retrieval/planning/prompting |
| Database | SQLite (WAL mode) | `hgshm.db`, `policy.db`, hierarchy/vector persistence |
| Validation | Pydantic v2 | Schema enforcement across all subsystems |
| Document Parsing | pdfplumber, python-docx | PDF/DOCX ingestion |
| Media Processing | Pillow, pytesseract | OCR for images and scanned documents |
| API | FastAPI + Uvicorn | REST surface across 20 routers |
| Fuzzy Matching | RapidFuzz | Entity/label deduplication in the memory graph |
| Testing | Pytest | 1,962 tests across every subsystem |

**Design constraints:**
- ✅ Fully local — no cloud API calls required at runtime
- ✅ Privacy-preserving — all data stays on device
- ✅ Consumer hardware — designed to run on a standard laptop
- ✅ Modular — every subsystem has a clean interface and can be replaced or ablated independently
- ✅ Gradient-free adaptation — ADMA policy learning requires no retraining loop

---

## 📁 Project Structure

```
blix_v03/
├── app.py                     # CLI entry point
├── pyproject.toml / requirements.txt
├── ARCHITECTURE.md            # Full design rationale
├── CHANGELOG_v0.3.*.md        # 16 versioned changelogs
│
├── core/                      # TutorAgent orchestration, memory scoring,
│                               #   hierarchy, graph, truth maintenance,
│                               #   contradiction resolution, embeddings
├── memory/
│   ├── hybrid/                # HGSHM: graph · vector · hierarchy ·
│   │                           #   consolidation · context builder
│   ├── system/, user/          # ADMA dual memory domains
│   └── manager.py
├── policy/                    # ADMA: models, store, reward engine,
│                               #   learner (bandits), optimizer,
│                               #   compiler, adaptive retriever/planner
├── planning/                   # Beam search planner, critic, replanner
├── workspace/                  # Global workspace, broadcast bus,
│                               #   attention, inner dialogue
├── causality/                  # Cause graphs, counterfactuals,
│                               #   causal/meta-causal reflection
├── reasoning/                  # Confidence modeling, temporal queries
├── metacognition/               # Self-model, strategy evolution,
│                               #   capability tracking
├── reflection/                  # Reflection/insight engines, MQL,
│                               #   goal tracking, consolidation
├── agents/                     # Executor, task runtime, failure
│                               #   memory, tool reliability
├── curiosity/, hypothesis/, experiments/  # Autonomous exploration
├── knowledge/                  # Document/media ingestion, research
│                               #   assistant, synthesis
├── verification/, specialists/ # Independent verifier + consensus
│                               #   voting specialists
├── world_model/, simulation/   # Latent world model, scenario ranking
├── graph/                      # Temporal graph primitives
├── retrieval/                   # Temporal + active-attention retrievers,
│                               #   cross-encoder reranker
├── learning/                    # Continual adapter, failure clusterer
├── tools/                      # Tool registry
├── llm/                        # Provider factory: Transformers / Ollama
├── evaluation/                  # StateBench metrics + CLI runner
├── api/                        # FastAPI app + 20 routers
├── schemas/, config/, utils/    # Pydantic models, settings, helpers
└── tests/                       # 1,962 tests across every subsystem
```

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.10+
python --version

# Option A: Ollama (local LLM backend)
# Install from https://ollama.com
ollama pull llama3.2        # or mistral, phi3, gemma2

# Option B: HuggingFace Transformers (fully local, no separate server)
# handled directly via `transformers` + `torch`
```

### Installation

```bash
git clone https://github.com/SAYANDUTTA8442/blix.ai.git
cd blix.ai/blix_v03

python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
# Optional extras:
pip install -e ".[transformers]"   # HF Transformers backend
pip install -e ".[api]"            # FastAPI server
pip install -e ".[documents]"      # PDF/DOCX ingestion
pip install -e ".[media]"          # Image OCR
```

### Quick Start — HGSHM Memory

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

### Quick Start — CLI Tutor Agent

```bash
python app.py
# /memory  /profile  /stats  /graph  /projects  /hierarchy  /eval
```

### Running the API

```bash
uvicorn api.server:app --reload
# Serves 20 routers: chat, memory, knowledge, reflection, graph,
# documents, stats, goals, reasoning-research, agent(s), temporal,
# metacognition, workspace, ml, causality, search, curiosity,
# world_model, simulation, specialists
```

### Running the Test Suite

```bash
pytest tests/ -q
# 1,962 passed
```

---

## 🗺️ Roadmap

### v0.3 — complete
- [x] Hierarchical memory (Raw → Session → Daily → Weekly → Project → Principle → WorldModel)
- [x] Hybrid Graph-Based Semantic Hierarchical Memory (HGSHM) — sqlite-vec + graph + hierarchy fusion
- [x] Truth Maintenance Engine — contradiction detection and belief repair
- [x] Causal reasoning package — cause graphs, counterfactuals, causal reflection
- [x] Workspace coordination — planner / verifier / evaluator / specialist consensus
- [x] Failure memory and replanning
- [x] Metacognition — self-model, strategy evolution, confidence calibration
- [x] Curiosity engine, hypothesis manager, experiment planner
- [x] Document + media ingestion (PDF, DOCX, HTML, OCR)
- [x] REST API layer (FastAPI, 20 routers)
- [x] **Adaptive Dual Memory Architecture (ADMA)** — Thompson-sampling policy learning, dynamic prompt compiler, dependency-injected ablation framework
- [x] StateBench benchmark suite — 1,962 passing tests

### v0.4 — Cognitive Kernel (in progress)
- [ ] Unify HGSHM + ADMA policy memory under a single kernel API
- [ ] Graph-augmented RAG as the default retrieval path
- [ ] Continual belief update loop driven by live user corrections
- [ ] Web UI (minimal, local) over the existing FastAPI routers

### v0.5 / v0.6 — Adaptive Intelligence → Autonomous Researcher
- [ ] Multi-agent coordination primitives
- [ ] Long-horizon goal tracking across the reflection + planning stack
- [ ] Autonomous experiment execution loop (curiosity → hypothesis → experiment → belief update, end to end)
- [ ] Publish StateBench as a standalone benchmark
- [ ] Academic paper on the Blix / ADMA architecture

### v1.0 — Cognitive Operating System
- [ ] Tool use and external action execution as a first-class subsystem
- [ ] Full multi-agent, multi-user deployment story

---

## 📄 Research Context

Blix began as an independent project in **June 2024** — a year before the author entered IIT Patna. It is the longest-running and most architecturally ambitious project in this portfolio, now on its 16th versioned release.

A **323-page architecture document** covers the full design rationale, subsystem specifications, and theoretical grounding for every component. A **41-page Minor Project I academic report** (Blix v0.3.1) documents the original 329 StateBench tests and their results; the subsequent v0.3.2–v0.3.16 changelogs extend this with causal reasoning, metacognition, curiosity-driven experimentation, and the ADMA policy-learning layer.

The architectural thinking in Blix directly informed the author's research internship at BITS Pilani (April–May 2026), where structured reasoning pipeline design — a core Blix concern — was applied to the ECOT-ERG empathetic dialogue framework, now under review at **EMNLP 2026**.

**Conceptual influences:** ACT-R cognitive architecture · Soar cognitive system · Global Workspace Theory · Truth Maintenance Systems (Doyle, 1979) · Retrieval-Augmented Generation · Chain-of-Thought reasoning · Contextual bandits / Thompson sampling · Direct Preference Optimization

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

**Before contributing:**
1. Open an issue describing the change you want to make
2. Wait for discussion and approval before submitting a PR
3. For major architectural changes, a design doc is expected

**Good first issues:** documentation improvements, new StateBench test cases, alternative embedding model integrations, additional Ollama model configurations, new policy arms for the ADMA bandit layer.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Sayan Dutta**
AI Researcher · BS-MS CSDA · IIT Patna · ORCID: 0009-0006-4747-8820

[![Portfolio](https://img.shields.io/badge/Portfolio-sayandutta.netlify.app-4f46e5?style=flat-square&logo=safari&logoColor=white)](https://sayandutta.netlify.app)
[![GitHub](https://img.shields.io/badge/GitHub-SAYANDUTTA8442-0f172a?style=flat-square&logo=github&logoColor=white)](https://github.com/SAYANDUTTA8442)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-sayandutta8653128442-0a66c2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/sayandutta8653128442)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Sayan--Dutta--19-00ccbb?style=flat-square&logo=researchgate&logoColor=white)](https://www.researchgate.net/profile/Sayan-Dutta-19)
[![Email](https://img.shields.io/badge/Email-sayandutta.developer@gmail.com-ea4335?style=flat-square&logo=gmail&logoColor=white)](mailto:sayandutta.developer@gmail.com)

---

<div align="center">

*Building intelligent systems that reason, remember, and recover.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:4f46e5,50:1e1b4b,100:0d1117&height=90&section=footer" width="100%"/>

</div>
