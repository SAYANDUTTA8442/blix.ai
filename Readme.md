<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,40:0f172a,80:1e1b4b,100:4f46e5&height=140&section=header&text=Blix&fontSize=56&fontColor=ffffff&fontAlignY=58&fontAlign=50&desc=Cognitive%20Agent%20Architecture%20%C2%B7%20Memory%20%C2%B7%20Reasoning%20%C2%B7%20Recovery&descSize=14&descAlignY=80&descColor=a5b4fc&animation=fadeIn" width="100%"/>

</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-a5b4fc?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-4f46e5?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467df?style=flat-square&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Status](https://img.shields.io/badge/Status-Active%20Development-22c55e?style=flat-square)]()
[![Version](https://img.shields.io/badge/Version-v0.3.1-a5b4fc?style=flat-square)]()
[![Tests](https://img.shields.io/badge/Tests-329%20documented-22c55e?style=flat-square)]()

<br/>

*Built by [Sayan Dutta](https://sayandutta.netlify.app) · AI Researcher · IIT Patna*

</div>

---

> **Blix is not a chatbot wrapper.**
> It is a ground-up cognitive agent architecture that explores what it takes to build an AI system that **reasons, remembers, and recovers from failure** in a principled way — combining persistent hierarchical memory, knowledge graphs, truth maintenance, belief tracking, autonomous replanning, and workspace-based reasoning coordination, all running locally on consumer hardware.

---

## Table of Contents

- [Why Blix Exists](#-why-blix-exists)
- [What Makes Blix Different](#-what-makes-blix-different)
- [Architecture](#-architecture)
  - [Memory System](#-memory-system)
  - [Knowledge Graph](#-knowledge-graph)
  - [Truth Maintenance Engine](#-truth-maintenance-engine)
  - [Workspace Coordination](#-workspace-coordination)
  - [Failure Memory & Replanning](#-failure-memory--replanning)
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

Most LLM applications today are stateless wrappers: they call an API, format a prompt, and return a response. Between sessions, they remember nothing. Within sessions, they accumulate a context window until it overflows. They have no model of what they believe, no ability to detect when a belief is wrong, and no mechanism to learn from failure.

**The problems Blix is built to solve:**

| Problem | What most systems do | What Blix does |
|---|---|---|
| Context loss between sessions | Forget everything | Persistent hierarchical memory with temporal retrieval |
| Contradictory knowledge | Accept any new claim silently | Truth Maintenance Engine detects and repairs inconsistencies |
| No model of belief | No explicit belief state | Belief tracking with confidence estimation |
| Static memory | Flat key-value or vector store | Knowledge graph with semantic traversal |
| Failure is unrecoverable | Crash or hallucinate | Failure memory + principled replanning |
| Cloud dependency | API calls required | Fully local via Ollama + FAISS |
| Monolithic design | Hard to extend or ablate | Modular — every component is independently testable |

---

## ✦ What Makes Blix Different

```
Most AI assistants:    User → Prompt → LLM → Response
                                  (stateless, amnesiac)

Blix:                  User
                         │
                    Conversation Manager
                         │
                  ┌──────┴────────────────────────────────┐
                  │                                       │
           Prompt Builder                    Memory Retrieval Layer
                  │                          ┌────────────┼────────────┐
                  │                   Episodic Mem   Semantic Mem   Belief Store
                  │                          └────────────┼────────────┘
                  │                                       │
                  │                              Knowledge Graph
                  │                                       │
                  │                          Truth Maintenance Engine
                  │                                       │
                  └──────────────┬────────────────────────┘
                                 │
                            Local LLM (Ollama)
                                 │
                          Workspace Coordinator
                         ┌───────┼───────┐
                    Planner  Verifier  Evaluator
                                 │
                          Response + Memory Update
                                 │
                         Failure Memory (if needed)
                                 │
                              Replan
```

---

## 🏗️ Architecture

Blix is organized into **six cooperating subsystems**. Each is independently testable and can be ablated, replaced, or extended without breaking the others.

---

### 🧠 Memory System

The foundation of Blix. Rather than a flat vector store, Blix implements a **hierarchical memory architecture** with three distinct layers:

```
┌─────────────────────────────────────────────────────┐
│                   MEMORY HIERARCHY                   │
├─────────────────────────────────────────────────────┤
│  Working Memory    │ Active context window           │
│                    │ Current task state              │
│                    │ Attention-weighted retrieval    │
├─────────────────────────────────────────────────────┤
│  Episodic Memory   │ Timestamped conversation turns  │
│                    │ Temporal decay modeling         │
│                    │ Confidence-weighted retrieval   │
│                    │ Session-boundary awareness      │
├─────────────────────────────────────────────────────┤
│  Semantic Memory   │ FAISS vector index              │
│                    │ Sentence-Transformer embeddings │
│                    │ Semantic similarity search      │
│                    │ Cross-session concept linkage   │
└─────────────────────────────────────────────────────┘
```

**Key properties:**
- **Temporal retrieval** — recency and relevance are weighted independently, not collapsed
- **Confidence estimation** — each stored memory carries a confidence score that decays and can be updated
- **Decay modeling** — older memories fade unless reinforced by repeated access or explicit pinning
- **Cross-session persistence** — memories survive process restarts via SQLite-backed storage

---

### 🕸️ Knowledge Graph

Blix maintains an explicit **entity-relation knowledge graph** alongside the vector memory layer. Where FAISS retrieves by semantic similarity, the knowledge graph retrieves by structured relationship traversal.

```
     [Sayan] ──authored──► [ECOT-ERG paper]
        │                        │
     studies-at              submitted-to
        │                        │
    [IIT Patna]            [EMNLP 2026]
        │
     member-of
        │
   [Google GSA]
```

- Entities and relations are extracted and stored during ingestion
- Graph traversal enables multi-hop reasoning ("who supervised the paper that Sayan co-authored?")
- Graph state is reconciled with the Truth Maintenance Engine on every write

---

### ⚖️ Truth Maintenance Engine (TME)

The TME is the component that makes Blix's knowledge *trustworthy*. Most AI systems accept any new claim without checking it against what they already believe. Blix does not.

```
New Claim Arrives
       │
       ▼
Consistency Check against Belief Store
       │
   ┌───┴────────────────┐
   │                    │
Consistent          Contradiction Detected
   │                    │
Store & propagate   ┌───┴───────────────┐
                    │                   │
              Resolve by           Flag for
              confidence            user review
                    │
              Update affected
              beliefs downstream
```

**Capabilities:**
- Detects direct contradictions in stored beliefs
- Propagates belief updates downstream through dependent beliefs
- Resolves conflicts by confidence score or recency heuristic
- Maintains a full audit trail of belief revisions

---

### 🖥️ Workspace Coordination

The Workspace is Blix's **executive layer** — the subsystem that coordinates planning, verification, and evaluation before committing a response.

```
Task arrives
     │
     ▼
┌─── Planner ──────────────────────────────┐
│  Decomposes task into subtasks            │
│  Assigns retrieval and reasoning steps    │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌─── Verifier ─────────────────────────────┐
│  Checks retrieved evidence               │
│  Validates logical consistency           │
│  Flags low-confidence steps              │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌─── Evaluator ────────────────────────────┐
│  Scores response quality                 │
│  Compares against task specification     │
│  Decides: commit response or replan      │
└───────────────────┬──────────────────────┘
                    │
             Commit / Replan
```

---

### 🔁 Failure Memory & Replanning

When a task fails — due to retrieval failure, contradiction, low evaluator score, or LLM refusal — Blix does not simply return an error. It **stores the failure** and uses it to inform future attempts.

```
Task fails
     │
     ▼
Failure logged to Failure Memory
  · Task description
  · Failure mode classification
  · State at time of failure
  · Attempted strategies
     │
     ▼
Replanning Engine
  · Consults Failure Memory for similar past failures
  · Selects alternative strategy
  · Adjusts confidence thresholds
     │
     ▼
Retry with modified plan
```

This is the component closest to **continual learning without gradient updates** — Blix improves its planning behavior from experience without retraining.

---

## 📐 StateBench

**StateBench** is Blix's internal benchmark suite for evaluating the core cognitive subsystems independently.

| Benchmark | What it evaluates |
|---|---|
| `state_tracking` | Does Blix correctly maintain and update belief state across a multi-turn sequence? |
| `contradiction_detection` | Does the TME catch planted contradictions in the knowledge base? |
| `retrieval_fidelity` | Does memory retrieval surface the most contextually relevant items? |
| `replan_success_rate` | When the first plan fails, does replanning recover to a correct response? |
| `temporal_decay` | Does memory correctly down-weight stale information over time? |
| `confidence_calibration` | Are confidence scores meaningful predictors of response accuracy? |

All 329 tests are documented in the **Minor Project I academic report (Blix v0.3.1, 41 pages)**.

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Language | Python 3.10+ | Core implementation |
| Deep Learning | PyTorch | Model training and inference hooks |
| LLM Runtime | Ollama | Local LLM inference (Llama, Mistral, Phi, Gemma) |
| Transformers | Hugging Face Transformers | Model loading, tokenization, embeddings |
| Vector Search | FAISS | Semantic memory retrieval |
| Embeddings | Sentence Transformers | Semantic encoding for memory and graph |
| Database | SQLite | Persistent memory, belief store, failure log |
| Validation | Pydantic | Schema enforcement across all subsystems |
| Document Parsing | PyMuPDF / pdfplumber | PDF and document ingestion pipeline |
| Evaluation | Custom StateBench | Subsystem-level benchmarking |

**Design constraints:**
- ✅ Fully local — no cloud API calls required at runtime
- ✅ Privacy-preserving — all data stays on device
- ✅ Consumer hardware — designed to run on a standard laptop (8GB RAM minimum)
- ✅ Modular — every subsystem has a clean interface and can be replaced independently

---

## 📁 Project Structure

```
blix.ai/
├── core/
│   ├── memory/
│   │   ├── working_memory.py        # Active context management
│   │   ├── episodic_memory.py       # Timestamped turn storage
│   │   ├── semantic_memory.py       # FAISS vector index
│   │   └── memory_manager.py        # Unified retrieval interface
│   ├── knowledge/
│   │   ├── knowledge_graph.py       # Entity-relation graph
│   │   └── graph_traversal.py       # Multi-hop query engine
│   ├── tme/
│   │   ├── belief_store.py          # Belief state management
│   │   ├── consistency_checker.py   # Contradiction detection
│   │   └── belief_updater.py        # Downstream propagation
│   ├── workspace/
│   │   ├── planner.py               # Task decomposition
│   │   ├── verifier.py              # Evidence validation
│   │   └── evaluator.py             # Response scoring
│   └── failure/
│       ├── failure_memory.py        # Failure log and retrieval
│       └── replanner.py             # Alternative strategy selection
├── inference/
│   ├── ollama_client.py             # Local LLM interface
│   └── prompt_builder.py           # Context-aware prompt assembly
├── ingestion/
│   ├── pdf_pipeline.py              # Document ingestion
│   └── ocr_fallback.py              # OCR for scanned documents
├── conversation/
│   ├── session_manager.py           # Multi-session handling
│   └── history.py                   # Persistent conversation log
├── evaluation/
│   └── statebench/                  # 329-test benchmark suite
├── config/
│   └── settings.py                  # Pydantic-validated config
├── tests/
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.10+
python --version

# Ollama (for local LLM inference)
# Install from https://ollama.com
ollama pull llama3.2        # or mistral, phi3, gemma2
```

### Installation

```bash
# Clone the repository
git clone https://github.com/SAYANDUTTA8442/blix.ai.git
cd blix.ai

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```python
from core.memory.memory_manager import MemoryManager
from core.tme.belief_store import BeliefStore
from core.workspace.planner import Planner
from inference.ollama_client import OllamaClient
from conversation.session_manager import SessionManager

# Initialize subsystems
memory = MemoryManager()
beliefs = BeliefStore()
llm = OllamaClient(model="llama3.2")
planner = Planner(memory=memory, beliefs=beliefs, llm=llm)
session = SessionManager(planner=planner)

# Start a session
response = session.chat("What do you remember about my last project?")
print(response)
```

### Running StateBench

```bash
# Run the full benchmark suite
python -m evaluation.statebench.run_all

# Run a specific subsystem benchmark
python -m evaluation.statebench.state_tracking
python -m evaluation.statebench.contradiction_detection
```

---

## 🗺️ Roadmap

### v0.3 (Current)
- [x] Hierarchical memory system (working / episodic / semantic)
- [x] FAISS-backed semantic retrieval
- [x] Knowledge graph with entity-relation storage
- [x] Truth Maintenance Engine — contradiction detection and belief repair
- [x] Workspace coordination (planner / verifier / evaluator)
- [x] Failure memory and replanning
- [x] Multi-session persistence (SQLite)
- [x] PDF and document ingestion pipeline
- [x] StateBench benchmark suite (329 tests)
- [x] Local LLM inference via Ollama

### v0.4 (In Progress)
- [ ] Graph-augmented RAG — joint vector + graph retrieval
- [ ] Confidence-calibrated response generation
- [ ] Continual belief update from user corrections
- [ ] Improved temporal decay modeling
- [ ] REST API layer (FastAPI)
- [ ] Web UI (minimal, local)

### v1.0 (Planned)
- [ ] Multi-agent coordination primitives
- [ ] Tool use and external action execution
- [ ] Long-horizon planning with goal tracking
- [ ] Published StateBench as standalone benchmark
- [ ] Academic paper on the Blix architecture

---

## 📄 Research Context

Blix began as an independent project in **June 2024** — a year before the author entered IIT Patna. It is the longest-running and most architecturally ambitious project in this portfolio.

A **323-page architecture document** has been produced covering the full design rationale, subsystem specifications, and theoretical grounding for every component. A **41-page Minor Project I academic report** (Blix v0.3.1) documents all 329 StateBench tests and their results.

The architectural thinking in Blix directly informed the author's research internship at BITS Pilani (April–May 2026), where structured reasoning pipeline design — a core Blix concern — was applied to the ECOT-ERG empathetic dialogue framework, now under review at **EMNLP 2026**.

**Conceptual influences:** ACT-R cognitive architecture · Soar cognitive system · Truth Maintenance Systems (Doyle, 1979) · Retrieval-Augmented Generation · Chain-of-Thought reasoning · Direct Preference Optimization

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

**Before contributing:**
1. Open an issue describing the change you want to make
2. Wait for discussion and approval before submitting a PR
3. For major architectural changes, a design doc is expected

**Good first issues:** documentation improvements, new StateBench test cases, alternative embedding model integrations, additional Ollama model configurations.

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
