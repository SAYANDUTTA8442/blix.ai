"""
Tests for Blix v0.3.3 — "Platformization & Knowledge Intelligence".

Tests cover:

1. InsightGenerationEngine (Feature 3)
2. API models validation
3. FastAPI endpoints via TestClient — all 7 router groups:
   /chat, /memory, /knowledge, /reflection, /graph, /documents, /stats, /goals
4. Streaming endpoint smoke test
5. BlixContext.dashboard_stats() structure
6. API error handling (404, 400, 415)

Architecture: tests inject a pre-built BlixContext with a tmp memory_dir,
bypassing the full LLM stack (heuristic fallbacks).
All tests are synchronous — we use FastAPI's HTTPX-backed TestClient.

Python 3.10 compatible — fully offline.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from api.context import BlixContext
from api.deps import set_context
from api.models import (
    ChatRequest, CreateGoalRequest, GoalBlockerRequest, GoalMilestoneRequest,
    GoalProgressRequest, MQLRequest, ReflectionRunRequest, SynthesisRequest,
    UpsertRelationRequest,
)
from api.server import create_app
from reflection.insight_engine import (
    ActionableInsight, InsightCategory, InsightGenerationEngine,
)
from schemas.memory_entry import MemoryEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(days_ago: float = 0.0) -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago)


def _entry(id: int, input: str = "q", output: str = "a",
           topics: list | None = None) -> MemoryEntry:
    return MemoryEntry(id=id, input=input, output=output,
                       timestamp=_ts(), topics=topics or [])


# ---------------------------------------------------------------------------
# BlixContext fixture (minimal — no LLM, no embeddings, tmp memory_dir)
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Minimal LLM that returns a deterministic string so tests stay offline."""

    def model_name(self) -> str:
        return "fake-0.3.3"

    def generate(self, prompt: str) -> str:
        return "This is a test reply from the fake LLM."


@pytest.fixture(scope="session")
def tmp_memory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("memory")


@pytest.fixture(scope="session")
def ctx(tmp_memory: Path) -> BlixContext:
    """
    Build a BlixContext that can operate offline.

    We patch the LLM after construction so that extraction/chat calls
    return deterministic strings without network I/O.
    """
    from config import settings as _settings

    # Point all storage paths to the temp directory
    _settings.settings.memory.conversations_file = tmp_memory / "conversations.json"
    _settings.settings.memory.profile_file = tmp_memory / "profile.json"
    _settings.settings.memory.learning_state_file = tmp_memory / "learning_state.json"
    _settings.settings.embed.embeddings_file = tmp_memory / "embeddings.npy"
    _settings.settings.embed.embedding_ids_file = tmp_memory / "embedding_ids.json"

    context = BlixContext(tmp_memory)
    context.llm = _FakeLLM()
    context.agent._llm = _FakeLLM()
    return context


@pytest.fixture(scope="session")
def client(ctx: BlixContext) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with the pre-built BlixContext injected."""
    app = create_app.__wrapped__(tmp_memory=ctx.memory_dir) if hasattr(create_app, "__wrapped__") else None

    # Build app manually, skipping lifespan (we set context directly)
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from api.routers.chat import router as chat_router
    from api.routers.memory import router as memory_router
    from api.routers.knowledge import router as knowledge_router
    from api.routers.reflection import router as reflection_router
    from api.routers.graph import router as graph_router
    from api.routers.documents import router as documents_router
    from api.routers.stats_goals import stats_router, goals_router

    test_app = FastAPI(title="Blix Test", version="0.3.3")
    test_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    test_app.include_router(chat_router)
    test_app.include_router(memory_router)
    test_app.include_router(knowledge_router)
    test_app.include_router(reflection_router)
    test_app.include_router(graph_router)
    test_app.include_router(documents_router)
    test_app.include_router(stats_router)
    test_app.include_router(goals_router)

    @test_app.get("/health")
    def health():
        return {"status": "ok", "version": "0.3.3"}

    set_context(ctx)
    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c


# ===========================================================================
# Feature 3 — InsightGenerationEngine
# ===========================================================================


class TestInsightGenerationEngine:
    @pytest.fixture
    def engine(self, tmp_path: Path) -> InsightGenerationEngine:
        return InsightGenerationEngine(tmp_path / "insights.json")

    def test_analyze_topic_trends_dominant(self, engine: InsightGenerationEngine) -> None:
        memories = [_entry(i, topics=["transformers"]) for i in range(10)] + \
                   [_entry(i + 100, topics=["cooking"]) for i in range(2)]
        results = engine.analyze_topic_trends(memories, min_fraction=0.3)
        assert len(results) >= 1
        assert any("transformers" in r.insight.lower() for r in results)
        assert all(r.category == InsightCategory.RESEARCH_INTEREST for r in results)

    def test_analyze_topic_trends_no_dominant(self, engine: InsightGenerationEngine) -> None:
        memories = [_entry(i, topics=[f"topic_{i}"]) for i in range(10)]
        results = engine.analyze_topic_trends(memories, min_fraction=0.6)
        assert results == []

    def test_analyze_bottlenecks_detects_blocker(self, engine: InsightGenerationEngine, tmp_path: Path) -> None:
        from reflection.goal_tracker import GoalTracker
        gt = GoalTracker(tmp_path / "goals.json")
        g1 = gt.create_goal("Goal A")
        g2 = gt.create_goal("Goal B")
        gt.add_blocker(g1.goal_id, "evaluation framework")
        gt.add_blocker(g2.goal_id, "evaluation framework")
        goals = gt.list_goals()
        results = engine.analyze_bottlenecks(goals)
        assert any("evaluation framework" in r.insight.lower() for r in results)
        assert all(r.category == InsightCategory.BOTTLENECK for r in results)

    def test_analyze_project_patterns_high_risk(self, engine: InsightGenerationEngine, tmp_path: Path) -> None:
        from reflection.project_intelligence import ProjectIntelligenceEngine
        pi = ProjectIntelligenceEngine(tmp_path / "pi.json")
        pi.add_risk("Blix", "risk one")
        pi.add_risk("Blix", "risk two")
        pi.add_risk("Blix", "risk three")
        results = engine.analyze_project_patterns(pi.list_all())
        assert any("Blix" in r.insight for r in results)
        assert all(r.category == InsightCategory.PROJECT_PATTERN for r in results)

    def test_analyze_activity_trend_detects_shift(self, engine: InsightGenerationEngine) -> None:
        older = [_entry(i, topics=["chatbots"]) for i in range(20)]
        recent = [_entry(i + 20, topics=["memory systems"]) for i in range(20)]
        memories = older + recent
        results = engine.analyze_activity_trend(memories, window=20)
        assert len(results) == 1
        assert results[0].category == InsightCategory.TREND
        assert "chatbots" in results[0].insight.lower() or "memory" in results[0].insight.lower()

    def test_analyze_activity_trend_no_shift_same_topic(self, engine: InsightGenerationEngine) -> None:
        memories = [_entry(i, topics=["nlp"]) for i in range(40)]
        results = engine.analyze_activity_trend(memories, window=20)
        assert results == []

    def test_generate_all_combines_analyses(self, engine: InsightGenerationEngine, tmp_path: Path) -> None:
        from reflection.goal_tracker import GoalTracker
        gt = GoalTracker(tmp_path / "goals.json")
        g = gt.create_goal("Build Blix")
        gt.add_blocker(g.goal_id, "missing API")
        memories = [_entry(i, topics=["nlp"]) for i in range(20)] + \
                   [_entry(i + 20, topics=["graphs"]) for i in range(20)]
        results = engine.generate_all(memories=memories, goals=gt.list_goals())
        assert len(results) >= 1
        assert engine.count == len(results)

    def test_llm_recommendation_used(self, tmp_path: Path) -> None:
        engine = InsightGenerationEngine(tmp_path / "insights.json", llm=_FakeLLM())
        memories = [_entry(i, topics=["transformers"]) for i in range(10)] + \
                   [_entry(i + 10, topics=["cooking"]) for i in range(2)]
        results = engine.analyze_topic_trends(memories)
        # With fake LLM, recommendation is a fixed string
        if results:
            assert results[0].recommendation  # recommendation phrased (non-empty from fake LLM)

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        e1 = InsightGenerationEngine(tmp_path / "i.json")
        memories = [_entry(i, topics=["nlp"]) for i in range(20)] + \
                   [_entry(i + 20, topics=["graphs"]) for i in range(20)]
        e1.generate_all(memories=memories)
        e2 = InsightGenerationEngine(tmp_path / "i.json")
        assert e2.count == e1.count

    def test_list_insights_by_category(self, engine: InsightGenerationEngine) -> None:
        engine._insights = [
            ActionableInsight(insight="x", category=InsightCategory.TREND),
            ActionableInsight(insight="y", category=InsightCategory.BOTTLENECK),
        ]
        trends = engine.list_insights(category=InsightCategory.TREND)
        assert len(trends) == 1
        assert trends[0].insight == "x"

    def test_latest_returns_newest(self, engine: InsightGenerationEngine) -> None:
        from datetime import timedelta as td
        engine._insights = [
            ActionableInsight(insight="old", category=InsightCategory.TREND,
                              created_at=(datetime.now(timezone.utc) - td(days=2)).isoformat()),
            ActionableInsight(insight="new", category=InsightCategory.TREND,
                              created_at=datetime.now(timezone.utc).isoformat()),
        ]
        latest = engine.latest(limit=1)
        assert latest[0].insight == "new"

    def test_actionable_insight_to_dict(self) -> None:
        ai = ActionableInsight(
            insight="User focuses on NLP.",
            category=InsightCategory.RESEARCH_INTEREST,
            confidence=0.85,
            evidence=["30 of 40 memories tagged nlp"],
            recommendation="Create a research knowledge base.",
        )
        d = ai.to_dict()
        assert d["category"] == "research_interest"
        assert d["confidence"] == 0.85
        assert d["evidence"]

    def test_to_insight_conversion(self) -> None:
        from reflection.reflection_engine import ReflectionScope
        ai = ActionableInsight(insight="NLP trend.", category=InsightCategory.TREND, confidence=0.7)
        ins = ai.to_insight(scope=ReflectionScope.BEHAVIOR, scope_ref="all")
        assert ins.confidence == 0.7
        assert "NLP trend" in ins.insight


# ===========================================================================
# API Model validation
# ===========================================================================


class TestAPIModels:
    def test_chat_request_min_length(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_chat_request_valid(self) -> None:
        req = ChatRequest(message="Hello Blix!")
        assert req.message == "Hello Blix!"

    def test_upsert_relation_request_confidence_range(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            UpsertRelationRequest(from_label="a", relation="uses", to_label="b", confidence=1.5)

    def test_synthesis_request_defaults(self) -> None:
        req = SynthesisRequest()
        assert req.include_projects is True
        assert req.include_facts is True


# ===========================================================================
# /health & /
# ===========================================================================


class TestHealthEndpoint:
    def test_health(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.3.3"


# ===========================================================================
# /chat
# ===========================================================================


class TestChatEndpoints:
    def test_post_chat(self, client: TestClient) -> None:
        r = client.post("/chat", json={"message": "What is attention mechanism?"})
        assert r.status_code == 200
        data = r.json()
        assert "reply" in data
        assert isinstance(data["reply"], str)
        assert len(data["reply"]) > 0

    def test_post_chat_with_session_id(self, client: TestClient) -> None:
        r = client.post("/chat", json={"message": "Hello", "session_id": "test-session-1"})
        assert r.status_code == 200
        assert r.json()["session_id"] == "test-session-1"

    def test_post_chat_empty_message_rejected(self, client: TestClient) -> None:
        r = client.post("/chat", json={"message": ""})
        assert r.status_code == 422

    def test_post_mql(self, client: TestClient) -> None:
        r = client.post("/chat/mql", json={"command": "show active goals"})
        assert r.status_code == 200
        data = r.json()
        assert data["matched"] is True
        assert "text" in data

    def test_post_mql_unrecognised(self, client: TestClient) -> None:
        r = client.post("/chat/mql", json={"command": "do something weird"})
        assert r.status_code == 200
        data = r.json()
        assert data["matched"] is False

    def test_post_chat_stream(self, client: TestClient) -> None:
        with client.stream("POST", "/chat/stream", json={"message": "Tell me about graphs."}) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            chunks = []
            for line in resp.iter_lines():
                if line.startswith("data:"):
                    payload = json.loads(line[5:].strip())
                    chunks.append(payload)
                    if payload.get("done"):
                        break
        assert any("token" in c for c in chunks)
        assert chunks[-1].get("done") is True


# ===========================================================================
# /memory
# ===========================================================================


class TestMemoryEndpoints:
    def test_list_memories_empty(self, client: TestClient, ctx: BlixContext) -> None:
        # Memory may already have entries from chat tests — just check shape
        r = client.get("/memory")
        assert r.status_code == 200
        data = r.json()
        assert "memories" in data
        assert "total" in data
        assert "page" in data

    def test_list_memories_pagination(self, client: TestClient) -> None:
        r = client.get("/memory?page=1&page_size=5")
        assert r.status_code == 200
        data = r.json()
        assert len(data["memories"]) <= 5
        assert data["page_size"] == 5

    def test_get_memory_by_id(self, client: TestClient, ctx: BlixContext) -> None:
        # Ensure at least one memory exists
        client.post("/chat", json={"message": "Remember this: transformers."})
        all_memories = ctx.memory_manager.get_all_memories()
        if not all_memories:
            pytest.skip("No memories available")
        m = all_memories[-1]
        r = client.get(f"/memory/{m.id}")
        assert r.status_code == 200
        assert r.json()["id"] == m.id

    def test_get_memory_not_found(self, client: TestClient) -> None:
        r = client.get("/memory/99999999")
        assert r.status_code == 404

    def test_search_memories(self, client: TestClient) -> None:
        client.post("/chat", json={"message": "Explain attention in transformers."})
        r = client.get("/memory/search?q=transformers&top_k=5")
        assert r.status_code == 200
        data = r.json()
        assert data["query"] == "transformers"
        assert isinstance(data["results"], list)

    def test_lifecycle_stats(self, client: TestClient) -> None:
        r = client.get("/memory/lifecycle")
        assert r.status_code == 200
        data = r.json()
        assert "active" in data

    def test_compress_memory(self, client: TestClient, ctx: BlixContext) -> None:
        client.post("/chat", json={"message": "Test compression target."})
        memories = ctx.memory_manager.get_all_memories()
        if not memories:
            pytest.skip("No memories")
        mid = memories[-1].id
        r = client.post(f"/memory/{mid}/compress")
        assert r.status_code == 200
        data = r.json()
        assert data["state"] == "compressed"
        assert data["memory_id"] == mid

    def test_compress_memory_not_found(self, client: TestClient) -> None:
        r = client.post("/memory/99999999/compress")
        assert r.status_code == 404


# ===========================================================================
# /knowledge
# ===========================================================================


class TestKnowledgeEndpoints:
    def test_list_facts_empty(self, client: TestClient) -> None:
        r = client.get("/knowledge/facts")
        assert r.status_code == 200
        data = r.json()
        assert "facts" in data
        assert "total" in data

    def test_strongest_facts(self, client: TestClient, ctx: BlixContext) -> None:
        # Seed a fact
        ctx.consolidation.consolidate("User deeply understands transformers", 1, topic="nlp")
        r = client.get("/knowledge/facts/strongest?top_k=5")
        assert r.status_code == 200
        facts = r.json()["facts"]
        assert len(facts) >= 1

    def test_list_facts_filter_by_topic(self, client: TestClient, ctx: BlixContext) -> None:
        ctx.consolidation.consolidate("User knows Python well", 2, topic="programming")
        r = client.get("/knowledge/facts?topic=programming")
        assert r.status_code == 200
        data = r.json()
        assert all(f["topic"] == "programming" for f in data["facts"])

    def test_synthesize(self, client: TestClient) -> None:
        r = client.post("/knowledge/synthesize", json={"include_projects": False, "include_facts": False})
        assert r.status_code == 200
        data = r.json()
        assert "report_id" in data
        assert "narrative" in data

    def test_list_reports(self, client: TestClient) -> None:
        client.post("/knowledge/synthesize", json={})
        r = client.get("/knowledge/reports")
        assert r.status_code == 200
        data = r.json()
        assert "reports" in data
        assert data["total"] >= 1

    def test_get_report(self, client: TestClient) -> None:
        r = client.post("/knowledge/synthesize", json={})
        report_id = r.json()["report_id"]
        r2 = client.get(f"/knowledge/reports/{report_id}")
        assert r2.status_code == 200
        assert r2.json()["report_id"] == report_id

    def test_get_report_not_found(self, client: TestClient) -> None:
        r = client.get("/knowledge/reports/nonexistent_xyz")
        assert r.status_code == 404


# ===========================================================================
# /reflection
# ===========================================================================


class TestReflectionEndpoints:
    def test_list_insights_empty_initially(self, client: TestClient) -> None:
        r = client.get("/reflection/insights")
        assert r.status_code == 200
        data = r.json()
        assert "insights" in data
        assert "total" in data

    def test_run_reflection_behavior(self, client: TestClient) -> None:
        # Ensure some memories exist first
        client.post("/chat", json={"message": "I love working with transformers."})
        r = client.post("/reflection/run", json={"scope": "behavior", "scope_ref": "test"})
        assert r.status_code == 200
        data = r.json()
        assert data["scope"] == "behavior"
        assert isinstance(data["insights"], list)

    def test_run_reflection_project(self, client: TestClient, ctx: BlixContext) -> None:
        ctx.project_manager.get_or_create("Blix")
        r = client.post("/reflection/run", json={"scope": "project", "scope_ref": "Blix"})
        assert r.status_code == 200
        assert r.json()["scope_ref"] == "Blix"

    def test_run_reflection_with_material(self, client: TestClient) -> None:
        r = client.post("/reflection/run", json={
            "scope": "session",
            "scope_ref": "test-session",
            "material": "The user worked extensively on knowledge graphs and embedding retrieval systems embedding graphs retrieval.",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["scope"] == "session"
        # Heuristic reflection should produce at least one insight
        assert len(data["insights"]) >= 1

    def test_generate_actionable_insights(self, client: TestClient) -> None:
        r = client.post("/reflection/insights/generate")
        assert r.status_code == 200
        data = r.json()
        assert "insights" in data
        assert isinstance(data["insights"], list)

    def test_list_actionable_insights(self, client: TestClient) -> None:
        r = client.get("/reflection/insights/actionable")
        assert r.status_code == 200
        data = r.json()
        assert "insights" in data

    def test_list_actionable_insights_by_category(self, client: TestClient, ctx: BlixContext) -> None:
        # Seed a trend insight
        from datetime import timezone
        ctx.insight_engine._insights.append(
            ActionableInsight(insight="NLP trend.", category=InsightCategory.TREND, confidence=0.8)
        )
        r = client.get("/reflection/insights/actionable?category=trend")
        assert r.status_code == 200
        data = r.json()
        assert all(i["category"] == "trend" for i in data["insights"])

    def test_list_insights_scope_filter(self, client: TestClient) -> None:
        client.post("/reflection/run", json={"scope": "daily", "scope_ref": "2025-07-15",
                                              "material": "daily content nlp nlp nlp"})
        r = client.get("/reflection/insights?scope=daily")
        assert r.status_code == 200


# ===========================================================================
# /graph
# ===========================================================================


class TestGraphEndpoints:
    def test_graph_snapshot(self, client: TestClient) -> None:
        r = client.get("/graph")
        assert r.status_code == 200
        data = r.json()
        assert "node_count" in data
        assert "edge_count" in data
        assert "nodes" in data

    def test_list_nodes(self, client: TestClient) -> None:
        r = client.get("/graph/nodes")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_list_nodes_filter_kind(self, client: TestClient, ctx: BlixContext) -> None:
        from core.memory_graph import EntityKind, GraphNode
        ctx.graph.add_node(GraphNode(id="test_person", kind=EntityKind.PERSON, label="Test Person"))
        r = client.get("/graph/nodes?kind=person")
        assert r.status_code == 200
        assert all(n["kind"] == "person" for n in r.json())

    def test_list_nodes_invalid_kind(self, client: TestClient) -> None:
        r = client.get("/graph/nodes?kind=invalidkind")
        assert r.status_code == 400

    def test_get_node(self, client: TestClient, ctx: BlixContext) -> None:
        from core.memory_graph import EntityKind, GraphNode
        ctx.graph.add_node(GraphNode(id="sayan_test", kind=EntityKind.PERSON, label="Sayan Test"))
        r = client.get("/graph/nodes/sayan_test")
        assert r.status_code == 200
        assert r.json()["node_id"] == "sayan_test"

    def test_get_node_not_found(self, client: TestClient) -> None:
        r = client.get("/graph/nodes/definitely_does_not_exist_xyz")
        assert r.status_code == 404

    def test_graph_path(self, client: TestClient, ctx: BlixContext) -> None:
        ctx.graph.upsert_relation(
            "Alice Test", from_kind="person", relation="works_on",
            to_label="Project X", to_kind="project",
        )
        ctx.graph.upsert_relation(
            "Project X", from_kind="project", relation="uses",
            to_label="Python Test", to_kind="skill",
        )
        from core.memory_graph import _slug
        r = client.get(f"/graph/path?from_id={_slug('Alice Test')}&to_id={_slug('Python Test')}")
        assert r.status_code == 200
        data = r.json()
        assert "from_id" in data

    def test_graph_path_no_connection(self, client: TestClient) -> None:
        r = client.get("/graph/path?from_id=nonexistent_a&to_id=nonexistent_b")
        assert r.status_code == 200
        data = r.json()
        assert data["path"] is None
        assert data["hops"] is None

    def test_centrality(self, client: TestClient) -> None:
        r = client.get("/graph/centrality?top_k=5")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_upsert_relation(self, client: TestClient) -> None:
        r = client.post("/graph/relations", json={
            "from_label": "Sayan API Test",
            "from_kind": "person",
            "relation": "works_on",
            "to_label": "Blix API",
            "to_kind": "project",
            "confidence": 0.9,
        })
        assert r.status_code == 200
        data = r.json()
        assert "node_count" in data
        assert data["node_count"] >= 1

    def test_upsert_relation_invalid_kind(self, client: TestClient) -> None:
        r = client.post("/graph/relations", json={
            "from_label": "A", "from_kind": "invalid_kind",
            "relation": "uses", "to_label": "B", "to_kind": "topic",
        })
        assert r.status_code == 400


# ===========================================================================
# /documents
# ===========================================================================


class TestDocumentEndpoints:
    def test_upload_txt(self, client: TestClient, tmp_path: Path) -> None:
        txt = tmp_path / "doc.txt"
        txt.write_text("Transformers use self-attention mechanisms to process sequences efficiently.")
        with txt.open("rb") as fh:
            r = client.post("/documents/upload", files={"file": ("doc.txt", fh, "text/plain")})
        assert r.status_code == 200
        data = r.json()
        assert data["format"] == "txt"
        assert data["chunk_count"] >= 1
        assert data["raw_text_length"] > 0

    def test_upload_md(self, client: TestClient, tmp_path: Path) -> None:
        md = tmp_path / "notes.md"
        md.write_text("# Attention\n\nAttention is a core concept in transformers.")
        with md.open("rb") as fh:
            r = client.post("/documents/upload", files={"file": ("notes.md", fh, "text/markdown")})
        assert r.status_code == 200
        assert r.json()["format"] == "md"

    def test_upload_unsupported_format(self, client: TestClient, tmp_path: Path) -> None:
        bad = tmp_path / "data.csv"
        bad.write_text("a,b,c")
        with bad.open("rb") as fh:
            r = client.post("/documents/upload", files={"file": ("data.csv", fh, "text/csv")})
        assert r.status_code == 415

    def test_list_documents(self, client: TestClient) -> None:
        r = client.get("/documents")
        assert r.status_code == 200
        data = r.json()
        assert "total_documents" in data
        assert "documents" in data


# ===========================================================================
# /stats
# ===========================================================================


class TestStatsEndpoint:
    def test_dashboard_stats(self, client: TestClient) -> None:
        r = client.get("/stats")
        assert r.status_code == 200
        data = r.json()
        assert "memory_count" in data
        assert "graph_nodes" in data
        assert "graph_edges" in data
        assert "knowledge_facts" in data
        assert "projects" in data
        assert "goals" in data
        assert "insights" in data
        assert "background" in data
        # All counts are non-negative integers
        for key in ("memory_count", "graph_nodes", "graph_edges", "knowledge_facts",
                    "projects", "goals", "active_goals", "semantic_clusters"):
            assert data[key] >= 0, f"{key} should be >= 0"

    def test_dashboard_stats_lifecycle(self, client: TestClient) -> None:
        r = client.get("/stats")
        lc = r.json()["lifecycle_state_counts"]
        assert "active" in lc
        assert "compressed" in lc
        assert "archived" in lc
        assert "deleted" in lc


# ===========================================================================
# /goals
# ===========================================================================


class TestGoalsEndpoints:
    def test_create_goal(self, client: TestClient) -> None:
        r = client.post("/goals", json={
            "title": "Build Blix v0.4",
            "description": "Next major version",
            "priority": 1,
            "related_project": "Blix",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["title"] == "Build Blix v0.4"
        assert data["priority"] == 1
        assert data["progress"] == 0

    def test_list_goals(self, client: TestClient) -> None:
        client.post("/goals", json={"title": "List test goal"})
        r = client.get("/goals")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1

    def test_list_goals_filter_active(self, client: TestClient) -> None:
        r = client.get("/goals?status=active")
        assert r.status_code == 200
        data = r.json()
        assert all(g["status"] == "active" for g in data["goals"])

    def test_list_goals_invalid_status(self, client: TestClient) -> None:
        r = client.get("/goals?status=invalid_status")
        assert r.status_code == 400

    def test_get_goal(self, client: TestClient) -> None:
        r = client.post("/goals", json={"title": "Fetch me by ID"})
        goal_id = r.json()["goal_id"]
        r2 = client.get(f"/goals/{goal_id}")
        assert r2.status_code == 200
        assert r2.json()["goal_id"] == goal_id

    def test_get_goal_not_found(self, client: TestClient) -> None:
        r = client.get("/goals/goal_99999")
        assert r.status_code == 404

    def test_set_progress(self, client: TestClient) -> None:
        r = client.post("/goals", json={"title": "Progress test goal"})
        goal_id = r.json()["goal_id"]
        r2 = client.patch(f"/goals/{goal_id}/progress", json={"progress": 72})
        assert r2.status_code == 200
        assert r2.json()["progress"] == 72

    def test_add_blocker(self, client: TestClient) -> None:
        r = client.post("/goals", json={"title": "Blocker test goal"})
        goal_id = r.json()["goal_id"]
        r2 = client.post(f"/goals/{goal_id}/blockers", json={"description": "missing infra"})
        assert r2.status_code == 200
        assert "missing infra" in r2.json()["blockers"]

    def test_resolve_blocker(self, client: TestClient) -> None:
        r = client.post("/goals", json={"title": "Resolve blocker goal"})
        goal_id = r.json()["goal_id"]
        client.post(f"/goals/{goal_id}/blockers", json={"description": "to be resolved"})
        # Use POST with content= for DELETE with body (TestClient limitation)
        r2 = client.request(
            "DELETE", f"/goals/{goal_id}/blockers",
            json={"description": "to be resolved"},
        )
        assert r2.status_code == 200
        assert "to be resolved" not in r2.json()["blockers"]

    def test_add_milestone(self, client: TestClient) -> None:
        r = client.post("/goals", json={"title": "Milestone test goal"})
        goal_id = r.json()["goal_id"]
        r2 = client.post(f"/goals/{goal_id}/milestones", json={"title": "Design phase"})
        assert r2.status_code == 200
        milestones = r2.json()["milestones"]
        assert any(m["title"] == "Design phase" for m in milestones)


# ===========================================================================
# BlixContext.dashboard_stats() unit test
# ===========================================================================


class TestBlixContextStats:
    def test_dashboard_stats_keys(self, ctx: BlixContext) -> None:
        stats = ctx.dashboard_stats()
        required_keys = [
            "memory_count", "embedding_index_size", "knowledge_facts",
            "projects", "graph_nodes", "graph_edges", "goals", "active_goals",
            "insights", "reflection_records", "knowledge_reports",
            "semantic_clusters", "lifecycle_state_counts",
            "contradictions_unresolved", "session_count",
            "daily_summaries", "weekly_summaries", "background",
        ]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_dashboard_stats_types(self, ctx: BlixContext) -> None:
        stats = ctx.dashboard_stats()
        int_keys = ["memory_count", "knowledge_facts", "graph_nodes", "graph_edges",
                    "goals", "insights"]
        for key in int_keys:
            assert isinstance(stats[key], int), f"{key} should be int"
        assert isinstance(stats["lifecycle_state_counts"], dict)
        assert isinstance(stats["background"], dict)
