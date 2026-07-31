"""
Tests for Blix v0.3.4 — "Knowledge Graph Reasoning & Cognitive Queries".

Covers:
1+2.  CognitiveQueryEngine   (natural-language queries + multi-hop inference)
3.    MQLv2Engine             (expression-style MQL)
5.    ResearchAssistant        (paper → structured notes)
6.    ExplainabilityEngine     (evidence chains)
7.    ReasoningEvaluator       (reasoning accuracy + graph coverage)
API.  /reason + /research endpoints

Python 3.10 compatible — fully offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from core.cognitive_query_engine import (
    CognitiveQueryEngine, QueryResult, ReasoningStep, ReasoningTrace,
)
from core.explainability import (
    ExplainabilityEngine, ExplainedResponse,
    FactEvidence, GraphEvidence, MemoryEvidence, InsightEvidence,
)
from core.graph_reasoner import GraphReasoner
from core.memory_graph import EntityKind, GraphEdge, GraphNode, MemoryGraph, RelationKind
from evaluation.reasoning import ReasoningCase, ReasoningEvaluator
from evaluation.blix_eval import ReasoningEvaluator as RE_from_blix_eval
from knowledge.document_processor import ProcessedDocument, DocumentFormat, DocumentChunk
from knowledge.research_assistant import ResearchAssistant, ResearchNotes, _heuristic_research_notes
from reflection.mql_v2 import (
    MQLv2Engine, MQLv2Parser, MQLv2Result,
)
from schemas.memory_entry import MemoryEntry
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(days_ago: float = 0.0) -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago)


def _entry(id: int, output: str = "a", topics: list | None = None,
           importance: float | None = None) -> MemoryEntry:
    return MemoryEntry(id=id, input="q", output=output, timestamp=_ts(),
                       topics=topics or [], importance=importance)


def _make_graph(tmp_path: Path) -> MemoryGraph:
    """Build a test graph: Sayan → works_on → Blix → uses → FastAPI/ChromaDB."""
    g = MemoryGraph(tmp_path / "graph.json")
    g.upsert_relation("Sayan", EntityKind.PERSON, RelationKind.WORKS_ON, "Blix", EntityKind.PROJECT, source_memory_id=1)
    g.upsert_relation("Blix", EntityKind.PROJECT, RelationKind.USES, "FastAPI", EntityKind.SKILL, source_memory_id=2)
    g.upsert_relation("Blix", EntityKind.PROJECT, RelationKind.USES, "ChromaDB", EntityKind.SKILL, source_memory_id=3)
    g.upsert_relation("Blix", EntityKind.PROJECT, RelationKind.USES, "Transformers", EntityKind.SKILL, source_memory_id=4)
    g.upsert_relation("Sayan", EntityKind.PERSON, RelationKind.STUDIES_AT, "IIT Patna", EntityKind.ORGANIZATION)
    g.upsert_relation("Alice", EntityKind.PERSON, RelationKind.COLLABORATES_WITH, "Sayan", EntityKind.PERSON)
    return g


def _make_doc(doc_id: str = "doc_001", title: str = "Test Paper") -> ProcessedDocument:
    return ProcessedDocument(
        doc_id=doc_id, title=title, format=DocumentFormat.TXT,
        summary="This paper presents a novel approach to memory retrieval.",
        key_findings=["Hierarchical memory improves recall by 20%.", "Graph augmentation reduces drift."],
        concepts=["hierarchical memory", "knowledge graph", "retrieval"],
        related_topics=["memory", "nlp"],
        entities=[("Sayan", "person"), ("IIT Patna", "organization")],
        chunks=[DocumentChunk(chunk_id="c0", text="Main content of the paper.", chunk_index=0)],
        raw_text_length=500,
    )


# ===========================================================================
# Feature 1+2 — CognitiveQueryEngine
# ===========================================================================


class TestCognitiveQueryEngine:
    @pytest.fixture
    def graph(self, tmp_path: Path) -> MemoryGraph:
        return _make_graph(tmp_path)

    @pytest.fixture
    def engine(self, graph: MemoryGraph) -> CognitiveQueryEngine:
        return CognitiveQueryEngine(graph=graph)

    # ------ Feature 1: direct graph queries ------

    def test_query_uses_outgoing(self, engine: CognitiveQueryEngine) -> None:
        result = engine.query("What does Blix use?")
        assert not result.is_empty()
        labels = {a.lower() for a in result.answer}
        assert "fastapi" in labels or "chromadb" in labels or "transformers" in labels

    def test_query_works_on(self, engine: CognitiveQueryEngine) -> None:
        result = engine.query("What does Sayan work on?")
        assert not result.is_empty()
        assert any("blix" in a.lower() for a in result.answer)

    def test_query_studies_at(self, engine: CognitiveQueryEngine) -> None:
        result = engine.query("Where does Sayan study?")
        assert not result.is_empty()
        assert any("iit" in a.lower() or "patna" in a.lower() for a in result.answer)

    def test_query_inverse_who_works_on(self, engine: CognitiveQueryEngine) -> None:
        result = engine.query("Who works on Blix?")
        assert not result.is_empty()
        assert any("sayan" in a.lower() for a in result.answer)

    def test_query_unknown_entity(self, engine: CognitiveQueryEngine) -> None:
        result = engine.query("What does XYZUnknownEntity use?")
        assert result.is_empty()
        assert "not found" in result.trace.explanation.lower()

    def test_query_trace_has_steps(self, engine: CognitiveQueryEngine) -> None:
        result = engine.query("What does Blix use?")
        if not result.is_empty():
            assert len(result.trace.steps) >= 1

    def test_query_result_to_dict(self, engine: CognitiveQueryEngine) -> None:
        result = engine.query("What does Blix use?")
        d = result.to_dict()
        assert "query" in d
        assert "answer" in d
        assert "trace" in d
        assert "steps" in d["trace"]

    def test_reasoning_trace_str(self) -> None:
        trace = ReasoningTrace(
            steps=[ReasoningStep("Sayan", "works_on", "Blix", confidence=0.9)],
            confidence=0.9,
            explanation="Test trace",
        )
        assert "works_on" in str(trace)
        assert "Sayan" in str(trace)

    def test_reasoning_trace_to_dict(self) -> None:
        trace = ReasoningTrace(
            steps=[ReasoningStep("A", "uses", "B", confidence=0.8)],
            source_memory_ids=[1, 2],
            confidence=0.8,
        )
        d = trace.to_dict()
        assert d["confidence"] == pytest.approx(0.8)
        assert len(d["steps"]) == 1
        assert d["source_memory_ids"] == [1, 2]

    # ------ Feature 2: multi-hop ------

    def test_multihop_finds_intermediates(self, engine: CognitiveQueryEngine) -> None:
        result = engine.multi_hop_query("Sayan", "FastAPI")
        assert not result.is_empty()
        assert any("blix" in a.lower() for a in result.answer)

    def test_multihop_no_path(self, engine: CognitiveQueryEngine) -> None:
        result = engine.multi_hop_query("IIT Patna", "FastAPI")
        assert result.is_empty()

    def test_multihop_unknown_start(self, engine: CognitiveQueryEngine) -> None:
        result = engine.multi_hop_query("Ghost", "FastAPI")
        assert result.is_empty()
        assert "not found" in result.trace.explanation.lower()

    def test_multihop_confidence_in_range(self, engine: CognitiveQueryEngine) -> None:
        result = engine.multi_hop_query("Sayan", "FastAPI")
        if not result.is_empty():
            assert 0.0 <= result.trace.confidence <= 1.0

    # ------ Feature 2: transitive inference ------

    def test_transitive_closure(self, engine: CognitiveQueryEngine) -> None:
        result = engine.infer_transitive("Blix", "uses", depth=1)
        assert not result.is_empty()
        labels = {a.lower() for a in result.answer}
        assert "fastapi" in labels or "chromadb" in labels

    def test_transitive_depth_2(self, engine: CognitiveQueryEngine, tmp_path: Path) -> None:
        # Add Sayan works_on Blix, Blix uses FastAPI → Sayan transitively connected to FastAPI via works_on
        g = _make_graph(tmp_path / "g2")
        # Add: FastAPI uses HTTP (depth 2 target)
        g.upsert_relation("FastAPI", EntityKind.SKILL, RelationKind.USES, "HTTP", EntityKind.SKILL)
        eng = CognitiveQueryEngine(g)
        result = eng.infer_transitive("Blix", "uses", depth=2)
        labels = {a.lower() for a in result.answer}
        # At depth 1: fastapi, chromadb, transformers; at depth 2: http
        assert "fastapi" in labels or "http" in labels

    def test_transitive_unknown_entity(self, engine: CognitiveQueryEngine) -> None:
        result = engine.infer_transitive("Ghost", "uses")
        assert result.is_empty()

    def test_transitive_unknown_relation(self, engine: CognitiveQueryEngine) -> None:
        result = engine.infer_transitive("Blix", "invalid_relation_xyz")
        assert result.is_empty()
        assert "unknown relation" in result.trace.explanation.lower()

    def test_transitive_explanation_populated(self, engine: CognitiveQueryEngine) -> None:
        result = engine.infer_transitive("Blix", "uses")
        assert result.trace.explanation != ""

    def test_transitive_steps_in_trace(self, engine: CognitiveQueryEngine) -> None:
        result = engine.infer_transitive("Blix", "uses", depth=1)
        if not result.is_empty():
            assert len(result.trace.steps) >= 1

    # ------ Query parsing ------

    def test_query_parse_uses(self, engine: CognitiveQueryEngine) -> None:
        parsed = engine._parse_query("What does Blix use?")
        assert parsed is not None
        subject, rel, inverse = parsed
        assert "blix" in subject.lower()
        assert rel == "uses"
        assert inverse is False

    def test_query_parse_inverse(self, engine: CognitiveQueryEngine) -> None:
        parsed = engine._parse_query("Who works on Blix?")
        assert parsed is not None
        _, _, inverse = parsed
        assert inverse is True

    def test_query_parse_unrecognised(self, engine: CognitiveQueryEngine) -> None:
        parsed = engine._parse_query("blah blah blah")
        # Should still match fallback pattern or return None
        # Either is acceptable — just don't raise
        assert parsed is None or isinstance(parsed, tuple)


# ===========================================================================
# Feature 3 — MQLv2Engine (expression-style queries)
# ===========================================================================


class TestMQLv2Parser:
    def test_memories_topics_contains(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse('memories where topics contains "nlp"')
        assert r is not None
        assert r[0] == "mem_topics_contains"
        assert r[1]["groups"][0] == "nlp"

    def test_memories_project_eq(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse('memories where project = "Blix"')
        assert r is not None
        assert r[0] == "mem_project_eq"

    def test_memories_importance_gte(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse("memories where importance >= 0.7")
        assert r is not None
        assert r[0] == "mem_importance_cmp"
        assert r[1]["groups"][0] == ">="
        assert r[1]["groups"][1] == "0.7"

    def test_facts_about(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse('facts about "transformers"')
        assert r is not None
        assert r[0] == "facts_about"

    def test_facts_min_confidence(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse("facts min_confidence = 0.8")
        assert r is not None
        assert r[0] == "facts_min_conf"

    def test_facts_topic_eq(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse('facts topic = "nlp"')
        assert r is not None
        assert r[0] == "facts_topic_eq"

    def test_insights_last_30_days(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse("insights last_30_days")
        assert r is not None
        assert r[0] == "insights_last_days"
        assert r[1]["groups"][0] == "30"

    def test_insights_category(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse('insights category = "trend"')
        assert r is not None
        assert r[0] == "insights_category"

    def test_goals_status(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse("goals status = active")
        assert r is not None
        assert r[0] == "goals_status"

    def test_goals_priority_lte(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse("goals priority <= 2")
        assert r is not None
        assert r[0] == "goals_priority_cmp"

    def test_graph_neighbours(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse('graph neighbours "Sayan"')
        assert r is not None
        assert r[0] == "graph_neighbours"

    def test_graph_path(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse('graph path "Sayan" to "FastAPI"')
        assert r is not None
        assert r[0] == "graph_path"

    def test_cognitive_query(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse('query "What does Blix use?"')
        assert r is not None
        assert r[0] == "cognitive_query"

    def test_infer_transitive(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse('infer "Blix" via "uses" depth 2')
        assert r is not None
        assert r[0] == "infer_transitive"

    def test_multihop(self) -> None:
        parser = MQLv2Parser()
        r = parser.parse('multihop "Sayan" to "FastAPI"')
        assert r is not None
        assert r[0] == "multihop"

    def test_unrecognised_returns_none(self) -> None:
        parser = MQLv2Parser()
        assert parser.parse("do something weird") is None


class TestMQLv2Engine:
    @pytest.fixture
    def engine(self, tmp_path: Path) -> MQLv2Engine:
        from reflection.goal_tracker import GoalTracker
        from reflection.consolidation_engine import ConsolidationEngine

        g = _make_graph(tmp_path)
        gt = GoalTracker(tmp_path / "goals.json")
        gt.create_goal("Build Blix v0.4", priority=1, related_project="Blix")
        ce = ConsolidationEngine(tmp_path / "facts.json", base_confidence=0.6)
        ce.consolidate("User prefers PyTorch", 1, topic="ml")
        cqe = CognitiveQueryEngine(graph=g)

        return MQLv2Engine(
            goal_tracker=gt,
            graph=g,
            graph_reasoner=GraphReasoner(g),
            cognitive_query_engine=cqe,
            consolidation_engine=ce,
        )

    def test_memories_topics_unavailable(self, engine: MQLv2Engine) -> None:
        r = engine.run('memories where topics contains "nlp"')
        assert r.matched
        # No memory_manager → unavailable message
        assert "MemoryManager" in r.text

    def test_facts_about(self, engine: MQLv2Engine) -> None:
        r = engine.run('facts about "pytorch"')
        assert r.matched
        assert "PyTorch" in r.text or "pytorch" in r.text.lower()

    def test_facts_min_confidence(self, engine: MQLv2Engine) -> None:
        r = engine.run("facts min_confidence = 0.5")
        assert r.matched
        assert "fact" in r.text.lower() or "No " in r.text

    def test_goals_status_active(self, engine: MQLv2Engine) -> None:
        r = engine.run("goals status = active")
        assert r.matched
        assert "Build Blix v0.4" in r.text

    def test_goals_priority_cmp(self, engine: MQLv2Engine) -> None:
        r = engine.run("goals priority <= 2")
        assert r.matched
        assert "Build Blix v0.4" in r.text

    def test_goals_project(self, engine: MQLv2Engine) -> None:
        r = engine.run('goals project = "Blix"')
        assert r.matched
        assert "Build Blix v0.4" in r.text

    def test_graph_neighbours(self, engine: MQLv2Engine) -> None:
        r = engine.run('graph neighbours "Sayan"')
        assert r.matched
        assert "Blix" in r.text or "IIT" in r.text

    def test_graph_neighbours_unknown(self, engine: MQLv2Engine) -> None:
        r = engine.run('graph neighbours "GhostEntity"')
        assert r.matched
        assert "not found" in r.text.lower()

    def test_graph_path(self, engine: MQLv2Engine) -> None:
        r = engine.run('graph path "Sayan" to "FastAPI"')
        assert r.matched
        assert "hop" in r.text.lower() or "path" in r.text.lower()

    def test_cognitive_query(self, engine: MQLv2Engine) -> None:
        r = engine.run('query "What does Blix use?"')
        assert r.matched
        assert r.trace is not None
        assert "fastapi" in r.text.lower() or "chromadb" in r.text.lower()

    def test_infer_transitive(self, engine: MQLv2Engine) -> None:
        r = engine.run('infer "Blix" via "uses" depth 1')
        assert r.matched
        assert "fastapi" in r.text.lower() or "chromadb" in r.text.lower()

    def test_multihop(self, engine: MQLv2Engine) -> None:
        r = engine.run('multihop "Sayan" to "FastAPI"')
        assert r.matched
        assert "blix" in r.text.lower() or "Blix" in r.text

    def test_fallback_to_v1_show_command(self, engine: MQLv2Engine) -> None:
        r = engine.run("show active goals")
        assert r.matched  # falls back to v0.3.2 MQLEngine
        assert "Build Blix v0.4" in r.text

    def test_is_mql_command(self, engine: MQLv2Engine) -> None:
        assert engine.is_mql_command('query "What does Blix use?"')
        assert engine.is_mql_command("memories where topics contains \"nlp\"")
        assert engine.is_mql_command("show active goals")
        assert not engine.is_mql_command("What is gradient descent?")

    def test_unrecognised_falls_back(self, engine: MQLv2Engine) -> None:
        r = engine.run("some completely unknown command xyz")
        assert not r.matched


# ===========================================================================
# Feature 5 — ResearchAssistant
# ===========================================================================


class TestResearchAssistant:
    @pytest.fixture
    def assistant(self, tmp_path: Path) -> ResearchAssistant:
        return ResearchAssistant(notes_file=tmp_path / "notes.json")

    def test_process_document(self, assistant: ResearchAssistant) -> None:
        doc = _make_doc()
        notes = assistant.process(doc)
        assert notes.doc_id == "doc_001"
        assert notes.title == "Test Paper"
        assert notes.summary
        assert assistant.count == 1

    def test_heuristic_extracts_findings(self) -> None:
        doc = _make_doc()
        doc.chunks = [DocumentChunk(
            chunk_id="c0", chunk_index=0,
            text=(
                "Abstract: We study memory systems.\n"
                "Methodology: We use hierarchical compression.\n"
                "Results: Accuracy improved by 15%. Recall improved.\n"
                "Limitations: Only tested on small datasets."
            )
        )]
        notes = _heuristic_research_notes(doc)
        assert notes.summary
        assert notes.methodology != ""

    def test_get_notes(self, assistant: ResearchAssistant) -> None:
        doc = _make_doc("doc_002")
        assistant.process(doc)
        notes = assistant.get("doc_002")
        assert notes is not None
        assert notes.doc_id == "doc_002"

    def test_get_notes_not_found(self, assistant: ResearchAssistant) -> None:
        assert assistant.get("nonexistent") is None

    def test_list_all_sorted_recent_first(self, assistant: ResearchAssistant) -> None:
        assistant.process(_make_doc("d1"))
        assistant.process(_make_doc("d2"))
        all_notes = assistant.list_all()
        assert len(all_notes) == 2
        # Most recent first
        assert all_notes[0].created_at >= all_notes[1].created_at

    def test_search_by_title(self, assistant: ResearchAssistant) -> None:
        assistant.process(_make_doc("d1", "Attention is All You Need"))
        results = assistant.search("attention")
        assert len(results) >= 1
        assert any("Attention" in n.title for n in results)

    def test_search_by_concept(self, assistant: ResearchAssistant) -> None:
        assistant.process(_make_doc("d1", "Graph Memory Paper"))
        results = assistant.search("knowledge graph")
        assert len(results) >= 1

    def test_persistence_roundtrip(self, tmp_path: Path) -> None:
        a1 = ResearchAssistant(tmp_path / "n.json")
        a1.process(_make_doc())
        a2 = ResearchAssistant(tmp_path / "n.json")
        assert a2.count == 1

    def test_llm_extraction_used(self, tmp_path: Path) -> None:
        class FakeLLM:
            def generate(self, prompt: str) -> str:
                return json.dumps({
                    "summary": "LLM summary of paper.",
                    "methodology": "We used transformer models.",
                    "key_findings": ["Attention improves recall."],
                    "limitations": ["Only tested on English text."],
                    "future_work": ["Test on multilingual data."],
                    "related_concepts": ["attention", "recall"],
                    "entities": [["BERT", "skill"]],
                    "related_topics": ["nlp"],
                    "confidence": 0.92,
                })
            def model_name(self) -> str:
                return "fake"

        assistant = ResearchAssistant(tmp_path / "n.json", llm=FakeLLM())
        doc = _make_doc()
        notes = assistant.process(doc)
        assert notes.summary == "LLM summary of paper."
        assert notes.methodology == "We used transformer models."
        assert notes.confidence == 0.92

    def test_llm_failure_falls_back(self, tmp_path: Path) -> None:
        class BadLLM:
            def generate(self, prompt: str) -> str:
                return "not json"
            def model_name(self) -> str:
                return "fake"

        assistant = ResearchAssistant(tmp_path / "n.json", llm=BadLLM())
        notes = assistant.process(_make_doc())
        # Heuristic fallback — should still produce something
        assert notes.summary != ""

    def test_integration_with_consolidation(self, tmp_path: Path) -> None:
        from reflection.consolidation_engine import ConsolidationEngine
        ce = ConsolidationEngine(tmp_path / "facts.json")
        assistant = ResearchAssistant(tmp_path / "n.json", consolidation_engine=ce)
        doc = _make_doc()
        assistant.process(doc)
        # Key findings should have been consolidated as facts
        assert ce.fact_count >= 1

    def test_integration_with_graph(self, tmp_path: Path) -> None:
        g = MemoryGraph(tmp_path / "graph.json")
        assistant = ResearchAssistant(tmp_path / "n.json", graph=g)
        doc = _make_doc()
        assistant.process(doc)
        # Entities should have been upserted into the graph
        assert g.node_count >= 1

    def test_to_dict(self) -> None:
        notes = ResearchNotes(
            doc_id="d1", title="Paper",
            summary="Summary.", methodology="Method.",
            key_findings=["Finding 1"],
            limitations=["Limit 1"],
            future_work=["Future 1"],
        )
        d = notes.to_dict()
        assert d["doc_id"] == "d1"
        assert d["key_findings"] == ["Finding 1"]

    def test_from_dict_roundtrip(self) -> None:
        original = ResearchNotes(
            doc_id="d1", title="Paper", summary="S", methodology="M",
            key_findings=["F1"], limitations=["L1"], future_work=["FW1"],
            entities=[("Sayan", "person")], related_topics=["nlp"], confidence=0.8,
        )
        restored = ResearchNotes.from_dict(original.to_dict())
        assert restored.doc_id == "d1"
        assert restored.confidence == pytest.approx(0.8)
        assert restored.entities == [("Sayan", "person")]


# ===========================================================================
# Feature 6 — ExplainabilityEngine
# ===========================================================================


class TestExplainabilityEngine:
    @pytest.fixture
    def engine_with_graph(self, tmp_path: Path) -> tuple:
        g = _make_graph(tmp_path)
        reasoner = GraphReasoner(g)
        engine = ExplainabilityEngine(graph=g, graph_reasoner=reasoner)
        return engine, g

    def test_explain_with_graph_from_trace(self, engine_with_graph: tuple) -> None:
        engine, g = engine_with_graph
        trace = ReasoningTrace(
            steps=[ReasoningStep("Blix", "uses", "FastAPI", confidence=0.9)],
            confidence=0.9,
            explanation="Blix uses FastAPI.",
        )
        result = engine.explain("FastAPI", "What does Blix use?", reasoning_trace=trace)
        assert isinstance(result, ExplainedResponse)
        assert len(result.graph_evidence) >= 1
        assert result.graph_evidence[0].confidence == pytest.approx(0.9)

    def test_explain_graph_evidence_from_query(self, engine_with_graph: tuple) -> None:
        engine, g = engine_with_graph
        # Entity "Blix" mentioned in query → edges from Blix found
        result = engine.explain("FastAPI, ChromaDB", "What does Blix use?")
        assert isinstance(result, ExplainedResponse)
        # Graph evidence should be found (Blix has outgoing edges)
        assert len(result.graph_evidence) >= 1

    def test_explain_with_no_components(self) -> None:
        engine = ExplainabilityEngine()
        result = engine.explain("answer", "query")
        assert isinstance(result, ExplainedResponse)
        assert result.total_evidence_count == 0
        assert result.overall_confidence == 0.0

    def test_overall_confidence_weighted(self, tmp_path: Path) -> None:
        result = ExplainedResponse(
            answer="test",
            memory_evidence=[MemoryEvidence(1, "excerpt", 0.8, ["nlp"])],
            fact_evidence=[FactEvidence("f1", "fact", 0.9, 5)],
        )
        conf = result.overall_confidence
        assert 0.0 < conf <= 1.0
        # Fact evidence is weighted 1.5 vs memory 1.0 — fact confidence dominates
        # Expected: (0.8*1.0 + 0.9*1.5) / (1.0+1.5) = (0.8 + 1.35) / 2.5 = 0.86
        assert conf == pytest.approx((0.8 * 1.0 + 0.9 * 1.5) / 2.5, abs=0.01)

    def test_total_evidence_count(self) -> None:
        result = ExplainedResponse(
            answer="test",
            memory_evidence=[MemoryEvidence(1, "x", 0.8)],
            fact_evidence=[FactEvidence("f1", "y", 0.9, 3)],
            graph_evidence=[GraphEvidence("A→B")],
        )
        assert result.total_evidence_count == 3

    def test_explain_str(self) -> None:
        result = ExplainedResponse(
            answer="FastAPI, ChromaDB",
            memory_evidence=[MemoryEvidence(14, "Used ChromaDB as embedding store", 0.85)],
            fact_evidence=[FactEvidence("fact_3", "Blix uses FastAPI", 0.88, 10)],
            graph_evidence=[GraphEvidence("Blix →[uses]→ FastAPI", confidence=0.95)],
        )
        text = result.explain_str()
        assert "FastAPI, ChromaDB" in text
        assert "Memory #14" in text
        assert "fact_3" in text
        assert "Graph evidence" in text

    def test_to_dict_structure(self) -> None:
        result = ExplainedResponse(answer="test")
        d = result.to_dict()
        assert "answer" in d
        assert "overall_confidence" in d
        assert "memory_evidence" in d
        assert "fact_evidence" in d
        assert "graph_evidence" in d

    def test_fact_evidence_to_dict(self) -> None:
        fe = FactEvidence("f1", "Blix uses FastAPI", 0.88, 10)
        d = fe.to_dict()
        assert d["type"] == "canonical_fact"
        assert d["fact_id"] == "f1"

    def test_graph_evidence_str(self) -> None:
        ge = GraphEvidence("Blix →[uses]→ FastAPI", confidence=0.95)
        assert "FastAPI" in str(ge)
        assert "0.95" in str(ge)

    def test_insight_evidence_to_dict(self) -> None:
        ie = InsightEvidence("User focuses on NLP.", confidence=0.8, scope="behavior")
        d = ie.to_dict()
        assert d["type"] == "insight"
        assert d["scope"] == "behavior"

    def test_explained_response_empty_no_components(self) -> None:
        engine = ExplainabilityEngine(max_memories=3, max_facts=3, max_graph_paths=3)
        result = engine.explain("answer", "What is attention?")
        assert result.answer == "answer"
        assert result.total_evidence_count == 0


# ===========================================================================
# Feature 7 — ReasoningEvaluator
# ===========================================================================


class TestReasoningEvaluator:
    def test_reasoning_accuracy_all_match(self) -> None:
        ev = ReasoningEvaluator()
        acc = ev.reasoning_accuracy(["FastAPI", "ChromaDB"], ["FastAPI", "ChromaDB"])
        assert acc == 1.0

    def test_reasoning_accuracy_partial(self) -> None:
        ev = ReasoningEvaluator()
        acc = ev.reasoning_accuracy(["FastAPI"], ["FastAPI", "ChromaDB"])
        assert acc == 0.5

    def test_reasoning_accuracy_none_match(self) -> None:
        ev = ReasoningEvaluator()
        acc = ev.reasoning_accuracy(["Unknown"], ["FastAPI"])
        assert acc == 0.0

    def test_reasoning_accuracy_empty_expected(self) -> None:
        ev = ReasoningEvaluator()
        assert ev.reasoning_accuracy(["anything"], []) == 1.0

    def test_reasoning_accuracy_case_insensitive(self) -> None:
        ev = ReasoningEvaluator()
        acc = ev.reasoning_accuracy(["fastapi"], ["FastAPI"])
        assert acc == 1.0

    def test_reasoning_precision_no_hallucination(self) -> None:
        ev = ReasoningEvaluator()
        prec = ev.reasoning_precision(["FastAPI", "ChromaDB"], ["FastAPI", "ChromaDB", "Redis"])
        assert prec == 1.0

    def test_reasoning_precision_with_hallucination(self) -> None:
        ev = ReasoningEvaluator()
        prec = ev.reasoning_precision(["FastAPI", "HallucinatedTool"], ["FastAPI"])
        assert prec == 0.5

    def test_graph_coverage_entities(self, tmp_path: Path) -> None:
        ev = ReasoningEvaluator()
        g = _make_graph(tmp_path)
        cov = ev.graph_coverage(g, ["Sayan", "Blix", "FastAPI"])
        assert cov == pytest.approx(1.0)

    def test_graph_coverage_missing_entity(self, tmp_path: Path) -> None:
        ev = ReasoningEvaluator()
        g = _make_graph(tmp_path)
        cov = ev.graph_coverage(g, ["Sayan", "Blix", "NonExistentTool"])
        assert cov < 1.0

    def test_graph_coverage_with_edges(self, tmp_path: Path) -> None:
        ev = ReasoningEvaluator()
        g = _make_graph(tmp_path)
        edges = [("Sayan", "works_on", "Blix"), ("Blix", "uses", "FastAPI")]
        cov = ev.graph_coverage(g, ["Sayan", "Blix", "FastAPI"], expected_edges=edges)
        assert cov > 0.9

    def test_graph_coverage_wrong_edge(self, tmp_path: Path) -> None:
        ev = ReasoningEvaluator()
        g = _make_graph(tmp_path)
        edges = [("Sayan", "uses", "Blix")]  # wrong relation (should be works_on)
        cov = ev.graph_coverage(g, ["Sayan", "Blix"], expected_edges=edges)
        assert cov < 1.0

    def test_path_accuracy_exact(self) -> None:
        ev = ReasoningEvaluator()
        assert ev.path_accuracy(2, 2) == 1.0

    def test_path_accuracy_within_tolerance(self) -> None:
        ev = ReasoningEvaluator()
        assert ev.path_accuracy(3, 2, tolerance=1) == 1.0

    def test_path_accuracy_outside_tolerance(self) -> None:
        ev = ReasoningEvaluator()
        assert ev.path_accuracy(5, 2, tolerance=1) == 0.0

    def test_path_accuracy_none_expected(self) -> None:
        ev = ReasoningEvaluator()
        assert ev.path_accuracy(3, None) == 1.0

    def test_path_accuracy_none_predicted(self) -> None:
        ev = ReasoningEvaluator()
        assert ev.path_accuracy(None, 2) == 0.0

    def test_inference_recall(self) -> None:
        ev = ReasoningEvaluator()
        recall = ev.inference_recall(["FastAPI", "ChromaDB", "Transformers"], ["FastAPI", "ChromaDB"])
        assert recall == 1.0

    def test_inference_recall_partial(self) -> None:
        ev = ReasoningEvaluator()
        recall = ev.inference_recall(["FastAPI"], ["FastAPI", "ChromaDB"])
        assert recall == 0.5

    def test_explainability_score_all_present(self) -> None:
        ev = ReasoningEvaluator()
        trace = ReasoningTrace(steps=[ReasoningStep("A", "uses", "B", 0.9)], confidence=0.9)
        result = ExplainedResponse(
            answer="test",
            memory_evidence=[MemoryEvidence(1, "x", 0.8)],
            fact_evidence=[FactEvidence("f1", "y", 0.9, 3)],
            graph_evidence=[GraphEvidence("A→B", confidence=0.9)],
            reasoning_trace=trace,
        )
        score = ev.explainability_score(result)
        assert score == pytest.approx(1.0)

    def test_explainability_score_partial(self) -> None:
        ev = ReasoningEvaluator()
        result = ExplainedResponse(
            answer="test",
            memory_evidence=[MemoryEvidence(1, "x", 0.8)],
        )
        score = ev.explainability_score(result)
        assert 0.0 < score < 1.0

    def test_explainability_score_empty(self) -> None:
        ev = ReasoningEvaluator()
        result = ExplainedResponse(answer="test")
        assert ev.explainability_score(result) == 0.0

    def test_evaluate_reasoning_query_fn(self, tmp_path: Path) -> None:
        g = _make_graph(tmp_path)
        cqe = CognitiveQueryEngine(graph=g)
        ev = ReasoningEvaluator()
        cases = [
            ReasoningCase(case_id="c1", query="What does Blix use?",
                          expected_answers=["FastAPI", "ChromaDB"]),
        ]
        results = ev.evaluate_reasoning(
            cases, query_fn=cqe.query, graph=g,
            expected_graph_entities=["Sayan", "Blix", "FastAPI"],
        )
        assert "reasoning_accuracy" in results
        assert "graph_coverage" in results
        assert results["reasoning_accuracy"] >= 0.0
        assert results["graph_coverage"] >= 0.0

    def test_evaluate_reasoning_transitive(self, tmp_path: Path) -> None:
        g = _make_graph(tmp_path)
        cqe = CognitiveQueryEngine(graph=g)
        ev = ReasoningEvaluator()
        cases = [
            ReasoningCase(
                case_id="c1", start_entity="Blix",
                transitive_relation="uses", transitive_depth=1,
                expected_transitive_nodes=["FastAPI"],
            ),
        ]
        results = ev.evaluate_reasoning(
            cases,
            infer_fn=lambda e, r, d: cqe.infer_transitive(e, r, depth=d),
        )
        assert "inference_recall" in results
        assert results["inference_recall"] >= 0.5

    def test_reasoning_case_dataclass(self) -> None:
        case = ReasoningCase(
            case_id="c1",
            query="What does Blix use?",
            expected_answers=["FastAPI"],
        )
        assert case.case_id == "c1"
        assert case.transitive_depth == 2  # default

    def test_reasoning_evaluator_in_blix_eval(self) -> None:
        assert ReasoningEvaluator is RE_from_blix_eval


# ===========================================================================
# API — /reason + /research endpoints
# ===========================================================================


class _FakeLLM:
    def model_name(self) -> str:
        return "fake-0.3.4"

    def generate(self, prompt: str) -> str:
        return "This is a test reply."


@pytest.fixture(scope="module")
def tmp_memory_v4(tmp_path_factory):
    return tmp_path_factory.mktemp("memory_v4")


@pytest.fixture(scope="module")
def ctx_v4(tmp_memory_v4):
    from config import settings as _settings
    _settings.settings.memory.conversations_file = tmp_memory_v4 / "conversations.json"
    _settings.settings.memory.profile_file = tmp_memory_v4 / "profile.json"
    _settings.settings.memory.learning_state_file = tmp_memory_v4 / "learning_state.json"
    _settings.settings.embed.embeddings_file = tmp_memory_v4 / "embeddings.npy"
    _settings.settings.embed.embedding_ids_file = tmp_memory_v4 / "embedding_ids.json"

    from api.context import BlixContext
    ctx = BlixContext(tmp_memory_v4)
    ctx.llm = _FakeLLM()
    ctx.agent._llm = _FakeLLM()
    # Seed the graph for query tests
    ctx.graph.upsert_relation("Sayan", EntityKind.PERSON, RelationKind.WORKS_ON, "Blix", EntityKind.PROJECT)
    ctx.graph.upsert_relation("Blix", EntityKind.PROJECT, RelationKind.USES, "FastAPI", EntityKind.SKILL)
    ctx.graph.upsert_relation("Blix", EntityKind.PROJECT, RelationKind.USES, "ChromaDB", EntityKind.SKILL)
    return ctx


@pytest.fixture(scope="module")
def client_v4(ctx_v4) -> Generator[TestClient, None, None]:
    from api.deps import set_context
    from api.routers.reasoning_research import reason_router, research_router
    from api.routers.chat import router as chat_router
    from api.routers.stats_goals import stats_router

    app = FastAPI(title="Blix Test v0.3.4")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(reason_router)
    app.include_router(research_router)
    app.include_router(chat_router)
    app.include_router(stats_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    set_context(ctx_v4)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestReasoningAPI:
    def test_cognitive_query_what_does_blix_use(self, client_v4: TestClient) -> None:
        r = client_v4.post("/reason/query", json={"query": "What does Blix use?", "explain": True})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert isinstance(data["answer"], list)
        assert "trace" in data
        assert "explanation" in data
        # Should find FastAPI or ChromaDB
        labels = [a.lower() for a in data["answer"]]
        assert "fastapi" in labels or "chromadb" in labels

    def test_cognitive_query_no_explain(self, client_v4: TestClient) -> None:
        r = client_v4.post("/reason/query", json={"query": "What does Blix use?", "explain": False})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "trace" not in data

    def test_cognitive_query_unknown_entity(self, client_v4: TestClient) -> None:
        r = client_v4.post("/reason/query", json={"query": "What does GhostXYZ use?"})
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == []
        assert data["is_empty"] is True

    def test_multihop_sayan_to_fastapi(self, client_v4: TestClient) -> None:
        r = client_v4.post("/reason/multihop", json={"start": "Sayan", "end": "FastAPI"})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "blix" in [a.lower() for a in data["answer"]]

    def test_multihop_no_path(self, client_v4: TestClient) -> None:
        r = client_v4.post("/reason/multihop", json={"start": "IIT Patna", "end": "ChromaDB"})
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == []

    def test_transitive_infer(self, client_v4: TestClient) -> None:
        r = client_v4.post("/reason/infer", json={"entity": "Blix", "relation": "uses", "depth": 1})
        assert r.status_code == 200
        data = r.json()
        labels = [a.lower() for a in data.get("answer", [])]
        assert "fastapi" in labels or "chromadb" in labels

    def test_explain_endpoint(self, client_v4: TestClient) -> None:
        r = client_v4.get("/reason/explain?q=What+does+Blix+use%3F&answer=FastAPI")
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "overall_confidence" in data
        assert "graph_evidence" in data

    def test_empty_query_rejected(self, client_v4: TestClient) -> None:
        r = client_v4.post("/reason/query", json={"query": ""})
        assert r.status_code == 422


class TestResearchAPI:
    def test_list_research_notes_empty(self, client_v4: TestClient) -> None:
        r = client_v4.get("/research")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "notes" in data

    def test_get_research_notes_not_found(self, client_v4: TestClient) -> None:
        r = client_v4.get("/research/nonexistent_doc_id")
        assert r.status_code == 404

    def test_search_research_notes_empty(self, client_v4: TestClient) -> None:
        r = client_v4.get("/research/search?q=attention")
        assert r.status_code == 200
        data = r.json()
        assert "query" in data
        assert data["query"] == "attention"

    def test_research_flow_process_and_retrieve(self, client_v4: TestClient, ctx_v4) -> None:
        # Process a document via ResearchAssistant directly, then verify API retrieves it
        doc = _make_doc("research_api_test_doc", "Attention Mechanisms in NLP")
        ctx_v4.research_assistant.process(doc)

        r = client_v4.get("/research/research_api_test_doc")
        assert r.status_code == 200
        data = r.json()
        assert data["doc_id"] == "research_api_test_doc"
        assert data["title"] == "Attention Mechanisms in NLP"

    def test_research_search_finds_processed(self, client_v4: TestClient, ctx_v4) -> None:
        doc = _make_doc("searchable_doc", "Graph-Based Memory Systems")
        ctx_v4.research_assistant.process(doc)

        r = client_v4.get("/research/search?q=graph")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1
