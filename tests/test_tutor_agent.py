"""
Tests for core/tutor_agent.py — v0.2

Mock LLM + stub SemanticRetriever — no Ollama/Transformers/network needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from core.memory_extractor import MemoryExtractor, ExtractionResult
from core.memory_manager import MemoryManager
from core.memory_retriever import MemoryRetriever
from core.prompt_builder import PromptBuilder
from core.semantic_retriever import SemanticRetriever
from core.tutor_agent import TutorAgent
from llm.base import LLMProvider
from schemas.memory_entry import MemoryEntry
from schemas.profile import Profile


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class MockLLM(LLMProvider):
    def __init__(self, reply: str = "Mocked reply.") -> None:
        self._reply = reply
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        return self._reply

    def model_name(self) -> str:
        return "mock"


class ErrorLLM(LLMProvider):
    def generate(self, prompt: str) -> str:
        raise RuntimeError("LLM offline")

    def model_name(self) -> str:
        return "error"


class JsonLLM(LLMProvider):
    """Returns fixed JSON for extraction tests."""

    def __init__(self, json_str: str) -> None:
        self._json = json_str

    def generate(self, prompt: str) -> str:
        return self._json

    def model_name(self) -> str:
        return "json-mock"


class StubEmbeddingStore:
    """No-op embedding store."""

    def __init__(self) -> None:
        self._ids: list[int] = []

    def search(self, query: str, top_k: Optional[int] = None) -> list:
        return []

    def add(self, entry_id: int, text: str) -> Optional[int]:
        self._ids.append(entry_id)
        return len(self._ids) - 1

    def rebuild(self, pairs: list) -> None:
        self._ids = [e for e, _ in pairs]

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def indexed_ids(self) -> list[int]:
        return list(self._ids)


def _retriever() -> SemanticRetriever:
    from core.embedding_store import EmbeddingStore

    class StubStore(EmbeddingStore):
        def __init__(self) -> None:
            self._ids: list[int] = []
            self._matrix = None
            self._embed_model = None
            self._tfidf = None
            self._tfidf_corpus: list[str] = []
            self._tfidf_fitted = False
            self._corpus: list[str] = []
            self._corpus_ids: list[int] = []
            self._pending_ids: list[int] = []
            self._pending_texts: list[str] = []

        def search(self, *a, **kw) -> list:
            return []

        def add(self, entry_id: int, text: str) -> Optional[int]:
            self._ids.append(entry_id)
            return len(self._ids) - 1

        def rebuild(self, pairs: list) -> None:
            self._ids = [e for e, _ in pairs]

        @property
        def size(self) -> int:
            return len(self._ids)

        @property
        def indexed_ids(self) -> list[int]:
            return list(self._ids)

    return SemanticRetriever(StubStore(), MemoryRetriever(recent_k=3))


@pytest.fixture
def mm(tmp_path: Path) -> MemoryManager:
    return MemoryManager(
        conversations_file=tmp_path / "c.json",
        profile_file=tmp_path / "p.json",
        learning_state_file=tmp_path / "l.json",
    )


@pytest.fixture
def agent(mm: MemoryManager) -> TutorAgent:
    return TutorAgent(
        llm=MockLLM(),
        memory_manager=mm,
        retriever=_retriever(),
        prompt_builder=PromptBuilder(),
        extractor=None,
    )


# ---------------------------------------------------------------------------
# Core chat
# ---------------------------------------------------------------------------


class TestChat:
    def test_returns_string(self, agent: TutorAgent) -> None:
        assert isinstance(agent.chat("What is AI?"), str)

    def test_saves_memory(self, agent: TutorAgent, mm: MemoryManager) -> None:
        agent.chat("question")
        assert mm.memory_count() == 1

    def test_correct_input_saved(self, agent: TutorAgent, mm: MemoryManager) -> None:
        agent.chat("Explain backprop")
        assert mm.get_all_memories()[0].input == "Explain backprop"

    def test_correct_output_saved(self, agent: TutorAgent, mm: MemoryManager) -> None:
        agent.chat("anything")
        assert mm.get_all_memories()[0].output == "Mocked reply."

    def test_multiple_turns_accumulate(self, agent: TutorAgent, mm: MemoryManager) -> None:
        for i in range(5):
            agent.chat(f"turn {i}")
        assert mm.memory_count() == 5

    def test_llm_error_raises(self, mm: MemoryManager) -> None:
        ag = TutorAgent(
            llm=ErrorLLM(), memory_manager=mm,
            retriever=_retriever(), prompt_builder=PromptBuilder(),
        )
        with pytest.raises(RuntimeError):
            ag.chat("question")

    def test_no_memory_saved_on_llm_error(self, mm: MemoryManager) -> None:
        ag = TutorAgent(
            llm=ErrorLLM(), memory_manager=mm,
            retriever=_retriever(), prompt_builder=PromptBuilder(),
        )
        try:
            ag.chat("question")
        except RuntimeError:
            pass
        assert mm.memory_count() == 0


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class TestIndexing:
    def test_entry_indexed_after_chat(self, agent: TutorAgent) -> None:
        agent.chat("index this")
        assert agent.index_size == 1

    def test_multiple_entries_indexed(self, agent: TutorAgent) -> None:
        for i in range(3):
            agent.chat(f"turn {i}")
        assert agent.index_size == 3


# ---------------------------------------------------------------------------
# CoT extraction integration
# ---------------------------------------------------------------------------


_EXTRACT_JSON = """{
  "facts": ["User is learning PyTorch autograd."],
  "topics": ["PyTorch", "Autograd"],
  "weak_topics": [],
  "learning_topics": ["PyTorch", "Autograd"],
  "strong_topics": [],
  "importance": 0.7,
  "profile_updates": {
    "name": "Sayan",
    "education": "",
    "new_interests": ["PyTorch"],
    "new_projects": [],
    "new_goals": []
  }
}"""


class TestExtractionIntegration:
    def _agent_with_extraction(self, mm: MemoryManager) -> TutorAgent:
        llm = JsonLLM(_EXTRACT_JSON)
        return TutorAgent(
            llm=llm,
            memory_manager=mm,
            retriever=_retriever(),
            prompt_builder=PromptBuilder(),
            extractor=MemoryExtractor(llm=llm, enabled=True),
        )

    def test_entry_enriched_with_topics(self, mm: MemoryManager) -> None:
        ag = self._agent_with_extraction(mm)
        ag.chat("How does PyTorch autograd work?")
        entry = mm.get_all_memories()[0]
        assert "PyTorch" in entry.topics

    def test_entry_enriched_with_facts(self, mm: MemoryManager) -> None:
        ag = self._agent_with_extraction(mm)
        ag.chat("How does PyTorch autograd work?")
        entry = mm.get_all_memories()[0]
        assert len(entry.extracted_facts) > 0

    def test_learning_state_updated(self, mm: MemoryManager) -> None:
        ag = self._agent_with_extraction(mm)
        ag.chat("Explain autograd")
        assert "PyTorch" in mm.learning_state.topics_in_progress

    def test_profile_auto_updated(self, mm: MemoryManager) -> None:
        ag = self._agent_with_extraction(mm)
        ag.chat("Explain autograd")
        assert mm.profile.name == "Sayan"

    def test_extraction_error_doesnt_crash_chat(self, mm: MemoryManager) -> None:
        class BrokenExtractor(MemoryExtractor):
            def extract(self, *a) -> ExtractionResult:
                raise ValueError("extraction exploded")

        ag = TutorAgent(
            llm=MockLLM(),
            memory_manager=mm,
            retriever=_retriever(),
            prompt_builder=PromptBuilder(),
            extractor=BrokenExtractor(MockLLM()),
        )
        reply = ag.chat("question")
        assert isinstance(reply, str)
        assert mm.memory_count() == 1


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_contains_question(self, agent: TutorAgent) -> None:
        p = agent.build_prompt("unique_xyz_question", [])
        assert "unique_xyz_question" in p

    def test_contains_profile_name(self, agent: TutorAgent, mm: MemoryManager) -> None:
        mm.profile = Profile(name="TestUser")
        p = agent.build_prompt("q", [])
        assert "TestUser" in p


# ---------------------------------------------------------------------------
# rebuild_index
# ---------------------------------------------------------------------------


class TestRebuildIndex:
    def test_rebuild_runs_without_error(self, agent: TutorAgent, mm: MemoryManager) -> None:
        mm.add_memory("q1", "a1")
        mm.add_memory("q2", "a2")
        agent.rebuild_index()   # must not raise
