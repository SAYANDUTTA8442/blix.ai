"""Tests for core/memory_retriever.py"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.memory_retriever import MemoryRetriever
from schemas.memory_entry import MemoryEntry


def _make_entry(id: int, input: str, output: str = "answer") -> MemoryEntry:
    return MemoryEntry(id=id, input=input, output=output, timestamp=datetime.now(timezone.utc).replace(tzinfo=None))


@pytest.fixture
def retriever() -> MemoryRetriever:
    return MemoryRetriever(
        recent_k=3,
        fuzzy_top_k=3,
        fuzzy_threshold=60.0,
        keyword_top_k=3,
    )


@pytest.fixture
def memories() -> list[MemoryEntry]:
    return [
        _make_entry(1, "What is gradient descent?", "Gradient descent is an optimisation algorithm."),
        _make_entry(2, "Explain backpropagation", "Backpropagation computes gradients via chain rule."),
        _make_entry(3, "How does attention work?", "Attention assigns weights to input tokens."),
        _make_entry(4, "What is Python?", "Python is a high-level programming language."),
        _make_entry(5, "Tell me about transformers", "Transformers use self-attention mechanisms."),
    ]


class TestRecent:
    def test_returns_last_k(self, retriever: MemoryRetriever, memories: list[MemoryEntry]) -> None:
        result = retriever.recent(memories, k=2)
        assert len(result) == 2
        assert result[-1].id == 5

    def test_returns_all_when_fewer_than_k(self, retriever: MemoryRetriever) -> None:
        small = [_make_entry(1, "q")]
        result = retriever.recent(small, k=5)
        assert len(result) == 1

    def test_empty_list(self, retriever: MemoryRetriever) -> None:
        assert retriever.recent([]) == []


class TestKeywordSearch:
    def test_finds_exact_match(self, retriever: MemoryRetriever, memories: list[MemoryEntry]) -> None:
        results = retriever.keyword_search(memories, "gradient descent")
        ids = [r.id for r in results]
        assert 1 in ids

    def test_case_insensitive(self, retriever: MemoryRetriever, memories: list[MemoryEntry]) -> None:
        results = retriever.keyword_search(memories, "PYTHON")
        ids = [r.id for r in results]
        assert 4 in ids

    def test_no_match_returns_empty(self, retriever: MemoryRetriever, memories: list[MemoryEntry]) -> None:
        results = retriever.keyword_search(memories, "quantum computing")
        assert results == []

    def test_respects_top_k(self, retriever: MemoryRetriever) -> None:
        many = [_make_entry(i, f"gradient step {i}") for i in range(10)]
        results = retriever.keyword_search(many, "gradient", top_k=3)
        assert len(results) <= 3


class TestFuzzySearch:
    def test_finds_semantically_close(self, retriever: MemoryRetriever, memories: list[MemoryEntry]) -> None:
        results = retriever.fuzzy_search(memories, "gradient descent optimization")
        ids = [r.id for r in results]
        assert 1 in ids

    def test_threshold_filters_low_scores(self, retriever: MemoryRetriever, memories: list[MemoryEntry]) -> None:
        results = retriever.fuzzy_search(memories, "quantum entanglement", threshold=95.0)
        assert results == []

    def test_empty_list(self, retriever: MemoryRetriever) -> None:
        assert retriever.fuzzy_search([], "anything") == []


class TestRetrieve:
    def test_returns_deduplicated_results(
        self, retriever: MemoryRetriever, memories: list[MemoryEntry]
    ) -> None:
        results = retriever.retrieve(memories, "gradient descent")
        ids = [r.id for r in results]
        assert len(ids) == len(set(ids)), "Duplicate ids found"

    def test_results_sorted_by_id(
        self, retriever: MemoryRetriever, memories: list[MemoryEntry]
    ) -> None:
        results = retriever.retrieve(memories, "transformer attention")
        ids = [r.id for r in results]
        assert ids == sorted(ids)

    def test_empty_memories(self, retriever: MemoryRetriever) -> None:
        assert retriever.retrieve([], "anything") == []

    def test_always_includes_recent(self, retriever: MemoryRetriever) -> None:
        """Even an off-topic query should include recent entries."""
        m = [_make_entry(i, f"topic {i}") for i in range(1, 10)]
        results = retriever.retrieve(m, "completely unrelated xyz")
        assert len(results) >= 1  # at least recent entries
