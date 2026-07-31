"""
Tests for core/semantic_retriever.py

Uses a stub EmbeddingStore so no model download is needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from core.embedding_store import EmbeddingStore
from core.memory_retriever import MemoryRetriever
from core.semantic_retriever import SemanticRetriever
from schemas.memory_entry import MemoryEntry


def _entry(id: int, text: str) -> MemoryEntry:
    return MemoryEntry(id=id, input=text, output="answer", timestamp=datetime.now(timezone.utc).replace(tzinfo=None))


class StubEmbeddingStore(EmbeddingStore):
    """
    Stub that returns pre-set search results without any real embeddings.
    """

    def __init__(self, search_results: list[tuple[int, float]]) -> None:
        # Do NOT call super().__init__() — we don't want file I/O
        self._search_results = search_results
        self._ids: list[int] = []
        self._matrix = None
        self._embed_model = None
        self._tfidf = None

    def search(self, query: str, top_k: Optional[int] = None) -> list[tuple[int, float]]:
        return self._search_results

    def add(self, entry_id: int, text: str) -> Optional[int]:
        self._ids.append(entry_id)
        return len(self._ids) - 1

    def rebuild(self, id_text_pairs: list[tuple[int, str]]) -> None:
        self._ids = [eid for eid, _ in id_text_pairs]

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def indexed_ids(self) -> list[int]:
        return list(self._ids)


@pytest.fixture
def memories() -> list[MemoryEntry]:
    return [
        _entry(1, "gradient descent optimization"),
        _entry(2, "backpropagation chain rule"),
        _entry(3, "attention mechanism transformers"),
        _entry(4, "python programming basics"),
        _entry(5, "matrix multiplication linear algebra"),
    ]


class TestSemanticRetrieverRetrieve:
    def test_returns_list(self, memories: list[MemoryEntry]) -> None:
        store = StubEmbeddingStore([(1, 0.9), (3, 0.75)])
        retriever = SemanticRetriever(store, MemoryRetriever(recent_k=2))
        result = retriever.retrieve(memories, "gradient")
        assert isinstance(result, list)

    def test_semantic_hits_included(self, memories: list[MemoryEntry]) -> None:
        store = StubEmbeddingStore([(1, 0.9), (3, 0.75)])
        retriever = SemanticRetriever(store, MemoryRetriever(recent_k=2))
        result = retriever.retrieve(memories, "gradient")
        ids = [m.id for m in result]
        assert 1 in ids
        assert 3 in ids

    def test_results_sorted_by_id(self, memories: list[MemoryEntry]) -> None:
        store = StubEmbeddingStore([(3, 0.9), (1, 0.7)])
        retriever = SemanticRetriever(store, MemoryRetriever(recent_k=1))
        result = retriever.retrieve(memories, "anything")
        ids = [m.id for m in result]
        assert ids == sorted(ids)

    def test_no_duplicates(self, memories: list[MemoryEntry]) -> None:
        # semantic and legacy both return entry 1
        store = StubEmbeddingStore([(1, 0.9)])
        retriever = SemanticRetriever(store, MemoryRetriever(recent_k=3))
        result = retriever.retrieve(memories, "gradient descent")
        ids = [m.id for m in result]
        assert len(ids) == len(set(ids))

    def test_empty_memories(self) -> None:
        store = StubEmbeddingStore([])
        retriever = SemanticRetriever(store, MemoryRetriever())
        assert retriever.retrieve([], "anything") == []


class TestSemanticRetrieverIndexEntry:
    def test_index_entry_increments_size(self, memories: list[MemoryEntry]) -> None:
        store = StubEmbeddingStore([])
        retriever = SemanticRetriever(store, MemoryRetriever())
        retriever.index_entry(memories[0])
        assert retriever.index_size == 1

    def test_rebuild_index_sets_size(self, memories: list[MemoryEntry]) -> None:
        store = StubEmbeddingStore([])
        retriever = SemanticRetriever(store, MemoryRetriever())
        retriever.rebuild_index(memories)
        assert retriever.index_size == len(memories)
