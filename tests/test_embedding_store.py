"""
Tests for core/embedding_store.py

Uses the TF-IDF fallback (no network/GPU).  A InMemoryStore subclass
overrides _sbert_encode to always return None so we stay in fallback mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from core.embedding_store import EmbeddingStore


class TFIDFStore(EmbeddingStore):
    """EmbeddingStore subclass that forces TF-IDF mode (no SBERT)."""

    def _get_sbert(self) -> None:           # type: ignore[override]
        return None

    def _sbert_encode(self, texts: list[str]) -> None:  # type: ignore[override]
        return None


def _store(tmp_path: Path, threshold: float = 0.0, top_k: int = 5) -> TFIDFStore:
    return TFIDFStore(
        embed_model_name="all-MiniLM-L6-v2",
        embeddings_file=tmp_path / "emb.npy",
        ids_file=tmp_path / "ids.json",
        threshold=threshold,
        top_k=top_k,
    )


class TestEmbeddingStoreAdd:
    def test_size_starts_zero(self, tmp_path: Path) -> None:
        assert _store(tmp_path).size == 0

    def test_add_two_entries(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.add(1, "gradient descent optimization learning rate")
        s.add(2, "backpropagation chain rule neural network")
        assert s.size == 2

    def test_indexed_ids_tracked(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.add(10, "alpha topic one sentence")
        s.add(11, "beta topic two sentence")
        assert set(s.indexed_ids) == {10, 11}

    def test_add_persists_files(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.add(1, "sentence alpha")
        s.add(2, "sentence beta")
        assert (tmp_path / "emb.npy").exists()
        assert (tmp_path / "ids.json").exists()

    def test_add_single_fails_gracefully(self, tmp_path: Path) -> None:
        """Single doc can't be TF-IDF embedded; size stays 0 — no crash."""
        s = _store(tmp_path)
        s.add(1, "only one document")
        # May or may not succeed depending on corpus size; must not raise
        assert s.size in (0, 1)


class TestEmbeddingStoreRemove:
    def test_remove_existing(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.add(1, "to remove sentence here")
        s.add(2, "to keep sentence here")
        assert s.remove(1) is True
        assert s.size == 1
        assert 1 not in s.indexed_ids

    def test_remove_missing_returns_false(self, tmp_path: Path) -> None:
        assert _store(tmp_path).remove(999) is False


class TestEmbeddingStoreSearch:
    def _seeded(self, tmp_path: Path) -> TFIDFStore:
        s = _store(tmp_path, threshold=0.0)
        s.add(1, "gradient descent optimization algorithm")
        s.add(2, "neural network deep learning")
        s.add(3, "python programming syntax")
        return s

    def test_search_returns_list(self, tmp_path: Path) -> None:
        s = self._seeded(tmp_path)
        assert isinstance(s.search("gradient"), list)

    def test_search_empty_store(self, tmp_path: Path) -> None:
        assert _store(tmp_path).search("anything") == []

    def test_result_format(self, tmp_path: Path) -> None:
        s = self._seeded(tmp_path)
        for eid, score in s.search("gradient"):
            assert isinstance(eid, int)
            assert 0.0 <= score <= 1.0


class TestEmbeddingStorePersistence:
    def test_reload_from_disk(self, tmp_path: Path) -> None:
        s1 = _store(tmp_path)
        s1.add(1, "persistent text alpha sentence")
        s1.add(2, "persistent text beta sentence")

        s2 = TFIDFStore(
            embed_model_name="all-MiniLM-L6-v2",
            embeddings_file=tmp_path / "emb.npy",
            ids_file=tmp_path / "ids.json",
            threshold=0.0,
            top_k=5,
        )
        assert s2.size == 2
        assert set(s2.indexed_ids) == {1, 2}


class TestEmbeddingStoreRebuild:
    def test_rebuild_sets_size(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        pairs = [(i, f"document {i} about topic machine learning") for i in range(1, 6)]
        s.rebuild(pairs)
        assert s.size == 5

    def test_rebuild_empty_clears(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.rebuild([])
        assert s.size == 0

    def test_rebuild_replaces_old_index(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.add(1, "old entry one alpha")
        s.add(2, "old entry two beta")
        new_pairs = [(10, "new doc alpha topic"), (11, "new doc beta topic")]
        s.rebuild(new_pairs)
        assert set(s.indexed_ids) == {10, 11}
