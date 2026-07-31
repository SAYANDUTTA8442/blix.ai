"""Tests for core/memory_manager.py"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.memory_manager import MemoryManager
from schemas.profile import Profile
from schemas.learning_state import LearningState


@pytest.fixture
def mm(tmp_path: Path) -> MemoryManager:
    """Fresh MemoryManager backed by temporary files."""
    return MemoryManager(
        conversations_file=tmp_path / "conversations.json",
        profile_file=tmp_path / "profile.json",
        learning_state_file=tmp_path / "learning_state.json",
    )


class TestMemoryManagerCRUD:
    def test_add_memory_increments_id(self, mm: MemoryManager) -> None:
        e1 = mm.add_memory("q1", "a1")
        e2 = mm.add_memory("q2", "a2")
        assert e1.id == 1
        assert e2.id == 2

    def test_get_all_returns_all(self, mm: MemoryManager) -> None:
        mm.add_memory("q1", "a1")
        mm.add_memory("q2", "a2")
        assert len(mm.get_all_memories()) == 2

    def test_memory_count(self, mm: MemoryManager) -> None:
        assert mm.memory_count() == 0
        mm.add_memory("q", "a")
        assert mm.memory_count() == 1

    def test_get_memory_by_id_found(self, mm: MemoryManager) -> None:
        entry = mm.add_memory("question", "answer")
        found = mm.get_memory_by_id(entry.id)
        assert found is not None
        assert found.input == "question"

    def test_get_memory_by_id_missing(self, mm: MemoryManager) -> None:
        assert mm.get_memory_by_id(999) is None

    def test_update_memory(self, mm: MemoryManager) -> None:
        entry = mm.add_memory("old question", "old answer")
        updated = mm.update_memory(entry.id, input="new question")
        assert updated is not None
        assert updated.input == "new question"

    def test_update_missing_returns_none(self, mm: MemoryManager) -> None:
        assert mm.update_memory(999, input="x") is None

    def test_delete_memory(self, mm: MemoryManager) -> None:
        entry = mm.add_memory("q", "a")
        result = mm.delete_memory(entry.id)
        assert result is True
        assert mm.memory_count() == 0

    def test_delete_missing_returns_false(self, mm: MemoryManager) -> None:
        assert mm.delete_memory(999) is False

    def test_persists_to_disk(self, tmp_path: Path) -> None:
        """Memories written by one instance are readable by another."""
        mm1 = MemoryManager(
            conversations_file=tmp_path / "c.json",
            profile_file=tmp_path / "p.json",
            learning_state_file=tmp_path / "l.json",
        )
        mm1.add_memory("persist this", "yes it persists")

        mm2 = MemoryManager(
            conversations_file=tmp_path / "c.json",
            profile_file=tmp_path / "p.json",
            learning_state_file=tmp_path / "l.json",
        )
        assert mm2.memory_count() == 1
        assert mm2.get_all_memories()[0].input == "persist this"

    def test_id_survives_delete_then_add(self, mm: MemoryManager) -> None:
        e1 = mm.add_memory("q1", "a1")
        e2 = mm.add_memory("q2", "a2")
        mm.delete_memory(e1.id)
        e3 = mm.add_memory("q3", "a3")
        # e3 id must be greater than e2 id
        assert e3.id > e2.id


class TestMemoryManagerProfile:
    def test_default_profile_empty(self, mm: MemoryManager) -> None:
        assert mm.profile.is_empty()

    def test_set_and_load_profile(self, tmp_path: Path) -> None:
        mm1 = MemoryManager(
            conversations_file=tmp_path / "c.json",
            profile_file=tmp_path / "p.json",
            learning_state_file=tmp_path / "l.json",
        )
        mm1.profile = Profile(name="Sayan", education="IIT Patna")

        mm2 = MemoryManager(
            conversations_file=tmp_path / "c.json",
            profile_file=tmp_path / "p.json",
            learning_state_file=tmp_path / "l.json",
        )
        assert mm2.profile.name == "Sayan"


class TestMemoryManagerLearningState:
    def test_default_learning_state_empty(self, mm: MemoryManager) -> None:
        assert mm.learning_state.total_count() == 0

    def test_set_and_load_learning_state(self, tmp_path: Path) -> None:
        mm1 = MemoryManager(
            conversations_file=tmp_path / "c.json",
            profile_file=tmp_path / "p.json",
            learning_state_file=tmp_path / "l.json",
        )
        mm1.learning_state = LearningState(topics_in_progress=["Transformers"])

        mm2 = MemoryManager(
            conversations_file=tmp_path / "c.json",
            profile_file=tmp_path / "p.json",
            learning_state_file=tmp_path / "l.json",
        )
        assert "Transformers" in mm2.learning_state.topics_in_progress
