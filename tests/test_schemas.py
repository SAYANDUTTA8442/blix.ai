"""Tests for all Pydantic schemas — v0.2 (includes new MemoryEntry fields)."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from schemas.memory_entry import MemoryEntry
from schemas.profile import Profile
from schemas.learning_state import LearningState


class TestMemoryEntry:
    def test_basic_creation(self) -> None:
        entry = MemoryEntry(id=1, input="What is AI?", output="AI is…")
        assert entry.id == 1
        assert isinstance(entry.timestamp, datetime)

    def test_v02_fields_default(self) -> None:
        entry = MemoryEntry(id=1, input="q", output="a")
        assert entry.embedding_id is None
        assert entry.extracted_facts == []
        assert entry.topics == []

    def test_v02_fields_populated(self) -> None:
        entry = MemoryEntry(
            id=1,
            input="q",
            output="a",
            embedding_id=42,
            extracted_facts=["User is studying NLP."],
            topics=["NLP", "Transformers"],
        )
        assert entry.embedding_id == 42
        assert "NLP" in entry.topics

    def test_importance_range(self) -> None:
        MemoryEntry(id=1, input="q", output="a", importance=0.0)
        MemoryEntry(id=1, input="q", output="a", importance=1.0)
        with pytest.raises(ValidationError):
            MemoryEntry(id=1, input="q", output="a", importance=1.1)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValidationError):
            MemoryEntry(id=1, input="", output="a")

    def test_roundtrip(self) -> None:
        entry = MemoryEntry(
            id=5,
            input="question",
            output="answer",
            extracted_facts=["Fact one."],
            topics=["Topic"],
            importance=0.6,
        )
        restored = MemoryEntry.model_validate(entry.model_dump())
        assert restored.id == 5
        assert restored.extracted_facts == ["Fact one."]
        assert restored.topics == ["Topic"]


class TestProfile:
    def test_defaults_empty(self) -> None:
        assert Profile().is_empty()

    def test_not_empty_with_name(self) -> None:
        assert not Profile(name="Sayan").is_empty()

    def test_full_profile(self) -> None:
        p = Profile(
            name="Sayan",
            education="IIT Patna",
            interests=["LLMs"],
            projects=["Blix"],
            goals=["Research"],
        )
        assert not p.is_empty()
        assert "LLMs" in p.interests


class TestLearningState:
    def test_defaults_empty(self) -> None:
        ls = LearningState()
        assert ls.total_count() == 0

    def test_all_topics_deduplicates(self) -> None:
        ls = LearningState(
            topics_learned=["Python"],
            topics_in_progress=["Python", "Transformers"],
        )
        assert ls.all_topics().count("Python") == 1

    def test_total_count_unique(self) -> None:
        ls = LearningState(
            topics_learned=["A"],
            topics_in_progress=["B", "C"],
            weak_topics=["D"],
            strong_topics=["A"],  # duplicate
        )
        assert ls.total_count() == 4
