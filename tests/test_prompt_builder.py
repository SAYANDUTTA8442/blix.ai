"""Tests for core/prompt_builder.py"""

from __future__ import annotations

from datetime import datetime, timezone

from core.prompt_builder import PromptBuilder
from schemas.memory_entry import MemoryEntry
from schemas.profile import Profile
from schemas.learning_state import LearningState


def _entry(id: int, input: str, output: str = "answer") -> MemoryEntry:
    return MemoryEntry(id=id, input=input, output=output, timestamp=datetime.now(timezone.utc).replace(tzinfo=None))


class TestPromptBuilder:
    def setup_method(self) -> None:
        self.builder = PromptBuilder()
        self.profile = Profile(
            name="Sayan",
            education="IIT Patna BS-MS CSDA",
            interests=["LLMs", "NLP"],
            projects=["Blix", "ECOT"],
            goals=["AI Research Engineer"],
        )
        self.state = LearningState(
            topics_learned=["Python"],
            topics_in_progress=["Transformers"],
            weak_topics=["Statistics"],
            strong_topics=["Linear Algebra"],
        )

    def test_output_is_string(self) -> None:
        prompt = self.builder.build("How does attention work?", self.profile, self.state, [])
        assert isinstance(prompt, str)

    def test_system_block_present(self) -> None:
        prompt = self.builder.build("q", self.profile, self.state, [])
        assert "Blix" in prompt

    def test_profile_injected(self) -> None:
        prompt = self.builder.build("q", self.profile, self.state, [])
        assert "Sayan" in prompt
        assert "IIT Patna" in prompt

    def test_learning_state_injected(self) -> None:
        prompt = self.builder.build("q", self.profile, self.state, [])
        assert "Transformers" in prompt
        assert "Statistics" in prompt

    def test_memories_injected(self) -> None:
        memories = [_entry(1, "What is backprop?", "Chain rule applied to neural nets.")]
        prompt = self.builder.build("q", self.profile, self.state, memories)
        assert "backprop" in prompt.lower() or "What is backprop" in prompt

    def test_question_injected(self) -> None:
        prompt = self.builder.build("How do LSTMs work?", self.profile, self.state, [])
        assert "How do LSTMs work?" in prompt

    def test_empty_profile_section_omitted(self) -> None:
        prompt = self.builder.build("q", Profile(), self.state, [])
        assert "User Profile" not in prompt

    def test_empty_learning_state_section_omitted(self) -> None:
        prompt = self.builder.build("q", self.profile, LearningState(), [])
        assert "Learning State" not in prompt

    def test_no_memories_section_omitted(self) -> None:
        prompt = self.builder.build("q", self.profile, self.state, [])
        assert "Past Interactions" not in prompt

    def test_memory_count_shown(self) -> None:
        memories = [_entry(1, "q1"), _entry(2, "q2")]
        prompt = self.builder.build("q", self.profile, self.state, memories)
        assert "2 retrieved" in prompt
