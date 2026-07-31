"""
Tests for core/memory_extractor.py — v0.2

Uses mock LLMs — no Transformers or Ollama needed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.memory_extractor import MemoryExtractor, ExtractionResult
from llm.base import LLMProvider
from schemas.memory_entry import MemoryEntry
from schemas.learning_state import LearningState
from schemas.profile import Profile


# ---------------------------------------------------------------------------
# Mock LLMs
# ---------------------------------------------------------------------------


class JsonLLM(LLMProvider):
    def __init__(self, json_str: str) -> None:
        self._json = json_str

    def generate(self, prompt: str) -> str:
        return self._json

    def model_name(self) -> str:
        return "json-mock"


class ErrorLLM(LLMProvider):
    def generate(self, prompt: str) -> str:
        raise RuntimeError("LLM offline")

    def model_name(self) -> str:
        return "error-mock"


# ---------------------------------------------------------------------------
# Shared JSON fixtures
# ---------------------------------------------------------------------------

_GOOD_JSON = """{
  "facts": ["User is studying Transformers at IIT Patna."],
  "topics": ["Transformers", "Attention Mechanism"],
  "weak_topics": [],
  "learning_topics": ["Transformers", "Attention Mechanism"],
  "strong_topics": [],
  "importance": 0.8,
  "profile_updates": {
    "name": "Sayan",
    "education": "IIT Patna BS-MS CSDA",
    "new_interests": ["NLP", "Transformers"],
    "new_projects": ["ECOT"],
    "new_goals": ["AI Research Engineer"]
  }
}"""

_JSON_IN_PROSE = """
STEP 1: User asked about backprop, seems to be learning it.
STEP 2: Topic is backpropagation.
STEP 3: weak signal — needed explanation.
STEP 4: 0.5
STEP 5: User asked about backpropagation basics.

{
  "facts": ["User asked about backpropagation basics."],
  "topics": ["Backpropagation"],
  "weak_topics": ["Backpropagation"],
  "learning_topics": [],
  "strong_topics": [],
  "importance": 0.5,
  "profile_updates": {"name": "", "education": "", "new_interests": [], "new_projects": [], "new_goals": []}
}
"""

_JSON_WITH_FENCE = """```json
{
  "facts": ["User is working on ECOT project."],
  "topics": ["NLP"],
  "weak_topics": [],
  "learning_topics": ["NLP"],
  "strong_topics": [],
  "importance": 0.6,
  "profile_updates": {"name": "", "education": "", "new_interests": ["NLP"], "new_projects": ["ECOT"], "new_goals": []}
}
```"""

_BAD_JSON = "I could not determine anything useful here."

_STRONG_JSON = """{
  "facts": ["User demonstrated mastery of linear algebra."],
  "topics": ["Linear Algebra"],
  "weak_topics": [],
  "learning_topics": [],
  "strong_topics": ["Linear Algebra"],
  "importance": 0.9,
  "profile_updates": {"name": "", "education": "", "new_interests": [], "new_projects": [], "new_goals": []}
}"""


def _ts() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------


class TestExtractionResult:
    def test_empty_factory(self) -> None:
        r = ExtractionResult.empty()
        assert r.facts == []
        assert r.topics == []
        assert r.importance == 0.0

    def test_importance_clamped_low(self) -> None:
        assert ExtractionResult([], [], [], [], [], -0.5).importance == 0.0

    def test_importance_clamped_high(self) -> None:
        assert ExtractionResult([], [], [], [], [], 1.9).importance == 1.0

    def test_equality(self) -> None:
        a = ExtractionResult.empty()
        b = ExtractionResult.empty()
        assert a == b


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------


class TestExtract:
    def test_parses_clean_json(self) -> None:
        r = MemoryExtractor(JsonLLM(_GOOD_JSON)).extract("q", "a")
        assert "Transformers" in r.topics
        assert r.importance == pytest.approx(0.8)
        assert len(r.facts) == 1

    def test_parses_json_in_prose(self) -> None:
        r = MemoryExtractor(JsonLLM(_JSON_IN_PROSE)).extract("q", "a")
        assert "Backpropagation" in r.weak_topics

    def test_parses_json_with_fence(self) -> None:
        r = MemoryExtractor(JsonLLM(_JSON_WITH_FENCE)).extract("q", "a")
        assert "NLP" in r.topics

    def test_returns_empty_on_bad_json(self) -> None:
        r = MemoryExtractor(JsonLLM(_BAD_JSON)).extract("q", "a")
        assert r.facts == []
        assert r.importance == 0.0

    def test_returns_empty_on_llm_error(self) -> None:
        r = MemoryExtractor(ErrorLLM()).extract("q", "a")
        assert r.facts == []

    def test_disabled_skips_llm(self) -> None:
        r = MemoryExtractor(ErrorLLM(), enabled=False).extract("q", "a")
        assert r.facts == []

    def test_profile_fields_parsed(self) -> None:
        r = MemoryExtractor(JsonLLM(_GOOD_JSON)).extract("q", "a")
        assert r.profile_name == "Sayan"
        assert r.profile_education == "IIT Patna BS-MS CSDA"
        assert "NLP" in r.profile_new_interests
        assert "ECOT" in r.profile_new_projects


# ---------------------------------------------------------------------------
# apply_to_entry()
# ---------------------------------------------------------------------------


def _entry(id: int = 1) -> MemoryEntry:
    return MemoryEntry(id=id, input="question", output="answer", timestamp=_ts())


class TestApplyToEntry:
    def test_facts_set(self) -> None:
        ext = MemoryExtractor(JsonLLM(_GOOD_JSON))
        result = ext.extract("q", "a")
        entry = ext.apply_to_entry(_entry(), result)
        assert len(entry.extracted_facts) > 0

    def test_topics_set(self) -> None:
        ext = MemoryExtractor(JsonLLM(_GOOD_JSON))
        result = ext.extract("q", "a")
        entry = ext.apply_to_entry(_entry(), result)
        assert "Transformers" in entry.topics

    def test_importance_set(self) -> None:
        ext = MemoryExtractor(JsonLLM(_GOOD_JSON))
        result = ext.extract("q", "a")
        entry = ext.apply_to_entry(_entry(), result)
        assert entry.importance == pytest.approx(0.8)

    def test_original_entry_unchanged(self) -> None:
        ext = MemoryExtractor(JsonLLM(_GOOD_JSON))
        result = ext.extract("q", "a")
        original = _entry()
        ext.apply_to_entry(original, result)
        assert original.extracted_facts == []  # model_copy is immutable

    def test_empty_result_leaves_entry_clean(self) -> None:
        ext = MemoryExtractor(JsonLLM(_BAD_JSON))
        result = ext.extract("q", "a")
        entry = ext.apply_to_entry(_entry(), result)
        assert entry.extracted_facts == []


# ---------------------------------------------------------------------------
# apply_to_learning_state()
# ---------------------------------------------------------------------------


class TestApplyToLearningState:
    def _ext(self) -> MemoryExtractor:
        return MemoryExtractor(JsonLLM("{}"))

    def _result(
        self,
        learning: list[str],
        weak: list[str],
        strong: list[str],
    ) -> ExtractionResult:
        return ExtractionResult([], learning + weak + strong, weak, learning, strong, 0.5)

    def test_learning_added_to_in_progress(self) -> None:
        state = LearningState()
        new = self._ext().apply_to_learning_state(state, self._result(["Transformers"], [], []))
        assert "Transformers" in new.topics_in_progress

    def test_weak_added_to_weak(self) -> None:
        state = LearningState()
        new = self._ext().apply_to_learning_state(state, self._result([], ["Statistics"], []))
        assert "Statistics" in new.weak_topics

    def test_strong_added_to_strong(self) -> None:
        state = LearningState(topics_in_progress=["Python"])
        new = self._ext().apply_to_learning_state(state, self._result([], [], ["Python"]))
        assert "Python" in new.strong_topics
        assert "Python" not in new.topics_in_progress

    def test_no_duplicate(self) -> None:
        state = LearningState(topics_in_progress=["Transformers"])
        new = self._ext().apply_to_learning_state(state, self._result(["Transformers"], [], []))
        assert new.topics_in_progress.count("Transformers") == 1

    def test_empty_result_unchanged(self) -> None:
        state = LearningState(topics_in_progress=["Python"])
        new = self._ext().apply_to_learning_state(state, ExtractionResult.empty())
        assert "Python" in new.topics_in_progress


# ---------------------------------------------------------------------------
# apply_to_profile()
# ---------------------------------------------------------------------------


class TestApplyToProfile:
    def _ext(self) -> MemoryExtractor:
        return MemoryExtractor(JsonLLM("{}"))

    def test_name_set_when_empty(self) -> None:
        profile = Profile()
        result = ExtractionResult([], [], [], [], [], 0.5, profile_name="Sayan")
        new = self._ext().apply_to_profile(profile, result)
        assert new.name == "Sayan"

    def test_name_not_overwritten(self) -> None:
        profile = Profile(name="Existing")
        result = ExtractionResult([], [], [], [], [], 0.5, profile_name="NewName")
        new = self._ext().apply_to_profile(profile, result)
        assert new.name == "Existing"

    def test_education_set_when_empty(self) -> None:
        profile = Profile()
        result = ExtractionResult([], [], [], [], [], 0.5, profile_education="IIT Patna")
        new = self._ext().apply_to_profile(profile, result)
        assert new.education == "IIT Patna"

    def test_interests_extended(self) -> None:
        profile = Profile(interests=["NLP"])
        result = ExtractionResult([], [], [], [], [], 0.5,
                                  profile_new_interests=["LLMs", "NLP"])
        new = self._ext().apply_to_profile(profile, result)
        assert "LLMs" in new.interests
        assert new.interests.count("NLP") == 1  # no duplicate

    def test_projects_extended(self) -> None:
        profile = Profile(projects=["Blix"])
        result = ExtractionResult([], [], [], [], [], 0.5,
                                  profile_new_projects=["ECOT", "Blix"])
        new = self._ext().apply_to_profile(profile, result)
        assert "ECOT" in new.projects
        assert new.projects.count("Blix") == 1

    def test_goals_extended(self) -> None:
        profile = Profile()
        result = ExtractionResult([], [], [], [], [], 0.5,
                                  profile_new_goals=["Publish paper"])
        new = self._ext().apply_to_profile(profile, result)
        assert "Publish paper" in new.goals

    def test_no_update_returns_same_profile(self) -> None:
        profile = Profile(name="Sayan", education="IIT")
        new = self._ext().apply_to_profile(profile, ExtractionResult.empty())
        assert new is profile  # identical object — nothing changed
