"""
MemoryExtractor — Chain-of-Thought automatic memory extraction.  v0.2

After every turn, this module asks the LLM to reason step-by-step about
what is worth remembering long-term.  Outputs:

* Factual sentences to store on the ``MemoryEntry``
* Topic labels + competency signals → ``LearningState``
* Personal facts (name, education, projects, goals) → ``Profile``
* Importance score (0.0 – 1.0)

Chain-of-Thought prompt structure
----------------------------------
  ROLE     — memory analyst for Blix
  STEP 1   — personal facts the user revealed
  STEP 2   — topics discussed
  STEP 3   — competency signal per topic
  STEP 4   — importance rating
  STEP 5   — 1-3 concise fact sentences
  OUTPUT   — ONLY a JSON object

Explicit reasoning steps force the model to self-correct before emitting
the final JSON, reducing hallucination on small local models.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from llm.base import LLMProvider
from schemas.memory_entry import MemoryEntry
from schemas.learning_state import LearningState
from schemas.profile import Profile
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# CoT extraction prompt
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """\
You are a memory analyst for an AI tutor called Blix.
Given one conversation exchange, extract structured information.
Think step-by-step before answering.

## Exchange
USER: {user_input}
BLIX: {assistant_output}

## Reasoning (work through EACH step before the JSON)

STEP 1 — PERSONAL FACTS
Did the user reveal anything personal?
- name, education level, school/university, age, location
- job, role, tools they use
- personal projects, goals, ambitions
Write a short note for each item found. Write "none" if nothing personal.

STEP 2 — TOPICS
List every distinct academic / technical topic discussed (e.g. "Transformers",
"PyTorch", "gradient descent"). Write "none" if purely conversational.

STEP 3 — COMPETENCY SIGNAL
For each topic from Step 2, assign ONE label:
  "weak"     — user was confused, made errors, needed basic help
  "learning" — user is engaged, asking questions, progressing
  "strong"   — user showed mastery, corrected something, gave insight
Write "none" if no signal.

STEP 4 — IMPORTANCE (0.0 to 1.0)
How worth remembering is this exchange overall?
  0.0 = trivial small-talk
  0.5 = useful context
  1.0 = critical fact / breakthrough
Give ONE float.

STEP 5 — FACTS TO REMEMBER
Write 1–3 short sentences (≤ 15 words each) capturing the most important
facts from this exchange. Be specific. Write "none" if nothing notable.

## Final JSON output
Emit ONLY the JSON below — no other text before or after it:

{{
  "facts": ["<sentence>", ...],
  "topics": ["<topic>", ...],
  "weak_topics": ["<topic>", ...],
  "learning_topics": ["<topic>", ...],
  "strong_topics": ["<topic>", ...],
  "importance": <float 0.0-1.0>,
  "profile_updates": {{
    "name": "<name or empty string>",
    "education": "<education or empty string>",
    "new_interests": ["<topic>", ...],
    "new_projects": ["<project>", ...],
    "new_goals": ["<goal>", ...]
  }}
}}
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class ExtractionResult:
    """Parsed output from one CoT extraction call."""

    __slots__ = (
        "facts",
        "topics",
        "weak_topics",
        "learning_topics",
        "strong_topics",
        "importance",
        "profile_name",
        "profile_education",
        "profile_new_interests",
        "profile_new_projects",
        "profile_new_goals",
    )

    def __init__(
        self,
        facts: list[str],
        topics: list[str],
        weak_topics: list[str],
        learning_topics: list[str],
        strong_topics: list[str],
        importance: float,
        profile_name: str = "",
        profile_education: str = "",
        profile_new_interests: Optional[list[str]] = None,
        profile_new_projects: Optional[list[str]] = None,
        profile_new_goals: Optional[list[str]] = None,
    ) -> None:
        self.facts = facts
        self.topics = topics
        self.weak_topics = weak_topics
        self.learning_topics = learning_topics
        self.strong_topics = strong_topics
        self.importance = max(0.0, min(1.0, importance))
        self.profile_name = profile_name
        self.profile_education = profile_education
        self.profile_new_interests = profile_new_interests or []
        self.profile_new_projects = profile_new_projects or []
        self.profile_new_goals = profile_new_goals or []

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtractionResult):
            return False
        return (
            self.facts == other.facts
            and self.topics == other.topics
            and self.importance == other.importance
        )

    @classmethod
    def empty(cls) -> "ExtractionResult":
        """No-op result used when extraction fails."""
        return cls([], [], [], [], [], 0.0)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class MemoryExtractor:
    """
    Runs CoT extraction on a conversation turn and returns structured facts.

    Parameters
    ----------
    llm:
        Any ``LLMProvider`` — the same one used for chat is fine; the
        extraction prompt is short so token cost is low.
    enabled:
        Set ``False`` to skip extraction entirely (no LLM call made).
    """

    def __init__(self, llm: LLMProvider, enabled: bool = True) -> None:
        self._llm = llm
        self._enabled = enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, user_input: str, assistant_output: str) -> ExtractionResult:
        """
        Run CoT extraction on one conversation turn.

        Returns ``ExtractionResult.empty()`` on any failure so the caller
        never needs to handle errors.

        Parameters
        ----------
        user_input:
            The user's message for this turn.
        assistant_output:
            Blix's reply for this turn.
        """
        if not self._enabled:
            return ExtractionResult.empty()

        prompt = _EXTRACT_PROMPT.format(
            user_input=user_input[:600],
            assistant_output=assistant_output[:600],
        )
        try:
            raw = self._llm.generate(prompt)
            return self._parse(raw)
        except Exception as exc:
            log.warning("MemoryExtractor.extract failed: %s", exc)
            return ExtractionResult.empty()

    def apply_to_entry(
        self,
        entry: MemoryEntry,
        result: ExtractionResult,
    ) -> MemoryEntry:
        """
        Merge extraction results into *entry* and return the updated copy.

        Uses Pydantic ``model_copy`` so the original is never mutated.
        """
        updates: dict = {
            "extracted_facts": result.facts,
            "topics": result.topics,
        }
        if result.importance > 0.0:
            updates["importance"] = result.importance
        return entry.model_copy(update=updates)

    def apply_to_learning_state(
        self,
        state: LearningState,
        result: ExtractionResult,
    ) -> LearningState:
        """
        Merge topic competency signals from *result* into *state*.

        New topics are added; existing ones are promoted or demoted when
        the new signal contradicts the current classification.
        """
        learned = list(state.topics_learned)
        in_progress = list(state.topics_in_progress)
        weak = list(state.weak_topics)
        strong = list(state.strong_topics)

        def _add_unique(lst: list[str], topic: str) -> None:
            if topic and topic.lower() not in ("none", "") and topic not in lst:
                lst.append(topic)

        def _remove_from_all(topic: str) -> None:
            for lst in (learned, in_progress, weak, strong):
                if topic in lst:
                    lst.remove(topic)

        for t in result.learning_topics:
            _remove_from_all(t)
            _add_unique(in_progress, t)

        for t in result.weak_topics:
            _remove_from_all(t)
            _add_unique(weak, t)

        for t in result.strong_topics:
            _remove_from_all(t)
            _add_unique(strong, t)

        return LearningState(
            topics_learned=learned,
            topics_in_progress=in_progress,
            weak_topics=weak,
            strong_topics=strong,
        )

    def apply_to_profile(
        self,
        profile: Profile,
        result: ExtractionResult,
    ) -> Profile:
        """
        Merge personal-fact signals from *result* into *profile*.

        Only non-empty values overwrite existing ones.  Lists (interests,
        projects, goals) are extended with new unique items.

        Parameters
        ----------
        profile:
            The current user profile.
        result:
            Output from ``extract()``.

        Returns
        -------
        Profile
            Updated profile (new Pydantic instance; original unchanged).
        """
        updates: dict = {}

        if result.profile_name and not profile.name:
            updates["name"] = result.profile_name

        if result.profile_education and not profile.education:
            updates["education"] = result.profile_education

        def _extend_unique(existing: list[str], new: list[str]) -> list[str]:
            combined = list(existing)
            for item in new:
                if item and item not in combined:
                    combined.append(item)
            return combined

        new_interests = _extend_unique(profile.interests, result.profile_new_interests)
        new_projects = _extend_unique(profile.projects, result.profile_new_projects)
        new_goals = _extend_unique(profile.goals, result.profile_new_goals)

        if new_interests != profile.interests:
            updates["interests"] = new_interests
        if new_projects != profile.projects:
            updates["projects"] = new_projects
        if new_goals != profile.goals:
            updates["goals"] = new_goals

        if not updates:
            return profile

        return profile.model_copy(update=updates)

    # ------------------------------------------------------------------
    # Private parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(raw: str) -> ExtractionResult:
        """
        Extract the JSON block from *raw* LLM output and parse it.

        The model may emit CoT reasoning text before the JSON block, so
        we search for the outermost ``{`` … ``}`` span.  Markdown code
        fences (` ```json `) are stripped first.
        """
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        cleaned = cleaned.replace("```", "")

        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            log.warning("MemoryExtractor: no JSON block found in output.")
            return ExtractionResult.empty()

        json_str = cleaned[start:end]
        try:
            data: dict = json.loads(json_str)
        except json.JSONDecodeError as exc:
            log.warning("MemoryExtractor: JSON parse error: %s", exc)
            return ExtractionResult.empty()

        def _clean_list(key: str) -> list[str]:
            raw_val = data.get(key, [])
            if not isinstance(raw_val, list):
                return []
            return [
                str(v).strip()
                for v in raw_val
                if str(v).strip().lower() not in ("none", "")
            ]

        def _clean_str(mapping: dict, key: str) -> str:
            val = mapping.get(key, "")
            s = str(val).strip() if val else ""
            return "" if s.lower() in ("none", "n/a", "") else s

        try:
            importance = float(data.get("importance", 0.0))
        except (TypeError, ValueError):
            importance = 0.0

        pu: dict = data.get("profile_updates", {}) or {}

        def _clean_list_from(mapping: dict, key: str) -> list[str]:
            """Like _clean_list but reads from an arbitrary dict (not top-level data)."""
            raw_val = mapping.get(key, [])
            if not isinstance(raw_val, list):
                return []
            return [
                str(v).strip()
                for v in raw_val
                if str(v).strip().lower() not in ("none", "")
            ]

        return ExtractionResult(
            facts=_clean_list("facts"),
            topics=_clean_list("topics"),
            weak_topics=_clean_list("weak_topics"),
            learning_topics=_clean_list("learning_topics"),
            strong_topics=_clean_list("strong_topics"),
            importance=importance,
            profile_name=_clean_str(pu, "name"),
            profile_education=_clean_str(pu, "education"),
            profile_new_interests=_clean_list_from(pu, "new_interests"),
            profile_new_projects=_clean_list_from(pu, "new_projects"),
            profile_new_goals=_clean_list_from(pu, "new_goals"),
        )
