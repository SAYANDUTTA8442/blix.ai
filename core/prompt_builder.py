"""
PromptBuilder — assembles the full prompt sent to the LLM each turn.  v0.2

Prompt structure
----------------
  SYSTEM BLOCK         — Blix persona + behaviour instructions
  USER PROFILE         — name, education, interests, projects, goals
  LEARNING STATE       — topics learned / in-progress / weak / strong
  RELEVANT MEMORIES    — retrieved past interactions with extracted facts
  CURRENT QUESTION     — the user's literal input for this turn

v0.2 additions
--------------
* Memory entries now render their ``extracted_facts`` so the LLM sees
  compact factual summaries alongside the raw exchange previews.

Python 3.10 compatible.
"""

from __future__ import annotations

from schemas.memory_entry import MemoryEntry
from schemas.profile import Profile
from schemas.learning_state import LearningState
from utils.helpers import truncate


_SYSTEM_PROMPT = """\
You are Blix, an AI Tutor and long-term mentor with persistent memory.

Core behaviours:
- You remember everything the user has told you across all past sessions
  and always personalise explanations to their background, goals, and
  current projects.
- You calibrate depth to the user's learning state: patient and scaffolded
  for weak topics, peer-level for strong ones.
- You are honest when uncertain and never fabricate information.
- You ask one focused follow-up question when clarification would
  meaningfully improve your answer.
- Keep responses focused and complete: depth over breadth, no padding.
- When referencing past conversations, be specific."""


class PromptBuilder:
    """
    Assembles the complete prompt string from structured context objects.

    Stateless — instantiate once and call ``build()`` on every turn.
    Each ``_render_*`` helper owns one prompt section.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        user_input: str,
        profile: Profile,
        learning_state: LearningState,
        relevant_memories: list[MemoryEntry],
    ) -> str:
        """
        Return the fully assembled, section-separated prompt string.

        Parameters
        ----------
        user_input:
            Raw text from the user for the current turn.
        profile:
            Persistent user profile.
        learning_state:
            Current learning-state snapshot.
        relevant_memories:
            Pre-retrieved entries from ``SemanticRetriever.retrieve()``.
        """
        sections = [
            _SYSTEM_PROMPT,
            self._render_profile(profile),
            self._render_learning_state(learning_state),
            self._render_memories(relevant_memories),
            self._render_question(user_input),
        ]
        return "\n\n".join(s for s in sections if s.strip())

    # ------------------------------------------------------------------
    # Private renderers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_profile(profile: Profile) -> str:
        if profile.is_empty():
            return ""
        lines: list[str] = ["## User Profile"]
        if profile.name:
            lines.append(f"Name: {profile.name}")
        if profile.education:
            lines.append(f"Education: {profile.education}")
        if profile.interests:
            lines.append("Interests: " + ", ".join(profile.interests))
        if profile.projects:
            lines.append("Active projects: " + ", ".join(profile.projects))
        if profile.goals:
            lines.append("Goals: " + ", ".join(profile.goals))
        if profile.notes:
            lines.append("Notes: " + " | ".join(profile.notes))
        return "\n".join(lines)

    @staticmethod
    def _render_learning_state(state: LearningState) -> str:
        parts: list[str] = []
        if state.topics_learned:
            parts.append("Already learned: " + ", ".join(state.topics_learned))
        if state.topics_in_progress:
            parts.append("Currently studying: " + ", ".join(state.topics_in_progress))
        if state.weak_topics:
            parts.append("Needs more work: " + ", ".join(state.weak_topics))
        if state.strong_topics:
            parts.append("Strong areas: " + ", ".join(state.strong_topics))
        if not parts:
            return ""
        return "## Learning State\n" + "\n".join(parts)

    @staticmethod
    def _render_memories(memories: list[MemoryEntry]) -> str:
        if not memories:
            return ""
        lines: list[str] = [
            f"## Relevant Past Interactions ({len(memories)} retrieved)"
        ]
        for m in memories:
            date_str = m.timestamp.strftime("%Y-%m-%d")
            imp = f"  importance={m.importance:.2f}" if m.importance is not None else ""
            lines.append(f"\n[{date_str} · id={m.id}{imp}]")
            lines.append(f"  You asked: {truncate(m.input, 120)}")
            lines.append(f"  Blix said: {truncate(m.output, 200)}")
            # v0.2: include extracted facts for richer context
            if m.extracted_facts:
                lines.append("  Key facts: " + " | ".join(m.extracted_facts[:3]))
            if m.topics:
                lines.append("  Topics: " + ", ".join(m.topics[:5]))
        return "\n".join(lines)

    @staticmethod
    def _render_question(user_input: str) -> str:
        return f"## Current Question\n{user_input}"
