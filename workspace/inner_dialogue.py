"""
Inner Dialogue — Blix v0.3.9  (New module 8)

A structured, multi-voice commentary pass over whatever currently
holds the workspace's attention — modeled after the spec's example:

    Planner:     Need strategy.
    Critic:      Plan confidence low.
    Self Model:  Math capability high.
    Reflection:  Previous failures indicate X.

Inspired by Chain-of-Thought and Minsky's Society of Mind: rather than
a single reasoning stream, several "voices" (one per cognitive module)
each contribute a short, targeted remark about the current workspace
focus, and ``InnerDialogue`` collects them into an ordered transcript
that downstream reasoning (or a human reviewing Blix's process) can
read as a coherent multi-perspective deliberation.

This module does NOT implement new reasoning — each voice is a thin
adapter that asks an already-existing module (StrategyManager,
PlanCritic, SelfModel, FailureMemory, ...) for its one-line take on
the current situation. ``InnerDialogue`` is purely the orchestration
and transcript-assembly layer.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional


@dataclass
class DialogueTurn:
    """One voice's contribution to the inner dialogue."""

    voice: str          # e.g. "Planner", "Critic", "Self Model", "Reflection"
    remark: str
    spoken_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {"voice": self.voice, "remark": self.remark, "spoken_at": self.spoken_at}


@dataclass
class DialogueTranscript:
    """A full inner-dialogue pass over one topic/situation."""

    topic: str
    turns: list[DialogueTurn] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"topic": self.topic, "turns": [t.to_dict() for t in self.turns]}

    def as_text(self) -> str:
        """Render the transcript as readable multi-line text, matching the spec's example format."""
        lines = [f"{t.voice}:\n{t.remark}" for t in self.turns]
        return "\n\n".join(lines)


# A "voice" is just a callable that takes a topic string and returns a
# short remark, or None if it has nothing to contribute this round.
VoiceFn = Callable[[str], Optional[str]]


class InnerDialogue:
    """
    Orchestrates a multi-voice commentary pass over a topic, where each
    voice is a thin adapter over an existing cognitive module.

    Parameters
    ----------
    voices:
        Ordered dict of voice_name -> callable(topic) -> Optional[str].
        Voices are consulted in registration order, matching the
        spec's example ordering (Planner, Critic, Self Model, Reflection).
    """

    def __init__(self, voices: Optional[dict[str, VoiceFn]] = None) -> None:
        self._voices: dict[str, VoiceFn] = voices or {}

    # ------------------------------------------------------------------
    # Voice registration
    # ------------------------------------------------------------------

    def register_voice(self, name: str, fn: VoiceFn) -> None:
        self._voices[name] = fn

    def registered_voices(self) -> list[str]:
        return list(self._voices.keys())

    # ------------------------------------------------------------------
    # Running a dialogue pass
    # ------------------------------------------------------------------

    def run(self, topic: str) -> DialogueTranscript:
        """
        Consult every registered voice about ``topic`` in order,
        skipping any that return ``None`` (nothing to contribute).
        """
        turns: list[DialogueTurn] = []
        for name, fn in self._voices.items():
            try:
                remark = fn(topic)
            except Exception:
                remark = None
            if remark:
                turns.append(DialogueTurn(voice=name, remark=remark))
        return DialogueTranscript(topic=topic, turns=turns)


# ---------------------------------------------------------------------------
# Standard voice adapters — thin wrappers over existing v0.3.x modules,
# matching the spec's literal example voices. Callers wire these in via
# InnerDialogue.register_voice(); none of this duplicates module logic.
# ---------------------------------------------------------------------------


def planner_voice(strategy_manager, ref_key: str) -> VoiceFn:
    """'Planner: Need strategy.' — reports whether a strategy switch is warranted."""

    def _voice(topic: str) -> Optional[str]:
        if strategy_manager.is_repeated_failure(ref_key):
            return "Need strategy. Repeated failures suggest the current approach should change."
        return "Proceeding with current plan; no strategic concerns flagged."

    return _voice


def critic_voice(plan_quality_score) -> VoiceFn:
    """'Critic: Plan confidence low.' — reports on plan confidence/risk if a score is available."""

    def _voice(topic: str) -> Optional[str]:
        if plan_quality_score is None:
            return None
        if plan_quality_score.is_low_confidence:
            return f"Plan confidence low ({plan_quality_score.confidence:.2f})."
        if plan_quality_score.is_high_risk:
            return f"Plan risk is elevated ({plan_quality_score.risk:.2f})."
        return f"Plan looks sound (confidence {plan_quality_score.confidence:.2f})."

    return _voice


def self_model_voice(self_model_store, domain: str) -> VoiceFn:
    """'Self Model: Math capability high.' — reports current capability belief for a domain."""

    def _voice(topic: str) -> Optional[str]:
        score = self_model_store.capability(domain)
        level = "high" if score >= 0.85 else "low" if score < 0.6 else "moderate"
        return f"{domain.capitalize()} capability {level} ({score:.2f})."

    return _voice


def reflection_voice(failure_memory) -> VoiceFn:
    """'Reflection: Previous failures indicate X.' — surfaces relevant past-failure context."""

    def _voice(topic: str) -> Optional[str]:
        if failure_memory is None or failure_memory.count == 0:
            return None
        common = failure_memory.most_common_failures(top_k=1)
        if not common:
            return None
        return f"Previous failures indicate caution around: {common[0].failure}"

    return _voice
