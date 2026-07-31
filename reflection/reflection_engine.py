"""
Reflection Engine — Blix v0.3.2  (Feature 1)

Generates insights from accumulated memories at multiple scopes:

    Memory → Reflection → Insights → Knowledge

Reflection scopes
------------------
* ``session``  — what happened in the most recent session
* ``daily``    — what happened today
* ``weekly``   — what happened this week
* ``project``  — progress/risk reflection for a named project
* ``behavior`` — patterns in how the user works/learns over time
* ``learning`` — what topics/skills are growing or stagnating

Each reflection produces one or more ``Insight`` objects:

    {
      "insight": "User's primary focus has shifted from chatbot
                   development to cognitive memory systems.",
      "confidence": 0.91
    }

Insights are persisted to ``memory/reflections.json`` and feed into
``KnowledgeSynthesisEngine`` (knowledge/synthesis.py) and the Memory
Query Language (``show reflections this week``).

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from llm.base import LLMProvider
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ReflectionScope(str, Enum):
    SESSION = "session"
    DAILY = "daily"
    WEEKLY = "weekly"
    PROJECT = "project"
    BEHAVIOR = "behavior"
    LEARNING = "learning"


@dataclass
class Insight:
    """A single piece of reflective knowledge derived from memories."""

    insight: str
    confidence: float = 0.5
    scope: ReflectionScope = ReflectionScope.SESSION
    scope_ref: str = ""               # e.g. session id, date, project name
    source_memory_ids: list[int] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "insight": self.insight,
            "confidence": self.confidence,
            "scope": self.scope.value,
            "scope_ref": self.scope_ref,
            "source_memory_ids": self.source_memory_ids,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Insight":
        return cls(
            insight=d["insight"],
            confidence=d.get("confidence", 0.5),
            scope=ReflectionScope(d.get("scope", "session")),
            scope_ref=d.get("scope_ref", ""),
            source_memory_ids=d.get("source_memory_ids", []),
            created_at=d.get("created_at", ""),
        )


@dataclass
class ReflectionRecord:
    """One reflection run — a batch of Insights with metadata."""

    scope: ReflectionScope
    scope_ref: str
    insights: list[Insight] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "scope": self.scope.value,
            "scope_ref": self.scope_ref,
            "insights": [i.to_dict() for i in self.insights],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReflectionRecord":
        return cls(
            scope=ReflectionScope(d["scope"]),
            scope_ref=d.get("scope_ref", ""),
            insights=[Insight.from_dict(i) for i in d.get("insights", [])],
            created_at=d.get("created_at", ""),
        )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_REFLECTION_PROMPT = """\
You are Blix's reflection module. Analyse the material below and produce
1-4 high-level INSIGHTS about the user — patterns, shifts in focus,
emerging interests, behavioral tendencies, or learning progress.

Each insight must be a single factual sentence about the USER (not about
the material itself), written in third person, e.g.
"User's primary focus has shifted from chatbot development to cognitive
memory systems."

Respond with ONLY a JSON array of objects, each with keys "insight" and
"confidence" (0.0-1.0):

[
  {{"insight": "...", "confidence": 0.91}}
]

Material ({scope}):
{material}
"""


# ---------------------------------------------------------------------------
# Reflection Engine
# ---------------------------------------------------------------------------


class ReflectionEngine:
    """
    Generates and persists ``Insight`` objects at multiple scopes.

    Parameters
    ----------
    reflections_file:
        Path to ``reflections.json``.
    llm:
        LLM provider for generating insights. If ``None``, falls back to
        a heuristic reflector (keyword-frequency based) so the engine
        stays usable offline / in tests.
    """

    def __init__(
        self,
        reflections_file: Path,
        llm: Optional[LLMProvider] = None,
    ) -> None:
        self._file = reflections_file
        self._llm = llm
        self._records: list[ReflectionRecord] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._records = [ReflectionRecord.from_dict(r) for r in raw]
            log.info("ReflectionEngine: loaded %d reflection records.", len(self._records))
        except Exception as exc:
            log.warning("ReflectionEngine: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in self._records], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Core reflection
    # ------------------------------------------------------------------

    def reflect(
        self,
        scope: ReflectionScope,
        scope_ref: str,
        material: str,
        source_memory_ids: Optional[list[int]] = None,
    ) -> ReflectionRecord:
        """
        Run a reflection pass over ``material`` and persist the result.

        Parameters
        ----------
        scope:
            Which reflection scope this run belongs to.
        scope_ref:
            Identifier for the scope instance (e.g. "session-12",
            "2025-07-15", "2025-W29", "Blix").
        material:
            Concatenated summaries / memory text to reflect on.
        source_memory_ids:
            Memory ids that contributed to this material (for traceability).
        """
        insights = self._generate_insights(scope, material)
        for ins in insights:
            ins.scope = scope
            ins.scope_ref = scope_ref
            ins.source_memory_ids = source_memory_ids or []

        record = ReflectionRecord(scope=scope, scope_ref=scope_ref, insights=insights)
        self._records.append(record)
        self._save()
        log.info(
            "ReflectionEngine: %s reflection for %r produced %d insight(s).",
            scope.value, scope_ref, len(insights),
        )
        return record

    def reflect_on_curiosity(
        self,
        curiosity_target: str,
        hypothesis_statement: str,
        experiment_outcome: Optional[str],
        learned: bool,
    ) -> "ReflectionRecord":
        """
        v0.3.13 — Reflect on a completed curiosity→experiment→outcome cycle:
        'Why was I curious about X? Did I learn from it?'

        Parameters
        ----------
        curiosity_target:
            The ``CuriositySignal.target`` that started the cycle.
        hypothesis_statement:
            The hypothesis that was tested.
        experiment_outcome:
            What the experiment actually found (None if no experiment ran).
        learned:
            Whether the hypothesis was SUPPORTED or useful knowledge was gained.
        """
        if learned:
            material = (
                f"Curiosity target: {curiosity_target}. "
                f"Hypothesis tested: {hypothesis_statement}. "
                f"Outcome: {experiment_outcome or 'no experiment recorded'}. "
                f"Result: knowledge gained — hypothesis supported."
            )
        else:
            material = (
                f"Curiosity target: {curiosity_target}. "
                f"Hypothesis tested: {hypothesis_statement}. "
                f"Outcome: {experiment_outcome or 'no experiment recorded'}. "
                f"Result: hypothesis not supported — this line of inquiry may need revision."
            )
        return self.reflect(ReflectionScope.LEARNING, curiosity_target[:60], material)

    def _generate_insights(self, scope: ReflectionScope, material: str) -> list[Insight]:
        if not material.strip():
            return []
        if self._llm is not None:
            return self._llm_insights(scope, material)
        return self._heuristic_insights(scope, material)

    def _llm_insights(self, scope: ReflectionScope, material: str) -> list[Insight]:
        prompt = _REFLECTION_PROMPT.format(scope=scope.value, material=material[:4000])
        try:
            raw = self._llm.generate(prompt).strip()  # type: ignore[union-attr]
            raw = _strip_code_fence(raw)
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("expected JSON array")
            results: list[Insight] = []
            for item in data:
                text = str(item.get("insight", "")).strip()
                if not text:
                    continue
                conf = float(item.get("confidence", 0.5))
                conf = max(0.0, min(1.0, conf))
                results.append(Insight(insight=text, confidence=conf))
            return results
        except Exception as exc:
            log.warning("ReflectionEngine: LLM insight generation failed (%s); using heuristic.", exc)
            return self._heuristic_insights(scope, material)

    def _heuristic_insights(self, scope: ReflectionScope, material: str) -> list[Insight]:
        """
        Offline fallback: extract the most frequent meaningful tokens
        and phrase a generic insight sentence.
        """
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", material.lower())
        stop = {
            "this", "that", "with", "from", "have", "been", "were", "they",
            "their", "about", "which", "would", "could", "should", "there",
            "session", "memory", "memories", "user", "blix",
        }
        freq: dict[str, int] = {}
        for w in words:
            if w not in stop:
                freq[w] = freq.get(w, 0) + 1
        if not freq:
            return []
        top = sorted(freq.items(), key=lambda kv: -kv[1])[:3]
        topics = ", ".join(t for t, _ in top)
        scope_label = {
            ReflectionScope.SESSION: "In this session",
            ReflectionScope.DAILY: "Today",
            ReflectionScope.WEEKLY: "This week",
            ReflectionScope.PROJECT: "On this project",
            ReflectionScope.BEHAVIOR: "Recently",
            ReflectionScope.LEARNING: "In recent learning",
        }.get(scope, "Recently")
        text = f"{scope_label}, the user's activity centred on: {topics}."
        confidence = min(0.6, 0.3 + 0.05 * top[0][1])
        return [Insight(insight=text, confidence=round(confidence, 2))]

    # ------------------------------------------------------------------
    # Convenience scope builders
    # ------------------------------------------------------------------

    def reflect_session(self, session_summary: object) -> ReflectionRecord:
        """Reflect on a single ``SessionSummary`` (from HierarchyManager)."""
        material = getattr(session_summary, "summary", str(session_summary))
        ref_id = getattr(session_summary, "id", "session")
        source_ids = getattr(session_summary, "raw_memory_ids", [])
        return self.reflect(ReflectionScope.SESSION, ref_id, material, source_ids)

    def reflect_daily(self, daily_summary: object) -> ReflectionRecord:
        material = getattr(daily_summary, "summary", str(daily_summary))
        ref_id = getattr(daily_summary, "date", "daily")
        return self.reflect(ReflectionScope.DAILY, ref_id, material)

    def reflect_weekly(self, weekly_summary: object) -> ReflectionRecord:
        material = getattr(weekly_summary, "summary", str(weekly_summary))
        ref_id = getattr(weekly_summary, "week_label", "weekly")
        return self.reflect(ReflectionScope.WEEKLY, ref_id, material)

    def reflect_project(self, project_summary: object) -> ReflectionRecord:
        """
        Reflect on a ``ProjectSummary`` — focuses on progress, goals,
        blockers, and trajectory.
        """
        p = project_summary
        material = (
            f"Project: {getattr(p, 'project_name', '?')}\n"
            f"Status: {getattr(p, 'current_status', '?')}\n"
            f"Goals: {', '.join(getattr(p, 'goals', []))}\n"
            f"Completed: {', '.join(getattr(p, 'completed_work', []))}\n"
            f"Next actions: {', '.join(getattr(p, 'next_actions', []))}\n"
            f"Milestones: {', '.join(getattr(p, 'milestones', []))}"
        )
        ref_id = getattr(p, "project_name", "project")
        return self.reflect(ReflectionScope.PROJECT, ref_id, material)

    def reflect_learning(self, learning_state: object) -> ReflectionRecord:
        """
        Reflect on a ``LearningState`` — which topics are growing,
        which are stagnating.
        """
        topics = getattr(learning_state, "topics", {})
        if not topics:
            return self.reflect(ReflectionScope.LEARNING, "learning", "")
        lines = []
        for topic, data in list(topics.items())[:20]:
            count = data.get("count", 0) if isinstance(data, dict) else getattr(data, "count", 0)
            lines.append(f"{topic}: discussed {count} time(s)")
        material = "\n".join(lines)
        return self.reflect(ReflectionScope.LEARNING, "learning", material)

    def reflect_behavior(self, memories: list) -> ReflectionRecord:
        """
        Reflect on user behavior patterns across a list of MemoryEntry objects
        (e.g. session timing, question style, recurring requests).
        """
        if not memories:
            return self.reflect(ReflectionScope.BEHAVIOR, "behavior", "")
        hours = [m.timestamp.hour for m in memories if getattr(m, "timestamp", None)]
        avg_hour = sum(hours) / len(hours) if hours else 12
        all_text = " ".join(
            (getattr(m, "input", "") + " " + getattr(m, "output", "")) for m in memories[-50:]
        )
        material = f"Average activity hour: {avg_hour:.0f}:00\n\n{all_text[:3000]}"
        ids = [m.id for m in memories[-50:]]
        return self.reflect(ReflectionScope.BEHAVIOR, "behavior", material, ids)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_records(
        self,
        scope: Optional[ReflectionScope] = None,
        scope_ref: Optional[str] = None,
    ) -> list[ReflectionRecord]:
        records = self._records
        if scope is not None:
            records = [r for r in records if r.scope == scope]
        if scope_ref is not None:
            records = [r for r in records if r.scope_ref == scope_ref]
        return records

    def get_recent_insights(
        self,
        scope: Optional[ReflectionScope] = None,
        limit: int = 10,
    ) -> list[Insight]:
        """Return the most recent insights, optionally filtered by scope."""
        all_insights: list[Insight] = []
        for r in self._records:
            if scope is not None and r.scope != scope:
                continue
            all_insights.extend(r.insights)
        all_insights.sort(key=lambda i: i.created_at, reverse=True)
        return all_insights[:limit]

    def get_insights_since(self, since: datetime) -> list[Insight]:
        """Return all insights created at or after ``since``."""
        result = []
        for r in self._records:
            for i in r.insights:
                try:
                    created = datetime.fromisoformat(i.created_at)
                except ValueError:
                    continue
                if created.replace(tzinfo=None) >= since.replace(tzinfo=None):
                    result.append(i)
        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def insight_count(self) -> int:
        return sum(len(r.insights) for r in self._records)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_code_fence(text: str) -> str:
    """Strip ```json ... ``` fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
