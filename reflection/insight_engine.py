"""
Insight Generation Engine — Blix v0.3.3  (Feature 3)

Upgrades reflection from "summary" to "insight":

    v0.3.2 reflection = Insight{insight: str, confidence: float}
    v0.3.3 insight     = ActionableInsight{
        insight, confidence, category, evidence, recommendation
    }

Categories
----------
* ``trend``               — recurring patterns across time
* ``bottleneck``           — recurring blockers/friction points
* ``research_interest``    — emerging/dominant topics of interest
* ``project_pattern``      — patterns in how projects progress

Example
-------
    {
      "insight": "Most conversations involve AI systems.",
      "category": "research_interest",
      "confidence": 0.84,
      "evidence": ["42 of 60 recent memories tagged with AI-related topics"],
      "recommendation": "Create a dedicated research knowledge base for AI systems."
    }

Design
------
``InsightGenerationEngine`` analyses aggregated data from across the
v0.3-v0.3.2 stack (memories, topics, goals, projects, canonical facts)
using heuristic statistical analysis, with an optional LLM pass for
natural-language recommendation phrasing.

This is additive to ``ReflectionEngine`` — ``ActionableInsight`` can be
converted to/from ``Insight`` for storage compatibility.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from llm.base import LLMProvider
from reflection.reflection_engine import Insight, ReflectionScope
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class InsightCategory(str, Enum):
    TREND = "trend"
    BOTTLENECK = "bottleneck"
    RESEARCH_INTEREST = "research_interest"
    PROJECT_PATTERN = "project_pattern"


@dataclass
class ActionableInsight:
    """An insight with supporting evidence and a concrete recommendation."""

    insight: str
    category: InsightCategory
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    recommendation: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "insight": self.insight,
            "category": self.category.value,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActionableInsight":
        return cls(
            insight=d["insight"],
            category=InsightCategory(d.get("category", "trend")),
            confidence=d.get("confidence", 0.5),
            evidence=d.get("evidence", []),
            recommendation=d.get("recommendation", ""),
            created_at=d.get("created_at", ""),
        )

    def to_insight(self, scope: ReflectionScope = ReflectionScope.BEHAVIOR, scope_ref: str = "") -> Insight:
        """Convert to a v0.3.2 ``Insight`` for storage in ``ReflectionEngine``."""
        text = self.insight
        if self.recommendation:
            text = f"{text} Recommendation: {self.recommendation}"
        return Insight(insight=text, confidence=self.confidence, scope=scope, scope_ref=scope_ref)


# ---------------------------------------------------------------------------
# Prompt for recommendation phrasing
# ---------------------------------------------------------------------------

_RECOMMENDATION_PROMPT = """\
You are Blix's insight engine. Given the following observation about the
user, write ONE concise, actionable recommendation sentence (imperative
mood, e.g. "Create a dedicated research knowledge base for X.").

Observation: {observation}
Evidence: {evidence}

Respond with ONLY the recommendation sentence, no preamble.
"""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class InsightGenerationEngine:
    """
    Generates ``ActionableInsight`` objects from aggregated v0.3-v0.3.2 state.

    Parameters
    ----------
    insights_file:
        Path to ``actionable_insights.json``.
    llm:
        Optional LLM for recommendation phrasing. Falls back to templated
        recommendations if ``None``.
    """

    def __init__(self, insights_file: Path, llm: Optional[LLMProvider] = None) -> None:
        self._file = insights_file
        self._llm = llm
        self._insights: list[ActionableInsight] = []
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
            self._insights = [ActionableInsight.from_dict(i) for i in raw]
            log.info("InsightGenerationEngine: loaded %d insight(s).", len(self._insights))
        except Exception as exc:
            log.warning("InsightGenerationEngine: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([i.to_dict() for i in self._insights], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    def analyze_topic_trends(
        self,
        memories: list,
        min_fraction: float = 0.3,
        recent_window: int = 60,
    ) -> list[ActionableInsight]:
        """
        Detect dominant topics across recent memories.

        If a single topic (or topic group) appears in ``>= min_fraction``
        of the recent window, emit a ``research_interest`` insight.
        """
        recent = memories[-recent_window:] if len(memories) > recent_window else memories
        if not recent:
            return []

        topic_counter: Counter[str] = Counter()
        for m in recent:
            for t in getattr(m, "topics", []):
                topic_counter[t.lower()] += 1

        if not topic_counter:
            return []

        results: list[ActionableInsight] = []
        total = len(recent)
        top_topic, top_count = topic_counter.most_common(1)[0]
        fraction = top_count / total

        if fraction >= min_fraction:
            observation = f"Most conversations involve {top_topic}."
            evidence = [f"{top_count} of {total} recent memories tagged with '{top_topic}'-related topics"]
            confidence = round(min(0.95, 0.5 + fraction), 2)
            recommendation = self._recommend(observation, evidence) or (
                f"Create a dedicated knowledge base for {top_topic}."
            )
            results.append(ActionableInsight(
                insight=observation,
                category=InsightCategory.RESEARCH_INTEREST,
                confidence=confidence,
                evidence=evidence,
                recommendation=recommendation,
            ))

        return results

    # ------------------------------------------------------------------
    # Bottleneck analysis
    # ------------------------------------------------------------------

    def analyze_bottlenecks(self, goals: list) -> list[ActionableInsight]:
        """
        Detect recurring blockers across active goals.

        If the same blocker description (or near-duplicate) appears
        across multiple goals, emit a ``bottleneck`` insight.
        """
        from reflection.goal_tracker import GoalStatus

        blocker_counter: Counter[str] = Counter()
        blocker_goals: dict[str, list[str]] = {}
        for g in goals:
            if g.status != GoalStatus.ACTIVE:
                continue
            for b in g.active_blockers:
                key = b.description.lower().strip()
                blocker_counter[key] += 1
                blocker_goals.setdefault(key, []).append(g.title)

        results: list[ActionableInsight] = []
        for blocker, count in blocker_counter.items():
            if count >= 1:
                observation = f"'{blocker}' is blocking progress on {count} goal(s)."
                evidence = [f"Affects: {', '.join(blocker_goals[blocker])}"]
                confidence = round(min(0.95, 0.4 + 0.15 * count), 2)
                recommendation = self._recommend(observation, evidence) or (
                    f"Prioritise resolving '{blocker}' to unblock dependent goals."
                )
                results.append(ActionableInsight(
                    insight=observation,
                    category=InsightCategory.BOTTLENECK,
                    confidence=confidence,
                    evidence=evidence,
                    recommendation=recommendation,
                ))
        return results

    # ------------------------------------------------------------------
    # Project progress patterns
    # ------------------------------------------------------------------

    def analyze_project_patterns(self, project_states: list) -> list[ActionableInsight]:
        """
        Detect patterns in project progress: stalled projects (risk_level
        escalated with no progress change recorded) or consistently
        high-risk projects.
        """
        from reflection.project_intelligence import RiskLevel

        results: list[ActionableInsight] = []
        for ps in project_states:
            if ps.risk_level == RiskLevel.HIGH:
                observation = f"Project '{ps.project_name}' has accumulated multiple risks."
                evidence = [f"{len(ps.risks)} open risk(s): {', '.join(ps.risks)}"]
                confidence = 0.8
                recommendation = self._recommend(observation, evidence) or (
                    f"Schedule a risk-review session for '{ps.project_name}' "
                    "before continuing new work."
                )
                results.append(ActionableInsight(
                    insight=observation,
                    category=InsightCategory.PROJECT_PATTERN,
                    confidence=confidence,
                    evidence=evidence,
                    recommendation=recommendation,
                ))
            elif ps.progress == 0 and ps.focus:
                observation = f"Project '{ps.project_name}' has a defined focus but no recorded progress."
                evidence = [f"Focus: {ps.focus}; progress: 0%"]
                recommendation = self._recommend(observation, evidence) or (
                    f"Define a first milestone for '{ps.project_name}' to start tracking progress."
                )
                results.append(ActionableInsight(
                    insight=observation,
                    category=InsightCategory.PROJECT_PATTERN,
                    confidence=0.6,
                    evidence=evidence,
                    recommendation=recommendation,
                ))
        return results

    # ------------------------------------------------------------------
    # Temporal trend analysis (activity over time)
    # ------------------------------------------------------------------

    def analyze_activity_trend(self, memories: list, window: int = 20) -> list[ActionableInsight]:
        """
        Compare the topic distribution of the most recent ``window``
        memories against the previous ``window`` to detect a shift
        in focus (e.g. "shifted from chatbot development to memory systems").
        """
        if len(memories) < window * 2:
            return []

        older = memories[-2 * window:-window]
        recent = memories[-window:]

        older_topics = Counter(t.lower() for m in older for t in getattr(m, "topics", []))
        recent_topics = Counter(t.lower() for m in recent for t in getattr(m, "topics", []))

        if not older_topics or not recent_topics:
            return []

        old_top = older_topics.most_common(1)[0][0]
        new_top = recent_topics.most_common(1)[0][0]

        if old_top == new_top:
            return []

        observation = f"User's primary focus has shifted from {old_top} to {new_top}."
        evidence = [
            f"Previous window dominant topic: '{old_top}' ({older_topics[old_top]} mentions)",
            f"Recent window dominant topic: '{new_top}' ({recent_topics[new_top]} mentions)",
        ]
        confidence = 0.7
        recommendation = self._recommend(observation, evidence) or (
            f"Review whether prior work on '{old_top}' should be archived or "
            f"connected to the new focus on '{new_top}'."
        )
        return [ActionableInsight(
            insight=observation,
            category=InsightCategory.TREND,
            confidence=confidence,
            evidence=evidence,
            recommendation=recommendation,
        )]

    # ------------------------------------------------------------------
    # Full pass
    # ------------------------------------------------------------------

    def generate_all(
        self,
        memories: list,
        goals: Optional[list] = None,
        project_states: Optional[list] = None,
    ) -> list[ActionableInsight]:
        """
        Run all analyses and persist newly-generated insights.

        Returns the list of newly-generated ``ActionableInsight`` objects
        (does not return previously-persisted ones).
        """
        new: list[ActionableInsight] = []
        new.extend(self.analyze_topic_trends(memories))
        new.extend(self.analyze_activity_trend(memories))
        if goals is not None:
            new.extend(self.analyze_bottlenecks(goals))
        if project_states is not None:
            new.extend(self.analyze_project_patterns(project_states))

        self._insights.extend(new)
        if new:
            self._save()
            log.info("InsightGenerationEngine: generated %d new insight(s).", len(new))
        return new

    # ------------------------------------------------------------------
    # Recommendation phrasing
    # ------------------------------------------------------------------

    def _recommend(self, observation: str, evidence: list[str]) -> str:
        if self._llm is None:
            return ""
        prompt = _RECOMMENDATION_PROMPT.format(
            observation=observation, evidence="; ".join(evidence)
        )
        try:
            text = self._llm.generate(prompt).strip()
            text = text.strip('"')
            return text[:300]
        except Exception as exc:
            log.warning("InsightGenerationEngine: LLM recommendation failed (%s)", exc)
            return ""

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def list_insights(self, category: Optional[InsightCategory] = None) -> list[ActionableInsight]:
        if category is None:
            return list(self._insights)
        return [i for i in self._insights if i.category == category]

    def latest(self, limit: int = 10) -> list[ActionableInsight]:
        return sorted(self._insights, key=lambda i: i.created_at, reverse=True)[:limit]

    @property
    def count(self) -> int:
        return len(self._insights)
