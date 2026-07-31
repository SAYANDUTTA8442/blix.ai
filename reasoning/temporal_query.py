"""
Temporal Query Engine — Blix v0.3.7  (New module 8)

Fixes the third bug the spec calls out: v0.3.4's ``CognitiveQueryEngine``
only answers "what is true now" — it has no concept of history. This
module adds the missing query types:

    "What was my favorite language in 2024?"     → historical query
    "How has my research evolved?"                 → evolution query
    "When did Blix adopt FastAPI?"                   → transition query
    "What changed during the last month?"             → recency/diff query
    "What is my favorite language?"                     → current query (delegates)

``TemporalQueryEngine`` sits alongside (not on top of)
``core.cognitive_query_engine.CognitiveQueryEngine`` — current-state
graph queries still go through the v0.3.4 engine; this module owns
everything that requires ``StateTracker``/``StateTransitionEngine``/
``TemporalGraph`` history.

Python 3.10 compatible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from core.state_tracker import StateTracker
from core.state_transition import StateTransitionEngine
from graph.temporal_graph import TemporalGraph
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class TemporalQueryResult:
    """Result of one temporal query."""

    query: str
    query_type: str
    answer: str
    timeline: list = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "answer": self.answer,
            "timeline": self.timeline,
            "explanation": self.explanation,
        }

    def is_empty(self) -> bool:
        return not self.answer


# ---------------------------------------------------------------------------
# Query patterns
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"what was (?:(.+?)'s\s+)?(?:my\s+)?(.+?)\s+in\s+(\d{4})\??$", re.I
    ), "historical_year"),
    (re.compile(
        r"how has (?:(.+?)'s\s+)?(?:my\s+)?(.+?)\s+evolved\??$", re.I
    ), "evolution"),
    (re.compile(
        r"when did\s+(.+?)\s+(?:adopt|start using|switch to|change to)\s+(.+?)\??$", re.I
    ), "transition"),
    (re.compile(
        r"what changed\s+(?:in|during)\s+the\s+last\s+(\d+)\s*(day|week|month|year)s?\??$", re.I
    ), "recent_changes"),
    (re.compile(
        r"what is (?:(.+?)'s\s+)?(?:my\s+)?(.+?)\??$", re.I
    ), "current"),
]

_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


class TemporalQueryEngine:
    """
    Answers natural-language queries about how Blix's knowledge has
    changed over time.

    Parameters
    ----------
    state_tracker:
        ``StateTracker`` for current/historical attribute lookups.
    transition_engine:
        ``StateTransitionEngine`` for transition/evolution queries.
    temporal_graph:
        Optional ``TemporalGraph`` for entity-relation evolution queries.
    default_entity:
        Fallback entity name when the query uses "my"/"I" without naming
        someone explicitly (e.g. "sayan" as the primary user).
    """

    def __init__(
        self,
        state_tracker: StateTracker,
        transition_engine: StateTransitionEngine,
        temporal_graph: Optional[TemporalGraph] = None,
        default_entity: str = "user",
    ) -> None:
        self._tracker = state_tracker
        self._transitions = transition_engine
        self._graph = temporal_graph
        self._default_entity = default_entity

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def query(self, text: str) -> TemporalQueryResult:
        text = text.strip()
        for pattern, qtype in _PATTERNS:
            m = pattern.match(text)
            if m is None:
                continue
            handler = getattr(self, f"_handle_{qtype}")
            return handler(text, m)
        return TemporalQueryResult(
            query=text, query_type="unrecognised", answer="",
            explanation="Query did not match any known temporal pattern.",
        )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _resolve_entity(self, raw: Optional[str]) -> str:
        if raw and raw.strip():
            return raw.strip()
        return self._default_entity

    def _handle_historical_year(self, text: str, m: re.Match) -> TemporalQueryResult:
        entity = self._resolve_entity(m.group(1))
        attribute = _slugify(m.group(2))
        year = m.group(3)
        timestamp = f"{year}-12-31T23:59:59"

        snap = self._tracker.at_time(entity, attribute, timestamp)
        if snap is None:
            return TemporalQueryResult(
                query=text, query_type="historical_year", answer="",
                explanation=f"No recorded value for {entity}.{attribute} during {year}.",
            )
        return TemporalQueryResult(
            query=text, query_type="historical_year",
            answer=snap.value,
            timeline=[snap.to_dict()],
            explanation=f"{entity}'s {attribute.replace('_', ' ')} in {year} was '{snap.value}'.",
        )

    def _handle_evolution(self, text: str, m: re.Match) -> TemporalQueryResult:
        entity = self._resolve_entity(m.group(1))
        attribute = _slugify(m.group(2))

        history = self._tracker.history(entity, attribute)
        if not history:
            return TemporalQueryResult(
                query=text, query_type="evolution", answer="",
                explanation=f"No history recorded for {entity}.{attribute}.",
            )

        chain = " → ".join(s.value for s in history)
        explanation = (
            f"{entity}'s {attribute.replace('_', ' ')} evolved through "
            f"{len(history)} stage(s): {chain}."
        )
        return TemporalQueryResult(
            query=text, query_type="evolution", answer=chain,
            timeline=[s.to_dict() for s in history],
            explanation=explanation,
        )

    def _handle_transition(self, text: str, m: re.Match) -> TemporalQueryResult:
        entity = m.group(1).strip()
        target_value = m.group(2).strip()

        all_transitions = self._transitions.history(entity)
        matches = [
            t for t in all_transitions
            if _slugify(t.to_value) == _slugify(target_value) or target_value.lower() in t.to_value.lower()
        ]
        if not matches:
            return TemporalQueryResult(
                query=text, query_type="transition", answer="",
                explanation=f"No recorded transition to '{target_value}' for {entity}.",
            )
        first = matches[0]
        return TemporalQueryResult(
            query=text, query_type="transition",
            answer=first.transitioned_at,
            timeline=[first.to_dict()],
            explanation=f"{entity} adopted '{target_value}' on {first.transitioned_at}.",
        )

    def _handle_recent_changes(self, text: str, m: re.Match) -> TemporalQueryResult:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        days = amount * _UNIT_DAYS.get(unit, 1)
        since = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)).isoformat()

        changes = self._transitions.transitions_since(since)
        if not changes:
            return TemporalQueryResult(
                query=text, query_type="recent_changes", answer="No changes recorded in that period.",
                explanation=f"No transitions found since {since}.",
            )

        summary = "; ".join(t.describe() for t in changes[:10])
        return TemporalQueryResult(
            query=text, query_type="recent_changes",
            answer=summary,
            timeline=[t.to_dict() for t in changes],
            explanation=f"{len(changes)} change(s) in the last {amount} {unit}(s).",
        )

    def _handle_current(self, text: str, m: re.Match) -> TemporalQueryResult:
        entity = self._resolve_entity(m.group(1))
        attribute = _slugify(m.group(2))

        snap = self._tracker.current(entity, attribute)
        if snap is None:
            return TemporalQueryResult(
                query=text, query_type="current", answer="",
                explanation=f"No currently tracked value for {entity}.{attribute}.",
            )
        return TemporalQueryResult(
            query=text, query_type="current",
            answer=snap.value,
            timeline=[snap.to_dict()],
            explanation=f"{entity}'s current {attribute.replace('_', ' ')} is '{snap.value}'.",
        )

    # ------------------------------------------------------------------
    # Direct API (non-NL) for programmatic use
    # ------------------------------------------------------------------

    def when_adopted(self, entity: str, attribute: str, value: str) -> Optional[str]:
        history = self._tracker.history(entity, attribute)
        for snap in history:
            if _slugify(snap.value) == _slugify(value):
                return snap.start_time
        return None

    def evolution_chain(self, entity: str, attribute: str) -> list:
        return [s.value for s in self._tracker.history(entity, attribute)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text
