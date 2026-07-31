"""
State Reflection Engine — Blix v0.3.7  (New module 9)

Fixes the fourth bug the spec calls out: v0.3.2's ``ReflectionEngine``
only reflects on CURRENT interests and CURRENT goals — it has no
concept of how those things got here. ``StateReflectionEngine``
generates evolution narratives across four dimensions:

    Interest Evolution    — how topics of interest have shifted over time
    Skill Evolution        — how tracked skills/tech-stack have changed
    Project Evolution       — how project focus/status has changed
    Identity Evolution        — how self-descriptive attributes have changed
                                  (role, affiliation, research focus, etc.)

Each dimension is generated from ``StateTransitionEngine`` history for
a configurable set of attribute names, rather than re-deriving anything
from raw memory text — this module is a thin synthesis layer over the
v0.3.7 state/transition substrate, in keeping with the "don't add new
memory layers" guidance: it reuses what Items 1–8 already store.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from core.state_transition import StateTransitionEngine
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Evolution report
# ---------------------------------------------------------------------------


@dataclass
class EvolutionEntry:
    """One attribute's evolution narrative."""

    attribute: str
    chain: list[str] = field(default_factory=list)
    transition_count: int = 0
    narrative: str = ""

    def to_dict(self) -> dict:
        return {
            "attribute": self.attribute,
            "chain": self.chain,
            "transition_count": self.transition_count,
            "narrative": self.narrative,
        }


@dataclass
class StateEvolutionReport:
    """
    Full evolution report for one entity across the four dimensions.
    """

    entity: str
    interest_evolution: list[EvolutionEntry] = field(default_factory=list)
    skill_evolution: list[EvolutionEntry] = field(default_factory=list)
    project_evolution: list[EvolutionEntry] = field(default_factory=list)
    identity_evolution: list[EvolutionEntry] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "interest_evolution": [e.to_dict() for e in self.interest_evolution],
            "skill_evolution": [e.to_dict() for e in self.skill_evolution],
            "project_evolution": [e.to_dict() for e in self.project_evolution],
            "identity_evolution": [e.to_dict() for e in self.identity_evolution],
            "generated_at": self.generated_at,
        }

    def has_any_evolution(self) -> bool:
        return bool(
            self.interest_evolution or self.skill_evolution
            or self.project_evolution or self.identity_evolution
        )

    def summary(self) -> str:
        parts = []
        for label, entries in (
            ("Interests", self.interest_evolution),
            ("Skills", self.skill_evolution),
            ("Projects", self.project_evolution),
            ("Identity", self.identity_evolution),
        ):
            if entries:
                parts.append(f"{label}: " + "; ".join(e.narrative for e in entries))
        return " | ".join(parts) if parts else "No tracked evolution yet."


# ---------------------------------------------------------------------------
# Attribute classification — which tracked attributes belong to which dimension
# ---------------------------------------------------------------------------

_DEFAULT_DIMENSION_ATTRIBUTES: dict[str, list[str]] = {
    "interest": ["research_focus", "favorite_topic", "interest", "topic_of_interest"],
    "skill": ["favorite_language", "tech_stack", "primary_skill", "programming_language", "skill"],
    "project": ["current_project", "project_status", "project_focus", "active_project"],
    "identity": ["role", "affiliation", "city", "job_title", "research_area", "identity"],
}


class StateReflectionEngine:
    """
    Synthesises evolution narratives from ``StateTransitionEngine`` history.

    Parameters
    ----------
    transition_engine:
        ``StateTransitionEngine`` to read transition history from.
    dimension_attributes:
        Optional override of which attribute names map to which of the
        four evolution dimensions. Defaults to a reasonable starter set;
        callers can extend per-deployment without touching this module.
    """

    def __init__(
        self,
        transition_engine: StateTransitionEngine,
        dimension_attributes: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self._transitions = transition_engine
        self._dims = dimension_attributes or _DEFAULT_DIMENSION_ATTRIBUTES

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self, entity: str) -> StateEvolutionReport:
        """Generate the full four-dimension evolution report for ``entity``."""
        report = StateEvolutionReport(entity=entity)
        report.interest_evolution = self._evolution_for_dimension(entity, "interest")
        report.skill_evolution = self._evolution_for_dimension(entity, "skill")
        report.project_evolution = self._evolution_for_dimension(entity, "project")
        report.identity_evolution = self._evolution_for_dimension(entity, "identity")
        return report

    def _evolution_for_dimension(self, entity: str, dimension: str) -> list[EvolutionEntry]:
        attributes = self._dims.get(dimension, [])
        entries = []
        for attribute in attributes:
            history = self._transitions.history(entity, attribute)
            if not history:
                continue
            chain = [history[0].from_value] if history[0].from_value else []
            chain.extend(t.to_value for t in history)
            entries.append(EvolutionEntry(
                attribute=attribute,
                chain=chain,
                transition_count=len([t for t in history if not t.is_initial]),
                narrative=self._narrative_for(entity, attribute, history),
            ))
        return entries

    def _narrative_for(self, entity: str, attribute: str, history: list) -> str:
        if not history:
            return ""
        label = attribute.replace("_", " ")
        if len(history) == 1:
            return f"{label} started as '{history[0].to_value}'."
        chain = " → ".join(
            ([history[0].from_value] if history[0].from_value else []) + [t.to_value for t in history]
        )
        real_transitions = [t for t in history if not t.is_initial]
        return f"{label} evolved through {len(real_transitions)} change(s): {chain}."

    # ------------------------------------------------------------------
    # Single-dimension convenience accessors
    # ------------------------------------------------------------------

    def interest_evolution(self, entity: str) -> list[EvolutionEntry]:
        return self._evolution_for_dimension(entity, "interest")

    def skill_evolution(self, entity: str) -> list[EvolutionEntry]:
        return self._evolution_for_dimension(entity, "skill")

    def project_evolution(self, entity: str) -> list[EvolutionEntry]:
        return self._evolution_for_dimension(entity, "project")

    def identity_evolution(self, entity: str) -> list[EvolutionEntry]:
        return self._evolution_for_dimension(entity, "identity")

    # ------------------------------------------------------------------
    # Recent shifts (cross-dimension, time-windowed)
    # ------------------------------------------------------------------

    def recent_shifts(self, entity: str, days: int = 30) -> list[dict]:
        """
        All attribute changes (across all four dimensions) for ``entity``
        within the last ``days`` days, tagged with which dimension they
        belong to.
        """
        since = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)).isoformat()
        changes = self._transitions.transitions_since(since, entity=entity)

        attr_to_dim: dict[str, str] = {}
        for dim, attrs in self._dims.items():
            for a in attrs:
                attr_to_dim[a] = dim

        results = []
        for t in changes:
            if t.is_initial:
                continue
            results.append({
                "dimension": attr_to_dim.get(t.attribute, "other"),
                "attribute": t.attribute,
                "from_value": t.from_value,
                "to_value": t.to_value,
                "transitioned_at": t.transitioned_at,
            })
        return results
