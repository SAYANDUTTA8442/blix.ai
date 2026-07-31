"""
Search Critic — Blix v0.3.12  (New module, "Imagination + Search")

After ``planning.beam_search.BeamSearchPlanner`` picks a trajectory,
``SearchCritic`` asks the questions a human reviewer would ask before
trusting that choice:

    Why choose this branch?
    What assumptions exist?
    What is risky?

Produces a ``DecisionExplanation`` — same severity vocabulary as
``planning.critic.PlanCritic`` (v0.3.6: INFO/WARNING/CRITICAL issues +
an overall verdict), reused deliberately rather than inventing a
second issue taxonomy. Where ``PlanCritic`` critiques a STATIC
``TaskGraph`` (missing tools, circular deps, known failures),
``SearchCritic`` critiques a SEARCHED trajectory: how much better the
winner was than the runner-ups (a razor-thin margin is itself a
warning — the search was nearly indifferent), how deep/shallow the
lookahead was, how large the risk swings were along the chosen path,
and whether the chosen trajectory's value estimate rests on a trained
or still-cold-starting ``ValueNetwork``.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from planning.beam_search import BeamSearchResult
from simulation.trajectory_graph import Trajectory
from world_model.value_network import ValueNetwork
from utils.logger import get_logger

log = get_logger(__name__)

_THIN_MARGIN_THRESHOLD = 0.05    # winner vs. best runner-up value gap below this is "nearly indifferent"
_HIGH_RISK_SWING_THRESHOLD = 0.4  # max risk delta along the trajectory above this is flagged


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SearchIssue:
    """One issue found by the Search Critic."""

    severity: IssueSeverity
    category: str   # "thin_margin" | "shallow_search" | "high_risk_swing" | "untrained_value_network" | "no_trajectory_found"
    message: str

    def to_dict(self) -> dict:
        return {"severity": self.severity.value, "category": self.category, "message": self.message}


@dataclass
class DecisionExplanation:
    """Full explanation of a beam search decision: why this branch, what assumptions, what's risky."""

    goal: str
    chosen_actions: list[str]
    why_chosen: str
    assumptions: list[str]
    issues: list[SearchIssue] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "goal": self.goal, "chosen_actions": self.chosen_actions, "why_chosen": self.why_chosen,
            "assumptions": self.assumptions, "issues": [i.to_dict() for i in self.issues],
            "generated_at": self.generated_at,
        }

    @property
    def has_critical(self) -> bool:
        return any(i.severity == IssueSeverity.CRITICAL for i in self.issues)


class SearchCritic:
    """
    Explains and critiques a ``BeamSearchResult``.

    Parameters
    ----------
    value_network:
        Optional ``ValueNetwork`` — used to note whether the search's
        value estimates rest on a trained model (an honest caveat when
        they don't).
    """

    def __init__(self, value_network: Optional[ValueNetwork] = None) -> None:
        self._value_network = value_network

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def explain(self, result: BeamSearchResult) -> DecisionExplanation:
        """Produce a full DecisionExplanation for a completed beam search pass."""
        if result.best_trajectory is None:
            return DecisionExplanation(
                goal=result.goal, chosen_actions=[], why_chosen="No trajectory found — search produced no viable candidates.",
                assumptions=[], issues=[SearchIssue(IssueSeverity.CRITICAL, "no_trajectory_found", "Beam search found no viable trajectory.")],
            )

        trajectory = result.best_trajectory
        issues = self._find_issues(result)
        why_chosen = self._explain_choice(result)
        assumptions = self._list_assumptions(trajectory)

        return DecisionExplanation(
            goal=result.goal, chosen_actions=trajectory.actions, why_chosen=why_chosen,
            assumptions=assumptions, issues=issues,
        )

    # ------------------------------------------------------------------
    # Issue detection
    # ------------------------------------------------------------------

    def _find_issues(self, result: BeamSearchResult) -> list[SearchIssue]:
        issues: list[SearchIssue] = []
        trajectory = result.best_trajectory

        if result.runner_up_trajectories:
            # Use stored VN scores when available (v0.3.13 fix: apples-to-apples comparison)
            if result.runner_up_values:
                best_runner_up_value = max(result.runner_up_values)
            else:
                # Legacy fallback: total_predicted_value_delta (less accurate, kept for compatibility)
                best_runner_up_value = max(
                    (self._estimate_trajectory_value(t) for t in result.runner_up_trajectories), default=None,
                )
            if best_runner_up_value is not None and (result.best_value - best_runner_up_value) < _THIN_MARGIN_THRESHOLD:
                issues.append(SearchIssue(
                    IssueSeverity.WARNING, "thin_margin",
                    f"Winning trajectory's value ({result.best_value:.3f}) is only marginally better than the "
                    f"best runner-up ({best_runner_up_value:.3f}) — the search was nearly indifferent between branches.",
                ))

        if trajectory.depth <= 1:
            issues.append(SearchIssue(
                IssueSeverity.INFO, "shallow_search",
                f"Search only looked ahead {trajectory.depth} step(s) — consider a deeper search for higher-stakes decisions.",
            ))

        risk_values = [n.state.risk for n in trajectory.nodes]
        if risk_values:
            risk_swing = max(risk_values) - min(risk_values)
            if risk_swing >= _HIGH_RISK_SWING_THRESHOLD:
                issues.append(SearchIssue(
                    IssueSeverity.WARNING, "high_risk_swing",
                    f"Risk varies by {risk_swing:.2f} across the chosen trajectory — some intermediate states are notably riskier than others.",
                ))

        if self._value_network is not None and not self._value_network.is_trained:
            issues.append(SearchIssue(
                IssueSeverity.INFO, "untrained_value_network",
                "The value network backing this search is still cold-starting (using heuristic fallback values), "
                "so trajectory rankings should be treated as provisional.",
            ))

        return issues

    @staticmethod
    def _estimate_trajectory_value(trajectory: Trajectory) -> float:
        """Cumulative predicted value delta as a rough comparison proxy for a runner-up's final value."""
        return trajectory.total_predicted_value_delta

    # ------------------------------------------------------------------
    # Explanation text
    # ------------------------------------------------------------------

    def _explain_choice(self, result: BeamSearchResult) -> str:
        trajectory = result.best_trajectory
        actions = " -> ".join(trajectory.actions) if trajectory.actions else "(no actions)"
        return (
            f"Chose the trajectory [{actions}] for goal '{result.goal}' — it scored highest "
            f"(value={result.best_value:.3f}) among {1 + len(result.runner_up_trajectories)} candidate "
            f"trajectories considered at search depth {trajectory.depth}."
        )

    def _list_assumptions(self, trajectory: Trajectory) -> list[str]:
        assumptions = [
            "Each action's predicted resulting state was supplied by the caller's action generator, "
            "not independently verified against ground truth.",
            "Trajectory value estimates come from the configured ValueNetwork, which may still be "
            "cold-starting (see issues) and is not a validated causal model.",
        ]
        if trajectory.depth > 1:
            assumptions.append(
                "Later steps assume earlier steps in the trajectory actually succeed as predicted — "
                "no replanning or re-evaluation between steps is modeled within this single search pass."
            )
        return assumptions
