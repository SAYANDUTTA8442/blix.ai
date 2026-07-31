"""
Scenario Evaluator — Blix v0.3.10  (New module 9)

Inspired by MuZero: upgrades scenario selection from manual picking
(the Planner currently just runs with whatever plan
``planning.planner.Planner`` produced) to value-based ranking:

    Scenario A
    Scenario B
    Scenario C
      ↓
    Value Network
      ↓
    Best scenario

A "scenario" here is any candidate ``world_model.latent_world_model.LatentState``
representing a possible situation Blix could be in if it took a given
plan/action — e.g. derived from
``planning.plan_evaluator.PlanQualityEvaluator.evaluate()`` output for
several alternative plans. ``ScenarioRanker`` scores each candidate via
``world_model.value_network.ValueNetwork`` and returns them ranked,
best first.

This module does not generate scenarios itself (that remains the
Planner's job, possibly producing multiple candidate plans via the
``metacognition.strategy_manager.ReasoningStrategy.TREE_OF_THOUGHT``
strategy) — it is purely the ranking/selection layer.

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from world_model.latent_world_model import LatentState
from world_model.value_network import ValueNetwork


@dataclass
class Scenario:
    """One candidate scenario to be evaluated and ranked."""

    name: str
    state: LatentState
    description: str = ""


@dataclass
class RankedScenario:
    """A scenario, scored by the value network."""

    scenario: Scenario
    value: float

    def to_dict(self) -> dict:
        return {"name": self.scenario.name, "description": self.scenario.description, "value": round(self.value, 4)}


class ScenarioRanker:
    """
    Ranks candidate scenarios by value-network-estimated value.

    Parameters
    ----------
    value_network:
        ``ValueNetwork`` — supplies V(state) for each candidate scenario.
    """

    def __init__(self, value_network: Optional[ValueNetwork] = None) -> None:
        self._value_network = value_network

    def rank(self, scenarios: list[Scenario]) -> list[RankedScenario]:
        """Score and sort scenarios best-first by estimated value."""
        if self._value_network is None or not scenarios:
            return [RankedScenario(scenario=s, value=0.5) for s in scenarios]
        ranked = [RankedScenario(scenario=s, value=self._value_network.value(s.state)) for s in scenarios]
        return sorted(ranked, key=lambda r: -r.value)

    def best(self, scenarios: list[Scenario]) -> Optional[RankedScenario]:
        """Return the single highest-value scenario, or None if no scenarios were given."""
        ranked = self.rank(scenarios)
        return ranked[0] if ranked else None
