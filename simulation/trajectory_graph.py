"""
Trajectory Graph — Blix v0.3.12  (New module, "Imagination + Search")

Upgrades single-hop cause-effect modeling
(``causality.cause_graph.CauseGraph``, v0.3.11 — "A causes B") and
single-step prediction (``world_model.latent_world_model.LatentWorldModel``,
v0.3.10 — z_t -> z_(t+1)) to multi-step FUTURES as first-class objects:

    State0
      ↓ Action
    State1
      ↓ Action
    State2

Schema:

    StateNode    — one point in a trajectory (wraps world_model.latent_world_model.LatentState)
    ActionEdge   — one action connecting two StateNodes
    Trajectory   — an ordered chain of StateNodes/ActionEdges from a start state

This module is pure data structure — it does not generate trajectories
itself (that's ``planning.beam_search.BeamSearchPlanner``'s job) or
score them (that's ``world_model.value_network.ValueNetwork`` /
``world_model.scenario_ranker.ScenarioRanker``, reused as-is). A
``Trajectory`` is, deliberately, an ``EpistemicStatus.PREDICTED`` or
``EpistemicStatus.COUNTERFACTUAL`` object — an imagined future, never
itself written to ``memory.beliefs.BeliefStore`` (same safeguard
pattern locked in v0.3.11's ``causality.counterfactual_engine``: no
import of ``memory.beliefs`` anywhere in this file).

Python 3.10 compatible.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from causality.epistemic_status import EpistemicStatus
from world_model.latent_world_model import LatentState
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class StateNode:
    """One point in an imagined trajectory — a LatentState plus trajectory-local bookkeeping."""

    state: LatentState
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    depth: int = 0          # steps from the trajectory's start state
    label: str = ""          # optional human-readable description, e.g. "after switching to ToT"

    def to_dict(self) -> dict:
        return {"node_id": self.node_id, "depth": self.depth, "label": self.label, "state": self.state.to_dict()}


@dataclass
class ActionEdge:
    """One action connecting two StateNodes within a trajectory."""

    from_node_id: str
    to_node_id: str
    action: str               # e.g. "switch_to_tree_of_thought", a ReasoningStrategy.value, or a tool name
    predicted_value_delta: float = 0.0   # expected change in value-network score from taking this action

    def to_dict(self) -> dict:
        return {
            "from_node_id": self.from_node_id, "to_node_id": self.to_node_id,
            "action": self.action, "predicted_value_delta": round(self.predicted_value_delta, 4),
        }


@dataclass
class Trajectory:
    """
    An ordered chain of StateNodes/ActionEdges from a start state —
    one imagined future. Always epistemically PREDICTED or
    COUNTERFACTUAL, never a record of something that actually happened
    (that would be an OBSERVED episode in existing memory layers).
    """

    nodes: list[StateNode] = field(default_factory=list)
    edges: list[ActionEdge] = field(default_factory=list)
    trajectory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    epistemic_status: EpistemicStatus = EpistemicStatus.PREDICTED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "epistemic_status": self.epistemic_status.value,
            "depth": self.depth,
            "created_at": self.created_at,
        }

    @property
    def depth(self) -> int:
        return len(self.edges)

    @property
    def start_node(self) -> Optional[StateNode]:
        return self.nodes[0] if self.nodes else None

    @property
    def end_node(self) -> Optional[StateNode]:
        return self.nodes[-1] if self.nodes else None

    @property
    def actions(self) -> list[str]:
        return [e.action for e in self.edges]

    @property
    def total_predicted_value_delta(self) -> float:
        return sum(e.predicted_value_delta for e in self.edges)


class TrajectoryBuilder:
    """
    Incrementally builds a ``Trajectory`` one step at a time — the
    typical way ``planning.beam_search.BeamSearchPlanner`` constructs
    candidate futures.

    Parameters
    ----------
    start_state:
        The ``LatentState`` the trajectory begins from.
    epistemic_status:
        Defaults to PREDICTED; pass COUNTERFACTUAL for explicit
        what-if exploration (matching the vocabulary already used by
        ``causality.counterfactual_engine``).
    """

    def __init__(self, start_state: LatentState, epistemic_status: EpistemicStatus = EpistemicStatus.PREDICTED) -> None:
        start_node = StateNode(state=start_state, depth=0, label="start")
        self._trajectory = Trajectory(nodes=[start_node], epistemic_status=epistemic_status)

    def step(self, action: str, next_state: LatentState, predicted_value_delta: float = 0.0, label: str = "") -> "TrajectoryBuilder":
        """Append one (action, resulting state) step to the trajectory being built."""
        current = self._trajectory.end_node
        next_node = StateNode(state=next_state, depth=current.depth + 1, label=label)
        edge = ActionEdge(from_node_id=current.node_id, to_node_id=next_node.node_id, action=action, predicted_value_delta=predicted_value_delta)
        self._trajectory.nodes.append(next_node)
        self._trajectory.edges.append(edge)
        return self

    def build(self) -> Trajectory:
        return self._trajectory


class TrajectoryGraph:
    """
    Holds multiple ``Trajectory`` objects in memory for comparison —
    e.g. all the candidate futures a beam search pass is currently
    considering. Deliberately NOT persisted to disk: trajectories are
    transient imagined futures, regenerated each planning pass, not a
    long-term memory layer (consistent with the project's "no more
    memory subsystems" constraint for this release).
    """

    def __init__(self) -> None:
        self._trajectories: dict[str, Trajectory] = {}

    def add(self, trajectory: Trajectory) -> None:
        self._trajectories[trajectory.trajectory_id] = trajectory

    def get(self, trajectory_id: str) -> Optional[Trajectory]:
        return self._trajectories.get(trajectory_id)

    def all_trajectories(self) -> list[Trajectory]:
        return list(self._trajectories.values())

    def deepest(self) -> Optional[Trajectory]:
        if not self._trajectories:
            return None
        return max(self._trajectories.values(), key=lambda t: t.depth)

    def clear(self) -> None:
        self._trajectories = {}

    @property
    def count(self) -> int:
        return len(self._trajectories)
