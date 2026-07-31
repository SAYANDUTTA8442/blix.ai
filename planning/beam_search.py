"""
Beam Search Planner — Blix v0.3.12  (New module, "Imagination + Search")

The highest-ROI module in this release per design spec. Upgrades plan
selection from "the Planner produces one plan and the Executor just
runs it" (v0.3.5-v0.3.10) to genuine multi-step lookahead search:

    Goal
      ↓
    Generate candidates
      ↓
    Evaluate trajectories
      ↓
    Choose best

Concretely: at each depth, every current beam (a partial
``simulation.trajectory_graph.Trajectory``) is expanded by every
candidate action (supplied by an injected, swappable
``ActionGenerator`` callable — this module does not hard-code what
actions exist), each resulting state is scored by
``world_model.value_network.ValueNetwork``, and only the top
``beam_width`` partial trajectories (by cumulative value) survive to
the next depth. After ``max_depth`` steps, the single highest-value
complete trajectory is returned.

== Explicit scope ==
This is beam search — NOT MCTS, NOT a learned policy/world-model
rollout. No upper-confidence-bound exploration term, no simulation
backpropagation, no tree reuse across calls. That's an explicit,
locked scope boundary for v0.3.12: beam search gets most of the
lookahead benefit of MCTS at a fraction of the complexity, and the
spec calls for exactly this trade-off ("Avoid implementing full
MuZero, Dreamer, or JEPA").

Python 3.10 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from causality.epistemic_status import EpistemicStatus
from simulation.trajectory_graph import Trajectory, TrajectoryBuilder
from world_model.latent_world_model import LatentState
from world_model.value_network import ValueNetwork
from utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_BEAM_WIDTH = 3
_DEFAULT_MAX_DEPTH = 3

# An ActionGenerator proposes candidate (action_name, resulting_state)
# pairs from a given state. Supplied by the caller — this module makes
# no assumption about what actions exist (ReasoningStrategy switches,
# tool choices, anything with a LatentState-shaped consequence).
ActionGenerator = Callable[[LatentState], list[tuple[str, LatentState]]]


@dataclass
class BeamSearchResult:
    """The outcome of one beam search pass: the best trajectory found, plus the runner-up beams."""

    goal: str
    best_trajectory: Optional[Trajectory]
    runner_up_trajectories: list[Trajectory] = field(default_factory=list)
    runner_up_values: list[float] = field(default_factory=list)   # VN scores matching runner_up_trajectories
    best_value: float = 0.0
    epistemic_status: EpistemicStatus = EpistemicStatus.PREDICTED
    searched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "best_trajectory": self.best_trajectory.to_dict() if self.best_trajectory else None,
            "runner_up_trajectories": [t.to_dict() for t in self.runner_up_trajectories],
            "runner_up_values": [round(v, 4) for v in self.runner_up_values],
            "best_value": round(self.best_value, 4),
            "epistemic_status": self.epistemic_status.value,
            "searched_at": self.searched_at,
        }


class BeamSearchPlanner:
    """
    Multi-step lookahead search over candidate action sequences,
    pruned to the top-K beams at each depth by ValueNetwork score.

    Parameters
    ----------
    value_network:
        ``ValueNetwork`` (v0.3.10) — scores every candidate state
        reached during search.
    beam_width:
        How many partial trajectories survive each depth (K in
        top-K). Wider beams explore more but cost more value-network
        evaluations.
    max_depth:
        How many action steps to search ahead.
    """

    def __init__(self, value_network: ValueNetwork, beam_width: int = _DEFAULT_BEAM_WIDTH, max_depth: int = _DEFAULT_MAX_DEPTH) -> None:
        self._value_network = value_network
        self._beam_width = beam_width
        self._max_depth = max_depth

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, goal: str, start_state: LatentState, action_generator: ActionGenerator, max_depth: Optional[int] = None,
    ) -> BeamSearchResult:
        """
        Run beam search from ``start_state`` toward ``goal``, expanding
        candidates via ``action_generator`` for up to ``max_depth``
        steps (defaults to the planner's configured ``max_depth``).
        """
        depth_limit = max_depth if max_depth is not None else self._max_depth

        beams: list[tuple[TrajectoryBuilder, LatentState, float]] = [
            (TrajectoryBuilder(start_state), start_state, self._value_network.value(start_state))
        ]

        for _ in range(depth_limit):
            candidates: list[tuple[TrajectoryBuilder, LatentState, float]] = []
            for builder, current_state, _cumulative_value in beams:
                actions = action_generator(current_state)
                for action_name, next_state in actions:
                    value = self._value_network.value(next_state)
                    value_delta = value - self._value_network.value(current_state)
                    # Clone the builder's trajectory-so-far by re-stepping a fresh builder
                    # from the same start, replaying prior actions, then this new one —
                    # simplest correct way to branch without mutating shared state.
                    new_builder = _clone_and_step(builder, action_name, next_state, value_delta)
                    candidates.append((new_builder, next_state, value))

            if not candidates:
                break  # no actions available — search ends early at this depth

            candidates.sort(key=lambda c: -c[2])
            beams = candidates[: self._beam_width]

        if not beams:
            return BeamSearchResult(goal=goal, best_trajectory=None, best_value=0.0)

        ranked = sorted(beams, key=lambda b: -b[2])
        best_builder, _, best_value = ranked[0]
        best_trajectory = best_builder.build()
        runner_ups = [b[0].build() for b in ranked[1:]]
        runner_up_values = [b[2] for b in ranked[1:]]

        return BeamSearchResult(goal=goal, best_trajectory=best_trajectory,
                                runner_up_trajectories=runner_ups, runner_up_values=runner_up_values,
                                best_value=best_value)


def _clone_and_step(builder: TrajectoryBuilder, action_name: str, next_state: LatentState, value_delta: float) -> TrajectoryBuilder:
    """
    Branch a TrajectoryBuilder: replay its existing trajectory into a
    fresh builder (so sibling beams at the same depth don't share
    mutable trajectory state), then append the new step.
    """
    existing = builder.build()
    cloned = TrajectoryBuilder(existing.start_node.state, epistemic_status=existing.epistemic_status)
    for node, edge in zip(existing.nodes[1:], existing.edges):
        cloned.step(edge.action, node.state, predicted_value_delta=edge.predicted_value_delta, label=node.label)
    cloned.step(action_name, next_state, predicted_value_delta=value_delta)
    return cloned
