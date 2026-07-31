"""
Value Function — Blix v0.3.10  (New module 13)

Inspired by AlphaZero: a learned ``V(state)`` estimating the expected
long-run value (eventual success likelihood) of being in a given
latent state, as opposed to ``world_model.latent_world_model.LatentWorldModel``'s
one-step-ahead predictions (plan success THIS step, tool failure THIS
step). ``ValueNetwork`` estimates value more holistically — "how good
is this situation overall" — which is what
``world_model.scenario_ranker.ScenarioRanker`` (Item 9) needs to
compare multiple candidate scenarios against each other.

Same honest scope as the Latent World Model (Item 1): a small, real,
trainable PyTorch MLP over the same compact ``LatentState`` vector, not
an AlphaZero-scale system — see ``world_model/latent_world_model.py``'s
module docstring for the full reasoning on why a small model is the
honest choice here.

Used by:
    planning.plan_evaluator.PlanQualityEvaluator   — as an additional confidence signal
    world_model.scenario_ranker.ScenarioRanker         — to rank candidate scenarios

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from pathlib import Path

from utils.logger import get_logger
from world_model.latent_world_model import LATENT_DIMENSIONS, LatentState

log = get_logger(__name__)

_MIN_EXAMPLES_TO_TRAIN = 25
_HIDDEN_DIM = 12


def _build_network():
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(len(LATENT_DIMENSIONS), _HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(_HIDDEN_DIM, 1),
        nn.Sigmoid(),
    )


class ValueNetwork:
    """
    Learns V(state): the expected long-run value of a latent state.

    Parameters
    ----------
    examples_file:
        Path to persist accumulated (state, eventual_value) training examples.
    min_examples_to_train:
        Minimum examples before the network is actually trained.
    """

    def __init__(self, examples_file: Path, min_examples_to_train: int = _MIN_EXAMPLES_TO_TRAIN) -> None:
        self._file = examples_file
        self._min_examples = min_examples_to_train
        self._examples: list[dict] = []
        self._net = None
        self._load()
        if len(self._examples) >= self._min_examples:
            self._train()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                self._examples = json.load(fh)
        except Exception:
            self._examples = []

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump(self._examples[-2000:], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe_outcome(self, state: LatentState, eventual_value: float) -> None:
        """
        Record one real (state, eventual_value) observation —
        ``eventual_value`` should reflect how well things ultimately
        turned out from this state (e.g. 1.0 if the goal was eventually
        achieved, 0.0 if it was abandoned/failed, or a continuous
        success-adjacent score).
        """
        self._examples.append({"state": state.as_vector(), "value": max(0.0, min(1.0, eventual_value))})
        self._save()
        if len(self._examples) >= self._min_examples:
            self._train()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self, epochs: int = 50) -> None:
        try:
            import torch
            X = torch.tensor([e["state"] for e in self._examples], dtype=torch.float32)
            y = torch.tensor([[e["value"]] for e in self._examples], dtype=torch.float32)
            net = _build_network()
            optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
            loss_fn = torch.nn.MSELoss()
            net.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                loss = loss_fn(net(X), y)
                loss.backward()
                optimizer.step()
            net.eval()
            self._net = net
        except Exception as exc:
            log.warning("ValueNetwork: training failed (%s) — staying in fallback mode.", exc)
            self._net = None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def value(self, state: LatentState) -> float:
        """
        Estimate V(state). Falls back to a simple blended heuristic of
        the state's own confidence/risk/capability fields when untrained.
        """
        if self._net is None:
            return max(0.0, min(1.0, 0.4 * state.confidence + 0.3 * state.capability_estimate - 0.3 * state.risk))
        try:
            import torch
            with torch.no_grad():
                x = torch.tensor([state.as_vector()], dtype=torch.float32)
                return float(self._net(x)[0][0].item())
        except Exception:
            return max(0.0, min(1.0, 0.4 * state.confidence + 0.3 * state.capability_estimate - 0.3 * state.risk))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._net is not None

    @property
    def sample_count(self) -> int:
        return len(self._examples)
