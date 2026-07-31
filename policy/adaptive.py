"""
Adaptive Retrieval and Planning — v0.3.16

PolicyLearner-driven configuration for HybridRetriever and BeamSearchPlanner.

Instead of fixed weights, retrieval and planning parameters are selected
by the bandit at runtime and updated from observable outcomes.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class AdaptiveRetriever:
    """
    Wraps HybridRetriever with policy-driven weight selection.

    On each retrieve() call:
      1. Ask PolicySelector for the best retrieval weight config
      2. Apply those weights to HybridRetriever
      3. Record latency + result quality as a RewardSignal
      4. Dispatch reward to PolicyLearner

    Parameters
    ----------
    hgshm : HGSHM
        The HGSHM instance containing the HybridRetriever.
    policy_selector : PolicySelector
        For weight selection.
    reward_engine : RewardEngine
        For dispatching retrieval quality rewards.
    """

    def __init__(self, hgshm: Any, policy_selector: Any,
                 reward_engine: Any = None) -> None:
        self._hgshm = hgshm
        self._selector = policy_selector
        self._reward = reward_engine
        self._current_policy_id: str | None = None

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        context: dict[str, Any] | None = None,
        **kwargs,
    ) -> Any:
        """
        Retrieve with policy-selected weights and auto-reward dispatch.
        """
        import time
        from memory.hybrid.retrieval.hybrid_retriever import HybridWeights

        context = context or {}

        # Select best retrieval policy
        weights_cfg = self._selector.get_retrieval_weights(context)
        policy = self._selector._learner.select_one(
            __import__("policy.models", fromlist=["PolicyType"]).PolicyType.RETRIEVAL_WEIGHTS,
            context=context,
        )
        if policy:
            self._current_policy_id = policy.policy_id

        # Apply weights to retriever
        try:
            hw = HybridWeights(**{k: v for k, v in weights_cfg.items()
                                  if hasattr(HybridWeights, k) or k in [
                                      "semantic", "vector", "graph_distance",
                                      "importance", "confidence", "recency",
                                      "hierarchy", "context_similarity",
                                      "attention", "belief_confidence",
                                      "planning_relevance"]})
            self._hgshm.hybrid_retriever._weights = hw.normalised()
        except Exception as exc:
            log.debug("AdaptiveRetriever: weight application failed: %s", exc)

        # Execute retrieval
        t0 = time.perf_counter()
        results = self._hgshm.hybrid_retriever.retrieve(query, top_k=top_k, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Dispatch reward
        if self._reward and results:
            mean_score = sum(r.final_score for r in results) / len(results)
            self._reward.on_retrieval(
                mean_score, len(results), latency_ms,
                policy_id=self._current_policy_id)

        return results

    @property
    def current_weights(self) -> dict[str, float]:
        return self._selector.get_retrieval_weights()


class AdaptivePlanner:
    """
    Wraps BeamSearchPlanner with policy-driven configuration.

    On each search() call:
      1. Ask PolicySelector for the best planner config
      2. Instantiate BeamSearchPlanner with those parameters
      3. Record best_value as a RewardSignal
      4. Dispatch to PolicyLearner

    Parameters
    ----------
    value_network : ValueNetwork
        The ValueNetwork for trajectory scoring.
    policy_selector : PolicySelector
        For planner config selection.
    reward_engine : RewardEngine
        For dispatching planning quality rewards.
    """

    def __init__(self, value_network: Any, policy_selector: Any,
                 reward_engine: Any = None) -> None:
        self._vn = value_network
        self._selector = policy_selector
        self._reward = reward_engine
        self._current_policy_id: str | None = None

    def search(
        self,
        goal: str,
        start_state: Any,
        action_generator: Any,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """
        Plan with policy-selected config and auto-reward dispatch.
        """
        from planning.beam_search import BeamSearchPlanner
        from policy.models import PolicyType

        context = context or {}
        planner_cfg = self._selector.get_planner_config(context)

        policy = self._selector._learner.select_one(
            PolicyType.PLANNER_CONFIG, context=context)
        if policy:
            self._current_policy_id = policy.policy_id

        bw    = int(planner_cfg.get("beam_width", 3))
        depth = int(planner_cfg.get("max_depth",  2))

        planner = BeamSearchPlanner(self._vn, beam_width=bw, max_depth=depth)
        result  = planner.search(goal, start_state, action_generator)

        if self._reward and result.best_value is not None:
            self._reward.on_planner(
                result.best_value,
                len(result.runner_up_trajectories) + 1,
                policy_id=self._current_policy_id,
            )

        return result

    @property
    def current_config(self) -> dict[str, Any]:
        return self._selector.get_planner_config()
