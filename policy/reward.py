"""
Reward Engine — computes and dispatches RewardSignals.

System rewards come from observable outcomes:
  benchmark scores, latency, verification success, planner success,
  token efficiency, regression stability, failure recovery.

User rewards come from interaction signals:
  answer accepted, correction given, task completed, followup asked,
  preference signal, goal advanced, repeated usage.

No RLHF. No human labellers. Only signals derivable from runtime
behaviour without external annotation.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from policy.models import RewardSignal, RewardType, PolicyDomain

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Reward normalisers
# ────────────────────────────────────────────────────────────────────

def _normalise_latency(latency_ms: float, target_ms: float = 500.0) -> float:
    """Convert latency to reward: 0ms→1.0, target_ms→0.5, 2×target→0.0."""
    import math
    return math.exp(-0.693 * latency_ms / target_ms)


def _normalise_token_count(tokens: int, budget: int = 2000) -> float:
    """Reward for staying under token budget."""
    if tokens <= 0:
        return 1.0
    return max(0.0, 1.0 - tokens / (budget * 2))


# ────────────────────────────────────────────────────────────────────
# System Reward Engine
# ────────────────────────────────────────────────────────────────────

class SystemRewardEngine:
    """
    Computes reward signals from system-observable outcomes.

    All methods return a RewardSignal ready to be dispatched to
    the PolicyLearner.
    """

    def benchmark_reward(
        self,
        score: float,
        benchmark_name: str,
        policy_id: str | None = None,
    ) -> RewardSignal:
        """Reward from a benchmark run (score already in [0,1])."""
        return RewardSignal(
            reward_type=RewardType.BENCHMARK_SCORE,
            value=score,
            context={"benchmark": benchmark_name},
            policy_id=policy_id,
            source="benchmark_runner",
        )

    def latency_reward(
        self,
        latency_ms: float,
        operation: str,
        target_ms: float = 500.0,
        policy_id: str | None = None,
    ) -> RewardSignal:
        value = _normalise_latency(latency_ms, target_ms)
        return RewardSignal(
            reward_type=RewardType.LATENCY,
            value=value,
            context={"operation": operation, "latency_ms": latency_ms, "target_ms": target_ms},
            policy_id=policy_id,
            source="performance_monitor",
        )

    def verification_reward(
        self,
        success: bool,
        n_verifications: int = 1,
        policy_id: str | None = None,
    ) -> RewardSignal:
        value = 1.0 if success else 0.0
        return RewardSignal(
            reward_type=RewardType.VERIFICATION_SUCCESS,
            value=value,
            context={"n_verifications": n_verifications},
            policy_id=policy_id,
            source="verification_engine",
        )

    def planner_reward(
        self,
        best_value: float,
        n_trajectories: int,
        policy_id: str | None = None,
    ) -> RewardSignal:
        """Reward from BeamSearchPlanner: best trajectory value."""
        value = min(1.0, best_value)
        return RewardSignal(
            reward_type=RewardType.PLANNER_SUCCESS,
            value=value,
            context={"best_value": best_value, "n_trajectories": n_trajectories},
            policy_id=policy_id,
            source="beam_search_planner",
        )

    def memory_quality_reward(
        self,
        retrieval_score: float,
        n_results: int,
        latency_ms: float,
        policy_id: str | None = None,
    ) -> RewardSignal:
        """Combined quality × speed reward for retrieval operations."""
        quality   = min(1.0, retrieval_score)
        speed     = _normalise_latency(latency_ms, target_ms=200.0)
        coverage  = min(1.0, n_results / 5.0)
        value = 0.5 * quality + 0.3 * speed + 0.2 * coverage
        return RewardSignal(
            reward_type=RewardType.MEMORY_QUALITY,
            value=value,
            context={"retrieval_score": retrieval_score, "n_results": n_results,
                     "latency_ms": latency_ms},
            policy_id=policy_id,
            source="hgshm_retriever",
        )

    def token_efficiency_reward(
        self,
        tokens_used: int,
        tokens_budget: int = 2000,
        policy_id: str | None = None,
    ) -> RewardSignal:
        value = _normalise_token_count(tokens_used, tokens_budget)
        return RewardSignal(
            reward_type=RewardType.TOKEN_EFFICIENCY,
            value=value,
            context={"tokens_used": tokens_used, "budget": tokens_budget},
            policy_id=policy_id,
            source="prompt_compiler",
        )

    def regression_reward(
        self,
        n_passing: int,
        n_total: int,
        policy_id: str | None = None,
    ) -> RewardSignal:
        value = n_passing / max(n_total, 1)
        return RewardSignal(
            reward_type=RewardType.REGRESSION_STABLE,
            value=value,
            context={"passing": n_passing, "total": n_total},
            policy_id=policy_id,
            source="regression_runner",
        )

    def failure_recovery_reward(
        self,
        recovered: bool,
        n_attempts: int = 1,
        policy_id: str | None = None,
    ) -> RewardSignal:
        value = 1.0 / n_attempts if recovered else 0.0
        return RewardSignal(
            reward_type=RewardType.FAILURE_RECOVERY,
            value=value,
            context={"recovered": recovered, "n_attempts": n_attempts},
            policy_id=policy_id,
            source="failure_handler",
        )


# ────────────────────────────────────────────────────────────────────
# User Reward Engine
# ────────────────────────────────────────────────────────────────────

class UserRewardEngine:
    """
    Computes reward signals from user interaction outcomes.

    Signals are derived from observable interaction events —
    no explicit ratings required.
    """

    def answer_accepted_reward(
        self,
        accepted: bool,
        user_id: str,
        policy_id: str | None = None,
    ) -> RewardSignal:
        return RewardSignal(
            reward_type=RewardType.ANSWER_ACCEPTED,
            value=1.0 if accepted else 0.2,  # not 0.0 — partial credit for attempt
            context={"user_id": user_id},
            policy_id=policy_id,
            source="user_interaction",
        )

    def correction_reward(
        self,
        was_corrected: bool,
        correction_severity: float = 0.5,
        user_id: str = "",
        policy_id: str | None = None,
    ) -> RewardSignal:
        """Correction = negative signal proportional to severity."""
        value = max(0.0, 1.0 - correction_severity) if was_corrected else 0.8
        return RewardSignal(
            reward_type=RewardType.CORRECTION_GIVEN,
            value=value,
            context={"user_id": user_id, "corrected": was_corrected,
                     "severity": correction_severity},
            policy_id=policy_id,
            source="user_interaction",
        )

    def task_completion_reward(
        self,
        completed: bool,
        n_turns: int,
        user_id: str = "",
        policy_id: str | None = None,
    ) -> RewardSignal:
        """Task completed in fewer turns = higher reward."""
        import math
        if not completed:
            return RewardSignal(
                reward_type=RewardType.TASK_COMPLETED, value=0.1,
                context={"user_id": user_id, "completed": False, "turns": n_turns},
                policy_id=policy_id, source="task_tracker")
        efficiency = math.exp(-0.3 * max(0, n_turns - 1))  # 1 turn = 1.0, 5 turns ≈ 0.3
        return RewardSignal(
            reward_type=RewardType.TASK_COMPLETED,
            value=min(1.0, 0.5 + 0.5 * efficiency),
            context={"user_id": user_id, "completed": True, "turns": n_turns},
            policy_id=policy_id,
            source="task_tracker",
        )

    def followup_reward(
        self,
        asked_followup: bool,
        user_id: str = "",
        policy_id: str | None = None,
    ) -> RewardSignal:
        """Followup question = user engagement (moderate positive signal)."""
        return RewardSignal(
            reward_type=RewardType.FOLLOWUP_ASKED,
            value=0.7 if asked_followup else 0.4,
            context={"user_id": user_id},
            policy_id=policy_id,
            source="user_interaction",
        )

    def preference_signal_reward(
        self,
        preference_value: float,
        preference_type: str,
        user_id: str = "",
        policy_id: str | None = None,
    ) -> RewardSignal:
        """Direct preference signal (e.g., thumbs up/down, rating)."""
        return RewardSignal(
            reward_type=RewardType.PREFERENCE_SIGNAL,
            value=max(0.0, min(1.0, preference_value)),
            context={"user_id": user_id, "preference_type": preference_type},
            policy_id=policy_id,
            source="user_preference",
        )

    def goal_advanced_reward(
        self,
        goal_progress: float,
        goal_id: str,
        user_id: str = "",
        policy_id: str | None = None,
    ) -> RewardSignal:
        return RewardSignal(
            reward_type=RewardType.GOAL_ADVANCED,
            value=max(0.0, min(1.0, goal_progress)),
            context={"user_id": user_id, "goal_id": goal_id},
            policy_id=policy_id,
            source="goal_tracker",
        )

    def repeated_usage_reward(
        self,
        session_count: int,
        user_id: str = "",
        policy_id: str | None = None,
    ) -> RewardSignal:
        """More sessions = stronger signal that personalization is working."""
        import math
        value = 1.0 - math.exp(-0.3 * session_count)
        return RewardSignal(
            reward_type=RewardType.REPEATED_USAGE,
            value=value,
            context={"user_id": user_id, "session_count": session_count},
            policy_id=policy_id,
            source="session_tracker",
        )


# ────────────────────────────────────────────────────────────────────
# Unified Reward Engine
# ────────────────────────────────────────────────────────────────────

class RewardEngine:
    """
    Facade combining SystemRewardEngine and UserRewardEngine.

    Dispatches computed RewardSignals to a PolicyLearner automatically
    if one is provided.
    """

    def __init__(self, learner: Any = None) -> None:
        self.system = SystemRewardEngine()
        self.user   = UserRewardEngine()
        self._learner = learner  # PolicyLearner | None

    def dispatch(self, reward: RewardSignal) -> None:
        """Send a reward signal to the PolicyLearner."""
        if self._learner is not None:
            self._learner.observe(reward)
        else:
            log.debug("RewardEngine: no learner attached, reward dropped: %s", reward.reward_type)

    def set_learner(self, learner: Any) -> None:
        self._learner = learner

    # ── Convenience dispatch methods ─────────────────────────────────

    def on_benchmark(self, score: float, name: str, policy_id: str | None = None) -> None:
        self.dispatch(self.system.benchmark_reward(score, name, policy_id))

    def on_latency(self, ms: float, op: str, target_ms: float = 500.0,
                   policy_id: str | None = None) -> None:
        self.dispatch(self.system.latency_reward(ms, op, target_ms, policy_id))

    def on_planner(self, best_value: float, n_traj: int, policy_id: str | None = None) -> None:
        self.dispatch(self.system.planner_reward(best_value, n_traj, policy_id))

    def on_retrieval(self, score: float, n_results: int, latency_ms: float,
                     policy_id: str | None = None) -> None:
        self.dispatch(self.system.memory_quality_reward(score, n_results, latency_ms, policy_id))

    def on_answer_accepted(self, accepted: bool, user_id: str,
                           policy_id: str | None = None) -> None:
        self.dispatch(self.user.answer_accepted_reward(accepted, user_id, policy_id))

    def on_task_completed(self, completed: bool, n_turns: int, user_id: str,
                          policy_id: str | None = None) -> None:
        self.dispatch(self.user.task_completion_reward(completed, n_turns, user_id, policy_id))

    def on_preference(self, value: float, ptype: str, user_id: str,
                      policy_id: str | None = None) -> None:
        self.dispatch(self.user.preference_signal_reward(value, ptype, user_id, policy_id))
