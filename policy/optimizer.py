"""
PolicyOptimizer — lifecycle management for PolicyRecords.

Responsibilities:
  • Periodic decay (prevent stale policies from dominating)
  • Policy aging (retire policies that haven't improved in N observations)
  • Policy replacement (spawn a mutated variant when a policy converges poorly)
  • Rollback (revert to a previous version when performance drops)
  • Convergence detection (decide when a policy has stabilised)
"""
from __future__ import annotations
import logging
import math
from datetime import datetime, timezone
from typing import Any

from policy.models import PolicyRecord, PolicyDomain, PolicyType
from policy.store import PolicyStore

log = logging.getLogger(__name__)


class PolicyOptimizer:
    """
    Manages the lifecycle of policies in the bandit pool.

    Parameters
    ----------
    policy_store : PolicyStore
    decay_factor : float
        Multiplicative decay applied to (α-1) and (β-1) each cycle.
    min_observations : int
        Minimum observations before a policy is eligible for aging/replacement.
    aging_threshold : float
        If a policy's confidence stays below this after min_observations,
        it is a candidate for replacement.
    convergence_window : int
        Number of recent versions to check for convergence.
    convergence_tolerance : float
        If confidence change across convergence_window is < this, policy
        is considered converged.
    """

    def __init__(
        self,
        policy_store: PolicyStore,
        decay_factor: float = 0.995,
        min_observations: int = 10,
        aging_threshold: float = 0.35,
        convergence_window: int = 5,
        convergence_tolerance: float = 0.02,
    ) -> None:
        self._store = policy_store
        self._decay = decay_factor
        self._min_obs = min_observations
        self._aging_t = aging_threshold
        self._conv_window = convergence_window
        self._conv_tol = convergence_tolerance

    # ── Decay ────────────────────────────────────────────────────────

    def decay_all(self) -> int:
        """Apply temporal decay to all active policies. Returns count."""
        policies = self._store.all_active(limit=1000)
        for p in policies:
            p.decay(self._decay)
            self._store.save(p)
        log.debug("PolicyOptimizer: decayed %d policies", len(policies))
        return len(policies)

    # ── Aging / retirement ────────────────────────────────────────────

    def retire_poor_performers(self) -> list[str]:
        """
        Retire policies that have enough observations but poor confidence.
        Returns list of retired policy_ids.
        """
        policies = self._store.all_active(limit=1000)
        retired = []
        for p in policies:
            if (p.total_observations >= self._min_obs and
                    p.confidence < self._aging_t):
                p.retire()
                self._store.save_version(p.snapshot(reason="retired:poor_performance"))
                self._store.save(p)
                retired.append(p.policy_id)
                log.info("PolicyOptimizer: retired %s (conf=%.3f)", p.name, p.confidence)
        return retired

    # ── Convergence detection ─────────────────────────────────────────

    def is_converged(self, policy_id: str) -> bool:
        """
        Returns True if a policy's confidence has stabilised.

        Checks the last `convergence_window` version snapshots.
        """
        history = self._store.get_history(policy_id)
        if len(history) < self._conv_window:
            return False
        recent = [v.mean_reward for v in history[-self._conv_window:]]
        spread = max(recent) - min(recent)
        return spread < self._conv_tol

    # ── Policy replacement / mutation ────────────────────────────────

    def spawn_mutant(
        self,
        parent: PolicyRecord,
        mutation_scale: float = 0.1,
    ) -> PolicyRecord | None:
        """
        Create a mutated variant of a poorly-performing policy.

        Numeric config values are perturbed by ±mutation_scale.
        The mutant starts with uniform prior Beta(1,1).
        """
        if not parent.config:
            return None

        import random
        new_config: dict[str, Any] = {}
        for k, v in parent.config.items():
            if isinstance(v, (int, float)):
                delta = v * mutation_scale * (2 * random.random() - 1)
                new_config[k] = max(0.0, v + delta)
            elif isinstance(v, bool):
                new_config[k] = not v if random.random() < 0.2 else v
            else:
                new_config[k] = v

        mutant = PolicyRecord(
            name=f"{parent.name}_mutant_{parent.version}",
            domain=parent.domain,
            policy_type=parent.policy_type,
            config=new_config,
            user_id=parent.user_id,
            tags=parent.tags + ["mutant"],
            metadata={"parent_id": parent.policy_id, "mutation_scale": mutation_scale},
        )
        self._store.save(mutant)
        log.info("PolicyOptimizer: spawned mutant %s from %s", mutant.name, parent.name)
        return mutant

    def evolve_poor_performers(self, mutation_scale: float = 0.1) -> list[PolicyRecord]:
        """
        Spawn mutated replacements for policies that are below the aging
        threshold and have enough observations.

        Must be called BEFORE ``retire_poor_performers()`` in any cycle
        so that candidates are still active (``all_active()`` finds them).
        Returns list of new PolicyRecords.
        """
        spawned = []
        candidates = self._store.all_active(limit=1000)
        for p in candidates:
            if (p.total_observations >= self._min_obs and
                    p.confidence < self._aging_t and
                    "mutant" not in p.tags):
                mutant = self.spawn_mutant(p, mutation_scale)
                if mutant:
                    spawned.append(mutant)
        return spawned

    # ── Rollback trigger ─────────────────────────────────────────────

    def check_rollback_needed(
        self,
        policy_id: str,
        lookback: int = 5,
        drop_threshold: float = 0.10,
    ) -> int | None:
        """
        Check if a recent confidence drop warrants rollback.

        Returns the version to roll back to, or None if not needed.
        """
        history = self._store.get_history(policy_id)
        if len(history) < lookback + 1:
            return None
        recent = history[-lookback:]
        older  = history[-(lookback * 2):-lookback] if len(history) >= lookback * 2 else []
        if not older:
            return None
        recent_mean = sum(v.mean_reward for v in recent) / len(recent)
        older_mean  = sum(v.mean_reward for v in older)  / len(older)
        if older_mean - recent_mean > drop_threshold:
            # Find the best historical version
            best = max(history, key=lambda v: v.mean_reward)
            log.warning(
                "PolicyOptimizer: rollback recommended for %s "
                "(recent=%.3f, older=%.3f \u2192 v%d)",
                policy_id[:8], recent_mean, older_mean, best.version)
            return best.version
        return None

    def auto_rollback(self, policy_id: str, learner: Any = None) -> PolicyRecord | None:
        """
        Automatically roll back a policy if performance has dropped.
        Returns the rolled-back PolicyRecord or None.
        """
        target_version = self.check_rollback_needed(policy_id)
        if target_version is None:
            return None
        result = self._store.rollback(policy_id, target_version)
        if result and learner:
            learner._cache_put(policy_id, result)
        return result

    # ── Full optimization cycle ───────────────────────────────────────

    def run_cycle(
        self,
        learner: Any = None,
        spawn_mutants: bool = True,
        check_rollbacks: bool = True,
    ) -> dict[str, Any]:
        """
        Run a complete optimization cycle.

        Correct order (ISSUE-019 fix):
          1. Decay all active policies.
          2. Evolve (spawn mutants from) poor performers — while they are
             still active so ``all_active()`` can find them.
          3. Retire poor performers — after mutants have been spawned.
          4. Check rollbacks.

        The previous order (retire then evolve) was a silent bug: retire
        marked candidates as is_active=False before evolve could query
        them, so evolve_poor_performers() always returned an empty list
        and mutants were never spawned from intended parents.

        Returns a summary dict with keys: decayed, mutants, retired,
        rolled_back.
        """
        decayed = self.decay_all()

        # Evolve BEFORE retire (ISSUE-019 fix)
        mutants: list[PolicyRecord] = []
        if spawn_mutants:
            mutants = self.evolve_poor_performers()

        retired = self.retire_poor_performers()

        rolled_back: list[str] = []
        if check_rollbacks and learner:
            all_policies = self._store.all_active(limit=200)
            for p in all_policies:
                if p.total_observations >= self._min_obs * 2:
                    rb = self.auto_rollback(p.policy_id, learner)
                    if rb:
                        rolled_back.append(rb.policy_id)

        summary = {
            "decayed":     decayed,
            "mutants":     len(mutants),
            "retired":     len(retired),
            "rolled_back": len(rolled_back),
        }
        log.info("PolicyOptimizer cycle: %s", summary)
        return summary
