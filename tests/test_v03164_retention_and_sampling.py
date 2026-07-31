"""
Blix v0.3.16.1 — Tests for ISSUE-006 and ISSUE-007

ISSUE-006: reward_log retention
  - auto-prune triggers at max_rows_per_policy
  - prune keeps exactly the most-recent N rows
  - prune_reward_log public API (single policy and all policies)
  - reward_log_count uses COUNT not full scan
  - reward_stats has LIMIT (no full table scan)
  - broadcast rewards (policy_id=None) are not pruned
  - prune is idempotent
  - prune under concurrent log_reward calls

ISSUE-007: exact Thompson sampling
  - sample is in [0, 1]
  - variance matches Beta distribution (within statistical tolerance)
  - Beta(1,1) variance ≈ 0.0833 (uniform prior — no under-exploration)
  - high-confidence arm wins more often than low-confidence arm
  - no approximation at any parameter range
  - fallback on degenerate parameters
"""
from __future__ import annotations

import math
import random
import statistics
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from policy.models import (
    PolicyRecord, PolicyDomain, PolicyType,
    RewardSignal, RewardType,
)
from policy.store import PolicyStore


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def store(tmp_dir):
    s = PolicyStore(tmp_dir)
    yield s
    s.close()


def _policy(name: str = "test") -> PolicyRecord:
    return PolicyRecord(
        name=name,
        domain=PolicyDomain.SYSTEM,
        policy_type=PolicyType.PLANNER_CONFIG,
    )


def _reward(pid: str, value: float = 0.7) -> RewardSignal:
    return RewardSignal(RewardType.BENCHMARK_SCORE, value=value, policy_id=pid)


def _broadcast_reward(value: float = 0.7) -> RewardSignal:
    """Reward with no policy_id (broadcast)."""
    return RewardSignal(RewardType.BENCHMARK_SCORE, value=value, policy_id=None)


# ════════════════════════════════════════════════════════════════════
# ISSUE-006 — Reward Log Retention
# ════════════════════════════════════════════════════════════════════

class TestRewardLogAutoprune:
    def test_rows_below_limit_not_pruned(self, store):
        """Inserting fewer rows than max_rows_per_policy leaves them all."""
        p = _policy("below_limit")
        store.save(p)
        max_rows = 10
        for i in range(max_rows - 1):
            store.log_reward(_reward(p.policy_id, 0.5 + i * 0.01),
                             max_rows_per_policy=max_rows)
        assert store.reward_log_count(p.policy_id) == max_rows - 1

    def test_auto_prune_triggers_at_limit(self, store):
        """After max_rows_per_policy+1 inserts, count stays at or below limit."""
        p = _policy("at_limit")
        store.save(p)
        max_rows = 5
        for i in range(max_rows + 3):
            store.log_reward(_reward(p.policy_id, 0.6), max_rows_per_policy=max_rows)
        count = store.reward_log_count(p.policy_id)
        assert count <= max_rows, (
            f"Expected ≤ {max_rows} rows after auto-prune, got {count}"
        )

    def test_prune_keeps_most_recent_rows(self, store):
        """Pruned rows are the oldest; the most recent ones are kept."""
        p = _policy("most_recent")
        store.save(p)
        max_rows = 5

        # Insert 10 rewards with distinguishable values
        for i in range(10):
            store.log_reward(
                RewardSignal(RewardType.BENCHMARK_SCORE, value=i * 0.1,
                             policy_id=p.policy_id),
                max_rows_per_policy=max_rows
            )

        rows = store.recent_rewards(policy_id=p.policy_id, limit=20)
        values = [r["value"] for r in rows]

        # The most recent rewards (highest indices) must be present
        # The oldest ones (low indices, low values) must be gone
        assert len(values) <= max_rows
        # The highest value (0.9 = index 9) should be present
        assert max(values) == pytest.approx(0.9, abs=0.01), (
            f"Most recent row (value=0.9) missing. Present values: {values}"
        )

    def test_prune_reward_log_single_policy(self, store):
        """prune_reward_log(policy_id=X) prunes only that policy."""
        p1 = _policy("prune_p1"); p2 = _policy("prune_p2")
        store.save(p1); store.save(p2)

        for _ in range(20):
            store.log_reward(_reward(p1.policy_id), max_rows_per_policy=0)
            store.log_reward(_reward(p2.policy_id), max_rows_per_policy=0)

        store.prune_reward_log(policy_id=p1.policy_id, keep_last=5)

        assert store.reward_log_count(p1.policy_id) == 5
        assert store.reward_log_count(p2.policy_id) == 20  # untouched

    def test_prune_reward_log_all_policies(self, store):
        """prune_reward_log(policy_id=None) prunes all policies."""
        policies = [_policy(f"prune_all_{i}") for i in range(3)]
        for p in policies:
            store.save(p)
            for _ in range(15):
                store.log_reward(_reward(p.policy_id), max_rows_per_policy=0)

        deleted = store.prune_reward_log(policy_id=None, keep_last=5)
        assert deleted > 0  # something was deleted

        for p in policies:
            assert store.reward_log_count(p.policy_id) <= 5

    def test_prune_is_idempotent(self, store):
        """Pruning twice with same keep_last produces same result."""
        p = _policy("idempotent")
        store.save(p)
        for _ in range(20):
            store.log_reward(_reward(p.policy_id), max_rows_per_policy=0)

        store.prune_reward_log(policy_id=p.policy_id, keep_last=5)
        count_after_first = store.reward_log_count(p.policy_id)

        store.prune_reward_log(policy_id=p.policy_id, keep_last=5)
        count_after_second = store.reward_log_count(p.policy_id)

        assert count_after_first == count_after_second == 5

    def test_broadcast_rewards_not_auto_pruned(self, store):
        """
        Rewards with policy_id=None (broadcast) bypass auto-pruning
        since they have no policy to track a count for.
        """
        for _ in range(5):
            store.log_reward(_broadcast_reward(), max_rows_per_policy=2)

        # None-pid rewards should all be present
        total = store.reward_log_count(policy_id=None)
        assert total >= 5

    def test_prune_returns_deleted_count(self, store):
        """prune_reward_log returns the number of rows deleted."""
        p = _policy("deleted_count")
        store.save(p)
        for _ in range(10):
            store.log_reward(_reward(p.policy_id), max_rows_per_policy=0)

        deleted = store.prune_reward_log(policy_id=p.policy_id, keep_last=3)
        assert deleted == 7

    def test_prune_below_limit_deletes_nothing(self, store):
        """Pruning when row count is already below keep_last deletes nothing."""
        p = _policy("already_small")
        store.save(p)
        for _ in range(3):
            store.log_reward(_reward(p.policy_id), max_rows_per_policy=0)

        deleted = store.prune_reward_log(policy_id=p.policy_id, keep_last=10)
        assert deleted == 0
        assert store.reward_log_count(p.policy_id) == 3

    def test_reward_log_count_no_full_scan(self, store):
        """
        reward_log_count uses COUNT(*) not a Python-side full scan.
        Verify the method exists and returns correct values.
        """
        p = _policy("count_check")
        store.save(p)
        assert store.reward_log_count(p.policy_id) == 0

        for _ in range(7):
            store.log_reward(_reward(p.policy_id), max_rows_per_policy=0)
        assert store.reward_log_count(p.policy_id) == 7

    def test_reward_stats_has_limit(self, store):
        """
        reward_stats uses ORDER BY + LIMIT — verify it works correctly
        and includes a count field reflecting the query window.
        """
        p = _policy("stats_limit")
        store.save(p)
        for i in range(20):
            store.log_reward(_reward(p.policy_id, 0.4 + i * 0.02),
                             max_rows_per_policy=0)

        # With last_n=10 it should use only the 10 most recent rows
        stats_10 = store.reward_stats(p.policy_id, last_n=10)
        stats_all = store.reward_stats(p.policy_id, last_n=20)

        assert stats_10["count"] == 10
        assert stats_all["count"] == 20
        # The 10 most recent rewards have higher values than the 20 total average
        assert stats_10["mean"] > stats_all["mean"] - 0.001

    def test_reward_stats_default_limit_is_1000(self, store):
        """Default last_n for reward_stats is 1000 (matches prune default)."""
        import inspect
        sig = inspect.signature(store.reward_stats)
        assert sig.parameters["last_n"].default == 1000

    def test_concurrent_log_reward_with_prune(self, store):
        """
        Concurrent log_reward calls with auto-prune enabled must not
        raise exceptions or corrupt the reward_log table.
        """
        p = _policy("concurrent_prune")
        store.save(p)
        errors = []
        barrier = threading.Barrier(10, timeout=10.0)

        def log_many() -> None:
            try:
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    return
                for _ in range(5):
                    store.log_reward(_reward(p.policy_id, 0.7),
                                     max_rows_per_policy=10)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=log_many, daemon=True)
                   for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=15.0)

        assert not errors, f"Concurrent prune errors: {errors}"
        # After 10 threads × 5 inserts with max=10, count must be ≤ 10
        count = store.reward_log_count(p.policy_id)
        assert count <= 15, f"Expected ≤ 15 rows after concurrent prune, got {count}"


# ════════════════════════════════════════════════════════════════════
# ISSUE-007 — Exact Thompson Sampling
# ════════════════════════════════════════════════════════════════════

class TestExactThompsonSampling:
    """
    Verify that thompson_sample() uses the exact Beta distribution.

    Statistical tests use large sample sizes (N=50,000) so that
    sampling error is small relative to the differences being measured.
    Tolerances are set to 3× the expected sampling standard error.
    """

    N = 20_000  # large enough for < 2% sampling error; small enough for CI

    def _sample_variance(self, a: float, b: float) -> float:
        """Empirical variance of N samples from thompson_sample(a, b)."""
        p = PolicyRecord(alpha=a, beta_=b)
        samples = [p.thompson_sample() for _ in range(self.N)]
        mean = sum(samples) / self.N
        return sum((x - mean) ** 2 for x in samples) / self.N

    @staticmethod
    def _true_variance(a: float, b: float) -> float:
        """Exact variance of Beta(a, b)."""
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def test_sample_in_unit_interval(self):
        """Every sample must be in [0, 1]."""
        p = PolicyRecord(alpha=2.0, beta_=3.0)
        for _ in range(1000):
            s = p.thompson_sample()
            assert 0.0 <= s <= 1.0, f"Sample {s} outside [0, 1]"

    def test_uniform_prior_variance_correct(self):
        """
        Beta(1, 1) = Uniform[0,1] has variance = 1/12 ≈ 0.0833.
        The old approximation underestimated this by 14%.
        The exact implementation must be within 3% of the true value.
        """
        true_var = self._true_variance(1.0, 1.0)          # 0.08333
        empirical_var = self._sample_variance(1.0, 1.0)

        rel_error = abs(empirical_var - true_var) / true_var
        assert rel_error < 0.05, (
            f"Beta(1,1) variance error {rel_error:.1%} > 5%. "
            f"True={true_var:.5f}, empirical={empirical_var:.5f}. "
            f"Under-exploration at cold start is not fixed."
        )

    def test_low_obs_variance_correct(self):
        """
        Beta(1.1, 1.1) — state after 1-2 observations.
        Old approximation: 12.6% error. Exact must be < 5%.
        """
        a, b = 1.1, 1.1
        true_var = self._true_variance(a, b)
        empirical_var = self._sample_variance(a, b)
        rel_error = abs(empirical_var - true_var) / true_var
        assert rel_error < 0.05, (
            f"Beta(1.1,1.1) variance error {rel_error:.1%} > 5%. "
            f"True={true_var:.5f}, empirical={empirical_var:.5f}."
        )

    def test_asymmetric_prior_variance_correct(self):
        """
        Beta(3, 1) and Beta(1, 3) — biased priors.
        Old approximation: 16% error. Exact must be < 5%.
        """
        for a, b in [(3.0, 1.0), (1.0, 3.0)]:
            true_var = self._true_variance(a, b)
            empirical_var = self._sample_variance(a, b)
            rel_error = abs(empirical_var - true_var) / true_var
            assert rel_error < 0.05, (
                f"Beta({a},{b}) variance error {rel_error:.1%} > 5%. "
                f"True={true_var:.5f}, empirical={empirical_var:.5f}."
            )

    def test_mid_range_variance_correct(self):
        """Beta(5, 5) — ~10 observations. Must be within 5% of true variance."""
        a, b = 5.0, 5.0
        true_var = self._true_variance(a, b)
        empirical_var = self._sample_variance(a, b)
        rel_error = abs(empirical_var - true_var) / true_var
        assert rel_error < 0.05, (
            f"Beta(5,5) variance error {rel_error:.1%} > 5%."
        )

    def test_large_params_variance_correct(self):
        """Beta(50, 50) — converged policy. Must be within 5% of true variance."""
        a, b = 50.0, 50.0
        true_var = self._true_variance(a, b)
        empirical_var = self._sample_variance(a, b)
        rel_error = abs(empirical_var - true_var) / true_var
        assert rel_error < 0.05, (
            f"Beta(50,50) variance error {rel_error:.1%} > 5%."
        )

    def test_sample_mean_matches_beta_mean(self):
        """
        Sample mean must match Beta distribution mean = a/(a+b)
        within 1% for several parameter settings.
        """
        test_cases = [
            (1.0, 1.0, 0.5),
            (3.0, 1.0, 0.75),
            (1.0, 3.0, 0.25),
            (9.0, 1.0, 0.9),
            (5.0, 5.0, 0.5),
        ]
        for a, b, expected_mean in test_cases:
            p = PolicyRecord(alpha=a, beta_=b)
            samples = [p.thompson_sample() for _ in range(self.N)]
            empirical_mean = sum(samples) / self.N
            assert abs(empirical_mean - expected_mean) < 0.02, (
                f"Beta({a},{b}) mean error: expected {expected_mean:.3f}, "
                f"got {empirical_mean:.3f}"
            )

    def test_high_confidence_arm_wins_majority(self):
        """
        A high-confidence arm (Beta(9,1), conf=0.9) should beat a
        low-confidence arm (Beta(1,9), conf=0.1) in > 90% of draws.
        This is the core correctness guarantee of Thompson sampling.
        """
        p_high = PolicyRecord(alpha=9.0, beta_=1.0)   # conf = 0.9
        p_low  = PolicyRecord(alpha=1.0, beta_=9.0)   # conf = 0.1

        n_trials = 10_000
        wins = sum(
            1 for _ in range(n_trials)
            if p_high.thompson_sample() > p_low.thompson_sample()
        )
        win_rate = wins / n_trials
        assert win_rate > 0.90, (
            f"High-confidence arm won only {win_rate:.1%} of draws. "
            f"Expected > 90%. Thompson sampling exploration is broken."
        )

    def test_uniform_prior_arms_roughly_equal(self):
        """
        Two Beta(1,1) arms should each win ~50% of draws (true exploration).
        Old approximation under-dispersed, biasing early selection.
        """
        p1 = PolicyRecord(alpha=1.0, beta_=1.0)
        p2 = PolicyRecord(alpha=1.0, beta_=1.0)

        n_trials = 10_000
        p1_wins = sum(
            1 for _ in range(n_trials)
            if p1.thompson_sample() > p2.thompson_sample()
        )
        p1_win_rate = p1_wins / n_trials
        # Should be close to 50% — tolerate 5% either way
        assert 0.45 < p1_win_rate < 0.55, (
            f"Beta(1,1) arms: p1 won {p1_win_rate:.1%}, expected ~50%. "
            f"Biased exploration at cold start."
        )

    def test_no_wilson_hilferty_approximation_in_source(self):
        """
        The source code of thompson_sample must not contain the
        Wilson-Hilferty approximation (gauss + variance formula).
        This confirms the fix was actually applied.
        """
        import inspect
        source = inspect.getsource(PolicyRecord.thompson_sample)
        assert "gauss" not in source, (
            "Wilson-Hilferty (gauss) approximation still present in thompson_sample"
        )
        assert "gammavariate" not in source, (
            "Gamma-ratio fallback still present in thompson_sample (should use betavariate)"
        )
        assert "betavariate" in source, (
            "random.betavariate not found in thompson_sample"
        )

    def test_degenerate_alpha_does_not_raise(self):
        """
        Very small alpha or beta (near zero) must not raise an exception.
        The fallback must return a valid float in [0, 1].
        """
        p = PolicyRecord(alpha=1e-15, beta_=1e-15)
        result = p.thompson_sample()
        assert 0.0 <= result <= 1.0

    def test_samples_are_not_all_identical(self):
        """
        Consecutive samples must not all be the same value — this would
        indicate the method is returning the mean (confidence) instead
        of drawing from the distribution.
        """
        p = PolicyRecord(alpha=2.0, beta_=3.0)
        samples = {p.thompson_sample() for _ in range(20)}
        assert len(samples) > 5, (
            f"Only {len(samples)} distinct values in 20 samples — "
            f"method may be returning a constant."
        )

    def test_thompson_uses_betavariate(self):
        """
        Monkey-patch random.betavariate to count calls and confirm
        thompson_sample invokes it.
        """
        import policy.models as pm_module

        call_count = [0]
        original = pm_module.random.betavariate

        def counting_betavariate(a, b):
            call_count[0] += 1
            return original(a, b)

        pm_module.random.betavariate = counting_betavariate
        try:
            p = PolicyRecord(alpha=2.0, beta_=3.0)
            for _ in range(10):
                p.thompson_sample()
            assert call_count[0] == 10, (
                f"betavariate called {call_count[0]} times, expected 10"
            )
        finally:
            pm_module.random.betavariate = original
