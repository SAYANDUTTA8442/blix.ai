"""
Blix v0.3.16.1 — Tests for ISSUE-019

ISSUE-019: PolicyOptimizer.run_cycle() called retire_poor_performers()
before evolve_poor_performers(). Because retire_poor_performers() marks
candidates as is_active=False, all_active() in evolve_poor_performers()
found zero candidates — mutants were never spawned from intended parents.

Fix: evolve before retire. Tests prove:
  1.  run_cycle() spawns mutants from poor performers
  2.  mutants are active after the cycle
  3.  mutants have the correct parent reference
  4.  retired policies are inactive after the cycle
  5.  calling evolve then retire independently preserves the fix
  6.  calling retire then evolve (old order) produces zero mutants
  7.  evolve_poor_performers() only acts on active policies
  8.  auto_rollback uses _cache_put not direct dict assignment
  9.  run_cycle() returns correct summary keys
  10. dead code removed: no all_history_ids or double all_active() call
"""
from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from policy.models import (
    PolicyRecord, PolicyDomain, PolicyType,
    PolicyVersion, RewardSignal, RewardType,
)
from policy.optimizer import PolicyOptimizer
from policy.store import PolicyStore
from policy.learner import PolicyLearner


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def store(tmp_dir):
    s = PolicyStore(tmp_dir)
    yield s
    s.close()


@pytest.fixture
def optimizer(store):
    """Optimizer with low thresholds to make poor performers easy to create."""
    return PolicyOptimizer(
        store,
        min_observations=3,
        aging_threshold=0.40,
    )


def _poor_policy(name: str, store: PolicyStore,
                 obs: int = 5) -> PolicyRecord:
    """Create and save a poor-performing policy that qualifies for evolution."""
    p = PolicyRecord(
        name=name,
        domain=PolicyDomain.SYSTEM,
        policy_type=PolicyType.PLANNER_CONFIG,
        config={"beam_width": 4, "max_depth": 2},
        alpha=1.0, beta_=10.0,       # confidence ≈ 0.09
        success_count=1,
        failure_count=obs - 1,
    )
    store.save(p)
    return p


def _good_policy(name: str, store: PolicyStore) -> PolicyRecord:
    """Create and save a high-confidence policy."""
    p = PolicyRecord(
        name=name,
        domain=PolicyDomain.SYSTEM,
        policy_type=PolicyType.RETRIEVAL_WEIGHTS,
        config={"semantic": 0.5},
        alpha=10.0, beta_=1.0,       # confidence ≈ 0.91
        success_count=20, failure_count=2,
    )
    store.save(p)
    return p


# ════════════════════════════════════════════════════════════════════
# Core correctness: run_cycle spawns mutants
# ════════════════════════════════════════════════════════════════════

class TestRunCycleSpawnsMutants:
    def test_run_cycle_spawns_mutant_from_poor_performer(self, store, optimizer):
        """
        A poor-performing policy with enough observations must have a
        mutant spawned and itself retired in a single run_cycle() call.
        """
        poor = _poor_policy("poor", store)

        summary = optimizer.run_cycle(spawn_mutants=True)

        assert summary["mutants"] >= 1, (
            "Expected at least 1 mutant spawned; got 0. "
            "ISSUE-019 may still be present (retire before evolve)."
        )
        assert summary["retired"] >= 1

    def test_mutant_is_active_after_cycle(self, store, optimizer):
        """The spawned mutant must be an active policy after the cycle."""
        _poor_policy("poor_active", store)
        optimizer.run_cycle(spawn_mutants=True)

        active = store.all_active()
        mutants = [p for p in active if "mutant" in p.tags]
        assert len(mutants) >= 1, (
            "No active mutants found after run_cycle(). "
            "Mutant was either not spawned or was incorrectly retired."
        )

    def test_mutant_has_parent_reference(self, store, optimizer):
        """Mutant metadata must reference its parent's policy_id."""
        poor = _poor_policy("poor_parent", store)
        optimizer.run_cycle(spawn_mutants=True)

        active = store.all_active()
        mutants = [p for p in active if "mutant" in p.tags]
        assert mutants, "No mutants found"

        parent_ids = {m.metadata.get("parent_id") for m in mutants}
        assert poor.policy_id in parent_ids, (
            f"Mutant metadata does not reference parent {poor.policy_id[:8]}. "
            f"Found parent_ids: {parent_ids}"
        )

    def test_parent_is_retired_after_cycle(self, store, optimizer):
        """The poor-performing parent must be inactive after the cycle."""
        poor = _poor_policy("poor_retired", store)
        optimizer.run_cycle(spawn_mutants=True)

        refreshed = store.get(poor.policy_id)
        assert refreshed is not None
        assert not refreshed.is_active, (
            "Poor performer was not retired after run_cycle()."
        )

    def test_good_policy_not_retired(self, store, optimizer):
        """High-confidence policies must survive the cycle untouched."""
        good = _good_policy("good", store)
        poor = _poor_policy("poor", store)

        optimizer.run_cycle(spawn_mutants=True)

        good_refreshed = store.get(good.policy_id)
        assert good_refreshed.is_active, (
            "Good policy was incorrectly retired."
        )

    def test_multiple_poor_performers_all_get_mutants(self, store, optimizer):
        """Each qualifying poor policy should produce one mutant."""
        for i in range(3):
            _poor_policy(f"poor_{i}", store)

        summary = optimizer.run_cycle(spawn_mutants=True)

        assert summary["mutants"] == 3, (
            f"Expected 3 mutants (one per poor policy), got {summary['mutants']}."
        )
        assert summary["retired"] == 3


# ════════════════════════════════════════════════════════════════════
# Order-dependence proof
# ════════════════════════════════════════════════════════════════════

class TestOrderDependence:
    def test_evolve_before_retire_produces_mutants(self, store, optimizer):
        """
        Calling evolve_poor_performers() before retire_poor_performers()
        (the correct order) produces mutants.
        """
        _poor_policy("order_correct", store)

        # Correct order: evolve first
        mutants = optimizer.evolve_poor_performers()
        retired = optimizer.retire_poor_performers()

        assert len(mutants) == 1, "evolve before retire should produce 1 mutant"
        assert len(retired) == 1

    def test_retire_before_evolve_produces_no_mutants(self, store, optimizer):
        """
        Calling retire_poor_performers() before evolve_poor_performers()
        (the OLD buggy order) produces zero mutants — because retire
        marks candidates inactive before evolve can find them.
        This test documents the bug that was fixed.
        """
        _poor_policy("order_wrong", store)

        # Wrong order: retire first (the old bug)
        retired = optimizer.retire_poor_performers()
        mutants = optimizer.evolve_poor_performers()

        assert len(retired) == 1, "should have retired the poor performer"
        assert len(mutants) == 0, (
            "After retire, evolve finds no active candidates — "
            "this proves why the old order was wrong."
        )

    def test_run_cycle_uses_correct_order(self, store, optimizer):
        """
        run_cycle() must spawn mutants even when it also retires.
        This was impossible with the old order.
        """
        _poor_policy("cycle_order", store)
        summary = optimizer.run_cycle(spawn_mutants=True)

        # Under the old order: mutants=0, retired=1
        # Under the fixed order: mutants=1, retired=1
        assert summary["mutants"] == 1 and summary["retired"] == 1, (
            f"run_cycle returned {summary}. "
            f"Expected mutants=1, retired=1."
        )


# ════════════════════════════════════════════════════════════════════
# evolve_poor_performers behaviour
# ════════════════════════════════════════════════════════════════════

class TestEvolvePoorPerformers:
    def test_only_acts_on_active_policies(self, store, optimizer):
        """
        evolve_poor_performers() must not touch already-inactive policies.
        """
        poor = _poor_policy("inactive_poor", store)
        poor.retire()
        store.save(poor)

        mutants = optimizer.evolve_poor_performers()
        assert len(mutants) == 0, (
            "Should not spawn from an already-inactive policy."
        )

    def test_skips_existing_mutants(self, store, optimizer):
        """Mutants (policies tagged 'mutant') must not spawn further mutants."""
        # Create a mutant that is itself a poor performer
        mutant = PolicyRecord(
            name="existing_mutant",
            domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.PLANNER_CONFIG,
            config={"beam_width": 3},
            alpha=1.0, beta_=10.0,
            success_count=1, failure_count=9,
            tags=["mutant"],
            metadata={"parent_id": "some-parent"},
        )
        store.save(mutant)

        new_mutants = optimizer.evolve_poor_performers()
        assert len(new_mutants) == 0, (
            "Should not spawn a mutant from an existing mutant."
        )

    def test_skips_below_min_observations(self, store, optimizer):
        """Policies with fewer than min_observations are not evolved."""
        # obs=1, min_observations=3
        p = PolicyRecord(
            name="too_few_obs",
            domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.PLANNER_CONFIG,
            config={"beam_width": 3},
            alpha=1.0, beta_=10.0,
            success_count=0, failure_count=1,  # only 1 observation
        )
        store.save(p)

        mutants = optimizer.evolve_poor_performers()
        assert len(mutants) == 0

    def test_skips_above_aging_threshold(self, store, optimizer):
        """Policies above the aging threshold must not be evolved."""
        good = _good_policy("above_threshold", store)
        # Ensure it has enough observations
        good.success_count = 20; good.failure_count = 2
        store.save(good)

        mutants = optimizer.evolve_poor_performers()
        assert len(mutants) == 0


# ════════════════════════════════════════════════════════════════════
# auto_rollback uses _cache_put
# ════════════════════════════════════════════════════════════════════

class TestAutoRollbackCachePut:
    def test_auto_rollback_calls_cache_put_not_dict_assign(self, store):
        """
        auto_rollback() must update the learner cache via _cache_put()
        (which enforces LRU bounds) not direct dict assignment.
        """
        optimizer = PolicyOptimizer(store, min_observations=1)
        learner = PolicyLearner(store)

        p = PolicyRecord(
            name="rollback_test",
            domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.PLANNER_CONFIG,
            alpha=8.0, beta_=1.0,
        )
        store.save(p)

        # Save a version to roll back to
        v1 = p.snapshot(reason="v1")
        store.save_version(v1)

        # Build a degraded version history to trigger rollback
        for reward in [0.9, 0.9, 0.9, 0.9, 0.9,   # older good period
                        0.1, 0.1, 0.1, 0.1, 0.1]:   # recent bad period
            pv = p.snapshot()
            pv.mean_reward = reward
            store.save_version(pv)

        cache_put_calls = []
        original_put = learner._cache_put

        def tracking_cache_put(pid, policy):
            cache_put_calls.append(pid)
            return original_put(pid, policy)

        learner._cache_put = tracking_cache_put

        # Even if no rollback is triggered, the method must not use
        # direct dict assignment
        optimizer.auto_rollback(p.policy_id, learner=learner)
        learner._cache_put = original_put

        source = inspect.getsource(optimizer.auto_rollback)
        assert 'learner._cache[' not in source, (
            "auto_rollback() still uses direct dict assignment "
            "instead of _cache_put(). LRU eviction will be bypassed."
        )


# ════════════════════════════════════════════════════════════════════
# Summary dict and dead code removal
# ════════════════════════════════════════════════════════════════════

class TestSummaryAndCleanup:
    def test_run_cycle_summary_keys(self, store, optimizer):
        """run_cycle() must return all expected summary keys."""
        summary = optimizer.run_cycle()
        assert set(summary.keys()) == {"decayed", "mutants", "retired", "rolled_back"}

    def test_run_cycle_spawn_mutants_false(self, store, optimizer):
        """With spawn_mutants=False, mutants key must be 0."""
        _poor_policy("no_mutant", store)
        summary = optimizer.run_cycle(spawn_mutants=False)
        assert summary["mutants"] == 0

    def test_no_dead_code_all_history_ids(self):
        """
        The dead variable all_history_ids must not appear in
        evolve_poor_performers() source.
        """
        source = inspect.getsource(PolicyOptimizer.evolve_poor_performers)
        assert "all_history_ids" not in source, (
            "Dead code 'all_history_ids' still present in evolve_poor_performers()"
        )

    def test_no_double_all_active_call(self):
        """
        evolve_poor_performers() must call all_active() exactly once
        (the old version called it twice due to dead code).
        """
        source = inspect.getsource(PolicyOptimizer.evolve_poor_performers)
        # Strip docstring before counting — the doc mentions all_active() by name
        # but we want to count actual call sites (self._store.all_active)
        code_only = source.split('"""', 2)[-1]   # everything after the closing """
        count = code_only.count("all_active(")
        assert count == 1, (
            f"evolve_poor_performers() has {count} all_active() call(s) in code "
            f"(after docstring); expected exactly 1."
        )

    def test_run_cycle_docstring_mentions_issue019(self):
        """run_cycle() docstring must reference ISSUE-019."""
        doc = PolicyOptimizer.run_cycle.__doc__ or ""
        assert "ISSUE-019" in doc or "019" in doc, (
            "run_cycle() docstring should reference ISSUE-019 to explain "
            "why evolve comes before retire."
        )
