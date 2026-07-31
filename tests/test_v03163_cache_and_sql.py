"""
Blix v0.3.16.1 — Tests for ISSUE-004 and ISSUE-005

ISSUE-004: Bounded LRU cache on PolicyLearner._cache
  - cache respects max size
  - LRU eviction order is correct
  - cache hit promotes to MRU
  - evicted entries are re-fetched from DB (not lost)
  - custom cache_max_size works
  - cache_max_size=1 edge case
  - cache size is bounded after large registration runs

ISSUE-005: Static SQL in PolicyStore.all_active()
  - all 8 filter combinations produce correct results
  - no f-string or dynamic string construction at call time
  - SQL is pre-compiled (all keys in _ACTIVE_SQL)
  - all parameter combinations produce structurally safe SQL
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from policy.models import (
    PolicyRecord, PolicyDomain, PolicyType,
    RewardSignal, RewardType,
)
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


def _sys_policy(name: str, ptype: PolicyType = PolicyType.PLANNER_CONFIG,
                user_id: str | None = None) -> PolicyRecord:
    return PolicyRecord(
        name=name,
        domain=PolicyDomain.SYSTEM,
        policy_type=ptype,
        user_id=user_id,
    )


def _usr_policy(name: str, ptype: PolicyType = PolicyType.ANSWER_STYLE,
                user_id: str = "alice") -> PolicyRecord:
    return PolicyRecord(
        name=name,
        domain=PolicyDomain.USER,
        policy_type=ptype,
        user_id=user_id,
    )


# ════════════════════════════════════════════════════════════════════
# ISSUE-004 — Bounded LRU Cache
# ════════════════════════════════════════════════════════════════════

class TestLRUCacheBounds:
    def test_cache_respects_max_size(self, store):
        """
        After inserting cache_max_size+5 entries, the cache must
        not exceed cache_max_size items.
        """
        max_size = 10
        learner = PolicyLearner(store, cache_max_size=max_size)
        for i in range(max_size + 5):
            p = _sys_policy(f"policy_{i}")
            store.save(p)
            learner._cache_put(p.policy_id, p)

        assert len(learner._cache) == max_size, (
            f"Cache size {len(learner._cache)} exceeds max_size {max_size}"
        )

    def test_lru_eviction_order(self, store):
        """
        With cache_max_size=3, inserting a 4th entry evicts the
        least recently used (first inserted, never accessed) entry.
        """
        learner = PolicyLearner(store, cache_max_size=3)
        policies = [_sys_policy(f"p{i}") for i in range(4)]
        for p in policies:
            store.save(p)

        # Insert first 3 (fills cache)
        for p in policies[:3]:
            learner._cache_put(p.policy_id, p)
        assert len(learner._cache) == 3

        # Insert 4th — should evict policies[0] (LRU)
        learner._cache_put(policies[3].policy_id, policies[3])
        assert len(learner._cache) == 3
        assert policies[0].policy_id not in learner._cache, (
            "Expected policies[0] (LRU) to be evicted"
        )
        assert policies[3].policy_id in learner._cache, (
            "Expected policies[3] (newest) to be present"
        )

    def test_cache_hit_promotes_to_mru(self, store):
        """
        Accessing an existing entry promotes it to MRU so it is
        evicted last.
        """
        learner = PolicyLearner(store, cache_max_size=3)
        policies = [_sys_policy(f"promote_{i}") for i in range(4)]
        for p in policies:
            store.save(p)

        # Insert first 3
        for p in policies[:3]:
            learner._cache_put(p.policy_id, p)

        # Access policies[0] — promotes it to MRU
        learner._cache_get(policies[0].policy_id)

        # Insert 4th — should evict policies[1] (now LRU), not policies[0]
        learner._cache_put(policies[3].policy_id, policies[3])
        assert policies[0].policy_id in learner._cache, (
            "policies[0] should survive (was promoted to MRU)"
        )
        assert policies[1].policy_id not in learner._cache, (
            "policies[1] should be evicted (LRU after promotion of policies[0])"
        )

    def test_cache_put_updates_existing_entry(self, store):
        """
        Putting the same key twice updates the value and promotes to MRU.
        Cache size must not grow.
        """
        learner = PolicyLearner(store, cache_max_size=5)
        p = _sys_policy("update_test")
        store.save(p)

        learner._cache_put(p.policy_id, p)
        size_before = len(learner._cache)

        p.alpha = 9.0  # mutate
        learner._cache_put(p.policy_id, p)  # update same key

        assert len(learner._cache) == size_before, "Cache size grew on update"
        cached = learner._cache_get(p.policy_id)
        assert cached is not None
        assert cached.alpha == pytest.approx(9.0)

    def test_evicted_entries_refetchable_from_db(self, store):
        """
        An evicted entry must still be fetchable from DB via _get_cached().
        """
        learner = PolicyLearner(store, cache_max_size=2)
        p_old = _sys_policy("old")
        p_new1 = _sys_policy("new1")
        p_new2 = _sys_policy("new2")
        store.save(p_old); store.save(p_new1); store.save(p_new2)

        learner._cache_put(p_old.policy_id, p_old)   # goes in first → LRU
        learner._cache_put(p_new1.policy_id, p_new1)  # fills cache
        learner._cache_put(p_new2.policy_id, p_new2)  # evicts p_old

        assert p_old.policy_id not in learner._cache

        # _get_cached should fall through to DB and re-populate cache
        refetched = learner._get_cached(p_old.policy_id)
        assert refetched is not None
        assert refetched.policy_id == p_old.policy_id

    def test_cache_miss_returns_none(self, store):
        """_cache_get returns None for a key not in cache."""
        learner = PolicyLearner(store, cache_max_size=10)
        result = learner._cache_get("nonexistent_id")
        assert result is None

    def test_cache_max_size_one(self, store):
        """Edge case: cache_max_size=1 holds exactly one entry."""
        learner = PolicyLearner(store, cache_max_size=1)
        p1 = _sys_policy("one"); p2 = _sys_policy("two")
        store.save(p1); store.save(p2)

        learner._cache_put(p1.policy_id, p1)
        assert len(learner._cache) == 1

        learner._cache_put(p2.policy_id, p2)
        assert len(learner._cache) == 1
        assert p2.policy_id in learner._cache
        assert p1.policy_id not in learner._cache

    def test_cache_bounded_after_register_defaults(self, store):
        """
        After register_defaults() (15 policies), the cache must not exceed
        cache_max_size even if max_size < 15.
        """
        max_size = 5
        learner = PolicyLearner(store, cache_max_size=max_size)
        learner.register_defaults()
        assert len(learner._cache) <= max_size

    def test_cache_size_after_many_observations(self, store):
        """
        After 100 reward observations, the cache must stay within bounds.
        """
        learner = PolicyLearner(store, cache_max_size=10)
        learner.register_defaults()

        policies = store.all_active()
        p = policies[0]
        for i in range(100):
            learner.observe(RewardSignal(
                RewardType.BENCHMARK_SCORE, value=0.8, policy_id=p.policy_id))

        assert len(learner._cache) <= 10

    def test_default_cache_max_size_is_1000(self, store):
        """Default cache_max_size must be 1000 (documented in the audit fix)."""
        learner = PolicyLearner(store)
        assert learner._cache_max_size == 1000

    def test_custom_cache_max_size_honoured(self, store):
        """Custom cache_max_size is stored and used."""
        learner = PolicyLearner(store, cache_max_size=42)
        assert learner._cache_max_size == 42

    def test_cache_is_ordered_dict(self, store):
        """Cache must be an OrderedDict for correct LRU eviction."""
        from collections import OrderedDict
        learner = PolicyLearner(store)
        assert isinstance(learner._cache, OrderedDict)


# ════════════════════════════════════════════════════════════════════
# ISSUE-005 — Static SQL (no f-string in all_active)
# ════════════════════════════════════════════════════════════════════

class TestStaticSQL:
    """
    Verify that PolicyStore.all_active() uses only precomputed static SQL
    and that all 8 filter combinations return correct results.
    """

    def test_all_8_sql_keys_precomputed(self):
        """
        PolicyStore._ACTIVE_SQL must contain exactly 8 entries covering
        all (domain?, policy_type?, user_id?) combinations.
        """
        assert len(PolicyStore._ACTIVE_SQL) == 8
        expected_keys = {
            (False, False, False),
            (False, False, True),
            (False, True,  False),
            (False, True,  True),
            (True,  False, False),
            (True,  False, True),
            (True,  True,  False),
            (True,  True,  True),
        }
        assert set(PolicyStore._ACTIVE_SQL.keys()) == expected_keys

    def test_no_fstring_sql_in_all_active(self):
        """
        The source of PolicyStore.all_active() must not build SQL via
        f-string interpolation.  Instead it must look up a precomputed
        static string from _ACTIVE_SQL.
        """
        import inspect
        source = inspect.getsource(PolicyStore.all_active)
        # No f-string SQL construction
        assert 'f"SELECT' not in source, "f-string SQL found in all_active()"
        assert "f'SELECT" not in source, "f-string SQL found in all_active()"
        # Must reference the static lookup dict
        assert "_ACTIVE_SQL" in source, "_ACTIVE_SQL lookup not found in all_active()"

    def test_all_active_no_filter(self, store):
        """all_active() with no args returns all active policies."""
        for i in range(3):
            store.save(_sys_policy(f"sys_{i}"))
        results = store.all_active()
        assert len(results) == 3

    def test_all_active_filter_by_domain(self, store):
        """Filter by domain=SYSTEM returns only system policies."""
        store.save(_sys_policy("sys"))
        store.save(_usr_policy("usr"))
        results = store.all_active(domain=PolicyDomain.SYSTEM)
        assert all(p.domain == PolicyDomain.SYSTEM for p in results)
        assert len(results) == 1

    def test_all_active_filter_by_policy_type(self, store):
        """Filter by policy_type returns only matching type."""
        store.save(_sys_policy("planner", PolicyType.PLANNER_CONFIG))
        store.save(_sys_policy("retrieval", PolicyType.RETRIEVAL_WEIGHTS))
        results = store.all_active(policy_type=PolicyType.PLANNER_CONFIG)
        assert all(p.policy_type == PolicyType.PLANNER_CONFIG for p in results)
        assert len(results) == 1

    def test_all_active_filter_by_user_id(self, store):
        """Filter by user_id returns user-specific and global (None) policies."""
        alice = _usr_policy("alice_pref", user_id="alice")
        bob   = _usr_policy("bob_pref",   user_id="bob")
        global_p = PolicyRecord(
            name="global", domain=PolicyDomain.USER,
            policy_type=PolicyType.ANSWER_STYLE, user_id=None)
        store.save(alice); store.save(bob); store.save(global_p)

        results = store.all_active(user_id="alice")
        names = {p.name for p in results}
        assert "alice_pref" in names, "alice's policy must be returned"
        assert "global"     in names, "global (user_id=None) policy must be returned"
        assert "bob_pref" not in names, "bob's policy must not be returned"

    def test_all_active_filter_domain_and_type(self, store):
        """Filter by both domain and policy_type."""
        store.save(_sys_policy("sys_plan", PolicyType.PLANNER_CONFIG))
        store.save(_sys_policy("sys_ret",  PolicyType.RETRIEVAL_WEIGHTS))
        store.save(_usr_policy("usr_plan", PolicyType.PLANNER_CONFIG))
        results = store.all_active(
            domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.PLANNER_CONFIG)
        assert len(results) == 1
        assert results[0].name == "sys_plan"

    def test_all_active_filter_domain_and_user(self, store):
        """Filter by both domain and user_id."""
        sys_p   = _sys_policy("sys")
        usr_alice = _usr_policy("alice", user_id="alice")
        usr_bob   = _usr_policy("bob",   user_id="bob")
        store.save(sys_p); store.save(usr_alice); store.save(usr_bob)
        results = store.all_active(domain=PolicyDomain.USER, user_id="alice")
        names = {p.name for p in results}
        assert "alice" in names
        assert "sys"   not in names
        assert "bob"   not in names

    def test_all_active_filter_type_and_user(self, store):
        """Filter by both policy_type and user_id."""
        p1 = PolicyRecord(name="style_alice", domain=PolicyDomain.USER,
                          policy_type=PolicyType.ANSWER_STYLE, user_id="alice")
        p2 = PolicyRecord(name="diff_alice",  domain=PolicyDomain.USER,
                          policy_type=PolicyType.DIFFICULTY_LEVEL, user_id="alice")
        store.save(p1); store.save(p2)
        results = store.all_active(
            policy_type=PolicyType.ANSWER_STYLE, user_id="alice")
        assert len(results) == 1
        assert results[0].name == "style_alice"

    def test_all_active_filter_all_three(self, store):
        """Filter by domain, policy_type, and user_id simultaneously."""
        target = PolicyRecord(
            name="target", domain=PolicyDomain.USER,
            policy_type=PolicyType.ANSWER_STYLE, user_id="carol")
        other1 = PolicyRecord(
            name="other1", domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.ANSWER_STYLE, user_id=None)
        other2 = PolicyRecord(
            name="other2", domain=PolicyDomain.USER,
            policy_type=PolicyType.DIFFICULTY_LEVEL, user_id="carol")
        store.save(target); store.save(other1); store.save(other2)
        results = store.all_active(
            domain=PolicyDomain.USER,
            policy_type=PolicyType.ANSWER_STYLE,
            user_id="carol")
        assert len(results) == 1
        assert results[0].name == "target"

    def test_all_active_sql_values_are_strings(self):
        """Every SQL value in _ACTIVE_SQL must be a non-empty string."""
        for key, (sql, param_keys) in PolicyStore._ACTIVE_SQL.items():
            assert isinstance(sql, str) and sql.strip(), (
                f"Key {key}: SQL is empty or not a string"
            )
            assert isinstance(param_keys, list), (
                f"Key {key}: param_keys is not a list"
            )
            # All param keys must be from the allowed set
            allowed = {"domain", "policy_type", "user_id"}
            for pk in param_keys:
                assert pk in allowed, (
                    f"Key {key}: unexpected param key {pk!r}"
                )

    def test_all_active_sql_no_user_input_in_column_names(self):
        """
        No SQL string in _ACTIVE_SQL may contain runtime-variable
        column names — all column references must be literals.
        Verify by checking that none of the SQL strings contain format
        specifiers or concatenation markers.
        """
        for key, (sql, _) in PolicyStore._ACTIVE_SQL.items():
            assert "{" not in sql, f"Key {key}: f-string brace in SQL: {sql!r}"
            assert "%" not in sql or "%" in ("alpha", "beta"), \
                f"Key {key}: % format in SQL: {sql!r}"

    def test_all_active_limit_respected(self, store):
        """The limit parameter is applied correctly."""
        for i in range(10):
            store.save(_sys_policy(f"lim_{i}"))
        results = store.all_active(limit=3)
        assert len(results) == 3

    def test_all_active_sorted_by_confidence(self, store):
        """Results must be sorted by confidence descending."""
        p_low  = PolicyRecord(name="low",  domain=PolicyDomain.SYSTEM,
                              policy_type=PolicyType.WORKSPACE_CONFIG,
                              alpha=1.0, beta_=9.0)
        p_mid  = PolicyRecord(name="mid",  domain=PolicyDomain.SYSTEM,
                              policy_type=PolicyType.WORKSPACE_CONFIG,
                              alpha=5.0, beta_=5.0)
        p_high = PolicyRecord(name="high", domain=PolicyDomain.SYSTEM,
                              policy_type=PolicyType.WORKSPACE_CONFIG,
                              alpha=9.0, beta_=1.0)
        for p in [p_low, p_mid, p_high]:
            store.save(p)
        results = store.all_active(policy_type=PolicyType.WORKSPACE_CONFIG)
        confs = [p.confidence for p in results]
        assert confs == sorted(confs, reverse=True), (
            f"Results not sorted by confidence: {confs}"
        )
        assert results[0].name == "high"
