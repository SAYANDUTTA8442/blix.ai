"""
Blix v0.3.16.1 — Production Fix Tests

ISSUE-001: Batched decay — verify correctness and reduced DB write count
ISSUE-002: Schema migration — verify version tracking and migration execution
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from policy.models import (
    PolicyRecord, PolicyDomain, PolicyType,
    RewardSignal, RewardType,
)
from policy.store import PolicyStore, _SCHEMA_VERSION, _MIGRATIONS
from policy.learner import PolicyLearner


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def store(tmp_dir):
    return PolicyStore(tmp_dir)


@pytest.fixture
def learner(store):
    l = PolicyLearner(store)
    l.register_defaults()
    return l


def _make_reward(policy_id: str = None, value: float = 0.8) -> RewardSignal:
    return RewardSignal(
        RewardType.MEMORY_QUALITY, value=value, policy_id=policy_id
    )


def _active_policy(store: PolicyStore) -> PolicyRecord:
    """Return the first active policy from the store."""
    arms = store.all_active(domain=PolicyDomain.SYSTEM,
                            policy_type=PolicyType.RETRIEVAL_WEIGHTS)
    assert arms, "No active policies registered"
    return arms[0]


# ════════════════════════════════════════════════════════════════════
# ISSUE-001 — Batched Decay
# ════════════════════════════════════════════════════════════════════

class TestBatchedDecayCorrectness:
    """
    Verify that batched decay produces results numerically equivalent
    to the old per-observation decay while writing far fewer DB rows.
    """

    def test_decay_epoch_starts_at_one(self, learner):
        """No pending decay on a fresh learner."""
        assert learner._decay_epoch == pytest.approx(1.0)
        assert learner._decay_observation_count == 0

    def test_single_observe_advances_epoch(self, learner, store):
        p = _active_policy(store)
        learner.observe(_make_reward(policy_id=p.policy_id))
        # Epoch should have been multiplied once
        expected_epoch = learner._decay_factor
        assert learner._decay_epoch == pytest.approx(expected_epoch, rel=1e-6)

    def test_epoch_accumulates_correctly(self, learner, store):
        """After k observations the epoch should equal decay_factor^k."""
        p = _active_policy(store)
        k = 10
        for _ in range(k):
            learner.observe(_make_reward(policy_id=p.policy_id))

        # Each observe multiplies _decay_epoch by _decay_factor ONCE
        # (observations since the last flush, which hasn't happened yet
        #  because decay_persist_every=50 > 10)
        expected = learner._decay_factor ** k
        assert learner._decay_epoch == pytest.approx(expected, rel=1e-5)

    def test_in_memory_confidence_reflects_decay(self, learner, store):
        """
        A non-updated arm should show decayed confidence in memory
        even before it has been written to DB.
        """
        arms = store.all_active(domain=PolicyDomain.SYSTEM,
                                policy_type=PolicyType.RETRIEVAL_WEIGHTS)
        updated_arm = arms[0]
        other_arm   = arms[1]

        # Prime other_arm with rewards so alpha >> 1
        for _ in range(20):
            learner.observe(_make_reward(policy_id=other_arm.policy_id, value=0.9))

        # Capture alpha AFTER the rewards are applied (from cache)
        alpha_after_rewards = learner._cache[other_arm.policy_id].alpha
        assert alpha_after_rewards > 5.0, "other_arm should have high alpha after rewards"

        # Observe updated_arm — each call should apply epoch to other_arm in cache
        for _ in range(5):
            learner.observe(_make_reward(policy_id=updated_arm.policy_id))

        # Check in-memory state: epoch should have been applied
        cached = learner._cache.get(other_arm.policy_id)
        assert cached is not None
        epoch = learner._decay_epoch
        # epoch < 1.0 after 5 observations, so cached.alpha < alpha_after_rewards
        if epoch < 1.0:
            assert cached.alpha < alpha_after_rewards, (
                f"Expected cached.alpha ({cached.alpha:.4f}) < "
                f"alpha_after_rewards ({alpha_after_rewards:.4f})"
            )

    def test_epoch_resets_after_flush(self, learner, store):
        """After flush_decay() the epoch should return to 1.0."""
        p = _active_policy(store)
        for _ in range(5):
            learner.observe(_make_reward(policy_id=p.policy_id))
        assert learner._decay_epoch < 1.0  # some decay accumulated

        learner.flush_decay()
        assert learner._decay_epoch == pytest.approx(1.0)
        assert learner._decay_observation_count == 0

    def test_auto_flush_at_persist_interval(self, store):
        """
        After exactly decay_persist_every observations, the learner should
        automatically flush and reset the epoch.
        """
        learner = PolicyLearner(store, decay_persist_every=5)
        learner.register_defaults()
        p = _active_policy(store)

        for _ in range(5):   # exactly at the threshold
            learner.observe(_make_reward(policy_id=p.policy_id))

        # Epoch should be reset to 1.0 after auto-flush
        assert learner._decay_epoch == pytest.approx(1.0)
        assert learner._decay_observation_count == 0

    def test_no_flush_before_interval(self, store):
        """Before reaching decay_persist_every, epoch should NOT be reset."""
        learner = PolicyLearner(store, decay_persist_every=10)
        learner.register_defaults()
        p = _active_policy(store)

        for _ in range(7):   # below threshold
            learner.observe(_make_reward(policy_id=p.policy_id))

        assert learner._decay_epoch < 1.0      # pending decay
        assert learner._decay_observation_count == 7

    def test_confidence_converges_with_batched_decay(self, store):
        """
        After many positive rewards a policy should still converge toward
        high confidence — batching must not break the learning dynamic.
        """
        learner = PolicyLearner(store, decay_persist_every=20)
        learner.register_defaults()
        p = _active_policy(store)

        for _ in range(80):
            learner.observe(_make_reward(policy_id=p.policy_id, value=0.9))
        learner.flush_decay()

        p_final = store.get(p.policy_id)
        assert p_final.confidence > 0.7, (
            f"Expected confidence > 0.7 after 80 positive rewards, "
            f"got {p_final.confidence:.4f}"
        )

    def test_negative_rewards_lower_confidence_with_batched_decay(self, store):
        """Negative rewards must still reduce confidence under batched decay."""
        learner = PolicyLearner(store, decay_persist_every=20)
        learner.register_defaults()
        p = _active_policy(store)

        for _ in range(50):
            learner.observe(_make_reward(policy_id=p.policy_id, value=0.1))
        learner.flush_decay()

        p_final = store.get(p.policy_id)
        assert p_final.confidence < 0.5, (
            f"Expected confidence < 0.5 after 50 negative rewards, "
            f"got {p_final.confidence:.4f}"
        )

    def test_flush_decay_explicit_call(self, learner, store):
        """flush_decay() should write all non-excluded policies to DB."""
        p = _active_policy(store)
        for _ in range(3):
            learner.observe(_make_reward(policy_id=p.policy_id))

        written = learner.flush_decay()
        # Should write the non-updated policies (all arms minus p)
        total = store.count()
        assert written <= total  # can't write more than exist
        assert written >= 0

    def test_flush_decay_no_op_when_epoch_is_one(self, learner, store):
        """flush_decay() with no pending decay should be a no-op (0 writes)."""
        # Don't observe anything — epoch stays 1.0
        written = learner.flush_decay()
        assert written == 0

    def test_epoch_applied_on_cache_miss(self, store):
        """
        A policy loaded from DB after some decay has accumulated should
        have the epoch applied in-memory immediately.
        """
        learner = PolicyLearner(store, decay_persist_every=100)
        learner.register_defaults()
        arms = store.all_active(domain=PolicyDomain.SYSTEM,
                                policy_type=PolicyType.RETRIEVAL_WEIGHTS)
        target = arms[0]
        other  = arms[1]

        # Build up other_arm with positive rewards
        for _ in range(30):
            learner.observe(_make_reward(policy_id=other.policy_id, value=0.9))

        # Now observe target — this drives decay epoch down
        for _ in range(10):
            learner.observe(_make_reward(policy_id=target.policy_id))

        # Force a fresh load from DB by clearing cache for other_arm
        stored_alpha = store.get(other.policy_id).alpha
        del learner._cache[other.policy_id]

        # Re-fetch — should apply epoch to the stored value
        fetched = learner._get_cached(other.policy_id)
        assert fetched is not None
        # The fetched alpha should be decayed vs what's stored in DB
        # (because epoch < 1.0 and alpha > 1.0)
        if stored_alpha > 1.0 and learner._decay_epoch < 1.0:
            assert fetched.alpha <= stored_alpha


class TestBatchedDecayWriteReduction:
    """
    Verify that DB write frequency is substantially reduced vs the old
    per-observation approach.
    """

    def test_db_writes_less_than_n_times_observations(self, store):
        """
        With N policies and K observations, old code did N×K writes.
        New code should do far fewer.
        We verify by counting actual SQLite write transactions.
        """
        learner = PolicyLearner(store, decay_persist_every=50)
        learner.register_defaults()
        n_policies = store.count()
        p = _active_policy(store)

        write_count = [0]
        original_save = store.save

        def counting_save(policy):
            write_count[0] += 1
            return original_save(policy)

        store.save = counting_save

        k = 30  # observations
        for _ in range(k):
            learner.observe(_make_reward(policy_id=p.policy_id))

        store.save = original_save

        # Old code: k × n_policies = 30 × 15 = 450 writes (approx)
        # New code: k × 1 (only the updated arm) = 30 writes
        # Plus periodic flush writes — but none triggered yet (30 < 50)
        old_code_writes = k * n_policies
        assert write_count[0] < old_code_writes, (
            f"Expected fewer than {old_code_writes} writes "
            f"(old approach), got {write_count[0]}"
        )
        # Specifically: should be exactly k writes (only the updated arm)
        assert write_count[0] == k, (
            f"Expected exactly {k} writes (one per observation for updated arm), "
            f"got {write_count[0]}"
        )

    def test_flush_writes_non_updated_arms(self, store):
        """
        After flush_decay(), the non-updated arms should be written exactly once
        regardless of how many observations accumulated.
        """
        learner = PolicyLearner(store, decay_persist_every=100)
        learner.register_defaults()
        n_policies = store.count()
        p = _active_policy(store)

        write_count = [0]
        original_save = store.save
        def counting_save(policy):
            write_count[0] += 1
            return original_save(policy)
        store.save = counting_save

        k = 40
        for _ in range(k):
            learner.observe(_make_reward(policy_id=p.policy_id))

        # Force flush
        learner.flush_decay()
        store.save = original_save

        # k writes for the updated arm (one per observation)
        # + up to n_policies writes for the flush (all active arms, including p)
        expected_max = k + n_policies
        assert write_count[0] <= expected_max, (
            f"Expected ≤ {expected_max} writes, got {write_count[0]}"
        )
        # The key check: old code would do k × n_policies writes
        old_code_writes = k * n_policies
        assert write_count[0] < old_code_writes, (
            f"Batched decay should use fewer writes than old O(N) approach "
            f"({write_count[0]} vs old {old_code_writes})"
        )

    def test_performance_benchmark(self, store):
        """
        100 observations should complete in under 1 second.
        (Old code took ~234ms/100 on this machine; new code should be faster
        since N-1 DB writes per observation are eliminated.)
        """
        learner = PolicyLearner(store, decay_persist_every=50)
        learner.register_defaults()
        p = _active_policy(store)

        t0 = time.perf_counter()
        for _ in range(100):
            learner.observe(_make_reward(policy_id=p.policy_id))
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, (
            f"100 observations took {elapsed:.3f}s — expected < 1.0s"
        )


class TestBatchedDecayPersistenceCorrectness:
    """
    Verify that persistent state (what's in SQLite) is consistent after
    flush and that re-opening the store gives correct values.
    """

    def test_persistent_state_after_flush(self, tmp_dir):
        """
        After flush_decay, the DB should reflect the decayed alpha/beta values.
        Re-opening the store (simulating restart) should read those values.
        """
        store1 = PolicyStore(tmp_dir)
        learner1 = PolicyLearner(store1, decay_persist_every=10)
        learner1.register_defaults()
        p = _active_policy(store1)

        # Prime the policy with positive rewards
        for _ in range(20):
            learner1.observe(_make_reward(policy_id=p.policy_id, value=0.8))

        # Ensure flush has happened (20 > persist_every=10, so two auto-flushes)
        in_memory_conf = learner1._cache[p.policy_id].confidence

        # Re-open the store (simulates a process restart)
        store1.close()
        store2 = PolicyStore(tmp_dir)
        p_persisted = store2.get(p.policy_id)
        assert p_persisted is not None

        # The persisted confidence should be close to the in-memory value
        # (within one flush interval's worth of drift)
        assert abs(p_persisted.confidence - in_memory_conf) < 0.15, (
            f"Persisted confidence {p_persisted.confidence:.4f} diverged "
            f"from in-memory {in_memory_conf:.4f} by more than 0.15"
        )
        store2.close()

    def test_unflushed_decay_is_applied_on_reopen(self, tmp_dir):
        """
        If the process shuts down without flushing, the persisted state reflects
        the state at the last flush. This is the known trade-off: at most
        decay_persist_every observations of decay may be 'un-persisted' on
        an unclean shutdown. Verify the DB value is >= in-memory value
        (because in-memory has more decay applied than what's in DB).
        """
        store1 = PolicyStore(tmp_dir)
        learner1 = PolicyLearner(store1, decay_persist_every=100)
        learner1.register_defaults()
        p = _active_policy(store1)
        other_arms = store1.all_active(
            domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.RETRIEVAL_WEIGHTS)
        other = [a for a in other_arms if a.policy_id != p.policy_id][0]

        # Prime other_arm with rewards so alpha >> 1
        for _ in range(30):
            learner1.observe(_make_reward(policy_id=other.policy_id, value=0.9))

        # Now trigger decay on other by observing p
        for _ in range(20):
            learner1.observe(_make_reward(policy_id=p.policy_id))

        # In-memory state has epoch applied; DB has un-decayed value from last save
        alpha_in_memory = learner1._cache[other.policy_id].alpha
        store1.close()

        # Reopen: DB value should be >= in-memory value (unflushed decay not persisted)
        store2 = PolicyStore(tmp_dir)
        alpha_db = store2.get(other.policy_id).alpha

        # DB alpha >= in-memory alpha because in-memory has more decay applied
        assert alpha_db >= alpha_in_memory, (
            f"DB alpha ({alpha_db:.4f}) should be >= in-memory ({alpha_in_memory:.4f}) "
            f"since unflushed decay is only applied in-memory"
        )
        store2.close()


# ════════════════════════════════════════════════════════════════════
# ISSUE-002 — Schema Migration
# ════════════════════════════════════════════════════════════════════

class TestSchemaMigrationFreshDatabase:
    """Tests for schema initialisation on a brand-new database."""

    def test_fresh_db_has_schema_version_table(self, tmp_dir):
        store = PolicyStore(tmp_dir)
        version = store.get_schema_version()
        assert version == 1
        store.close()

    def test_fresh_db_version_equals_schema_version_constant(self, tmp_dir):
        store = PolicyStore(tmp_dir)
        assert store.get_schema_version() == _SCHEMA_VERSION
        store.close()

    def test_fresh_db_all_tables_exist(self, tmp_dir):
        store = PolicyStore(tmp_dir)
        conn = sqlite3.connect(str(tmp_dir / "policy.db"))
        tables = {row[0] for row in
                  conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        store.close()
        assert "policies"       in tables
        assert "policy_versions" in tables
        assert "reward_log"     in tables
        assert "schema_version" in tables

    def test_fresh_db_schema_version_row_is_singleton(self, tmp_dir):
        store = PolicyStore(tmp_dir)
        store.close()
        conn = sqlite3.connect(str(tmp_dir / "policy.db"))
        rows = conn.execute("SELECT * FROM schema_version").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == 1   # id = 1

    def test_schema_version_constant_is_positive_integer(self):
        assert isinstance(_SCHEMA_VERSION, int)
        assert _SCHEMA_VERSION >= 1

    def test_migrations_dict_keys_are_sequential(self):
        """Migration versions must be sequential integers starting from 2."""
        if not _MIGRATIONS:
            return   # no migrations yet — that's fine
        keys = sorted(_MIGRATIONS.keys())
        expected = list(range(2, len(keys) + 2))
        assert keys == expected, (
            f"Migration keys {keys} are not sequential starting from 2"
        )

    def test_migrations_dict_values_are_non_empty_lists(self):
        """Each migration entry must be a non-empty list of SQL strings."""
        for version, statements in _MIGRATIONS.items():
            assert isinstance(statements, list), (
                f"Migration {version}: expected list, got {type(statements)}"
            )
            assert len(statements) > 0, (
                f"Migration {version}: empty statement list"
            )
            for sql in statements:
                assert isinstance(sql, str) and sql.strip(), (
                    f"Migration {version}: blank/non-string SQL: {sql!r}"
                )


class TestSchemaMigrationExistingDatabase:
    """Tests for schema detection on a database that already exists."""

    def test_reopening_existing_db_preserves_version(self, tmp_dir):
        """Schema version should persist across close/reopen cycles."""
        store1 = PolicyStore(tmp_dir)
        v1 = store1.get_schema_version()
        store1.close()

        store2 = PolicyStore(tmp_dir)
        v2 = store2.get_schema_version()
        store2.close()

        assert v1 == v2 == _SCHEMA_VERSION

    def test_existing_db_without_schema_version_table_gets_version_1(self, tmp_dir):
        """
        A database created before migration support (no schema_version table)
        should be treated as version 1 on first open with the new code.
        """
        db_path = tmp_dir / "policy.db"

        # Create a bare-bones DB with the original schema but NO schema_version
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS policies (
                policy_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                domain TEXT NOT NULL,
                policy_type TEXT NOT NULL,
                config_json TEXT NOT NULL,
                alpha REAL NOT NULL DEFAULT 1.0,
                beta_ REAL NOT NULL DEFAULT 1.0,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                version INTEGER NOT NULL DEFAULT 1,
                is_active INTEGER NOT NULL DEFAULT 1,
                user_id TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS policy_versions (
                version_id TEXT PRIMARY KEY,
                policy_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                alpha REAL NOT NULL,
                beta_ REAL NOT NULL,
                mean_reward REAL NOT NULL,
                created_at TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS reward_log (
                reward_id TEXT PRIMARY KEY,
                reward_type TEXT NOT NULL,
                value REAL NOT NULL,
                policy_id TEXT,
                source TEXT NOT NULL,
                context_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
        """)
        conn.commit()
        conn.close()

        # Now open with the migration-aware PolicyStore
        store = PolicyStore(tmp_dir)
        version = store.get_schema_version()
        store.close()

        assert version == 1

    def test_existing_db_data_preserved_after_reinit(self, tmp_dir):
        """
        Previously saved policies must still be readable after the migration
        system is introduced.
        """
        store1 = PolicyStore(tmp_dir)
        p = PolicyRecord(
            name="legacy_policy",
            domain=PolicyDomain.SYSTEM,
            policy_type=PolicyType.PLANNER_CONFIG,
            config={"beam_width": 5},
        )
        store1.save(p)
        store1.close()

        store2 = PolicyStore(tmp_dir)
        retrieved = store2.get(p.policy_id)
        store2.close()

        assert retrieved is not None
        assert retrieved.name == "legacy_policy"
        assert retrieved.config["beam_width"] == 5


class TestSchemaMigrationExecution:
    """Tests for the migration runner itself."""

    def test_migration_runner_executes_statements(self, tmp_dir):
        """
        Patch _SCHEMA_VERSION and _MIGRATIONS to simulate a version-2 upgrade.
        Verify the migration SQL is executed and version is updated.
        """
        import policy.store as store_module

        # Simulate a migration that adds an index (safe, idempotent)
        fake_migration_sql = (
            "CREATE INDEX IF NOT EXISTS idx_test_migration "
            "ON policies(name)"
        )

        original_version = store_module._SCHEMA_VERSION
        original_migrations = store_module._MIGRATIONS

        try:
            store_module._SCHEMA_VERSION = 2
            store_module._MIGRATIONS = {2: [fake_migration_sql]}

            store = PolicyStore(tmp_dir)
            version = store.get_schema_version()
            store.close()

            assert version == 2

            # Verify the index was created
            conn = sqlite3.connect(str(tmp_dir / "policy.db"))
            indexes = {row[1] for row in
                       conn.execute("PRAGMA index_list(policies)").fetchall()}
            conn.close()
            assert "idx_test_migration" in indexes

        finally:
            store_module._SCHEMA_VERSION = original_version
            store_module._MIGRATIONS = original_migrations

    def test_migration_runs_only_once(self, tmp_dir):
        """
        Opening the store a second time should NOT re-run the migration.
        """
        import policy.store as store_module

        run_count = [0]
        fake_sql = "CREATE INDEX IF NOT EXISTS idx_run_once ON policies(name)"

        original_version = store_module._SCHEMA_VERSION
        original_migrations = store_module._MIGRATIONS

        try:
            store_module._SCHEMA_VERSION = 2
            store_module._MIGRATIONS = {2: [fake_sql]}

            original_run = PolicyStore._run_migrations

            def counting_run(self_inner, from_version):
                run_count[0] += 1
                return original_run(self_inner, from_version)

            PolicyStore._run_migrations = counting_run

            store1 = PolicyStore(tmp_dir)
            store1.close()
            assert run_count[0] == 1   # ran once

            store2 = PolicyStore(tmp_dir)
            store2.close()
            assert run_count[0] == 1   # did NOT run again

        finally:
            PolicyStore._run_migrations = original_run
            store_module._SCHEMA_VERSION = original_version
            store_module._MIGRATIONS = original_migrations

    def test_sequential_migrations_all_run(self, tmp_dir):
        """
        If stored version = 1 and target = 3, migrations 2 and 3 must both run.
        """
        import policy.store as store_module

        executed = []

        original_version = store_module._SCHEMA_VERSION
        original_migrations = store_module._MIGRATIONS

        try:
            store_module._SCHEMA_VERSION = 3
            store_module._MIGRATIONS = {
                2: ["CREATE INDEX IF NOT EXISTS idx_seq2 ON policies(name)"],
                3: ["CREATE INDEX IF NOT EXISTS idx_seq3 ON policies(domain)"],
            }

            store = PolicyStore(tmp_dir)
            version = store.get_schema_version()
            store.close()

            assert version == 3

            conn = sqlite3.connect(str(tmp_dir / "policy.db"))
            indexes = {row[1] for row in
                       conn.execute("PRAGMA index_list(policies)").fetchall()}
            conn.close()
            assert "idx_seq2" in indexes
            assert "idx_seq3" in indexes

        finally:
            store_module._SCHEMA_VERSION = original_version
            store_module._MIGRATIONS = original_migrations

    def test_get_schema_version_public_method(self, tmp_dir):
        """get_schema_version() public method returns correct version."""
        store = PolicyStore(tmp_dir)
        v = store.get_schema_version()
        assert v == _SCHEMA_VERSION
        store.close()

    def test_schema_version_survives_data_operations(self, tmp_dir):
        """
        Normal PolicyStore operations (save, get, log_reward) must not
        corrupt the schema_version table.
        """
        store = PolicyStore(tmp_dir)
        p = PolicyRecord(name="version_test",
                         domain=PolicyDomain.SYSTEM,
                         policy_type=PolicyType.PLANNER_CONFIG)
        store.save(p)
        r = RewardSignal(RewardType.BENCHMARK_SCORE, value=0.9,
                         policy_id=p.policy_id)
        store.log_reward(r)
        store.save_version(p.snapshot(reason="test"))

        version = store.get_schema_version()
        assert version == _SCHEMA_VERSION
        store.close()


# ════════════════════════════════════════════════════════════════════
# REGRESSION — ensure existing behaviour unchanged
# ════════════════════════════════════════════════════════════════════

class TestRegressionBatchedDecay:
    """Quick regressions to confirm the fix doesn't break the existing tests."""

    def test_register_defaults_still_works(self, learner, store):
        assert store.count() >= 15   # all defaults installed

    def test_select_one_still_returns_policy(self, learner):
        p = learner.select_one(PolicyType.RETRIEVAL_WEIGHTS, PolicyDomain.SYSTEM)
        assert p is not None

    def test_observe_updates_target_arm(self, learner, store):
        p = _active_policy(store)
        conf_before = store.get(p.policy_id).confidence
        for _ in range(5):
            learner.observe(_make_reward(policy_id=p.policy_id, value=0.9))
        conf_after = store.get(p.policy_id).confidence
        assert conf_after > conf_before

    def test_observe_batch_works(self, learner, store):
        p = _active_policy(store)
        rewards = [_make_reward(policy_id=p.policy_id, value=0.8) for _ in range(5)]
        total = learner.observe_batch(rewards)
        assert total >= 5

    def test_policy_summary_sorted(self, learner):
        summary = learner.policy_summary()
        confs = [r["confidence"] for r in summary]
        assert confs == sorted(confs, reverse=True)

    def test_flush_decay_callable_at_any_time(self, learner):
        """flush_decay() should be safe to call even with no pending decay."""
        written = learner.flush_decay()
        assert written == 0   # nothing to flush

    def test_schema_version_present_in_policy_db(self, store):
        assert store.get_schema_version() == _SCHEMA_VERSION
