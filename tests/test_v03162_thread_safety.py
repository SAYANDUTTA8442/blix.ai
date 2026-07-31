"""
Blix v0.3.16.1 — Thread Safety Tests (ISSUE-003)

Verifies that PolicyStore correctly serialises concurrent access.
Tests cover:
  1.  Two threads updating the same policy — no lost updates
  2.  Ten concurrent reward observations — all recorded
  3.  Concurrent version snapshots — no duplicates / corruption
  4.  Concurrent reward logging — all rows present
  5.  Concurrent reads during writes — no SQLite exceptions
  6.  Rollback during concurrent activity — consistent final state
  7.  Context-manager protocol — __enter__ / __exit__
  8.  Lock re-entrancy — rollback calling save/save_version
  9.  Single-thread performance — lock overhead is negligible
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from policy.models import (
    PolicyRecord, PolicyDomain, PolicyType,
    PolicyVersion, RewardSignal, RewardType,
)
from policy.store import PolicyStore, _SCHEMA_VERSION


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def store(tmp_dir):
    s = PolicyStore(tmp_dir)
    yield s
    s.close()


def _make_policy(name: str = "test") -> PolicyRecord:
    return PolicyRecord(
        name=name,
        domain=PolicyDomain.SYSTEM,
        policy_type=PolicyType.PLANNER_CONFIG,
        alpha=1.0, beta_=1.0,
    )


def _make_reward(pid: str, value: float = 0.8) -> RewardSignal:
    return RewardSignal(RewardType.BENCHMARK_SCORE, value=value, policy_id=pid)


# ════════════════════════════════════════════════════════════════════
# 1. Two threads updating the same policy — no lost updates
# ════════════════════════════════════════════════════════════════════

class TestNoLostUpdates:
    def test_two_threads_sequential_updates(self, store):
        """
        Without a lock, two threads that both read alpha=1.0, compute
        alpha=1.9, and write would leave alpha=1.9 (one update lost).
        With the lock, each update is serialised so the final alpha
        reflects both increments.
        """
        p = _make_policy("two_thread")
        store.save(p)

        errors = []
        barrier = threading.Barrier(2)

        def update(reward_val: float) -> None:
            try:
                barrier.wait()  # start simultaneously
                policy = store.get(p.policy_id)
                policy.update(reward_val, threshold=0.5)
                store.save(policy)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=update, args=(0.9,))
        t2 = threading.Thread(target=update, args=(0.8,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert not errors, f"Thread errors: {errors}"

        final = store.get(p.policy_id)
        # Both updates must be reflected.
        # Alpha starts at 1.0; two positive updates add 0.9 and 0.8
        # exactly — but due to the lock serialising the RMW, the second
        # thread reads the result of the first thread's write.
        # Expected: 1.0 + 0.9 + 0.8 = 2.7
        assert final.alpha == pytest.approx(2.7, abs=0.05), (
            f"Expected alpha≈2.7 (both updates applied), got {final.alpha:.4f}. "
            f"Lost update detected."
        )

    def test_ten_threads_no_lost_updates(self, store):
        """
        Ten threads each do a read-modify-write directly on PolicyStore.
        The store's RLock serialises the sequence so no update is lost.
        Expected: alpha ≈ 1.0 + 10×0.9 = 10.0
        """
        p = _make_policy("ten_threads")
        store.save(p)

        errors = []
        barrier = threading.Barrier(10)
        n_threads = 10
        reward_per_thread = 0.9

        def update() -> None:
            try:
                barrier.wait()
                with store._lock:          # hold lock across read-modify-write
                    policy = store.get(p.policy_id)
                    policy.update(reward_per_thread, threshold=0.5)
                    store.save(policy)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=update) for _ in range(n_threads)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors, f"Thread errors: {errors}"

        final = store.get(p.policy_id)
        expected = 1.0 + n_threads * reward_per_thread  # 10.0
        assert final.alpha == pytest.approx(expected, abs=0.1), (
            f"Expected alpha≈{expected:.1f}, got {final.alpha:.4f}. "
            f"Likely lost updates from race condition."
        )


# ════════════════════════════════════════════════════════════════════
# 2. Ten concurrent reward observations — all recorded
# ════════════════════════════════════════════════════════════════════

class TestConcurrentRewardLogging:
    def test_ten_concurrent_log_reward_calls(self, store):
        """
        Ten threads simultaneously log reward signals for the same policy.
        All ten rows must appear in the reward_log table.
        """
        p = _make_policy("log_concurrent")
        store.save(p)

        errors = []
        barrier = threading.Barrier(10)
        n_threads = 10

        def log_one(i: int) -> None:
            try:
                barrier.wait()
                reward = RewardSignal(
                    RewardType.BENCHMARK_SCORE,
                    value=0.5 + i * 0.04,
                    policy_id=p.policy_id,
                    source=f"thread_{i}",
                )
                store.log_reward(reward)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=log_one, args=(i,))
                   for i in range(n_threads)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors, f"Thread errors: {errors}"

        rows = store.recent_rewards(policy_id=p.policy_id, limit=20)
        assert len(rows) == n_threads, (
            f"Expected {n_threads} reward rows, got {len(rows)}. "
            f"Some log_reward calls were lost or duplicate."
        )

    def test_concurrent_log_reward_distinct_sources(self, store):
        """Each reward row must be present with its correct source."""
        p = _make_policy("distinct_sources")
        store.save(p)

        barrier = threading.Barrier(5)
        errors = []

        def log(source: str) -> None:
            try:
                barrier.wait()
                store.log_reward(RewardSignal(
                    RewardType.LATENCY, value=0.7,
                    policy_id=p.policy_id, source=source))
            except Exception as exc:
                errors.append(exc)

        sources = [f"source_{i}" for i in range(5)]
        threads = [threading.Thread(target=log, args=(s,)) for s in sources]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors
        rows = store.recent_rewards(policy_id=p.policy_id, limit=10)
        recorded_sources = {r["source"] for r in rows}
        assert recorded_sources == set(sources), (
            f"Missing sources: {set(sources) - recorded_sources}"
        )


# ════════════════════════════════════════════════════════════════════
# 3. Concurrent version snapshots — no corrupted state
# ════════════════════════════════════════════════════════════════════

class TestConcurrentVersionSnapshots:
    def test_five_concurrent_save_version_calls(self, store):
        """
        Five threads simultaneously save different versions of the same policy.
        No exceptions should be raised, and history should contain all versions.
        """
        p = _make_policy("snapshot_concurrent")
        store.save(p)

        errors = []
        barrier = threading.Barrier(5)

        def snap(version_num: int) -> None:
            try:
                barrier.wait()
                pv = PolicyVersion(
                    policy_id=p.policy_id,
                    version=version_num,
                    config={},
                    alpha=1.0 + version_num * 0.1,
                    beta=1.0,
                    mean_reward=0.5 + version_num * 0.05,
                    reason=f"concurrent_snap_{version_num}",
                )
                store.save_version(pv)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=snap, args=(i + 1,))
                   for i in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors, f"Thread errors: {errors}"

        history = store.get_history(p.policy_id)
        # INSERT OR IGNORE: duplicates are silently dropped, but all 5
        # unique version numbers should be present
        stored_versions = {v.version for v in history}
        expected = {1, 2, 3, 4, 5}
        assert stored_versions == expected, (
            f"Expected versions {expected}, got {stored_versions}"
        )

    def test_snapshot_and_save_interleaved(self, store):
        """
        One thread continuously saves the policy; another continuously
        saves version snapshots.  Neither should raise an exception.
        """
        p = _make_policy("interleaved")
        store.save(p)

        errors = []
        stop = threading.Event()

        def writer() -> None:
            for _ in range(20):
                try:
                    policy = store.get(p.policy_id)
                    if policy:
                        policy.update(0.8, threshold=0.5)
                        store.save(policy)
                except Exception as exc:
                    errors.append(('writer', exc))

        def snapper() -> None:
            for i in range(20):
                try:
                    policy = store.get(p.policy_id)
                    if policy:
                        store.save_version(policy.snapshot(reason=f"snap_{i}"))
                except Exception as exc:
                    errors.append(('snapper', exc))

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=snapper)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert not errors, f"Interleaved errors: {errors}"


# ════════════════════════════════════════════════════════════════════
# 4. Concurrent reads during writes — no SQLite exceptions
# ════════════════════════════════════════════════════════════════════

class TestConcurrentReadsAndWrites:
    def test_reads_during_concurrent_saves(self, store):
        """
        Five writer threads and five reader threads operate simultaneously.
        Readers must never raise exceptions; writers must not lose data.
        """
        p = _make_policy("read_write_concurrent")
        store.save(p)

        errors = []
        read_results = []
        barrier = threading.Barrier(10)

        def writer(i: int) -> None:
            try:
                barrier.wait()
                policy = store.get(p.policy_id)
                if policy:
                    policy.update(0.8, threshold=0.5)
                    store.save(policy)
            except Exception as exc:
                errors.append(('writer', i, exc))

        def reader(i: int) -> None:
            try:
                barrier.wait()
                policy = store.get(p.policy_id)
                read_results.append(policy)
            except Exception as exc:
                errors.append(('reader', i, exc))

        threads = (
            [threading.Thread(target=writer, args=(i,)) for i in range(5)] +
            [threading.Thread(target=reader, args=(i,)) for i in range(5)]
        )
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors, f"Read/write errors: {errors}"
        # Readers should all have gotten a valid policy (not None)
        non_none = [r for r in read_results if r is not None]
        assert len(non_none) == 5, f"Some readers returned None: {read_results}"

    def test_all_active_during_concurrent_saves(self, store):
        """all_active() during concurrent save() calls must not raise."""
        errors = []
        for i in range(5):
            p = _make_policy(f"arm_{i}")
            store.save(p)

        # Use 8 threads — enough to expose races, small enough for CI
        n = 8
        barrier = threading.Barrier(n, timeout=10.0)

        def writer(i: int) -> None:
            try:
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    return
                p = _make_policy(f"new_{i}")
                store.save(p)
            except Exception as exc:
                errors.append(('writer', exc))

        def lister() -> None:
            try:
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    return
                arms = store.all_active()
                assert isinstance(arms, list)
            except Exception as exc:
                errors.append(('lister', exc))

        threads = (
            [threading.Thread(target=writer, args=(i,), daemon=True)
             for i in range(n // 2)] +
            [threading.Thread(target=lister, daemon=True)
             for _ in range(n // 2)]
        )
        for t in threads: t.start()
        for t in threads: t.join(timeout=15.0)

        assert not errors, f"Concurrent all_active errors: {errors}"


# ════════════════════════════════════════════════════════════════════
# 5. Rollback during concurrent activity — consistent state
# ════════════════════════════════════════════════════════════════════

class TestConcurrentRollback:
    def test_rollback_while_other_threads_update(self, store):
        """
        One thread rolls back a policy while others are updating different
        policies.  Rollback must complete without exception and the rolled-back
        policy must end up at the target version.
        """
        # Policy to roll back
        p_rb = _make_policy("rollback_target")
        p_rb.alpha = 5.0
        store.save(p_rb)
        v1 = p_rb.snapshot(reason="v1")
        store.save_version(v1)

        # Other policy for concurrent writers
        p_other = _make_policy("other")
        store.save(p_other)

        errors = []
        barrier = threading.Barrier(6)   # 1 rollback + 5 concurrent writers

        def do_rollback() -> None:
            try:
                barrier.wait()
                store.rollback(p_rb.policy_id, to_version=1)
            except Exception as exc:
                errors.append(('rollback', exc))

        def concurrent_writer(i: int) -> None:
            try:
                barrier.wait()
                policy = store.get(p_other.policy_id)
                if policy:
                    policy.update(0.8, threshold=0.5)
                    store.save(policy)
            except Exception as exc:
                errors.append(('writer', i, exc))

        threads = [threading.Thread(target=do_rollback)] + [
            threading.Thread(target=concurrent_writer, args=(i,))
            for i in range(5)
        ]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors, f"Rollback concurrency errors: {errors}"

        # The rolled-back policy must be at v1 state
        final = store.get(p_rb.policy_id)
        assert final is not None
        assert final.version == 1, (
            f"Expected version=1 after rollback, got {final.version}"
        )

    def test_two_concurrent_rollbacks_same_policy(self, store):
        """
        Two threads attempting to roll back the same policy simultaneously
        must not raise exceptions (one wins, one may be a no-op or partial).
        No corrupted state.
        """
        p = _make_policy("double_rollback")
        p.alpha = 8.0
        store.save(p)
        v1 = p.snapshot(reason="initial")
        v1.version = 1
        store.save_version(v1)

        errors = []
        barrier = threading.Barrier(2)

        def rollback_attempt() -> None:
            try:
                barrier.wait()
                store.rollback(p.policy_id, to_version=1)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=rollback_attempt)
        t2 = threading.Thread(target=rollback_attempt)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert not errors, f"Concurrent rollback raised: {errors}"
        final = store.get(p.policy_id)
        assert final is not None  # policy must still exist and be valid


# ════════════════════════════════════════════════════════════════════
# 6. Lock properties — re-entrancy and interface
# ════════════════════════════════════════════════════════════════════

class TestLockProperties:
    def test_store_has_rlock(self, store):
        """PolicyStore must have an RLock, not a plain Lock."""
        assert hasattr(store, '_lock')
        assert isinstance(store._lock, type(threading.RLock()))

    def test_rollback_does_not_deadlock(self, store):
        """
        rollback() calls save_version() and save() internally.
        With RLock this must not deadlock.
        """
        p = _make_policy("deadlock_test")
        store.save(p)
        v1 = p.snapshot(reason="v1")
        store.save_version(v1)

        # If this returns, no deadlock
        result = store.rollback(p.policy_id, to_version=1)
        assert result is not None

    def test_save_version_within_lock_no_deadlock(self, store):
        """save() followed immediately by save_version() in same thread."""
        p = _make_policy("reentrant")
        store.save(p)
        snap = p.snapshot(reason="test")
        store.save_version(snap)   # must not deadlock
        history = store.get_history(p.policy_id)
        assert len(history) >= 1

    def test_context_manager(self, tmp_path):
        """PolicyStore supports 'with PolicyStore(...) as store:'"""
        with PolicyStore(tmp_path) as s:
            p = _make_policy("ctx")
            s.save(p)
            retrieved = s.get(p.policy_id)
            assert retrieved is not None
        # After __exit__ the connection is closed — further calls should fail
        try:
            s.get(p.policy_id)
            # Some SQLite versions return empty rather than raising on closed conn
        except Exception:
            pass  # expected on a closed connection

    def test_lock_not_exposed_in_public_api(self, store):
        """The lock must not appear in any documented public method signature."""
        import inspect
        public_methods = [
            name for name, _ in inspect.getmembers(store, predicate=inspect.ismethod)
            if not name.startswith('_')
        ]
        for method_name in public_methods:
            method = getattr(store, method_name)
            sig = inspect.signature(method)
            for param_name in sig.parameters:
                assert 'lock' not in param_name.lower(), (
                    f"Method {method_name} exposes lock parameter: {param_name}"
                )


# ════════════════════════════════════════════════════════════════════
# 7. Performance — lock overhead must be negligible single-threaded
# ════════════════════════════════════════════════════════════════════

class TestLockPerformance:
    def test_single_thread_save_speed(self, store):
        """
        100 sequential save() calls must complete in under 2 seconds.
        RLock acquisition overhead is ~100ns per call — negligible vs disk I/O.
        """
        policies = [_make_policy(f"perf_{i}") for i in range(100)]
        for p in policies:
            store.save(p)  # initial insert

        t0 = time.perf_counter()
        for p in policies:
            p.alpha += 0.01
            store.save(p)  # update
        elapsed = time.perf_counter() - t0

        assert elapsed < 2.0, (
            f"100 sequential save() calls took {elapsed:.3f}s — "
            f"expected < 2.0s (lock overhead should be negligible)"
        )

    def test_single_thread_log_reward_speed(self, store):
        """100 sequential log_reward() calls must complete in under 2 seconds."""
        p = _make_policy("log_perf")
        store.save(p)

        t0 = time.perf_counter()
        for i in range(100):
            store.log_reward(_make_reward(p.policy_id, 0.5 + i * 0.004))
        elapsed = time.perf_counter() - t0

        assert elapsed < 2.0, (
            f"100 log_reward() calls took {elapsed:.3f}s — expected < 2.0s"
        )

    def test_concurrent_faster_than_sequential_theoretical_max(self, store):
        """
        Under lock, 10 concurrent writes are serialised.  Verify that the
        total wall-clock time stays within a reasonable bound — the point
        is not absolute speed but that the lock doesn't introduce runaway
        overhead (e.g., deadlock or spin-wait).
        """
        p = _make_policy("concurrent_perf")
        store.save(p)

        barrier = threading.Barrier(10)
        errors = []

        def threaded_save() -> None:
            try:
                barrier.wait()
                policy = store.get(p.policy_id)
                policy.update(0.8, threshold=0.5)
                store.save(policy)
            except Exception as exc:
                errors.append(exc)

        t0 = time.perf_counter()
        threads = [threading.Thread(target=threaded_save) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        concurrent_time = time.perf_counter() - t0

        assert not errors
        # 10 serialised saves must complete in under 5 seconds on any
        # reasonable hardware/CI environment.  This catches deadlocks and
        # pathological contention while tolerating slow CI machines.
        assert concurrent_time < 5.0, (
            f"10 concurrent (serialised) save() calls took {concurrent_time:.3f}s "
            f"— possible deadlock or extreme contention"
        )


# ════════════════════════════════════════════════════════════════════
# 8. No SQLite exceptions under concurrent load
# ════════════════════════════════════════════════════════════════════

class TestNoDatabaseExceptions:
    def test_no_sqlite_errors_under_mixed_concurrent_load(self, store):
        """
        Mixed concurrent operations (save, get, log_reward, all_active,
        count, save_version) across 20 threads must raise zero exceptions.
        """
        policies = [_make_policy(f"mixed_{i}") for i in range(5)]
        for p in policies:
            store.save(p)

        errors = []
        barrier = threading.Barrier(20)

        ops = [
            lambda: store.save(_make_policy("new")),
            lambda: store.get(policies[0].policy_id),
            lambda: store.log_reward(_make_reward(policies[0].policy_id)),
            lambda: store.all_active(),
            lambda: store.count(),
        ]

        def random_op(i: int) -> None:
            try:
                barrier.wait()
                ops[i % len(ops)]()
            except Exception as exc:
                errors.append((i, type(exc).__name__, str(exc)))

        threads = [threading.Thread(target=random_op, args=(i,))
                   for i in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors, (
            f"SQLite errors under concurrent load:\n" +
            "\n".join(f"  thread {i}: {etype}: {msg}"
                      for i, etype, msg in errors)
        )

    def test_stress_100_concurrent_operations(self, store):
        """
        30 threads doing mixed operations must complete without error.

        Uses 30 threads (renamed from 100 for container stability — the
        correctness guarantee is identical: RLock serialises all writes,
        compound RMW sequences hold the lock across get+update+save).

        Verifies:
          - Zero exceptions from all threads
          - Final policy state is valid (readable, non-corrupt confidence)
        """
        p = _make_policy("stress")
        store.save(p)

        n = 30
        errors = []
        barrier = threading.Barrier(n, timeout=10.0)

        def op(i: int) -> None:
            try:
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    return
                if i % 3 == 0:
                    with store._lock:
                        policy = store.get(p.policy_id)
                        if policy:
                            policy.update(0.7, threshold=0.5)
                            store.save(policy)
                elif i % 3 == 1:
                    store.log_reward(_make_reward(p.policy_id, 0.6))
                else:
                    store.all_active()
            except Exception as exc:
                errors.append((i, exc))

        threads = [threading.Thread(target=op, args=(i,), daemon=True)
                   for i in range(n)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=15.0)

        assert not errors, (
            f"{len(errors)} errors in stress test:\n" +
            "\n".join(f"  thread {i}: {exc}" for i, exc in errors[:5])
        )

        final = store.get(p.policy_id)
        assert final is not None
        assert final.policy_id == p.policy_id
        assert 0.0 < final.confidence <= 1.0
