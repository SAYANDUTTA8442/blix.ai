"""
PolicyStore — persistent storage for policies and their version history.

Uses a separate SQLite database (policy.db) co-located with hgshm.db.
Supports full CRUD, version history, rollback, and domain-scoped queries.

Thread Safety (ISSUE-003)
-------------------------
PolicyStore is fully thread-safe.  A single ``threading.RLock`` (``_lock``)
serialises every write operation and every read-modify-write sequence.

Why WAL mode alone is insufficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SQLite WAL mode serialises *database-level writes* — it prevents two
``sqlite3.Connection.execute()`` calls from corrupting the file
simultaneously.  But the race condition in policy learning is at the
*application layer*, not the database layer:

    Thread A: policy = store.get(pid)   # read  alpha=1.0
    Thread B: policy = store.get(pid)   # read  alpha=1.0
    Thread A: policy.update(0.9)        # compute alpha=1.9
    Thread B: policy.update(0.8)        # compute alpha=1.8
    Thread A: store.save(policy)        # write alpha=1.9
    Thread B: store.save(policy)        # write alpha=1.8  ← A's update LOST

Both writes succeed at the SQLite level (WAL handles that), but Thread B
overwrites Thread A's computed result.  The only fix is to serialise the
entire read-compute-write sequence at the Python level.

Why RLock, not Lock
~~~~~~~~~~~~~~~~~~~~
``RLock`` allows the same thread to acquire the lock multiple times without
deadlocking.  This is necessary because several public methods call other
public methods internally (e.g. ``rollback()`` calls ``get()``,
``save_version()``, and ``save()``) — all of which also acquire the lock.
With a plain ``Lock``, the first re-entrant acquisition would deadlock.

Lock scope
~~~~~~~~~~
- All write methods (``save``, ``save_version``, ``log_reward``, ``delete``,
  ``rollback``, ``_set_schema_version``, ``_run_migrations``, ``close``) hold
  the lock for their entire duration.
- ``rollback()`` holds the lock across the full read-modify-write sequence.
- Read-only methods (``get``, ``all_active``, ``count``, ``get_history``,
  ``recent_rewards``, ``reward_stats``) do NOT hold the lock.  SQLite WAL
  mode provides snapshot-consistent reads; the lock would add unnecessary
  contention without preventing any real race (reads have no application-level
  side effects).
"""
from __future__ import annotations
import json
import logging
import math
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from policy.models import (
    PolicyRecord, PolicyVersion, PolicyDomain, PolicyType, RewardSignal
)

log = logging.getLogger(__name__)

# ── Schema migration infrastructure (ISSUE-002) ──────────────────────────────
#
# _SCHEMA_VERSION is the current target version.  Increment this and add an
# entry to _MIGRATIONS whenever the schema needs to change.
#
# Rules:
#   - Each entry in _MIGRATIONS is a list of SQL statements executed in order
#     within a single transaction.
#   - Statements must be idempotent where possible (use IF NOT EXISTS / IF EXISTS).
#   - Never modify or remove older migration entries; only append new ones.
#   - The initial schema (version 1) is created by _SCHEMA below.
#     _MIGRATIONS only contains changes made AFTER the initial schema.
#
# Example for adding a column in a future version:
#   _MIGRATIONS = {
#       2: ["ALTER TABLE policies ADD COLUMN priority REAL NOT NULL DEFAULT 0.5"],
#       3: ["CREATE INDEX IF NOT EXISTS idx_pol_priority ON policies(priority)"],
#   }

_SCHEMA_VERSION: int = 1

_MIGRATIONS: dict[int, list[str]] = {
    # No migrations yet — current schema is version 1.
    # Future versions add entries here:
    #   2: ["ALTER TABLE policies ADD COLUMN ..."],
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS policies (
    policy_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    domain        TEXT NOT NULL,
    policy_type   TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    alpha         REAL NOT NULL DEFAULT 1.0,
    beta_         REAL NOT NULL DEFAULT 1.0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    version       INTEGER NOT NULL DEFAULT 1,
    is_active     INTEGER NOT NULL DEFAULT 1,
    user_id       TEXT,
    tags_json     TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pol_domain   ON policies(domain);
CREATE INDEX IF NOT EXISTS idx_pol_type     ON policies(policy_type);
CREATE INDEX IF NOT EXISTS idx_pol_user     ON policies(user_id);
CREATE INDEX IF NOT EXISTS idx_pol_active   ON policies(is_active);
CREATE INDEX IF NOT EXISTS idx_pol_conf     ON policies(alpha, beta_);

CREATE TABLE IF NOT EXISTS policy_versions (
    version_id  TEXT PRIMARY KEY,
    policy_id   TEXT NOT NULL,
    version     INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    alpha       REAL NOT NULL,
    beta_       REAL NOT NULL,
    mean_reward REAL NOT NULL,
    created_at  TEXT NOT NULL,
    reason      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_pv_policy ON policy_versions(policy_id, version);

CREATE TABLE IF NOT EXISTS reward_log (
    reward_id   TEXT PRIMARY KEY,
    reward_type TEXT NOT NULL,
    value       REAL NOT NULL,
    policy_id   TEXT,
    source      TEXT NOT NULL,
    context_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    timestamp   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rlog_policy ON reward_log(policy_id);
CREATE INDEX IF NOT EXISTS idx_rlog_type   ON reward_log(reward_type);
CREATE INDEX IF NOT EXISTS idx_rlog_ts     ON reward_log(timestamp);

CREATE TABLE IF NOT EXISTS schema_version (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL
);
"""


class PolicyStore:
    """
    Persistent storage for PolicyRecord, PolicyVersion, and RewardSignal.

    Thread-safe: all write operations are serialised by an internal RLock.
    See module docstring for a detailed explanation of the locking strategy.

    Parameters
    ----------
    memory_dir : Path
        Directory where policy.db will be stored.
    """

    def __init__(self, memory_dir: Path) -> None:
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = memory_dir / "policy.db"
        # RLock: re-entrant so methods that call other locking methods
        # (e.g. rollback → get → save_version → save) do not deadlock.
        self._lock = threading.RLock()
        self._conn = self._connect()
        self._init_schema()
        # In-memory row counter for auto-pruning reward_log (ISSUE-006).
        # Key: policy_id → approximate count of rows in reward_log.
        # Initialised to 0; the first prune trigger queries the real count.
        # Survives only within a process lifetime — safe because prune is
        # idempotent and the counter drifts at most by max_rows_per_policy.
        self._reward_log_counts: dict[str, int] = {}
        log.debug("PolicyStore initialised at %s", self._db_path)

    # ── Context manager support ──────────────────────────────────────

    def __enter__(self) -> "PolicyStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        """
        Initialise schema and run any pending migrations.

        Sequence (ISSUE-002):
          1. Create all tables (idempotent — IF NOT EXISTS).
          2. Read the stored schema version (0 if schema_version is empty).
          3. If stored version < _SCHEMA_VERSION, run each pending migration
             in a dedicated transaction and update the version counter.

        On a fresh database, step 1 creates everything at version 1 and we
        write version=1 immediately.  On an existing database that pre-dates
        the migration system, the schema_version table will be empty after
        step 1; we treat that as version 1 (matching the original schema).

        The lock is held for the entire sequence so that two threads
        opening the same DB path simultaneously cannot race on migrations.
        (RLock: safe to call _set_schema_version/_run_migrations below.)
        """
        with self._lock:
            with self._conn:
                self._conn.executescript(_SCHEMA)

            current = self._get_schema_version()
            if current == 0:
                self._set_schema_version(1)
                current = 1
                log.debug("PolicyStore: schema_version initialised to 1")

            if current < _SCHEMA_VERSION:
                self._run_migrations(current)

        log.debug("PolicyStore: schema version %d (target %d)", current, _SCHEMA_VERSION)

    def _get_schema_version(self) -> int:
        """Return the stored schema version, or 0 if the table is empty."""
        row = self._conn.execute(
            "SELECT version FROM schema_version WHERE id = 1"
        ).fetchone()
        return int(row["version"]) if row else 0

    def _set_schema_version(self, version: int) -> None:
        """Upsert the schema version record.  Called under self._lock."""
        # Note: callers (_init_schema, _run_migrations) already hold
        # self._lock.  RLock permits re-entry — no deadlock.
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO schema_version (id, version) VALUES (1, ?)",
                    (version,)
                )

    def _run_migrations(self, from_version: int) -> None:
        """
        Execute all migrations from from_version+1 up to _SCHEMA_VERSION,
        each in its own transaction.  Updates schema_version after each step
        so a crash mid-migration leaves the DB in a consistent versioned state.

        Called under self._lock from _init_schema.  RLock permits re-entry
        into _set_schema_version without deadlock.
        """
        with self._lock:
            for target in range(from_version + 1, _SCHEMA_VERSION + 1):
                statements = _MIGRATIONS.get(target, [])
                log.info(
                    "PolicyStore: running migration to schema version %d "
                    "(%d statement(s))", target, len(statements)
                )
                try:
                    with self._conn:
                        for sql in statements:
                            self._conn.execute(sql)
                except sqlite3.Error as exc:
                    log.error(
                        "PolicyStore: migration to version %d failed: %s",
                        target, exc
                    )
                    raise RuntimeError(
                        f"Schema migration to version {target} failed: {exc}"
                    ) from exc
                self._set_schema_version(target)
                log.info("PolicyStore: migrated to schema version %d", target)

    def get_schema_version(self) -> int:
        """Public accessor for the current stored schema version."""
        return self._get_schema_version()

    # ── Policy CRUD ─────────────────────────────────────────────────

    def save(self, policy: PolicyRecord) -> None:
        """Persist a PolicyRecord.  Thread-safe: acquires self._lock."""
        d = policy.to_dict()
        params = {
            **d,
            "config_json":   json.dumps(d["config"]),
            "tags_json":     json.dumps(d["tags"]),
            "metadata_json": json.dumps(d["metadata"]),
        }
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute("""
                        INSERT OR REPLACE INTO policies VALUES (
                            :policy_id,:name,:domain,:policy_type,:config_json,
                            :alpha,:beta_,:success_count,:failure_count,:version,
                            :is_active,:user_id,:tags_json,:metadata_json,
                            :created_at,:updated_at
                        )
                    """, params)
            except sqlite3.Error as exc:
                log.error(
                    "PolicyStore.save failed for policy %s (%s): %s",
                    policy.policy_id[:8], policy.name, exc
                )
                raise

    def get(self, policy_id: str) -> PolicyRecord | None:
        row = self._conn.execute(
            "SELECT * FROM policies WHERE policy_id=?", (policy_id,)
        ).fetchone()
        return self._row_to_policy(row) if row else None

    def delete(self, policy_id: str) -> bool:
        """Delete a policy by ID.  Thread-safe: acquires self._lock."""
        with self._lock:
            try:
                with self._conn:
                    cur = self._conn.execute(
                        "DELETE FROM policies WHERE policy_id=?", (policy_id,))
                return cur.rowcount > 0
            except sqlite3.Error as exc:
                log.error("PolicyStore.delete failed for %s: %s", policy_id[:8], exc)
                raise

    # ── Precomputed static SQL for all_active() (ISSUE-005) ──────────
    #
    # Dynamic WHERE clause construction via f-string interpolation is a
    # structural injection risk: safe today (all clause bodies are hardcoded
    # literals), but one future developer adding a tag filter with an f-string
    # would introduce SQL injection with no code-review signal.
    #
    # We eliminate the pattern entirely by precomputing all 8 possible
    # WHERE variants as module-level constants keyed by (domain?, type?, user?).
    # all_active() does a dict lookup — no string construction at call time.
    _ACTIVE_SQL = {
        # key: (has_domain, has_policy_type, has_user_id)
        (False, False, False): (
            "SELECT * FROM policies WHERE is_active=1"
            " ORDER BY alpha/(alpha+beta_) DESC LIMIT ?",
            []
        ),
        (False, False, True): (
            "SELECT * FROM policies WHERE is_active=1"
            " AND (user_id=? OR user_id IS NULL)"
            " ORDER BY alpha/(alpha+beta_) DESC LIMIT ?",
            ["user_id"]
        ),
        (False, True, False): (
            "SELECT * FROM policies WHERE is_active=1"
            " AND policy_type=?"
            " ORDER BY alpha/(alpha+beta_) DESC LIMIT ?",
            ["policy_type"]
        ),
        (False, True, True): (
            "SELECT * FROM policies WHERE is_active=1"
            " AND policy_type=? AND (user_id=? OR user_id IS NULL)"
            " ORDER BY alpha/(alpha+beta_) DESC LIMIT ?",
            ["policy_type", "user_id"]
        ),
        (True, False, False): (
            "SELECT * FROM policies WHERE is_active=1"
            " AND domain=?"
            " ORDER BY alpha/(alpha+beta_) DESC LIMIT ?",
            ["domain"]
        ),
        (True, False, True): (
            "SELECT * FROM policies WHERE is_active=1"
            " AND domain=? AND (user_id=? OR user_id IS NULL)"
            " ORDER BY alpha/(alpha+beta_) DESC LIMIT ?",
            ["domain", "user_id"]
        ),
        (True, True, False): (
            "SELECT * FROM policies WHERE is_active=1"
            " AND domain=? AND policy_type=?"
            " ORDER BY alpha/(alpha+beta_) DESC LIMIT ?",
            ["domain", "policy_type"]
        ),
        (True, True, True): (
            "SELECT * FROM policies WHERE is_active=1"
            " AND domain=? AND policy_type=? AND (user_id=? OR user_id IS NULL)"
            " ORDER BY alpha/(alpha+beta_) DESC LIMIT ?",
            ["domain", "policy_type", "user_id"]
        ),
    }

    def all_active(
        self,
        domain: PolicyDomain | None = None,
        policy_type: PolicyType | None = None,
        user_id: str | None = None,
        limit: int = 500,
    ) -> list[PolicyRecord]:
        """
        Return active policies matching the given filters.

        Uses precomputed static SQL strings (ISSUE-005) — no f-string
        or dynamic string construction at call time.
        """
        key = (domain is not None, policy_type is not None, user_id is not None)
        sql, param_keys = self._ACTIVE_SQL[key]

        # Map each named parameter key to its bound value
        value_map = {
            "domain":      domain.value      if domain      else None,
            "policy_type": policy_type.value if policy_type else None,
            "user_id":     user_id,
        }
        params = [value_map[k] for k in param_keys] + [limit]

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_policy(r) for r in rows]

    def count(self, domain: PolicyDomain | None = None) -> int:
        if domain:
            return self._conn.execute(
                "SELECT COUNT(*) FROM policies WHERE domain=? AND is_active=1",
                (domain.value,)
            ).fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM policies WHERE is_active=1"
        ).fetchone()[0]

    def _row_to_policy(self, row: sqlite3.Row) -> PolicyRecord:
        d = dict(row)
        d["config"]   = json.loads(d.pop("config_json", "{}"))
        d["tags"]     = json.loads(d.pop("tags_json",   "[]"))
        d["metadata"] = json.loads(d.pop("metadata_json", "{}"))
        return PolicyRecord.from_dict(d)

    # ── Version history ──────────────────────────────────────────────

    def save_version(self, version: PolicyVersion) -> None:
        """Persist a PolicyVersion snapshot.  Thread-safe: acquires self._lock."""
        d = version.to_dict()
        params = {**d, "config_json": json.dumps(d["config"]), "beta_": d["beta"]}
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute("""
                        INSERT OR IGNORE INTO policy_versions VALUES (
                            :version_id,:policy_id,:version,:config_json,
                            :alpha,:beta_,:mean_reward,:created_at,:reason
                        )
                    """, params)
            except sqlite3.Error as exc:
                log.error(
                    "PolicyStore.save_version failed for policy %s v%d: %s",
                    version.policy_id[:8], version.version, exc
                )
                raise

    def get_history(self, policy_id: str) -> list[PolicyVersion]:
        rows = self._conn.execute(
            "SELECT * FROM policy_versions WHERE policy_id=? ORDER BY version ASC",
            (policy_id,)
        ).fetchall()
        versions = []
        for row in rows:
            d = dict(row)
            d["config"] = json.loads(d.pop("config_json", "{}"))
            d["beta"] = d.pop("beta_", 1.0)   # DB column is beta_, field is beta
            versions.append(PolicyVersion.from_dict(d))
        return versions

    def rollback(self, policy_id: str, to_version: int) -> PolicyRecord | None:
        """
        Restore a policy to a previous version.

        Thread-safe: the entire read-modify-write sequence is held under
        self._lock.  RLock allows nested acquisition by save_version/save.

        Sequence (atomic from callers' perspective):
          1. Read the target PolicyVersion from history.
          2. Read the current PolicyRecord.
          3. Snapshot the current state (pre-rollback).
          4. Apply the historical state and save.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM policy_versions WHERE policy_id=? AND version=?",
                (policy_id, to_version)
            ).fetchone()
            if not rows:
                return None
            d = dict(rows)
            d["config"] = json.loads(d.pop("config_json", "{}"))
            d["beta"] = d.pop("beta_", 1.0)
            pv = PolicyVersion.from_dict(d)

            policy = self.get(policy_id)
            if not policy:
                return None

            try:
                # Snapshot current state before overwriting (re-entrant: OK)
                self.save_version(policy.snapshot(reason=f"pre-rollback to v{to_version}"))
                policy.config     = pv.config
                policy.alpha      = pv.alpha
                policy.beta_      = pv.beta
                policy.version    = to_version
                policy.updated_at = datetime.now(timezone.utc).isoformat()
                self.save(policy)  # re-entrant: OK
            except sqlite3.Error as exc:
                log.error(
                    "PolicyStore.rollback failed for %s → v%d: %s",
                    policy_id[:8], to_version, exc
                )
                raise

            log.info("PolicyStore: rolled back %s to version %d", policy_id[:8], to_version)
            return policy

    # ── Reward log ───────────────────────────────────────────────────

    def log_reward(self, reward: RewardSignal, max_rows_per_policy: int = 1000) -> None:
        """
        Append a RewardSignal to the log.  Thread-safe: acquires self._lock.

        Auto-pruning (ISSUE-006)
        ------------------------
        ``reward_log`` previously grew without bound — at 1,000 signals/day
        the table would accumulate 365,000 rows/year with no deletion.

        After every insert, this method increments a per-policy counter and
        prunes the oldest rows when the count exceeds ``max_rows_per_policy``
        (default 1,000).  Pruning deletes in batches of 10% of the limit to
        amortise the DELETE cost across many inserts.

        The counter is tracked in-memory (``_reward_log_counts``) and
        initialised lazily from the DB on first use, so it survives restarts
        only approximately — but pruning is idempotent and safe to re-run.

        Parameters
        ----------
        reward : RewardSignal
            The reward to log.
        max_rows_per_policy : int
            Maximum rows retained per policy_id.  Oldest rows are deleted
            when this limit is exceeded.  Pass ``0`` to disable auto-pruning.
        """
        pid = reward.policy_id  # may be None (broadcast reward)
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute("""
                        INSERT INTO reward_log VALUES (?,?,?,?,?,?,?,?)
                    """, (str(uuid.uuid4()), reward.reward_type.value, reward.value,
                          pid, reward.source,
                          json.dumps(reward.context), json.dumps(reward.metadata),
                          reward.timestamp))
            except sqlite3.Error as exc:
                log.error(
                    "PolicyStore.log_reward failed (type=%s, value=%.3f): %s",
                    reward.reward_type.value, reward.value, exc
                )
                raise

            # Auto-prune
            if max_rows_per_policy > 0 and pid is not None:
                self._reward_log_counts[pid] = self._reward_log_counts.get(pid, 0) + 1
                if self._reward_log_counts[pid] > max_rows_per_policy:
                    self._prune_reward_log(pid, max_rows_per_policy)

    def _prune_reward_log(self, policy_id: str, keep_last: int) -> int:
        """
        Delete the oldest rows for ``policy_id``, keeping at most ``keep_last``.

        Called under self._lock (already held by log_reward).
        Returns the number of rows deleted.
        """
        try:
            with self._conn:
                cur = self._conn.execute("""
                    DELETE FROM reward_log
                    WHERE policy_id = ?
                      AND reward_id NOT IN (
                          SELECT reward_id FROM reward_log
                          WHERE policy_id = ?
                          ORDER BY timestamp DESC
                          LIMIT ?
                      )
                """, (policy_id, policy_id, keep_last))
            deleted = cur.rowcount
            if deleted:
                # Reset counter to accurate post-prune value
                self._reward_log_counts[policy_id] = keep_last
                log.debug(
                    "PolicyStore: pruned %d old reward_log rows for policy %s",
                    deleted, policy_id[:8]
                )
            return deleted
        except sqlite3.Error as exc:
            log.warning("PolicyStore._prune_reward_log failed: %s", exc)
            return 0

    def prune_reward_log(self, policy_id: str | None = None,
                         keep_last: int = 1000) -> int:
        """
        Public API: prune the reward log for one policy or all policies.

        Thread-safe: acquires self._lock.

        Parameters
        ----------
        policy_id : str | None
            Policy to prune.  ``None`` prunes all policies.
        keep_last : int
            Number of most-recent rows to retain per policy.

        Returns
        -------
        int
            Total number of rows deleted.
        """
        with self._lock:
            if policy_id is not None:
                return self._prune_reward_log(policy_id, keep_last)

            # Prune all policies
            try:
                rows = self._conn.execute(
                    "SELECT DISTINCT policy_id FROM reward_log WHERE policy_id IS NOT NULL"
                ).fetchall()
            except sqlite3.Error as exc:
                log.warning("PolicyStore.prune_reward_log (all) failed: %s", exc)
                return 0

            total = 0
            for row in rows:
                total += self._prune_reward_log(row["policy_id"], keep_last)
            return total

    def reward_log_count(self, policy_id: str | None = None) -> int:
        """
        Return the number of reward_log rows for a policy (or total if None).
        Uses a COUNT query — no full table scan.
        """
        if policy_id is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM reward_log WHERE policy_id=?",
                (policy_id,)
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM reward_log"
            ).fetchone()
        return int(row[0]) if row else 0

    def recent_rewards(
        self,
        policy_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if policy_id:
            rows = self._conn.execute(
                "SELECT * FROM reward_log WHERE policy_id=? "
                "ORDER BY timestamp DESC LIMIT ?", (policy_id, limit)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM reward_log ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["context"]  = json.loads(d.pop("context_json",  "{}"))
            d["metadata"] = json.loads(d.pop("metadata_json", "{}"))
            results.append(d)
        return results

    def reward_stats(self, policy_id: str,
                     last_n: int = 1000) -> dict[str, float]:
        """
        Return mean, std, min, max of the most recent reward values for a policy.

        Parameters
        ----------
        policy_id : str
        last_n : int
            Consider only the most recent ``last_n`` rows (default 1,000).
            This matches the auto-prune default, so the result is always
            representative of the retained window.  Previously this method
            performed a full table scan with no LIMIT (ISSUE-006).
        """
        rows = self._conn.execute(
            "SELECT value FROM reward_log WHERE policy_id=? "
            "ORDER BY timestamp DESC LIMIT ?",
            (policy_id, last_n)
        ).fetchall()
        values = [r["value"] for r in rows]
        if not values:
            return {"mean": 0.5, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
        return {
            "mean": mean, "std": math.sqrt(variance),
            "min": min(values), "max": max(values), "count": len(values)
        }

    def close(self) -> None:
        """Close the database connection.  Thread-safe: acquires self._lock."""
        with self._lock:
            self._conn.close()
