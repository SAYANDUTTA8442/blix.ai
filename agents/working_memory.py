"""
Working Memory — Blix v0.3.5  (Module 8)

Short-term memory for one active agent execution:

    current_task, current_state, tool_outputs, intermediate_reasoning

Entries have a TTL (steps) and are evicted automatically.
All contents are lost when the agent finishes — long-term facts are
persisted via MemoryWriteTool or ReflectionLoop.update_memory().

Python 3.10 compatible.
"""

from __future__ import annotations

from typing import Any, Optional

from agents.types import WorkingMemoryEntry
from utils.logger import get_logger

log = get_logger(__name__)


class WorkingMemory:
    """
    Key-value store with TTL-based eviction for one agent execution.

    Parameters
    ----------
    max_entries:
        Maximum number of active entries before oldest are evicted.
    default_ttl:
        Default TTL (agent steps) for new entries.
    """

    def __init__(self, max_entries: int = 50, default_ttl: int = 20) -> None:
        self._store: dict[str, WorkingMemoryEntry] = {}
        self._max = max_entries
        self._default_ttl = default_ttl
        self._step = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def set(
        self,
        key: str,
        value: Any,
        task_id: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Store a value under ``key``."""
        if len(self._store) >= self._max:
            self._evict_oldest()
        self._store[key] = WorkingMemoryEntry(
            key=key,
            value=value,
            task_id=task_id,
            ttl_steps=ttl or self._default_ttl,
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key (None if missing or expired)."""
        entry = self._store.get(key)
        if entry is None or entry.is_expired():
            return default
        return entry.value

    def has(self, key: str) -> bool:
        entry = self._store.get(key)
        return entry is not None and not entry.is_expired()

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()
        self._step = 0

    # ------------------------------------------------------------------
    # Task-scoped helpers
    # ------------------------------------------------------------------

    def set_task_output(self, task_id: str, output: str) -> None:
        """Persist the output of a completed task."""
        self.set(f"task_{task_id}_output", output, task_id=task_id, ttl=30)

    def get_task_output(self, task_id: str) -> Optional[str]:
        return self.get(f"task_{task_id}_output")

    def set_current_task(self, task_id: str, title: str) -> None:
        self.set("current_task_id", task_id, ttl=1000)
        self.set("current_task_title", title, ttl=1000)

    def get_current_task_id(self) -> Optional[str]:
        return self.get("current_task_id")

    # ------------------------------------------------------------------
    # Step tick (call once per agent step)
    # ------------------------------------------------------------------

    def tick(self) -> int:
        """
        Advance the step counter and evict expired entries.
        Returns number of entries evicted.
        """
        self._step += 1
        to_evict = [k for k, e in self._store.items() if e.is_expired()]
        for k in to_evict:
            del self._store[k]
            log.debug("WorkingMemory: evicted '%s' (expired)", k)
        for entry in self._store.values():
            entry.age_steps += 1
        if to_evict:
            log.debug("WorkingMemory: tick %d, evicted %d entry(ies).", self._step, len(to_evict))
        return len(to_evict)

    # ------------------------------------------------------------------
    # Context snapshot for tool injection
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """
        Return a flat dict of all non-expired entries for injection
        into tool ``context`` parameters.
        """
        return {k: e.value for k, e in self._store.items() if not e.is_expired()}

    # ------------------------------------------------------------------
    # Stats / helpers
    # ------------------------------------------------------------------

    def _evict_oldest(self) -> None:
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
        del self._store[oldest_key]
        log.debug("WorkingMemory: evicted oldest entry '%s'", oldest_key)

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def step(self) -> int:
        return self._step

    def summary(self) -> str:
        keys = [k for k, e in self._store.items() if not e.is_expired()]
        return f"WorkingMemory: {len(keys)} active entries, step={self._step}"
