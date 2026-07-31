"""
Memory Lifecycle & Forgetting — Blix v0.3.1  (Issue 2)

Addresses: "You never truly forget memories."

Implements a four-stage lifecycle:

    active → compressed → archived → deleted

Each ``MemoryEntry`` gains a ``lifecycle_state`` field.
``MemoryLifecycleManager`` drives transitions based on configurable rules.

Forgetting policy (Ebbinghaus-inspired)
----------------------------------------
* ``active``     — normal retrieval, full embedding, full prompt injection.
* ``compressed`` — raw text replaced with its extracted_facts summary;
                   embedding preserved; still retrievable.
* ``archived``   — removed from hot retrieval pool; stored in cold archive;
                   can be restored if explicitly queried.
* ``deleted``    — permanently removed from all storage.

Transition triggers
-------------------
* active → compressed :  age > compress_after_days AND access_count < compress_min_access
* compressed → archived: age > archive_after_days AND access_count < archive_min_access
* archived → deleted   : age > delete_after_days (hard cutoff)

All thresholds are configurable.  Defaults are generous (compress: 90d,
archive: 365d, delete: never by default).

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifecycle state
# ---------------------------------------------------------------------------


class LifecycleState(str, Enum):
    ACTIVE = "active"
    COMPRESSED = "compressed"
    ARCHIVED = "archived"
    DELETED = "deleted"


# ---------------------------------------------------------------------------
# Policy config
# ---------------------------------------------------------------------------


@dataclass
class ForgettingPolicy:
    """
    Thresholds that drive lifecycle transitions.

    Set ``delete_after_days=None`` to disable permanent deletion.
    """

    compress_after_days: float = 90.0
    compress_min_access: int = 3        # don't compress if accessed ≥ this many times
    archive_after_days: float = 365.0
    archive_min_access: int = 1         # don't archive if recently accessed
    delete_after_days: Optional[float] = None   # None = never delete
    importance_protect_threshold: float = 0.8   # never compress/archive high-importance memories


# ---------------------------------------------------------------------------
# Lifecycle record (stored alongside MemoryEntry)
# ---------------------------------------------------------------------------


@dataclass
class LifecycleRecord:
    """Tracks lifecycle metadata for one MemoryEntry."""

    memory_id: int
    state: LifecycleState = LifecycleState.ACTIVE
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    compressed_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    compressed_summary: str = ""   # replaces raw text when compressed

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "state": self.state.value,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "compressed_at": self.compressed_at.isoformat() if self.compressed_at else None,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
            "compressed_summary": self.compressed_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LifecycleRecord":
        def _dt(v: Optional[str]) -> Optional[datetime]:
            return datetime.fromisoformat(v) if v else None
        return cls(
            memory_id=d["memory_id"],
            state=LifecycleState(d.get("state", "active")),
            access_count=d.get("access_count", 0),
            last_accessed=_dt(d.get("last_accessed")),
            compressed_at=_dt(d.get("compressed_at")),
            archived_at=_dt(d.get("archived_at")),
            deleted_at=_dt(d.get("deleted_at")),
            compressed_summary=d.get("compressed_summary", ""),
        )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class MemoryLifecycleManager:
    """
    Manages the lifecycle of all MemoryEntry objects.

    Parameters
    ----------
    lifecycle_file:
        Path to ``lifecycle.json`` where LifecycleRecord objects are persisted.
    policy:
        ``ForgettingPolicy`` controlling thresholds.
    """

    def __init__(
        self,
        lifecycle_file: Path,
        policy: Optional[ForgettingPolicy] = None,
    ) -> None:
        self._file = lifecycle_file
        self._policy = policy or ForgettingPolicy()
        self._records: dict[int, LifecycleRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for item in raw:
                rec = LifecycleRecord.from_dict(item)
                self._records[rec.memory_id] = rec
            log.info("LifecycleManager: loaded %d records.", len(self._records))
        except Exception as exc:
            log.warning("LifecycleManager: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in self._records.values()], fh, indent=2)

    # ------------------------------------------------------------------
    # Record management
    # ------------------------------------------------------------------

    def get_or_create(self, memory_id: int) -> LifecycleRecord:
        if memory_id not in self._records:
            self._records[memory_id] = LifecycleRecord(memory_id=memory_id)
        return self._records[memory_id]

    def record_access(self, memory_id: int) -> None:
        """Increment access count and update last_accessed timestamp."""
        rec = self.get_or_create(memory_id)
        rec.access_count += 1
        rec.last_accessed = datetime.now(timezone.utc).replace(tzinfo=None)
        self._save()

    def get_state(self, memory_id: int) -> LifecycleState:
        return self.get_or_create(memory_id).state

    def get_access_count(self, memory_id: int) -> int:
        return self.get_or_create(memory_id).access_count

    # ------------------------------------------------------------------
    # Transition API
    # ------------------------------------------------------------------

    def compress(self, memory_id: int, summary: str) -> None:
        """Transition a memory to the COMPRESSED state."""
        rec = self.get_or_create(memory_id)
        if rec.state != LifecycleState.ACTIVE:
            return
        rec.state = LifecycleState.COMPRESSED
        rec.compressed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        rec.compressed_summary = summary
        self._save()
        log.info("Lifecycle: memory %d compressed.", memory_id)

    def archive(self, memory_id: int) -> None:
        """Transition a memory to the ARCHIVED state."""
        rec = self.get_or_create(memory_id)
        if rec.state not in (LifecycleState.ACTIVE, LifecycleState.COMPRESSED):
            return
        rec.state = LifecycleState.ARCHIVED
        rec.archived_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self._save()
        log.info("Lifecycle: memory %d archived.", memory_id)

    def delete(self, memory_id: int) -> None:
        """Mark a memory as DELETED."""
        rec = self.get_or_create(memory_id)
        rec.state = LifecycleState.DELETED
        rec.deleted_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self._save()
        log.info("Lifecycle: memory %d deleted.", memory_id)

    def restore(self, memory_id: int) -> None:
        """Restore an archived memory to ACTIVE."""
        rec = self.get_or_create(memory_id)
        if rec.state == LifecycleState.ARCHIVED:
            rec.state = LifecycleState.ACTIVE
            self._save()
            log.info("Lifecycle: memory %d restored to active.", memory_id)

    # ------------------------------------------------------------------
    # Batch GC pass (run periodically, e.g. on session end)
    # ------------------------------------------------------------------

    def run_gc(
        self,
        memories: list,  # list[MemoryEntry] — typed as list to avoid circular import
    ) -> dict[str, list[int]]:
        """
        Run one garbage-collection pass over all memories.

        Returns a report dict with lists of ids that were transitioned.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        p = self._policy
        report: dict[str, list[int]] = {
            "compressed": [], "archived": [], "deleted": [],
        }

        for m in memories:
            age_days = (now - m.timestamp).total_seconds() / 86400.0
            rec = self.get_or_create(m.id)
            importance = m.importance or 0.0

            # Never lifecycle high-importance memories
            if importance >= p.importance_protect_threshold:
                continue

            if rec.state == LifecycleState.ACTIVE:
                if (
                    age_days > p.compress_after_days
                    and rec.access_count < p.compress_min_access
                ):
                    summary = " ".join(m.extracted_facts[:3]) or m.output[:200]
                    self.compress(m.id, summary)
                    report["compressed"].append(m.id)

            elif rec.state == LifecycleState.COMPRESSED:
                if (
                    age_days > p.archive_after_days
                    and rec.access_count < p.archive_min_access
                ):
                    self.archive(m.id)
                    report["archived"].append(m.id)

            elif rec.state == LifecycleState.ARCHIVED:
                if p.delete_after_days and age_days > p.delete_after_days:
                    self.delete(m.id)
                    report["deleted"].append(m.id)

        if any(report.values()):
            log.info(
                "Lifecycle GC: compressed=%d archived=%d deleted=%d",
                len(report["compressed"]), len(report["archived"]), len(report["deleted"]),
            )
        return report

    # ------------------------------------------------------------------
    # Hot pool filter (used by SemanticRetriever / MemoryManager)
    # ------------------------------------------------------------------

    def filter_active(self, memories: list) -> list:
        """Return only memories in ACTIVE or COMPRESSED state."""
        allowed = {LifecycleState.ACTIVE, LifecycleState.COMPRESSED}
        return [m for m in memories if self.get_state(m.id) in allowed]

    def filter_archived(self, memories: list) -> list:
        """Return only ARCHIVED memories (for cold search)."""
        return [m for m in memories if self.get_state(m.id) == LifecycleState.ARCHIVED]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def state_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {s.value: 0 for s in LifecycleState}
        for rec in self._records.values():
            counts[rec.state.value] += 1
        return counts
