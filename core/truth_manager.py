"""
Truth Maintenance Engine — Blix v0.3.7  (New module 3)

Introduces ``TruthStatus`` — the missing dimension from v0.3.1's binary
winner/loser contradiction handling:

    ACTIVE       — currently believed true
    SUPERSEDED   — was true, has been explicitly replaced by a newer belief
    HISTORICAL   — was true for a known time window, naturally ended (no conflict)
    CONFLICTING  — currently unresolved; evidence exists on multiple sides
    ARCHIVED     — no longer relevant to surface at all (manually retired)

``TruthManager`` owns the ``TruthRecord`` store (one per Belief or
StateSnapshot id) and exposes the four operations the spec calls for:

    replace(old_id, new_id)   — old → SUPERSEDED, new → ACTIVE
    merge(id_a, id_b)          — collapse near-duplicate beliefs into one
    archive(id)                 — retire a record from active consideration
    resolve(id, status)          — direct status assignment (e.g. CONFLICTING → ACTIVE)

This module is intentionally storage-agnostic about WHAT it's tracking
truth for — it works against any id string, so both
``memory.beliefs.Belief`` and ``core.state_tracker.StateSnapshot`` can
share one ``TruthManager`` instance.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# TruthStatus
# ---------------------------------------------------------------------------


class TruthStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    HISTORICAL = "historical"
    CONFLICTING = "conflicting"
    ARCHIVED = "archived"


# ---------------------------------------------------------------------------
# TruthRecord
# ---------------------------------------------------------------------------


@dataclass
class TruthRecord:
    """
    Tracks the ``TruthStatus`` of one belief/snapshot id over time.

    Fields
    ------
    record_id:
        The id of the thing this truth record is about (a Belief id or
        StateSnapshot id — caller's choice of namespace).
    status:
        Current ``TruthStatus``.
    superseded_by:
        If status is SUPERSEDED, the id of the record that replaced it.
    merged_into:
        If this record was merged away, the id of the surviving record.
    history:
        Chronological list of (status, timestamp, note) status changes.
    """

    record_id: str
    status: TruthStatus = TruthStatus.ACTIVE
    superseded_by: Optional[str] = None
    merged_into: Optional[str] = None
    history: list[tuple] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "status": self.status.value,
            "superseded_by": self.superseded_by,
            "merged_into": self.merged_into,
            "history": [list(h) for h in self.history],
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TruthRecord":
        return cls(
            record_id=d["record_id"],
            status=TruthStatus(d.get("status", "active")),
            superseded_by=d.get("superseded_by"),
            merged_into=d.get("merged_into"),
            history=[tuple(h) for h in d.get("history", [])],
            updated_at=d.get("updated_at", ""),
        )


# ---------------------------------------------------------------------------
# Truth Manager
# ---------------------------------------------------------------------------


class TruthManager:
    """
    Maintains ``TruthRecord`` for every tracked belief/snapshot id and
    performs the four core truth-maintenance operations.

    Parameters
    ----------
    truth_file:
        Path to ``truth_records.json``.
    """

    def __init__(self, truth_file: Path) -> None:
        self._file = truth_file
        self._records: dict[str, TruthRecord] = {}
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
                rec = TruthRecord.from_dict(item)
                self._records[rec.record_id] = rec
            log.info("TruthManager: loaded %d record(s).", len(self._records))
        except Exception as exc:
            log.warning("TruthManager: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in self._records.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Registration / lookup
    # ------------------------------------------------------------------

    def ensure(self, record_id: str, initial_status: TruthStatus = TruthStatus.ACTIVE) -> TruthRecord:
        """Get the TruthRecord for ``record_id``, creating it if absent."""
        if record_id not in self._records:
            self._records[record_id] = TruthRecord(record_id=record_id, status=initial_status)
            self._save()
        return self._records[record_id]

    def get(self, record_id: str) -> Optional[TruthRecord]:
        return self._records.get(record_id)

    def status_of(self, record_id: str) -> TruthStatus:
        rec = self._records.get(record_id)
        return rec.status if rec else TruthStatus.ACTIVE

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def replace(self, old_id: str, new_id: str, note: str = "") -> None:
        """
        ``old_id`` → SUPERSEDED, ``new_id`` → ACTIVE.

        Used when a newer belief/state cleanly replaces an older one
        (the "Delhi → Kolkata" case — Replacement, not parallel truth).
        """
        old = self.ensure(old_id)
        new = self.ensure(new_id)
        self._set_status(old, TruthStatus.SUPERSEDED, note or f"superseded by {new_id}")
        old.superseded_by = new_id
        self._set_status(new, TruthStatus.ACTIVE, note or f"replaces {old_id}")
        self._save()
        log.info("TruthManager: replace — %s SUPERSEDED by %s", old_id, new_id)

    def merge(self, id_a: str, id_b: str, surviving_id: Optional[str] = None, note: str = "") -> str:
        """
        Collapse two near-duplicate beliefs into one (the "AI" /
        "Artificial Intelligence" case).

        Returns the surviving record_id. If ``surviving_id`` is not
        given, ``id_a`` survives.
        """
        survivor = surviving_id or id_a
        loser = id_b if survivor == id_a else id_a

        survivor_rec = self.ensure(survivor)
        loser_rec = self.ensure(loser)

        self._set_status(survivor_rec, TruthStatus.ACTIVE, note or f"merged with {loser}")
        self._set_status(loser_rec, TruthStatus.SUPERSEDED, note or f"merged into {survivor}")
        loser_rec.merged_into = survivor
        loser_rec.superseded_by = survivor
        self._save()
        log.info("TruthManager: merge — %s and %s → survivor=%s", id_a, id_b, survivor)
        return survivor

    def archive(self, record_id: str, note: str = "") -> None:
        """Retire a record from active consideration without declaring a winner."""
        rec = self.ensure(record_id)
        self._set_status(rec, TruthStatus.ARCHIVED, note or "archived")
        self._save()
        log.info("TruthManager: archive — %s", record_id)

    def resolve(self, record_id: str, status: TruthStatus, note: str = "") -> None:
        """Directly assign a status (e.g. moving CONFLICTING → ACTIVE once resolved)."""
        rec = self.ensure(record_id)
        self._set_status(rec, status, note)
        self._save()
        log.info("TruthManager: resolve — %s → %s", record_id, status.value)

    def mark_conflicting(self, id_a: str, id_b: str, note: str = "") -> None:
        """Flag two records as being in unresolved conflict (needs evidence comparison)."""
        rec_a = self.ensure(id_a)
        rec_b = self.ensure(id_b)
        self._set_status(rec_a, TruthStatus.CONFLICTING, note or f"conflicts with {id_b}")
        self._set_status(rec_b, TruthStatus.CONFLICTING, note or f"conflicts with {id_a}")
        self._save()
        log.info("TruthManager: marked conflicting — %s vs %s", id_a, id_b)

    def mark_historical(self, record_id: str, note: str = "") -> None:
        """Mark a record as HISTORICAL — was true, ended naturally, no conflict involved."""
        rec = self.ensure(record_id)
        self._set_status(rec, TruthStatus.HISTORICAL, note or "naturally ended")
        self._save()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_status(self, rec: TruthRecord, status: TruthStatus, note: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        rec.status = status
        rec.updated_at = now
        rec.history.append((status.value, now, note))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def all_with_status(self, status: TruthStatus) -> list[TruthRecord]:
        return [r for r in self._records.values() if r.status == status]

    def is_active(self, record_id: str) -> bool:
        return self.status_of(record_id) == TruthStatus.ACTIVE

    @property
    def count(self) -> int:
        return len(self._records)
