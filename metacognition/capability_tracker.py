"""
Capability Tracker — Blix v0.3.8  (New module 5)

Tracks Blix's actual track record per domain (coding, research, math,
planning, tool usage, ...) from task outcomes, and is the thing that
keeps ``metacognition.self_model.SelfModel`` honest — capability scores
aren't hand-set once, they're continuously re-derived from results.

    capability_tracker.record_outcome("coding", success=True)
    capability_tracker.record_outcome("legal_reasoning", success=False)
    ...
    capability_tracker.accuracy("coding")          # 0.93
    capability_tracker.sync_to_self_model(self_model_store)

This module owns the raw outcome counts and accuracy computation;
``SelfModel`` owns the resulting snapshot that other modules read from.
Keeping them separate means the Self Model can also hold qualitative,
non-derived information (preferences, manually-flagged known limits)
that wouldn't make sense to compute from a success/failure ratio alone.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-domain record
# ---------------------------------------------------------------------------


@dataclass
class CapabilityRecord:
    """Outcome counts for one tracked domain."""

    domain: str
    successes: int = 0
    failures: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def accuracy(self) -> float:
        """Neutral 0.5 prior when untested, matching ToolReliabilityRegistry convention."""
        if self.total == 0:
            return 0.5
        return self.successes / self.total

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "successes": self.successes,
            "failures": self.failures,
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CapabilityRecord":
        return cls(
            domain=d["domain"], successes=d.get("successes", 0), failures=d.get("failures", 0),
            last_updated=d.get("last_updated", ""),
        )


# ---------------------------------------------------------------------------
# Capability Tracker
# ---------------------------------------------------------------------------

# Default tracked domains, matching the spec's example set. Callers can
# record outcomes for any domain string; these are just the ones
# pre-seeded so `accuracy()` returns a neutral prior immediately rather
# than requiring a first observation.
_DEFAULT_DOMAINS = [
    "coding", "research", "math", "planning_accuracy", "tool_usage_accuracy",
]


class CapabilityTracker:
    """
    Tracks per-domain task outcomes and derives accuracy scores.

    Parameters
    ----------
    capability_file:
        Path to ``capability_tracker.json``.
    min_samples_for_confidence:
        How many recorded outcomes are needed before an accuracy score
        is considered reliable enough to act on (mirrors
        ``ToolReliabilityRegistry.is_confident()``).
    """

    def __init__(self, capability_file: Path, min_samples_for_confidence: int = 5) -> None:
        self._file = capability_file
        self._min_samples = min_samples_for_confidence
        self._records: dict[str, CapabilityRecord] = {}
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
                rec = CapabilityRecord.from_dict(item)
                self._records[rec.domain] = rec
            log.info("CapabilityTracker: loaded %d domain(s).", len(self._records))
        except Exception as exc:
            log.warning("CapabilityTracker: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in self._records.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_outcome(self, domain: str, success: bool) -> CapabilityRecord:
        """Record one task outcome (success/failure) for a domain. Updated after every task."""
        domain = domain.lower().strip()
        rec = self._records.get(domain)
        if rec is None:
            rec = CapabilityRecord(domain=domain)
            self._records[domain] = rec
        if success:
            rec.successes += 1
        else:
            rec.failures += 1
        rec.last_updated = datetime.now(timezone.utc).isoformat()
        self._save()
        return rec

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def accuracy(self, domain: str) -> float:
        domain = domain.lower().strip()
        rec = self._records.get(domain)
        return rec.accuracy if rec is not None else 0.5

    def is_confident(self, domain: str) -> bool:
        domain = domain.lower().strip()
        rec = self._records.get(domain)
        return rec is not None and rec.total >= self._min_samples

    def all_records(self) -> list[CapabilityRecord]:
        return list(self._records.values())

    def weakest_domains(self, top_k: int = 5) -> list[CapabilityRecord]:
        """Lowest-accuracy domains with enough samples to trust the score."""
        confident = [r for r in self._records.values() if r.total >= self._min_samples]
        return sorted(confident, key=lambda r: r.accuracy)[:top_k]

    def strongest_domains(self, top_k: int = 5) -> list[CapabilityRecord]:
        confident = [r for r in self._records.values() if r.total >= self._min_samples]
        return sorted(confident, key=lambda r: -r.accuracy)[:top_k]

    @property
    def tracked_domain_count(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Sync to SelfModel
    # ------------------------------------------------------------------

    def sync_to_self_model(self, self_model_store) -> int:
        """
        Push every confidently-measured domain accuracy into a
        ``metacognition.self_model.SelfModelStore``.

        Only syncs domains with enough samples (``min_samples_for_confidence``)
        so the Self Model doesn't get noisy single-observation scores.
        Returns the number of domains synced.
        """
        synced = 0
        for rec in self._records.values():
            if rec.total >= self._min_samples:
                self_model_store.set_capability(rec.domain, rec.accuracy)
                synced += 1
        return synced
