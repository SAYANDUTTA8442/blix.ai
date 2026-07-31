"""
Confidence Manager — Blix v0.3.8  (New module 3a)

Fixes the binary-belief problem: before this module, things were either
believed or not (``belief=True``). ``ConfidenceManager`` makes confidence
a first-class quantity attached to ANY tracked thing — a belief, a plan,
a tool selection, a verification outcome — and gives every cognitive
module a uniform way to read and update it:

    belief.confidence = 0.91
    plan.confidence    = 0.74
    tool.confidence      = 0.83

Rather than re-inventing confidence storage for every kind of object,
``ConfidenceManager`` is a generic namespaced store: any caller can
register a confidence score under (namespace, ref_id) — e.g.
("belief", belief_id), ("plan", graph_id), ("tool", tool_name) — and
read it back uniformly. This is what lets confidence propagate cleanly
through ``core.state_tracker.StateTracker``, ``core.truth_manager.TruthManager``,
the Planner, the Executor, and the Verifier without each of those
modules needing their own bespoke confidence field semantics.

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
# Confidence record
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceRecord:
    """One tracked confidence score."""

    namespace: str           # "belief" | "plan" | "tool" | "state" | ...
    ref_id: str
    score: float = 0.5
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    history: list[tuple] = field(default_factory=list)   # [(score, timestamp, reason), ...]

    def to_dict(self) -> dict:
        return {
            "namespace": self.namespace,
            "ref_id": self.ref_id,
            "score": round(self.score, 4),
            "updated_at": self.updated_at,
            "history": [list(h) for h in self.history],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConfidenceRecord":
        return cls(
            namespace=d["namespace"], ref_id=d["ref_id"], score=d.get("score", 0.5),
            updated_at=d.get("updated_at", ""),
            history=[tuple(h) for h in d.get("history", [])],
        )


# ---------------------------------------------------------------------------
# Confidence Manager
# ---------------------------------------------------------------------------


class ConfidenceManager:
    """
    Generic namespaced confidence store.

    Parameters
    ----------
    confidence_file:
        Path to ``confidence_records.json``.
    decay_per_day:
        How much confidence decays per day without reinforcement (models
        "I haven't re-confirmed this in a while" uncertainty growth).
        Defaults to 0 (no decay) since most callers reinforce explicitly.
    """

    def __init__(self, confidence_file: Path, decay_per_day: float = 0.0) -> None:
        self._file = confidence_file
        self._decay = decay_per_day
        self._records: dict[tuple, ConfidenceRecord] = {}
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
                rec = ConfidenceRecord.from_dict(item)
                self._records[(rec.namespace, rec.ref_id)] = rec
            log.info("ConfidenceManager: loaded %d record(s).", len(self._records))
        except Exception as exc:
            log.warning("ConfidenceManager: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in self._records.values()], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def set(self, namespace: str, ref_id: str, score: float, reason: str = "") -> ConfidenceRecord:
        """Directly set the confidence for (namespace, ref_id)."""
        score = max(0.0, min(1.0, score))
        key = (namespace, ref_id)
        now = datetime.now(timezone.utc).isoformat()
        rec = self._records.get(key)
        if rec is None:
            rec = ConfidenceRecord(namespace=namespace, ref_id=ref_id, score=score)
            self._records[key] = rec
        else:
            rec.score = score
            rec.updated_at = now
        rec.history.append((score, now, reason))
        self._save()
        return rec

    def get(self, namespace: str, ref_id: str, default: float = 0.5) -> float:
        """Read confidence for (namespace, ref_id), applying decay if configured."""
        rec = self._records.get((namespace, ref_id))
        if rec is None:
            return default
        if self._decay <= 0:
            return rec.score
        age_days = _days_since(rec.updated_at)
        decayed = max(0.0, rec.score - self._decay * age_days)
        return decayed

    def adjust(self, namespace: str, ref_id: str, delta: float, reason: str = "") -> float:
        """Adjust confidence by a delta (positive reinforces, negative weakens)."""
        current = self.get(namespace, ref_id)
        new_score = max(0.0, min(1.0, current + delta))
        self.set(namespace, ref_id, new_score, reason=reason)
        return new_score

    def reinforce(self, namespace: str, ref_id: str, amount: float = 0.1, reason: str = "reinforced") -> float:
        return self.adjust(namespace, ref_id, amount, reason=reason)

    def weaken(self, namespace: str, ref_id: str, amount: float = 0.15, reason: str = "weakened") -> float:
        return self.adjust(namespace, ref_id, -amount, reason=reason)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def all_in_namespace(self, namespace: str) -> list[ConfidenceRecord]:
        return [r for r in self._records.values() if r.namespace == namespace]

    def low_confidence(self, namespace: Optional[str] = None, threshold: float = 0.4) -> list[ConfidenceRecord]:
        records = self._records.values()
        if namespace:
            records = [r for r in records if r.namespace == namespace]
        return [r for r in records if self.get(r.namespace, r.ref_id) < threshold]

    def mean_confidence(self, namespace: Optional[str] = None) -> float:
        records = list(self._records.values())
        if namespace:
            records = [r for r in records if r.namespace == namespace]
        if not records:
            return 0.5
        return sum(self.get(r.namespace, r.ref_id) for r in records) / len(records)

    @property
    def count(self) -> int:
        return len(self._records)


def _days_since(iso_timestamp: str) -> float:
    try:
        then = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00")).replace(tzinfo=None)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        return max(0.0, (now - then).total_seconds() / 86400.0)
    except Exception:
        return 0.0
