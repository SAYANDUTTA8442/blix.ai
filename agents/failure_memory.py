"""
Failure Memory — Blix v0.3.6  (Upgrade 4)

Persists structured failure records so future planning avoids repeating
known mistakes:

    {
      "task": "build api",
      "failure": "schema mismatch",
      "fix": "update response model"
    }

Without this, failures are forgotten the moment a run ends (the v0.3.5
``ExecutionHistoryEntry`` log records *that* something failed but isn't
queried by the planner). ``FailureMemory`` is queried by:

* ``planning.replanner.Replanner`` — to choose an alternative tool/approach
* ``planning.critic.PlanCritic``   — to flag tasks resembling past failures

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Failure record
# ---------------------------------------------------------------------------


@dataclass
class FailureRecord:
    """
    One structured failure observation.

    Fields
    ------
    task_title:
        The task title/description that failed (normalised for matching).
    tool:
        The tool that was used when the failure occurred.
    failure:
        Short description of what went wrong.
    fix:
        Suggested or applied fix, if known (filled in by the Replanner
        once a retry succeeds with a different approach).
    goal:
        The parent goal, for context.
    occurrences:
        How many times this same failure pattern has been seen.
    """

    task_title: str
    tool: str
    failure: str
    fix: str = ""
    goal: str = ""
    occurrences: int = 1
    first_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "task_title": self.task_title,
            "tool": self.tool,
            "failure": self.failure,
            "fix": self.fix,
            "goal": self.goal,
            "occurrences": self.occurrences,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FailureRecord":
        return cls(
            task_title=d["task_title"],
            tool=d.get("tool", ""),
            failure=d.get("failure", ""),
            fix=d.get("fix", ""),
            goal=d.get("goal", ""),
            occurrences=d.get("occurrences", 1),
            first_seen=d.get("first_seen", ""),
            last_seen=d.get("last_seen", ""),
        )


# ---------------------------------------------------------------------------
# Failure Memory store
# ---------------------------------------------------------------------------


class FailureMemory:
    """
    Persistent store of failure records, queryable by task similarity.

    Parameters
    ----------
    failures_file:
        Path to ``failure_memory.json``.
    similarity_threshold:
        Token-overlap (Jaccard) threshold for considering two task
        descriptions "the same kind of task" for failure lookup.
    """

    def __init__(
        self,
        failures_file: Path,
        similarity_threshold: float = 0.4,
    ) -> None:
        self._file = failures_file
        self._threshold = similarity_threshold
        self._records: list[FailureRecord] = []
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
            self._records = [FailureRecord.from_dict(r) for r in raw]
            log.info("FailureMemory: loaded %d record(s).", len(self._records))
        except Exception as exc:
            log.warning("FailureMemory: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in self._records], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        task_title: str,
        tool: str,
        failure: str,
        goal: str = "",
        fix: str = "",
    ) -> FailureRecord:
        """
        Record a failure, merging into an existing similar record if found
        (incrementing ``occurrences``), else creating a new one.
        """
        existing = self._find_similar(task_title, tool)
        if existing is not None:
            existing.occurrences += 1
            existing.last_seen = datetime.now(timezone.utc).isoformat()
            if fix and not existing.fix:
                existing.fix = fix
            self._save()
            return existing

        record = FailureRecord(task_title=task_title, tool=tool, failure=failure, goal=goal, fix=fix)
        self._records.append(record)
        self._save()
        log.info("FailureMemory: recorded new failure for task '%s' (tool=%s)", task_title, tool)
        return record

    def record_fix(self, task_title: str, tool: str, fix: str) -> None:
        """Attach a known-good fix to an existing failure record."""
        existing = self._find_similar(task_title, tool)
        if existing is not None:
            existing.fix = fix
            self._save()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def similar_failures(self, task_title: str, tool: Optional[str] = None) -> list[FailureRecord]:
        """
        Return all failure records whose task_title is similar to the
        given task, optionally filtered to a specific tool.
        """
        results = []
        for r in self._records:
            if tool is not None and r.tool != tool:
                continue
            if _jaccard(task_title, r.task_title) >= self._threshold:
                results.append(r)
        return sorted(results, key=lambda r: -r.occurrences)

    def has_known_failure(self, task_title: str, tool: Optional[str] = None) -> bool:
        return len(self.similar_failures(task_title, tool)) > 0

    def suggest_fix(self, task_title: str, tool: Optional[str] = None) -> Optional[str]:
        """Return the most relevant known fix, if any."""
        matches = [r for r in self.similar_failures(task_title, tool) if r.fix]
        if not matches:
            return None
        return matches[0].fix

    def _find_similar(self, task_title: str, tool: str) -> Optional[FailureRecord]:
        for r in self._records:
            if r.tool == tool and _jaccard(task_title, r.task_title) >= self._threshold:
                return r
        return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._records)

    def most_common_failures(self, top_k: int = 5) -> list[FailureRecord]:
        return sorted(self._records, key=lambda r: -r.occurrences)[:top_k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_STOP = {"a", "an", "the", "is", "are", "to", "for", "of", "in", "on", "and",
         "or", "with", "create", "build", "make"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0
