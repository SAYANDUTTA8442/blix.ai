"""
Tool Reliability Registry — Blix v0.3.6  (Upgrade 5)

Persists tool success-rates ACROSS agent runs (complementing the
per-run ``ToolReliabilityStats`` in ``agents.state.AgentState``, which
resets every run).

    {"tool": "web_search", "success_rate": 0.92}

The Tool Selection Engine (v0.3.5 ``ToolRegistry.select_tool``) can
combine this with ``can_handle()`` scores so historically unreliable
tools are deprioritised even if they look like a good keyword match.

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
# Persistent reliability record
# ---------------------------------------------------------------------------


@dataclass
class ToolReliabilityRecord:
    """Cross-run reliability tracking for one tool."""

    tool_name: str
    successes: int = 0
    failures: int = 0
    total_duration_ms: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.5  # neutral prior for untested tools
        return self.successes / self.total

    @property
    def mean_duration_ms(self) -> float:
        return self.total_duration_ms / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "tool": self.tool_name,
            "success_rate": round(self.success_rate, 4),
            "successes": self.successes,
            "failures": self.failures,
            "total": self.total,
            "mean_duration_ms": round(self.mean_duration_ms, 1),
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolReliabilityRecord":
        return cls(
            tool_name=d["tool"],
            successes=d.get("successes", 0),
            failures=d.get("failures", 0),
            total_duration_ms=d.get("mean_duration_ms", 0.0) * d.get("total", 0),
            last_updated=d.get("last_updated", ""),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolReliabilityRegistry:
    """
    Persistent cross-run tool reliability tracker.

    Parameters
    ----------
    reliability_file:
        Path to ``tool_reliability.json``.
    min_samples_for_confidence:
        Below this many observations, reliability scores are treated as
        provisional (the Tool Selection Engine should not heavily
        penalise a tool that's only failed once or twice).
    """

    def __init__(
        self,
        reliability_file: Path,
        min_samples_for_confidence: int = 5,
    ) -> None:
        self._file = reliability_file
        self._min_samples = min_samples_for_confidence
        self._records: dict[str, ToolReliabilityRecord] = {}
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
                rec = ToolReliabilityRecord.from_dict(item)
                self._records[rec.tool_name] = rec
            log.info("ToolReliabilityRegistry: loaded %d tool record(s).", len(self._records))
        except Exception as exc:
            log.warning("ToolReliabilityRegistry: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in self._records.values()], fh, indent=2)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, tool_name: str, success: bool, duration_ms: float = 0.0) -> None:
        """Record one tool execution outcome."""
        rec = self._records.setdefault(tool_name, ToolReliabilityRecord(tool_name=tool_name))
        if success:
            rec.successes += 1
        else:
            rec.failures += 1
        rec.total_duration_ms += duration_ms
        rec.last_updated = datetime.now(timezone.utc).isoformat()
        self._save()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, tool_name: str) -> Optional[ToolReliabilityRecord]:
        return self._records.get(tool_name)

    def success_rate(self, tool_name: str) -> float:
        """Reliability score, 0.5 (neutral) if unseen or under-sampled."""
        rec = self._records.get(tool_name)
        if rec is None:
            return 0.5
        return rec.success_rate

    def is_confident(self, tool_name: str) -> bool:
        """Whether enough samples exist to trust this tool's reliability score."""
        rec = self._records.get(tool_name)
        return rec is not None and rec.total >= self._min_samples

    def rank_tools_by_reliability(self, tool_names: list[str]) -> list[tuple[str, float]]:
        """Return (tool_name, success_rate) sorted descending."""
        scored = [(name, self.success_rate(name)) for name in tool_names]
        return sorted(scored, key=lambda t: -t[1])

    def least_reliable(self, top_k: int = 5) -> list[ToolReliabilityRecord]:
        confident = [r for r in self._records.values() if r.total >= self._min_samples]
        return sorted(confident, key=lambda r: r.success_rate)[:top_k]

    def most_reliable(self, top_k: int = 5) -> list[ToolReliabilityRecord]:
        confident = [r for r in self._records.values() if r.total >= self._min_samples]
        return sorted(confident, key=lambda r: -r.success_rate)[:top_k]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def tracked_tool_count(self) -> int:
        return len(self._records)

    def all_records(self) -> list[ToolReliabilityRecord]:
        return list(self._records.values())
