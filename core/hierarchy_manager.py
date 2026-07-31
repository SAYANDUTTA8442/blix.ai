"""
HierarchyManager — orchestrates the memory compression pipeline.  v0.3

Manages:
    Raw MemoryEntry → SessionSummary → DailySummary → WeeklySummary

Each compression step calls the LLM to generate a human-readable summary.
Summaries are persisted to JSON files in ``memory/hierarchy/``.

Python 3.10 compatible.
"""
# DEPRECATED — core.hierarchy_manager (ISSUE-009)
#
# This module is superseded by memory.hybrid.hierarchy.hierarchy_manager.
# The class ``HierarchyManager`` here is the v0.3.x implementation;
# ``memory.hybrid.hierarchy.hierarchy_manager.HierarchyManager`` is the v0.3.15+ HGSHM implementation.
#
# These are different classes with different APIs. Callers that need
# the v0.3.15+ version must update their imports:
#
#     # Old (this file — legacy):
#     from core.hierarchy_manager import HierarchyManager
#
#     # New (HGSHM-backed):
#     from memory.hybrid.hierarchy.hierarchy_manager import HierarchyManager
#
# This file will be removed in v0.4. Do not add new callers.
# Issue: https://github.com/blix/blix/issues/9
#


from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from llm.base import LLMProvider
from schemas.memory_entry import MemoryEntry
from schemas.memory_layers import (
    DailySummary,
    MemoryLayerKind,
    SessionSummary,
    WeeklySummary,
)
from utils.logger import get_logger

log = get_logger(__name__)

_SESSION_PROMPT = """\
You are a memory summarizer. Summarize the following conversation session \
into 2-3 sentences that capture the key topics, decisions, and outcomes. \
Be specific and factual. Respond with only the summary text.

Session turns:
{turns}
"""

_DAILY_PROMPT = """\
You are a memory summarizer. Summarize the following session summaries \
for one day into 2-3 sentences. Focus on the overall theme and key work done. \
Respond with only the summary text.

Sessions:
{sessions}
"""

_WEEKLY_PROMPT = """\
You are a memory summarizer. Summarize the following daily summaries \
for one week into 2-3 sentences. Identify major themes and progress. \
Respond with only the summary text.

Days:
{days}
"""


class HierarchyManager:
    """
    Manages all levels of the memory hierarchy.

    Parameters
    ----------
    hierarchy_dir:
        Directory where ``sessions.json``, ``daily.json``, ``weekly.json``
        are stored.
    llm:
        LLM provider used for summary generation (optional — if None,
        a simple concatenation fallback is used so tests stay offline).
    """

    def __init__(self, hierarchy_dir: Path, llm: Optional[LLMProvider] = None) -> None:
        self._dir = hierarchy_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._llm = llm

        self._sessions: dict[str, SessionSummary] = {}
        self._daily: dict[str, DailySummary] = {}
        self._weekly: dict[str, WeeklySummary] = {}

        self._load_all()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        self._sessions = self._load_file("sessions.json", SessionSummary)     # type: ignore[type-var]
        self._daily = self._load_file("daily.json", DailySummary)              # type: ignore[type-var]
        self._weekly = self._load_file("weekly.json", WeeklySummary)           # type: ignore[type-var]

    def _load_file(self, filename: str, model_cls: type) -> dict:
        path = self._dir / filename
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            result = {}
            for item in raw:
                obj = model_cls.model_validate(item)
                result[obj.id] = obj
            return result
        except Exception as exc:
            log.warning("HierarchyManager: could not load %s (%s)", filename, exc)
            return {}

    def _save_file(self, filename: str, data: dict) -> None:
        path = self._dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                [v.model_dump() for v in data.values()],
                fh, indent=2, default=_json_default, ensure_ascii=False,
            )

    def _save_all(self) -> None:
        self._save_file("sessions.json", self._sessions)
        self._save_file("daily.json", self._daily)
        self._save_file("weekly.json", self._weekly)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session_summary(
        self,
        session_index: int,
        memories: list[MemoryEntry],
    ) -> SessionSummary:
        """
        Compress a list of raw memories into a ``SessionSummary``.

        Parameters
        ----------
        session_index:
            Monotonic session counter.
        memories:
            All ``MemoryEntry`` objects from this session.
        """
        session_id = f"session-{session_index}"
        turns_text = "\n".join(
            f"[{m.timestamp.strftime('%H:%M')}] User: {m.input[:120]}\nBlix: {m.output[:200]}"
            for m in memories
        )
        summary_text = self._summarize(_SESSION_PROMPT.format(turns=turns_text))

        started_at = memories[0].timestamp if memories else None
        ended_at = memories[-1].timestamp if memories else None

        ss = SessionSummary(
            id=session_id,
            session_index=session_index,
            summary=summary_text,
            source_ids=[session_id],
            raw_memory_ids=[m.id for m in memories],
            topics=self._extract_topics(memories),
            started_at=started_at,
            ended_at=ended_at,
            turn_count=len(memories),
        )
        self._sessions[session_id] = ss
        self._save_file("sessions.json", self._sessions)
        log.info("HierarchyManager: created %s (%d turns).", session_id, len(memories))
        return ss

    def roll_up_daily(self, date_str: str) -> Optional[DailySummary]:
        """
        Aggregate all session summaries for *date_str* (``YYYY-MM-DD``)
        into a ``DailySummary``.
        """
        # Find sessions whose ended_at date matches
        matching = [
            s for s in self._sessions.values()
            if s.ended_at and s.ended_at.strftime("%Y-%m-%d") == date_str
        ]
        if not matching:
            return None

        sessions_text = "\n".join(f"- {s.summary}" for s in matching)
        summary_text = self._summarize(_DAILY_PROMPT.format(sessions=sessions_text))

        daily_id = f"daily-{date_str}"
        ds = DailySummary(
            id=daily_id,
            date=date_str,
            summary=summary_text,
            session_ids=[s.id for s in matching],
            session_count=len(matching),
            topics=list({t for s in matching for t in s.topics}),
        )
        self._daily[daily_id] = ds
        self._save_file("daily.json", self._daily)
        log.info("HierarchyManager: created daily summary for %s.", date_str)
        return ds

    def roll_up_weekly(self, week_label: str) -> Optional[WeeklySummary]:
        """
        Aggregate daily summaries for *week_label* (``YYYY-WXX``) into a
        ``WeeklySummary``.
        """
        matching = [
            d for d in self._daily.values()
            if _date_to_week(d.date) == week_label
        ]
        if not matching:
            return None

        days_text = "\n".join(f"- [{d.date}] {d.summary}" for d in matching)
        summary_text = self._summarize(_WEEKLY_PROMPT.format(days=days_text))

        weekly_id = f"weekly-{week_label}"
        ws = WeeklySummary(
            id=weekly_id,
            week_label=week_label,
            summary=summary_text,
            daily_ids=[d.id for d in matching],
            daily_count=len(matching),
            topics=list({t for d in matching for t in d.topics}),
        )
        self._weekly[weekly_id] = ws
        self._save_file("weekly.json", self._weekly)
        log.info("HierarchyManager: created weekly summary for %s.", week_label)
        return ws

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> Optional[SessionSummary]:
        return self._sessions.get(session_id)

    def get_latest_sessions(self, n: int = 5) -> list[SessionSummary]:
        sessions = sorted(
            self._sessions.values(),
            key=lambda s: s.session_index,
            reverse=True,
        )
        return sessions[:n]

    def get_daily(self, date_str: str) -> Optional[DailySummary]:
        return self._daily.get(f"daily-{date_str}")

    def get_weekly(self, week_label: str) -> Optional[WeeklySummary]:
        return self._weekly.get(f"weekly-{week_label}")

    def get_hierarchy_context(self, max_sessions: int = 3) -> str:
        """
        Return a compact context string suitable for injection into prompts.

        Includes recent session summaries for close context and the latest
        weekly summary for broader background.
        """
        lines: list[str] = []
        recent = self.get_latest_sessions(max_sessions)
        if recent:
            lines.append("## Recent Session Summaries")
            for s in reversed(recent):
                lines.append(f"  [{s.session_index}] {s.summary}")

        # Latest weekly
        if self._weekly:
            latest_week = max(self._weekly.values(), key=lambda w: w.week_label)
            lines.append(f"\n## Weekly Overview ({latest_week.week_label})")
            lines.append(f"  {latest_week.summary}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def daily_count(self) -> int:
        return len(self._daily)

    @property
    def weekly_count(self) -> int:
        return len(self._weekly)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _summarize(self, prompt: str) -> str:
        """Call LLM or fall back to simple first-line extraction."""
        if self._llm is None:
            # Offline fallback: return first non-empty line of prompt body
            for line in prompt.splitlines():
                line = line.strip()
                if line and not line.startswith("You are") and len(line) > 10:
                    return line[:200]
            return "Summary unavailable (no LLM)."
        try:
            return self._llm.generate(prompt).strip()[:500]
        except Exception as exc:
            log.warning("HierarchyManager: summary generation failed (%s)", exc)
            return "Summary generation failed."

    @staticmethod
    def _extract_topics(memories: list[MemoryEntry]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for m in memories:
            for t in m.topics:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _date_to_week(date_str: str) -> str:
    """Convert ``'YYYY-MM-DD'`` to ISO week label ``'YYYY-WXX'``."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        iso_cal = d.isocalendar()
        return f"{iso_cal[0]}-W{iso_cal[1]:02d}"
    except ValueError:
        return "unknown"


def _json_default(obj: object) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not serialisable: {type(obj)!r}")
