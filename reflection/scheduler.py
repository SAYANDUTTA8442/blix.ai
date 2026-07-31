"""
Reflection Scheduler — Blix v0.3.2  (Feature 9)

Automatic cognitive maintenance: determines when each reflection scope
is "due" and dispatches the appropriate ``ReflectionEngine`` calls.

Schedule
--------
    Every session  → Session Reflection
    Every day      → Daily Reflection
    Every week     → Weekly Reflection
    Every month    → Deep Reflection (multi-week synthesis)

Design
------
``ReflectionScheduler`` is intentionally synchronous and side-effect-free
in its scheduling decisions (``due_*`` methods return booleans); the
actual reflection work is dispatched via small callables so it can be
run either inline or submitted to ``BackgroundProcessor`` (v0.3 Feature 6).

State (last-run timestamps) persists to ``reflection_schedule.json``.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schedule state
# ---------------------------------------------------------------------------


@dataclass
class ScheduleState:
    """Tracks the last time each reflection scope ran."""

    last_session: Optional[str] = None
    last_daily: Optional[str] = None
    last_weekly: Optional[str] = None
    last_monthly: Optional[str] = None
    session_count_since_daily: int = 0

    def to_dict(self) -> dict:
        return {
            "last_session": self.last_session,
            "last_daily": self.last_daily,
            "last_weekly": self.last_weekly,
            "last_monthly": self.last_monthly,
            "session_count_since_daily": self.session_count_since_daily,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduleState":
        return cls(
            last_session=d.get("last_session"),
            last_daily=d.get("last_daily"),
            last_weekly=d.get("last_weekly"),
            last_monthly=d.get("last_monthly"),
            session_count_since_daily=d.get("session_count_since_daily", 0),
        )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class ReflectionScheduler:
    """
    Determines when scheduled reflections are due and tracks last-run times.

    Parameters
    ----------
    schedule_file:
        Path to ``reflection_schedule.json``.
    now_fn:
        Callable returning the current ``datetime`` (UTC, naive). Injectable
        for testing; defaults to ``datetime.now(timezone.utc)``.
    """

    def __init__(
        self,
        schedule_file: Path,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._file = schedule_file
        self._now = now_fn or (lambda: datetime.now(timezone.utc).replace(tzinfo=None))
        self._state = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> ScheduleState:
        if not self._file.exists():
            return ScheduleState()
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                return ScheduleState.from_dict(json.load(fh))
        except Exception as exc:
            log.warning("ReflectionScheduler: load failed (%s)", exc)
            return ScheduleState()

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump(self._state.to_dict(), fh, indent=2)

    # ------------------------------------------------------------------
    # Due checks
    # ------------------------------------------------------------------

    def due_session(self) -> bool:
        """Every session is always due for session reflection."""
        return True

    def due_daily(self) -> bool:
        """Due once per calendar day (UTC)."""
        return self._is_due(self._state.last_daily, lambda dt: dt.date() != self._now().date())

    def due_weekly(self) -> bool:
        """Due once per ISO week."""
        def changed(dt: datetime) -> bool:
            return dt.isocalendar()[:2] != self._now().isocalendar()[:2]
        return self._is_due(self._state.last_weekly, changed)

    def due_monthly(self) -> bool:
        """Due once per calendar month."""
        def changed(dt: datetime) -> bool:
            now = self._now()
            return (dt.year, dt.month) != (now.year, now.month)
        return self._is_due(self._state.last_monthly, changed)

    def _is_due(self, last: Optional[str], changed: Callable[[datetime], bool]) -> bool:
        if last is None:
            return True
        try:
            dt = datetime.fromisoformat(last)
        except ValueError:
            return True
        return changed(dt)

    # ------------------------------------------------------------------
    # Mark-run
    # ------------------------------------------------------------------

    def mark_session_run(self) -> None:
        now = self._now()
        self._state.last_session = now.isoformat()
        self._state.session_count_since_daily += 1
        self._save()

    def mark_daily_run(self) -> None:
        self._state.last_daily = self._now().isoformat()
        self._state.session_count_since_daily = 0
        self._save()

    def mark_weekly_run(self) -> None:
        self._state.last_weekly = self._now().isoformat()
        self._save()

    def mark_monthly_run(self) -> None:
        self._state.last_monthly = self._now().isoformat()
        self._save()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def run_due(
        self,
        on_session: Optional[Callable[[], None]] = None,
        on_daily: Optional[Callable[[], None]] = None,
        on_weekly: Optional[Callable[[], None]] = None,
        on_monthly: Optional[Callable[[], None]] = None,
    ) -> list[str]:
        """
        Check all schedules and invoke the corresponding callback for any
        that are due, marking them as run.

        Returns the list of scope names that were triggered, e.g.
        ``["session", "daily"]``.

        Each callback is wrapped in try/except so a failure in one
        reflection scope doesn't prevent others from running (failure
        isolation, consistent with ``BackgroundProcessor``).
        """
        triggered: list[str] = []

        if self.due_session() and on_session is not None:
            self._safe_run("session", on_session)
            self.mark_session_run()
            triggered.append("session")
        elif self.due_session():
            self.mark_session_run()
            triggered.append("session")

        if self.due_daily():
            if on_daily is not None:
                self._safe_run("daily", on_daily)
            self.mark_daily_run()
            triggered.append("daily")

        if self.due_weekly():
            if on_weekly is not None:
                self._safe_run("weekly", on_weekly)
            self.mark_weekly_run()
            triggered.append("weekly")

        if self.due_monthly():
            if on_monthly is not None:
                self._safe_run("monthly", on_monthly)
            self.mark_monthly_run()
            triggered.append("monthly")

        if triggered:
            log.info("ReflectionScheduler: triggered %s", ", ".join(triggered))
        return triggered

    def _safe_run(self, scope: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            log.exception("ReflectionScheduler: %s reflection failed", scope)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def state(self) -> ScheduleState:
        return self._state
