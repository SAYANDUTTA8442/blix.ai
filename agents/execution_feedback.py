"""
Execution Feedback Loop — Blix v0.3.8  (New module 8)

The connective tissue between "a run just finished" and "Blix's
self-knowledge updated accordingly". Every ``AgentRunResult`` (and,
optionally, individual task-level outcomes) gets distilled into a
compact ``FeedbackEntry`` (success, failure, duration, confidence) and
fanned out to:

    agents.failure_memory.FailureMemory          (already existed, v0.3.6)
    metacognition.capability_tracker.CapabilityTracker  (NEW, v0.3.8)
    metacognition.self_model.SelfModelStore        (NEW, v0.3.8, via CapabilityTracker sync)

This module does not duplicate what those three already do — it is
purely the feedback-routing layer that decides WHAT gets reported to
WHOM after a run, so ``AgentExecutor`` doesn't need to know about
metacognition internals directly.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agents.failure_memory import FailureMemory
from metacognition.capability_tracker import CapabilityTracker
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feedback entry
# ---------------------------------------------------------------------------


@dataclass
class FeedbackEntry:
    """One distilled record of a task or run outcome."""

    domain: str
    success: bool
    duration_secs: float = 0.0
    confidence: float = 0.5
    task_title: str = ""
    tool: str = ""
    failure_reason: str = ""
    goal: str = ""
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "success": self.success,
            "duration_secs": round(self.duration_secs, 3),
            "confidence": round(self.confidence, 3),
            "task_title": self.task_title,
            "tool": self.tool,
            "failure_reason": self.failure_reason,
            "goal": self.goal,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackEntry":
        return cls(
            domain=d["domain"], success=d["success"],
            duration_secs=d.get("duration_secs", 0.0), confidence=d.get("confidence", 0.5),
            task_title=d.get("task_title", ""), tool=d.get("tool", ""),
            failure_reason=d.get("failure_reason", ""), goal=d.get("goal", ""),
            recorded_at=d.get("recorded_at", ""),
        )


# ---------------------------------------------------------------------------
# Execution Feedback Loop
# ---------------------------------------------------------------------------

# Crude keyword → domain classifier, used only when the caller doesn't
# supply an explicit domain. Mirrors planning.planner.GoalParser's
# domain detection so the two stay roughly consistent without a hard
# dependency between them.
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "coding": ["code", "implement", "function", "bug", "script", "api", "endpoint", "refactor"],
    "research": ["research", "paper", "survey", "literature", "investigate", "search for"],
    "math": ["calculate", "compute", "equation", "math", "proof", "formula"],
    "writing": ["write", "draft", "essay", "summarize", "article"],
}


def _infer_domain(task_title: str) -> str:
    text = task_title.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return domain
    return "general"


class ExecutionFeedbackLoop:
    """
    Records execution outcomes and fans them out to downstream
    self-knowledge stores.

    Parameters
    ----------
    feedback_file:
        Path to ``execution_feedback.json`` — full feedback log.
    failure_memory:
        Optional ``FailureMemory`` — failures are also recorded here
        (mirroring v0.3.6 behavior, kept for backwards-compatible callers
        that already feed FailureMemory directly).
    capability_tracker:
        Optional ``CapabilityTracker`` — every outcome updates the
        relevant domain's accuracy.
    """

    def __init__(
        self,
        feedback_file: Path,
        failure_memory: Optional[FailureMemory] = None,
        capability_tracker: Optional[CapabilityTracker] = None,
    ) -> None:
        self._file = feedback_file
        self._failure_memory = failure_memory
        self._capability_tracker = capability_tracker
        self._entries: list[FeedbackEntry] = []
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
            self._entries = [FeedbackEntry.from_dict(e) for e in raw]
            log.info("ExecutionFeedbackLoop: loaded %d entr(y/ies).", len(self._entries))
        except Exception as exc:
            log.warning("ExecutionFeedbackLoop: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        # Cap the persisted log to the most recent 1000 entries to keep the file bounded.
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([e.to_dict() for e in self._entries[-1000:]], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_task_outcome(
        self,
        task_title: str,
        success: bool,
        duration_secs: float = 0.0,
        confidence: float = 0.5,
        tool: str = "",
        failure_reason: str = "",
        goal: str = "",
        domain: Optional[str] = None,
    ) -> FeedbackEntry:
        """Record one task-level outcome and fan it out to downstream stores."""
        resolved_domain = domain or _infer_domain(task_title)
        entry = FeedbackEntry(
            domain=resolved_domain, success=success, duration_secs=duration_secs,
            confidence=confidence, task_title=task_title, tool=tool,
            failure_reason=failure_reason, goal=goal,
        )
        self._entries.append(entry)
        self._save()

        if self._capability_tracker is not None:
            self._capability_tracker.record_outcome(resolved_domain, success)

        if not success and self._failure_memory is not None:
            self._failure_memory.record(task_title, tool or "unknown_tool", failure_reason or "unspecified failure", goal=goal)

        return entry

    def record_run_result(self, result, domain: Optional[str] = None) -> list[FeedbackEntry]:
        """
        Record feedback for every task in an ``AgentRunResult.graph``,
        inferring success per-task from final ``TaskStatus``.

        This is the typical integration point: call once after
        ``AgentExecutor.run()`` returns.
        """
        from agents.types import TaskStatus

        entries = []
        for task in result.graph.tasks:
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED):
                continue  # still pending/blocked — nothing to report yet
            success = task.status == TaskStatus.COMPLETED
            entry = self.record_task_outcome(
                task_title=task.title,
                success=success,
                duration_secs=result.duration_secs / max(1, len(result.graph.tasks)),
                confidence=result.agent_state.get("confidence", 0.5) if result.agent_state else 0.5,
                tool=task.tool_hint or "",
                failure_reason=task.error or "",
                goal=result.goal,
                domain=domain,
            )
            entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def recent(self, limit: int = 20, domain: Optional[str] = None) -> list[FeedbackEntry]:
        entries = self._entries
        if domain:
            entries = [e for e in entries if e.domain == domain]
        return entries[-limit:]

    def success_rate(self, domain: Optional[str] = None) -> float:
        entries = self._entries
        if domain:
            entries = [e for e in entries if e.domain == domain]
        if not entries:
            return 0.5
        return sum(1 for e in entries if e.success) / len(entries)

    def mean_confidence(self, domain: Optional[str] = None) -> float:
        entries = self._entries
        if domain:
            entries = [e for e in entries if e.domain == domain]
        if not entries:
            return 0.5
        return sum(e.confidence for e in entries) / len(entries)

    @property
    def count(self) -> int:
        return len(self._entries)
