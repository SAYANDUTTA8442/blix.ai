"""
Reflection Loop — Blix v0.3.5  (Module 7)

After every task execution, the reflection loop:

1. Evaluates the Observation quality
2. Decides whether to accept, retry, or skip
3. Generates a reflection note (insight)
4. Persists to ExecutionHistory
5. Updates long-term memory (via ReflectionEngine or MemoryManager)

This closes the cognitive loop:

    Action → Result → Evaluation → Improvement

The improvement manifests as:
    * Retry hints fed back to the Executor
    * Reflection notes stored in ReflectionEngine
    * Execution history enriching future consolidation

Python 3.10 compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agents.types import ExecutionHistoryEntry, Observation, Task, TaskStatus
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reflection decision
# ---------------------------------------------------------------------------


@dataclass
class ReflectionDecision:
    """
    The output of one reflection cycle for a single task.

    Fields
    ------
    action:
        "accept" — use this result and mark task complete
        "retry"  — attempt the task again (with retry_hint)
        "skip"   — mark task as skipped (e.g. too many failures)
    retry_hint:
        Modification hint passed to the Executor on retry.
    note:
        Human-readable reflection note (persisted to history).
    quality_score:
        Adopted from the Observation.
    """

    action: str              # "accept" | "retry" | "skip"
    retry_hint: str = ""
    note: str = ""
    quality_score: float = 0.5

    def should_retry(self) -> bool:
        return self.action == "retry"

    def should_skip(self) -> bool:
        return self.action == "skip"

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "retry_hint": self.retry_hint,
            "note": self.note,
            "quality_score": round(self.quality_score, 3),
        }


# ---------------------------------------------------------------------------
# Reflection loop
# ---------------------------------------------------------------------------


class ReflectionLoop:
    """
    Evaluates task execution observations and drives improvement.

    Parameters
    ----------
    history_file:
        Path to ``execution_history.json`` for persistence.
    reflection_engine:
        Optional v0.3.2 ``ReflectionEngine`` — accepts reflection notes
        as insights.
    memory_manager:
        Optional ``MemoryManager`` — persists high-value results as memories.
    llm:
        Optional LLM for richer reflection notes.
    max_retries:
        Maximum retry attempts per task before skipping.
    quality_threshold:
        Observations scoring below this trigger a retry suggestion.
    """

    def __init__(
        self,
        history_file: Path,
        reflection_engine: Optional[object] = None,
        memory_manager: Optional[object] = None,
        llm: Optional[object] = None,
        max_retries: int = 2,
        quality_threshold: float = 0.3,
    ) -> None:
        self._file = history_file
        self._reflection_engine = reflection_engine
        self._mm = memory_manager
        self._llm = llm
        self._max_retries = max_retries
        self._quality_threshold = quality_threshold
        self._history: list[ExecutionHistoryEntry] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._history = [ExecutionHistoryEntry(**e) for e in data]
            log.info("ReflectionLoop: loaded %d history entries.", len(self._history))
        except Exception as exc:
            log.warning("ReflectionLoop: load failed (%s)", exc)

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with self._file.open("w", encoding="utf-8") as fh:
            json.dump([e.to_dict() for e in self._history[-500:]], fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Core reflection
    # ------------------------------------------------------------------

    def reflect(
        self,
        task: Task,
        observation: Observation,
        goal: str = "",
    ) -> ReflectionDecision:
        """
        Evaluate one task observation and return a ``ReflectionDecision``.

        The decision drives whether the Executor accepts, retries, or skips.
        """
        quality = observation.quality_score
        attempts = task.attempts

        # Decide action
        if observation.success and quality >= self._quality_threshold:
            action = "accept"
        elif observation.retry_suggested and attempts < self._max_retries:
            action = "retry"
        elif attempts >= self._max_retries:
            action = "skip"
            log.info("ReflectionLoop: task %s exceeded max retries (%d) → skip", task.task_id, self._max_retries)
        else:
            action = "accept" if quality > 0.1 else "retry"

        # Generate reflection note
        note = self._generate_note(task, observation, action)
        retry_hint = observation.retry_hint if action == "retry" else ""

        decision = ReflectionDecision(
            action=action,
            retry_hint=retry_hint,
            note=note,
            quality_score=quality,
        )

        # Persist to history
        entry = ExecutionHistoryEntry(
            goal=goal,
            task_id=task.task_id,
            task_title=task.title,
            tool=observation.tool_name,
            result_summary=observation.summary[:300],
            success=observation.success,
            quality_score=quality,
            reflection_note=note,
        )
        self._history.append(entry)
        self._save()

        # Downstream updates
        self._update_reflection_engine(note, task.title, quality)
        if action == "accept" and quality >= 0.5:
            self._update_memory(task, observation, goal)

        log.info(
            "ReflectionLoop: task=%s action=%s quality=%.2f retries=%d",
            task.task_id, action, quality, attempts,
        )
        return decision

    def _generate_note(
        self, task: Task, obs: Observation, action: str
    ) -> str:
        """Generate a human-readable reflection note."""
        if self._llm is not None:
            return self._llm_note(task, obs, action)
        return self._heuristic_note(task, obs, action)

    def _heuristic_note(self, task: Task, obs: Observation, action: str) -> str:
        quality_label = "high" if obs.quality_score >= 0.7 else "medium" if obs.quality_score >= 0.4 else "low"
        if action == "accept":
            return (
                f"Task '{task.title}' completed via {obs.tool_name} "
                f"with {quality_label} quality ({obs.quality_score:.2f}). "
                + (obs.retry_hint or "")
            )
        elif action == "retry":
            return (
                f"Task '{task.title}' output was {quality_label} quality. "
                f"Retrying: {obs.retry_hint}"
            )
        else:
            return (
                f"Task '{task.title}' skipped after {task.attempts} attempt(s). "
                f"Last error: {task.error[:100]}" if task.error else
                f"Task '{task.title}' skipped after {task.attempts} attempt(s)."
            )

    def _llm_note(self, task: Task, obs: Observation, action: str) -> str:
        prompt = (
            f"Write one concise sentence reflecting on this agent action:\n"
            f"Task: {task.title}\n"
            f"Tool: {obs.tool_name}\n"
            f"Success: {obs.success}\n"
            f"Quality: {obs.quality_score:.2f}\n"
            f"Decision: {action}\n"
            f"Observation: {obs.summary[:200]}\n"
            "Write only the reflection sentence."
        )
        try:
            return self._llm.generate(prompt).strip()[:300]  # type: ignore[union-attr]
        except Exception as exc:
            log.warning("ReflectionLoop LLM note failed (%s)", exc)
            return self._heuristic_note(task, obs, action)

    def _update_reflection_engine(self, note: str, task_title: str, quality: float) -> None:
        if self._reflection_engine is None or not note:
            return
        try:
            from reflection.reflection_engine import ReflectionScope
            self._reflection_engine.reflect(  # type: ignore[union-attr]
                ReflectionScope.SESSION,
                f"agent_task_{task_title[:20]}",
                note,
            )
        except Exception as exc:
            log.debug("ReflectionLoop: reflection_engine update failed (%s)", exc)

    def _update_memory(self, task: Task, obs: Observation, goal: str) -> None:
        """Persist high-quality task results to long-term memory."""
        if self._mm is None:
            return
        try:
            input_text = f"[Agent] {goal}: {task.title}"
            output_text = obs.summary[:500]
            self._mm.add_memory(input_text, output_text)  # type: ignore[union-attr]
        except Exception as exc:
            log.debug("ReflectionLoop: memory update failed (%s)", exc)

    # ------------------------------------------------------------------
    # History retrieval
    # ------------------------------------------------------------------

    def get_history(self, goal: Optional[str] = None, limit: int = 50) -> list[ExecutionHistoryEntry]:
        history = self._history
        if goal:
            history = [e for e in history if goal.lower() in e.goal.lower()]
        return history[-limit:]

    def success_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(1 for e in self._history if e.success) / len(self._history)

    def mean_quality(self) -> float:
        if not self._history:
            return 0.0
        return sum(e.quality_score for e in self._history) / len(self._history)

    @property
    def history_count(self) -> int:
        return len(self._history)
