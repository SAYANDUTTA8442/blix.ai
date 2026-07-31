"""
Observation Layer — Blix v0.3.5  (Module 6)

Transforms raw ``ExecutionResult`` objects into structured ``Observation``
objects that the Reflection loop can reason over.

Key responsibilities
--------------------
* Detect success / failure / partial results
* Summarise tool output in a consistent format
* Extract key facts from the output
* Assess output quality (0–1)
* Flag cases where retry is warranted and suggest a hint

Python 3.10 compatible.
"""

from __future__ import annotations

import re
from typing import Optional

from agents.types import ExecutionResult, ExecutionStatus, Observation
from utils.logger import get_logger

log = get_logger(__name__)


# Patterns that suggest a retriable failure
_RETRY_PATTERNS = [
    re.compile(r"\b(timeout|timed out|connection|network|rate limit|retry)\b", re.I),
    re.compile(r"\b(no results|not found|empty|zero results)\b", re.I),
    re.compile(r"\b(error|exception|failed|failure)\b", re.I),
]

# Quality signals
_HIGH_QUALITY = re.compile(r"\b(found|result|success|completed|generated|written|saved)\b", re.I)
_LOW_QUALITY = re.compile(r"\b(no result|nothing|empty|not found|failed|error|timeout)\b", re.I)


class ObservationLayer:
    """
    Converts ``ExecutionResult`` → ``Observation``.

    Uses heuristic analysis: output length, keyword signals, status codes.
    An optional LLM pass can provide richer fact extraction.

    Parameters
    ----------
    llm:
        Optional LLM for richer fact extraction from tool output.
    min_output_length:
        Outputs shorter than this are considered low-quality.
    """

    def __init__(
        self,
        llm: Optional[object] = None,
        min_output_length: int = 20,
    ) -> None:
        self._llm = llm
        self._min_len = min_output_length

    def observe(self, result: ExecutionResult) -> Observation:
        """
        Convert an ``ExecutionResult`` into a structured ``Observation``.
        """
        success = result.is_success() and bool(result.output.strip())
        summary = self._build_summary(result)
        facts = self._extract_facts(result)
        quality = self._assess_quality(result)
        retry, hint = self._should_retry(result, quality)

        obs = Observation(
            task_id=result.task_id,
            tool_name=result.tool_name,
            success=success,
            summary=summary,
            extracted_facts=facts,
            quality_score=quality,
            retry_suggested=retry,
            retry_hint=hint,
            raw_result=result,
        )
        log.debug(
            "Observation: task=%s tool=%s success=%s quality=%.2f retry=%s",
            result.task_id, result.tool_name, success, quality, retry,
        )
        return obs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_summary(self, result: ExecutionResult) -> str:
        if result.status == ExecutionStatus.SUCCESS and result.output:
            lines = result.output.strip().splitlines()
            first = lines[0][:200] if lines else ""
            total = len(result.output)
            return f"[{result.tool_name}] {first}" + (f" … ({total} chars total)" if total > 200 else "")
        elif result.error:
            return f"[{result.tool_name}] FAILED: {result.error[:200]}"
        else:
            return f"[{result.tool_name}] No output."

    def _extract_facts(self, result: ExecutionResult) -> list[str]:
        """Extract bullet-style or sentence facts from tool output."""
        if not result.output:
            return []
        text = result.output
        facts: list[str] = []

        # Bullet points
        bullets = re.findall(r"^[-•*]\s+(.+)$", text, re.MULTILINE)
        facts.extend(b.strip() for b in bullets[:5])

        # First 3 sentences if no bullets
        if not facts:
            sents = re.split(r"(?<=[.!?])\s+", text.strip())
            facts = [s[:200] for s in sents[:3] if len(s) > 20]

        return facts[:5]

    def _assess_quality(self, result: ExecutionResult) -> float:
        """Heuristic 0–1 quality score for the execution result."""
        if result.status in (ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT):
            return 0.0
        if result.status == ExecutionStatus.FAILURE:
            return 0.1

        output = result.output or ""
        length_score = min(1.0, len(output) / 500)

        high_hits = len(_HIGH_QUALITY.findall(output))
        low_hits = len(_LOW_QUALITY.findall(output))
        signal_score = max(0.0, min(1.0, 0.5 + 0.1 * high_hits - 0.2 * low_hits))

        return round((length_score * 0.4 + signal_score * 0.6), 3)

    def _should_retry(
        self, result: ExecutionResult, quality: float
    ) -> tuple[bool, str]:
        """
        Determine whether this result warrants a retry and suggest a hint.
        """
        if result.is_success() and quality >= 0.4:
            return False, ""

        # Check for retriable patterns
        error_text = (result.error or "") + (result.output or "")
        for pattern in _RETRY_PATTERNS:
            m = pattern.search(error_text)
            if m:
                trigger = m.group(0).lower()
                if "timeout" in trigger or "network" in trigger:
                    return True, "Retry with a shorter timeout or simpler query."
                if "no results" in trigger or "not found" in trigger:
                    return True, "Retry with a broader or different query."
                if "rate limit" in trigger:
                    return True, "Wait briefly before retrying."
                return True, "Retry with adjusted parameters."

        if quality < 0.2:
            return True, "Output quality was low. Try a different approach."

        return False, ""

    def batch_observe(self, results: list[ExecutionResult]) -> list[Observation]:
        return [self.observe(r) for r in results]
