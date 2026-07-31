"""
Verification Engine — Blix v0.3.6  (Upgrade 2)

Inserts a verification gate between execution and acceptance:

    v0.3.5:  Execute → Observe → Reflect (accept/retry/skip)
    v0.3.6:  Execute → Observe → VERIFY → Reflect (accept/retry/skip)

Verification runs structural/semantic checks on a task's output BEFORE
the Reflection loop decides whether to accept it. A task that "looks"
successful (non-empty output, no error) can still FAIL verification —
e.g. a "create an API" task whose output never mentions a route, or a
JSON-producing task whose output isn't valid JSON.

Verifiers are pluggable: ``Verifier`` is the ABC, ``VerificationEngine``
dispatches to all verifiers and aggregates results. Built-in verifiers:

* ``NonEmptyVerifier``      — output isn't trivially empty
* ``SchemaVerifier``        — output matches an expected JSON schema (if task.metadata["expected_schema"] set)
* ``KeywordPresenceVerifier`` — output mentions required keywords (task.metadata["required_keywords"])
* ``CodeSyntaxVerifier``    — output (or task result) is syntactically valid Python, when applicable

Python 3.10 compatible.
"""

from __future__ import annotations

import ast
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agents.types import ExecutionResult, Task
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"   # verifier not applicable to this task


@dataclass
class VerificationCheck:
    """Result of a single verifier."""

    verifier_name: str
    status: VerificationStatus
    message: str = ""

    def to_dict(self) -> dict:
        return {"verifier": self.verifier_name, "status": self.status.value, "message": self.message}


@dataclass
class VerificationReport:
    """Aggregated result of all verifiers run on one task result."""

    task_id: str
    checks: list[VerificationCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no verifier explicitly FAILED (SKIPPED checks don't count against it)."""
        return not any(c.status == VerificationStatus.FAILED for c in self.checks)

    @property
    def failed_checks(self) -> list[VerificationCheck]:
        return [c for c in self.checks if c.status == VerificationStatus.FAILED]

    def summary(self) -> str:
        if self.passed:
            ran = sum(1 for c in self.checks if c.status != VerificationStatus.SKIPPED)
            return f"Verification passed ({ran} check(s))."
        reasons = "; ".join(c.message for c in self.failed_checks)
        return f"Verification failed: {reasons}"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary(),
        }


# ---------------------------------------------------------------------------
# Verifier ABC
# ---------------------------------------------------------------------------


class Verifier(ABC):
    """Base class for all verifiers."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def applies_to(self, task: Task, result: ExecutionResult) -> bool:
        """Whether this verifier should run for the given task/result."""
        ...

    @abstractmethod
    def verify(self, task: Task, result: ExecutionResult) -> VerificationCheck:
        ...


# ---------------------------------------------------------------------------
# Built-in verifiers
# ---------------------------------------------------------------------------


class NonEmptyVerifier(Verifier):
    """Fails if the output is empty or trivially short."""

    def __init__(self, min_length: int = 5) -> None:
        self._min_length = min_length

    @property
    def name(self) -> str:
        return "non_empty"

    def applies_to(self, task: Task, result: ExecutionResult) -> bool:
        return True  # always applicable

    def verify(self, task: Task, result: ExecutionResult) -> VerificationCheck:
        if len((result.output or "").strip()) >= self._min_length:
            return VerificationCheck(self.name, VerificationStatus.PASSED)
        return VerificationCheck(
            self.name, VerificationStatus.FAILED,
            f"Output is empty or too short (<{self._min_length} chars).",
        )


class SchemaVerifier(Verifier):
    """
    Verifies that the output is valid JSON and matches an expected
    set of required top-level keys.

    Activated when ``task.metadata["expected_schema"]`` is a list of
    required key names.
    """

    @property
    def name(self) -> str:
        return "schema"

    def applies_to(self, task: Task, result: ExecutionResult) -> bool:
        return "expected_schema" in task.metadata

    def verify(self, task: Task, result: ExecutionResult) -> VerificationCheck:
        required_keys = task.metadata.get("expected_schema", [])
        try:
            data = json.loads(result.output)
        except (json.JSONDecodeError, TypeError):
            return VerificationCheck(self.name, VerificationStatus.FAILED, "Output is not valid JSON.")

        if not isinstance(data, dict):
            return VerificationCheck(self.name, VerificationStatus.FAILED, "Output JSON is not an object.")

        missing = [k for k in required_keys if k not in data]
        if missing:
            return VerificationCheck(
                self.name, VerificationStatus.FAILED,
                f"Missing required schema key(s): {missing}",
            )
        return VerificationCheck(self.name, VerificationStatus.PASSED)


class KeywordPresenceVerifier(Verifier):
    """
    Verifies the output mentions all required keywords.

    Activated when ``task.metadata["required_keywords"]`` is a non-empty list.
    """

    @property
    def name(self) -> str:
        return "keyword_presence"

    def applies_to(self, task: Task, result: ExecutionResult) -> bool:
        return bool(task.metadata.get("required_keywords"))

    def verify(self, task: Task, result: ExecutionResult) -> VerificationCheck:
        keywords = task.metadata.get("required_keywords", [])
        output_lower = (result.output or "").lower()
        missing = [k for k in keywords if k.lower() not in output_lower]
        if missing:
            return VerificationCheck(
                self.name, VerificationStatus.FAILED,
                f"Missing required keyword(s): {missing}",
            )
        return VerificationCheck(self.name, VerificationStatus.PASSED)


class CodeSyntaxVerifier(Verifier):
    """
    Verifies that Python code blocks in the output are syntactically valid.

    Applies to tasks where ``task.tool_hint == "python_tool"`` or output
    contains a ```python fence.
    """

    @property
    def name(self) -> str:
        return "code_syntax"

    def applies_to(self, task: Task, result: ExecutionResult) -> bool:
        return task.tool_hint == "python_tool" or "```python" in (result.output or "")

    def verify(self, task: Task, result: ExecutionResult) -> VerificationCheck:
        code = self._extract_code(result.output or "")
        if not code:
            return VerificationCheck(self.name, VerificationStatus.SKIPPED, "No code block found.")
        try:
            ast.parse(code)
            return VerificationCheck(self.name, VerificationStatus.PASSED)
        except SyntaxError as exc:
            return VerificationCheck(
                self.name, VerificationStatus.FAILED, f"Syntax error: {exc}",
            )

    def _extract_code(self, text: str) -> str:
        m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Verification Engine
# ---------------------------------------------------------------------------


class VerificationEngine:
    """
    Runs all applicable verifiers on a task's execution result.

    Parameters
    ----------
    verifiers:
        List of ``Verifier`` instances. Defaults to all built-ins.
    """

    def __init__(self, verifiers: Optional[list[Verifier]] = None) -> None:
        self._verifiers = verifiers if verifiers is not None else [
            NonEmptyVerifier(),
            SchemaVerifier(),
            KeywordPresenceVerifier(),
            CodeSyntaxVerifier(),
        ]

    def verify(self, task: Task, result: ExecutionResult) -> VerificationReport:
        checks: list[VerificationCheck] = []
        for verifier in self._verifiers:
            if verifier.applies_to(task, result):
                try:
                    checks.append(verifier.verify(task, result))
                except Exception as exc:
                    log.warning("VerificationEngine: verifier '%s' raised (%s)", verifier.name, exc)
                    checks.append(VerificationCheck(
                        verifier.name, VerificationStatus.FAILED, f"Verifier error: {exc}",
                    ))
            else:
                checks.append(VerificationCheck(verifier.name, VerificationStatus.SKIPPED))

        report = VerificationReport(task_id=task.task_id, checks=checks)
        log.debug("VerificationEngine: task=%s passed=%s", task.task_id, report.passed)
        return report

    def add_verifier(self, verifier: Verifier) -> None:
        self._verifiers.append(verifier)

    @property
    def verifier_names(self) -> list[str]:
        return [v.name for v in self._verifiers]
