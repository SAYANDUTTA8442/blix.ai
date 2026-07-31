"""
Planning Layer — Blix v0.3.5  (Modules 1 & 2)

Modules
-------
``GoalParser``        — extracts structured goal intent from natural language
``TaskDecomposer``    — converts a goal into a TaskGraph (subtasks + deps)
``Planner``           — orchestrates GoalParser + TaskDecomposer
``MilestoneTracker``  — syncs TaskGraph progress to GoalTracker (v0.3.2)

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agents.types import Task, TaskGraph, TaskStatus
from llm.base import LLMProvider
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# GoalParser
# ---------------------------------------------------------------------------


@dataclass
class ParsedGoal:
    """Structured representation of a user's goal intent."""

    raw_input: str
    title: str
    description: str
    domain: str = ""         # e.g. "research", "coding", "writing", "analysis"
    complexity: str = "medium"   # "simple" | "medium" | "complex"
    estimated_tasks: int = 3
    requires_web: bool = False
    requires_code: bool = False
    requires_files: bool = False


_DOMAIN_PATTERNS = {
    "research":  re.compile(r"\b(research|paper|literature|survey|study|academic)\b", re.I),
    "coding":    re.compile(r"\b(code|script|implement|build|develop|program|function)\b", re.I),
    "writing":   re.compile(r"\b(write|draft|essay|document|article|report|summary)\b", re.I),
    "analysis":  re.compile(r"\b(analyse|analyze|evaluate|assess|compare|benchmark)\b", re.I),
    "data":      re.compile(r"\b(data|dataset|csv|table|plot|chart|statistics)\b", re.I),
}

_COMPLEXITY_SIGNALS = {
    "simple":  re.compile(r"\b(quick|simple|brief|short|just|only|single)\b", re.I),
    "complex": re.compile(r"\b(comprehensive|full|complete|detailed|thorough|all|entire)\b", re.I),
}


class GoalParser:
    """
    Extracts structured goal intent from a natural-language goal statement.

    Uses heuristic pattern matching (always available) with an optional LLM
    pass for richer intent extraction.
    """

    def __init__(self, llm: Optional[LLMProvider] = None) -> None:
        self._llm = llm

    def parse(self, raw_input: str) -> ParsedGoal:
        if self._llm is not None:
            return self._llm_parse(raw_input)
        return self._heuristic_parse(raw_input)

    def _heuristic_parse(self, raw: str) -> ParsedGoal:
        domain = "general"
        for d, pattern in _DOMAIN_PATTERNS.items():
            if pattern.search(raw):
                domain = d
                break

        complexity = "medium"
        for level, pattern in _COMPLEXITY_SIGNALS.items():
            if pattern.search(raw):
                complexity = level
                break

        task_count = {"simple": 2, "medium": 4, "complex": 6}.get(complexity, 4)

        requires_web = bool(re.search(r"\b(search|find|latest|current|web|online|lookup)\b", raw, re.I))
        requires_code = bool(re.search(r"\b(code|script|program|implement|function|python)\b", raw, re.I))
        requires_files = bool(re.search(r"\b(file|document|pdf|read|write|save|load)\b", raw, re.I))

        # Title: first sentence or first 80 chars
        title = re.split(r"[.!?]", raw.strip())[0][:80].strip()

        return ParsedGoal(
            raw_input=raw,
            title=title or raw[:80],
            description=raw,
            domain=domain,
            complexity=complexity,
            estimated_tasks=task_count,
            requires_web=requires_web,
            requires_code=requires_code,
            requires_files=requires_files,
        )

    def _llm_parse(self, raw: str) -> ParsedGoal:
        prompt = f"""\
Analyse this goal and respond with ONLY a JSON object:
{{
  "title": "short title (≤10 words)",
  "description": "full goal description",
  "domain": "research|coding|writing|analysis|data|general",
  "complexity": "simple|medium|complex",
  "estimated_tasks": 3,
  "requires_web": false,
  "requires_code": false,
  "requires_files": false
}}

Goal: {raw}"""
        try:
            raw_resp = self._llm.generate(prompt).strip()  # type: ignore[union-attr]
            raw_resp = _strip_fence(raw_resp)
            data = json.loads(raw_resp)
            return ParsedGoal(
                raw_input=raw,
                title=str(data.get("title", raw[:80])),
                description=str(data.get("description", raw)),
                domain=str(data.get("domain", "general")),
                complexity=str(data.get("complexity", "medium")),
                estimated_tasks=int(data.get("estimated_tasks", 4)),
                requires_web=bool(data.get("requires_web", False)),
                requires_code=bool(data.get("requires_code", False)),
                requires_files=bool(data.get("requires_files", False)),
            )
        except Exception as exc:
            log.warning("GoalParser LLM failed (%s); using heuristic.", exc)
            return self._heuristic_parse(raw)


# ---------------------------------------------------------------------------
# TaskDecomposer
# ---------------------------------------------------------------------------


_DECOMPOSE_PROMPT = """\
You are a task planner. Decompose the following goal into 2-7 concrete subtasks.

Respond with ONLY a JSON array of task objects:
[
  {{
    "title": "Short task title",
    "description": "Detailed description of what to do",
    "depends_on": [],
    "tool_hint": "web_search|memory_search|llm|python_tool|file_tool|synthesis|reasoning|null"
  }}
]

Rules:
- Each task should be independently executable
- Use "depends_on" to list task indices (0-based) that must complete first
- tool_hint should reflect which tool best fits each task
- Keep tasks concrete and actionable

Goal: {goal}
Domain: {domain}
Complexity: {complexity}
"""

_HEURISTIC_TEMPLATES: dict[str, list[dict]] = {
    "research": [
        {"title": "Search for relevant information", "description": "Search the web for recent work on {goal}.", "tool_hint": "web_search", "depends_on": []},
        {"title": "Review existing knowledge", "description": "Search memory for past information on {goal}.", "tool_hint": "memory_search", "depends_on": []},
        {"title": "Analyse findings", "description": "Analyse the gathered information about {goal}.", "tool_hint": "llm", "depends_on": [0, 1]},
        {"title": "Synthesise knowledge report", "description": "Synthesise a unified report on {goal}.", "tool_hint": "synthesis", "depends_on": [2]},
    ],
    "coding": [
        {"title": "Understand requirements", "description": "Analyse the coding requirements for {goal}.", "tool_hint": "llm", "depends_on": []},
        {"title": "Design solution", "description": "Design a solution approach for {goal}.", "tool_hint": "llm", "depends_on": [0]},
        {"title": "Implement solution", "description": "Write Python code to implement {goal}.", "tool_hint": "python_tool", "depends_on": [1]},
        {"title": "Save result", "description": "Save the implementation result to memory.", "tool_hint": "memory_write", "depends_on": [2]},
    ],
    "writing": [
        {"title": "Research topic", "description": "Gather information about {goal}.", "tool_hint": "memory_search", "depends_on": []},
        {"title": "Create outline", "description": "Create a structured outline for {goal}.", "tool_hint": "llm", "depends_on": [0]},
        {"title": "Write draft", "description": "Write the full draft for {goal}.", "tool_hint": "llm", "depends_on": [1]},
    ],
    "analysis": [
        {"title": "Gather data", "description": "Collect relevant data for {goal}.", "tool_hint": "memory_search", "depends_on": []},
        {"title": "Analyse data", "description": "Analyse the collected data for {goal}.", "tool_hint": "llm", "depends_on": [0]},
        {"title": "Synthesise findings", "description": "Synthesise findings for {goal}.", "tool_hint": "synthesis", "depends_on": [1]},
    ],
    "general": [
        {"title": "Gather context", "description": "Search for relevant context on {goal}.", "tool_hint": "memory_search", "depends_on": []},
        {"title": "Process information", "description": "Process gathered information for {goal}.", "tool_hint": "llm", "depends_on": [0]},
        {"title": "Produce result", "description": "Produce the final result for {goal}.", "tool_hint": "llm", "depends_on": [1]},
    ],
}


class TaskDecomposer:
    """
    Transforms a ``ParsedGoal`` into a ``TaskGraph``.

    Uses domain-specific heuristic templates (offline) or LLM decomposition.
    """

    def __init__(self, llm: Optional[LLMProvider] = None) -> None:
        self._llm = llm

    def decompose(self, goal: ParsedGoal) -> TaskGraph:
        if self._llm is not None:
            return self._llm_decompose(goal)
        return self._heuristic_decompose(goal)

    def _heuristic_decompose(self, goal: ParsedGoal) -> TaskGraph:
        template = _HEURISTIC_TEMPLATES.get(goal.domain, _HEURISTIC_TEMPLATES["general"])
        graph = TaskGraph(goal=goal.raw_input)
        task_ids: list[str] = []

        for idx, tpl in enumerate(template):
            desc = tpl["description"].format(goal=goal.title)
            deps = [task_ids[i] for i in tpl.get("depends_on", []) if i < len(task_ids)]
            task = Task(
                title=tpl["title"],
                description=desc,
                tool_hint=tpl.get("tool_hint"),
                depends_on=deps,
            )
            graph.add_task(task)
            task_ids.append(task.task_id)
        return graph

    def _llm_decompose(self, goal: ParsedGoal) -> TaskGraph:
        prompt = _DECOMPOSE_PROMPT.format(
            goal=goal.raw_input, domain=goal.domain, complexity=goal.complexity
        )
        try:
            raw = _strip_fence(self._llm.generate(prompt).strip())  # type: ignore[union-attr]
            task_list = json.loads(raw)
            if not isinstance(task_list, list):
                raise ValueError("expected JSON array")
            graph = TaskGraph(goal=goal.raw_input)
            task_ids: list[str] = []
            for item in task_list[:8]:
                raw_deps = item.get("depends_on", [])
                deps = [task_ids[i] for i in raw_deps if isinstance(i, int) and i < len(task_ids)]
                task = Task(
                    title=str(item.get("title", "Task"))[:80],
                    description=str(item.get("description", "")),
                    tool_hint=item.get("tool_hint") or None,
                    depends_on=deps,
                )
                graph.add_task(task)
                task_ids.append(task.task_id)
            return graph
        except Exception as exc:
            log.warning("TaskDecomposer LLM failed (%s); using heuristic.", exc)
            return self._heuristic_decompose(goal)


# ---------------------------------------------------------------------------
# Planner — orchestrates GoalParser + TaskDecomposer
# ---------------------------------------------------------------------------


class Planner:
    """
    Combines GoalParser and TaskDecomposer into a single planning step.

        natural_language_goal
            ↓ GoalParser
        ParsedGoal
            ↓ TaskDecomposer
        TaskGraph
    """

    def __init__(self, llm: Optional[LLMProvider] = None) -> None:
        self._parser = GoalParser(llm=llm)
        self._decomposer = TaskDecomposer(llm=llm)

    def plan(self, goal_text: str) -> tuple[ParsedGoal, TaskGraph]:
        """
        Parse and decompose a goal into an executable TaskGraph.

        Returns
        -------
        (ParsedGoal, TaskGraph)
        """
        parsed = self._parser.parse(goal_text)
        graph = self._decomposer.decompose(parsed)
        log.info(
            "Planner: '%s' → %d task(s), domain=%s, complexity=%s",
            parsed.title[:50], len(graph.tasks), parsed.domain, parsed.complexity,
        )
        return parsed, graph


# ---------------------------------------------------------------------------
# MilestoneTracker — syncs TaskGraph → GoalTracker (v0.3.2)
# ---------------------------------------------------------------------------


class MilestoneTracker:
    """
    Keeps a v0.3.2 ``GoalTracker`` goal in sync with a ``TaskGraph``.

    Call ``sync()`` after each task completion to update milestone status.

    Parameters
    ----------
    goal_tracker:
        The v0.3.2 ``GoalTracker`` instance.
    """

    def __init__(self, goal_tracker: object) -> None:
        self._gt = goal_tracker

    def create_goal_from_graph(self, graph: TaskGraph) -> str:
        """
        Create a GoalTracker goal mirroring the TaskGraph.
        Returns the created goal_id.
        """
        goal = self._gt.create_goal(  # type: ignore[union-attr]
            title=graph.goal[:100],
            description=f"Agent execution: {len(graph.tasks)} planned tasks.",
            priority=2,
        )
        for task in graph.tasks:
            self._gt.add_milestone(goal.goal_id, task.title)  # type: ignore[union-attr]
        return goal.goal_id

    def sync(self, goal_id: str, graph: TaskGraph) -> None:
        """Mark completed/failed tasks as milestones in GoalTracker."""
        for task in graph.tasks:
            if task.status == TaskStatus.COMPLETED:
                try:
                    self._gt.complete_item(goal_id, task.title)  # type: ignore[union-attr]
                except Exception:
                    pass

    def update_blockers(self, goal_id: str, graph: TaskGraph) -> None:
        """Add failed tasks as blockers in GoalTracker."""
        for task in graph.tasks:
            if task.status == TaskStatus.FAILED and task.error:
                try:
                    self._gt.add_blocker(goal_id, task.error[:100])  # type: ignore[union-attr]
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
